// H4 추세(trend_trader.py) 전략 모듈. Infinox_Bots.mq4가 include
// 종목마다 CTrend 하나 (금 = 차트 종목, 이더리움 = ETHUSD 등). 매직·상태 파일을 따로 씀
// - 4시간봉 완성마다 판단. 종가가 직전 25봉 최고가 위면 매수, 최저가 아래면 매도
// - 매수는 종가가 직전 12봉 최저가 아래, 매도는 직전 12봉 최고가 위면 청산 (청산한 봉에서는 새로 진입 안 함)
// - 손절: 진입가 ± 3 x ATR(14, 4시간봉) 서버 설정, 익절 없음, 주말에도 보유
// - 수량: 손절 시 잔고의 TrendRiskPct% 손실 (0.01 내림, 최소 0.01, 최대 TrendMaxLot)
// - 구름 필터: 돌파 방향이 4시간봉 일목 구름(9/26/52, 지금 봉 위치에 그려진 구름)과 같을 때만 진입.
//   아니면 그림자 포지션: 진입한 것처럼 청산 신호나 손절가(± 3 ATR)까지 새 진입 안 함 (trend_trader.py와 같음)
// 상태: MQL4\Files\<상태 파일> (그림자 포지션)

#define T_N_ENTRY 25
#define T_N_EXIT 12
#define T_SL_ATR 3.0
#define T_SLIP_USD 0.50
#define T_CLOUD_FILTER true

class CTrend : public CStrat
{
public:
   string   sym, tag, st;
   int      magic;
   bool     ok;
   datetime okTry;
   int      ticket;
   datetime lastBar;
   double   openLot, openPx;
   int      openSide;
   bool     trendExit;
   int      shSide;                  // 그림자 포지션 방향 (0 = 없음)
   double   shSl, shEntry;

   void Setup(string s, string panelName, string file, int mg, bool on)
   {
      sym = s;
      name = panelName;
      tag = panelName;
      st = file;
      magic = mg;
      defOn = on;
      ticket = -1;
      lastBar = 0;
      shSide = 0;
      okTry = TimeLocal();
      ok = UseSymbol(sym);
      if (!ok)
         Print("[", tag, "] 종목 ", sym, "을 찾을 수 없어 이 전략은 쉽니다");
   }

   int    Magic() { return magic; }
   string Sym() { return sym; }
   double Bid1() { return MarketInfo(sym, MODE_BID); }
   double Ask1() { return MarketInfo(sym, MODE_ASK); }

   string Desc()
   {
      return StringFormat("%s 4시간봉 %d봉 돌파 추세추종, %d봉 반대 돌파 청산, 손절 %g ATR, 거래당 위험 %g%%%s", sym,
                          T_N_ENTRY, T_N_EXIT, T_SL_ATR, TrendRiskPct, T_CLOUD_FILTER ? ", 4시간봉 구름 방향과 같을 때만 진입" : "");
   }

   void SaveShadow()
   {
      StSet(st, "sh_side", IntegerToString(shSide));
      StSet(st, "sh_sl", DoubleToString(shSl, 2));
      StSet(st, "sh_entry", DoubleToString(shEntry, 2));
   }

   // 그림자 끝: 가상 결과(R)를 알리고 지움
   void EndShadow(string why, double px)
   {
      double risk = MathAbs(shEntry - shSl);
      double r = risk > 0 ? (px - shEntry) * shSide / risk : 0;
      Print(StringFormat("[%s] 그림자 %s 끝 (%s) %.2f → %.2f, 가상 %+.2fR", tag, shSide == 1 ? "BUY" : "SELL", why, shEntry, px, r));
      Send(tag, StringFormat("👻 그림자 %s 끝 (%s)\n%.2f → %.2f, 가상 결과 %+.2fR (%s)\n다음 신호부터 다시 진입합니다",
                             shSide == 1 ? "BUY" : "SELL", why, shEntry, px, r, r > 0 ? "건너뛴 게 손해" : "건너뛴 게 이득"));
      shSide = 0;
      SaveShadow();
   }

   // 마지막 완성 봉 위치에 그려진 구름 (선행1·2를 26봉 민 값). 데이터가 모자라면 false
   bool Cloud(const double &h[], const double &l[], int n, double &top, double &bot)
   {
      int k = n - 1 - 26;             // 26봉 전 봉에서 계산한 선행값
      if (k < 52)
         return false;
      double h9 = h[k], l9 = l[k], h26 = h[k], l26 = l[k], h52 = h[k], l52 = l[k];
      for (int i = k - 51; i <= k; i++)
      {
         if (i > k - 9)
         {
            h9 = MathMax(h9, h[i]);
            l9 = MathMin(l9, l[i]);
         }
         if (i > k - 26)
         {
            h26 = MathMax(h26, h[i]);
            l26 = MathMin(l26, l[i]);
         }
         h52 = MathMax(h52, h[i]);
         l52 = MathMin(l52, l[i]);
      }
      double a = ((h9 + l9) / 2 + (h26 + l26) / 2) / 2, b = (h52 + l52) / 2;
      top = MathMax(a, b);
      bot = MathMin(a, b);
      return true;
   }

   void CheckClosed()
   {
      if (ticket < 0 || MyTicketSym(sym, magic) == ticket)
         return;
      double profit, px;
      string reason;
      if (!ClosedInfo(ticket, profit, px, reason))
         return;
      if (trendExit || reason == "CLOSE")
         reason = "추세종료";
      double bal = AccountBalance();
      double pct = bal - profit > 0 ? profit / (bal - profit) * 100 : 0;
      Send(tag, StringFormat("%s %s %s %.2f lot\n%.2f → %.2f\n손익 %+.2f$ (%+.2f%%)\n%s\n%s",
                             profit > 0 ? "✅" : "❌", reason, openSide == 1 ? "BUY" : "SELL", openLot, openPx, px, profit, pct,
                             AccountLine(), SummaryLine(magic, sym)));
      ticket = -1;
   }

   void Place(int side, double atr)
   {
      double slDist = T_SL_ATR * atr;
      double raw = AccountBalance() * TrendRiskPct / 100 / (slDist * PerLot(sym));
      double step = MarketInfo(sym, MODE_LOTSTEP);
      double lot = MathFloor(raw / step + 1e-9) * step;
      bool forced = lot < MarketInfo(sym, MODE_MINLOT);
      lot = NormalizeDouble(MathMin(MathMin(MathMax(lot, MarketInfo(sym, MODE_MINLOT)), TrendMaxLot), MarketInfo(sym, MODE_MAXLOT)), 2);
      if (!MarginOkSym(tag, sym, lot))
         return;
      double fill;
      int tk = OpenMarketSym(tag, sym, side, lot, slDist, 0, magic, "trend H4 N25", T_SLIP_USD, fill);
      if (tk < 0)
         return;
      ticket = tk;
      openLot = lot;
      openPx = fill;
      openSide = side;
      trendExit = false;
      double risk = slDist * lot * PerLot(sym);
      Send(tag, StringFormat("%s 진입 %.2f lot\n가격 %.2f | 손절 %.2f (익절 없음, 추세 끝나면 청산)\n손절 시 손실 약 %.0f$ (%.2f%%)%s\n%s\n%s",
                             side == 1 ? "🟢 매수" : "🔴 매도", lot, fill, fill - side * slDist, risk, risk / AccountBalance() * 100,
                             forced ? "\n⚠️ 최소 수량이라 목표 위험보다 큼" : "", AccountLine(), SummaryLine(magic, sym)));
   }

   // 청산 신호는 꺼져 있어도 처리, 진입은 on일 때만
   void OnBar(datetime bar, bool on)
   {
      datetime t[];
      double h[], l[], c[], a[];
      int n = LoadBarsSym(sym, PERIOD_H4, 300, t, h, l, c);
      if (n <= T_N_ENTRY + 20 || t[n - 1] != bar)
         return;
      double cl = c[n - 1];
      double hh = h[n - 2], ll = l[n - 2];
      for (int i = n - 1 - T_N_ENTRY; i < n - 1; i++)
      {
         hh = MathMax(hh, h[i]);
         ll = MathMin(ll, l[i]);
      }
      double xh = h[n - 2], xl = l[n - 2];
      for (int i = n - 1 - T_N_EXIT; i < n - 1; i++)
      {
         xh = MathMax(xh, h[i]);
         xl = MathMin(xl, l[i]);
      }
      Atr(h, l, c, n, 14, a);
      double atr = a[n - 1];
      int entry = cl > hh ? 1 : cl < ll ? -1 : 0;
      Print(StringFormat("[%s] 4시간봉 %s: 종가 %.2f | 진입선 %.2f~%.2f | 청산선 %.2f~%.2f | ATR %.2f%s", tag,
                         TimeToString(bar, TIME_DATE | TIME_MINUTES), cl, ll, hh, xl, xh, atr,
                         entry == 0 ? "" : entry == 1 ? " -> 진입 신호 매수" : " -> 진입 신호 매도"));
      // 1) 청산 신호 (그림자 포지션 포함)
      bool exited = false;
      if (shSide != 0 && ((shSide == 1 && cl < xl) || (shSide == -1 && cl > xh)))
      {
         RefreshRates();
         EndShadow("추세종료", shSide == 1 ? Bid1() : Ask1());
         exited = true;
      }
      int tk = MyTicketSym(sym, magic);
      if (tk >= 0 && OrderSelect(tk, SELECT_BY_TICKET))
      {
         bool isBuy = OrderType() == OP_BUY;
         if ((isBuy && cl < xl) || (!isBuy && cl > xh))
         {
            trendExit = true;
            if (CloseTicket(tag, tk, T_SLIP_USD))
            {
               exited = true;
               CheckClosed();
            }
         }
      }
      // 2) 진입 신호 (포지션이 없고 이번 봉에서 청산하지 않았을 때만)
      if (!on || entry == 0 || exited || MyTicketSym(sym, magic) >= 0)
         return;
      if (shSide != 0)
      {
         Print("[", tag, "] 그림자 포지션 중이라 신호 무시");
         return;
      }
      double top, bot;
      if (T_CLOUD_FILTER && Cloud(h, l, n, top, bot) && !(entry == 1 ? cl > top : cl < bot))
      {
         RefreshRates();
         double px = entry == 1 ? Ask1() : Bid1();
         shSide = entry;
         shEntry = px;
         shSl = NormalizeDouble(px - entry * T_SL_ATR * atr, 2);
         SaveShadow();
         string where = cl >= bot && cl <= top ? "구름 안" : "구름 반대편";
         Print(StringFormat("[%s] 종가 %.2f가 %s (구름 %.2f~%.2f) -> 진입 안 함, 그림자 손절가 %.2f", tag, cl, where, bot, top, shSl));
         Send(tag, StringFormat("👻 %s 신호지만 종가 %.2f가 4시간봉 %s (구름 %.2f~%.2f)\n진입하지 않고, 추세가 끝나거나 손절가 %.2f에 닿을 때까지 다음 신호를 쉽니다",
                                entry == 1 ? "매수" : "매도", cl, where, bot, top, shSl));
         return;
      }
      Place(entry, atr);
   }

   //---- Infinox_Bots.mq4에서 부름
   void Init()
   {
      if (!ok)
         return;
      lastBar = iTime(sym, PERIOD_H4, 1);     // 켜진 뒤 새로 완성되는 봉부터 판단
      shSide = (int)StringToInteger(StGet(st, "sh_side", "0"));
      shSl = StringToDouble(StGet(st, "sh_sl", "0"));
      shEntry = StringToDouble(StGet(st, "sh_entry", "0"));
      ticket = MyTicketSym(sym, magic);
      if (ticket >= 0 && OrderSelect(ticket, SELECT_BY_TICKET))
      {
         openLot = OrderLots();
         openPx = OrderOpenPrice();
         openSide = OrderType() == OP_BUY ? 1 : -1;
         Print("[", tag, "] 기존 포지션 이어받음: 티켓 ", ticket);
      }
   }

   string StartText()
   {
      if (!ok)
         return StringFormat("⚠️ 종목 %s을 찾을 수 없어 이 전략은 쉽니다 (MT4 시장 감시 창에 종목이 있는지 확인)", sym);
      return "전략: " + Desc();
   }

   void Timer(bool on)
   {
      if (!ok)
      {
         if (TimeLocal() - okTry >= 60)          // 처음에 못 찾은 종목: 1분마다 다시 (주말·시세 받기 전)
         {
            okTry = TimeLocal();
            ok = UseSymbol(sym);
            if (ok)
            {
               Print("[", tag, "] 종목 ", sym, " 찾음 -> 시작");
               Init();
            }
         }
         if (!ok)
            return;
      }
      CheckClosed();
      if (shSide != 0)
      {
         RefreshRates();
         if (shSide == 1 ? Bid1() <= shSl : Ask1() >= shSl)
            EndShadow("손절", shSl);
      }
      datetime bar = iTime(sym, PERIOD_H4, 1);
      if (bar == 0 || bar == lastBar)
         return;
      lastBar = bar;
      OnBar(bar, on);
   }
};
