// H4 추세(trend_trader.py) 전략 모듈. Infinox_Bots.mq4가 include
// - 4시간봉 완성마다 판단. 종가가 직전 25봉 최고가 위면 매수, 최저가 아래면 매도
// - 매수는 종가가 직전 12봉 최저가 아래, 매도는 직전 12봉 최고가 위면 청산 (청산한 봉에서는 새로 진입 안 함)
// - 손절: 진입가 ± 3 x ATR(14, 4시간봉) 서버 설정, 익절 없음, 주말에도 보유
// - 수량: 손절 시 잔고의 TrendRiskPct% 손실 (0.01 내림, 최소 0.01, 최대 TrendMaxLot)

#define T_N_ENTRY 25
#define T_N_EXIT 12
#define T_SL_ATR 3.0
#define T_SLIP_USD 0.50

string   t_tag = "H4 추세";
int      t_ticket = -1;
datetime t_lastBar = 0;
double   t_openLot = 0, t_openPx = 0;
int      t_openSide = 0;
bool     t_trendExit = false;

string T_Desc()
{
   return StringFormat("4시간봉 %d봉 돌파 추세추종, %d봉 반대 돌파 청산, 손절 %g ATR, 거래당 위험 %g%%",
                       T_N_ENTRY, T_N_EXIT, T_SL_ATR, TrendRiskPct);
}

void T_CheckClosed()
{
   if (t_ticket < 0 || MyTicket(MagicTrend) == t_ticket)
      return;
   double profit, px;
   string reason;
   if (!ClosedInfo(t_ticket, profit, px, reason))
      return;
   if (t_trendExit || reason == "CLOSE")
      reason = "추세종료";
   double bal = AccountBalance();
   double pct = bal - profit > 0 ? profit / (bal - profit) * 100 : 0;
   Send(t_tag, StringFormat("%s %s %s %.2f lot\n%.2f → %.2f\n손익 %+.2f$ (%+.2f%%)\n%s\n%s",
                            profit > 0 ? "✅" : "❌", reason, t_openSide == 1 ? "BUY" : "SELL", t_openLot, t_openPx, px, profit, pct,
                            AccountLine(), SummaryLine(MagicTrend)));
   t_ticket = -1;
}

void T_Place(int side, double atr)
{
   double slDist = T_SL_ATR * atr;
   double raw = AccountBalance() * TrendRiskPct / 100 / (slDist * Contract());
   double step = MarketInfo(Symbol(), MODE_LOTSTEP);
   double lot = MathFloor(raw / step + 1e-9) * step;
   bool forced = lot < MarketInfo(Symbol(), MODE_MINLOT);
   lot = NormalizeDouble(MathMin(MathMin(MathMax(lot, MarketInfo(Symbol(), MODE_MINLOT)), TrendMaxLot),
                                 MarketInfo(Symbol(), MODE_MAXLOT)), 2);
   if (!MarginOk(t_tag, lot))
      return;
   double fill;
   int tk = OpenMarket(t_tag, side, lot, slDist, 0, MagicTrend, "trend H4 N25", T_SLIP_USD, fill);
   if (tk < 0)
      return;
   t_ticket = tk;
   t_openLot = lot;
   t_openPx = fill;
   t_openSide = side;
   t_trendExit = false;
   double risk = slDist * lot * Contract();
   Send(t_tag, StringFormat("%s 진입 %.2f lot\n가격 %.2f | 손절 %.2f (익절 없음, 추세 끝나면 청산)\n손절 시 손실 약 %.0f$ (%.2f%%)%s\n%s\n%s",
                            side == 1 ? "🟢 매수" : "🔴 매도", lot, fill, fill - side * slDist, risk, risk / AccountBalance() * 100,
                            forced ? "\n⚠️ 최소 수량이라 목표 위험보다 큼" : "", AccountLine(), SummaryLine(MagicTrend)));
}

// 청산 신호는 꺼져 있어도 처리, 진입은 on일 때만
void T_OnBar(datetime bar, bool on)
{
   datetime t[];
   double h[], l[], c[], a[];
   int n = LoadBars(PERIOD_H4, 300, t, h, l, c);
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
   Print(StringFormat("[H4 추세] 4시간봉 %s: 종가 %.2f | 진입선 %.2f~%.2f | 청산선 %.2f~%.2f | ATR %.2f%s",
                      TimeToString(bar, TIME_DATE | TIME_MINUTES), cl, ll, hh, xl, xh, atr,
                      entry == 0 ? "" : entry == 1 ? " -> 진입 신호 매수" : " -> 진입 신호 매도"));
   // 1) 청산 신호
   bool exited = false;
   int tk = MyTicket(MagicTrend);
   if (tk >= 0 && OrderSelect(tk, SELECT_BY_TICKET))
   {
      bool isBuy = OrderType() == OP_BUY;
      if ((isBuy && cl < xl) || (!isBuy && cl > xh))
      {
         t_trendExit = true;
         if (CloseTicket(t_tag, tk, T_SLIP_USD))
         {
            exited = true;
            T_CheckClosed();
         }
      }
   }
   // 2) 진입 신호 (포지션이 없고 이번 봉에서 청산하지 않았을 때만)
   if (on && entry != 0 && !exited && MyTicket(MagicTrend) < 0)
      T_Place(entry, atr);
}

//---------------------------------------------------------------- Infinox_Bots.mq4에서 부름
void T_Init()
{
   t_lastBar = iTime(NULL, PERIOD_H4, 1);     // 켜진 뒤 새로 완성되는 봉부터 판단
   t_ticket = MyTicket(MagicTrend);
   if (t_ticket >= 0 && OrderSelect(t_ticket, SELECT_BY_TICKET))
   {
      t_openLot = OrderLots();
      t_openPx = OrderOpenPrice();
      t_openSide = OrderType() == OP_BUY ? 1 : -1;
      Print("[H4 추세] 기존 포지션 이어받음: 티켓 ", t_ticket);
   }
}

string T_StartText() { return "전략: " + T_Desc(); }

void T_Timer(bool on)
{
   T_CheckClosed();
   datetime bar = iTime(NULL, PERIOD_H4, 1);
   if (bar == 0 || bar == t_lastBar)
      return;
   t_lastBar = bar;
   T_OnBar(bar, on);
}
