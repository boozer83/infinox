// SAR 단타(demo_trader.py) / 단타 2마틴(sar_martin_trader.py) 전략 모듈. Infinox_Bots.mq4가 include
// 같은 신호·필터를 쓰는 CSar 두 개 (매직·상태 파일·lot 계산만 다름)
// 규칙은 demo_trader.py와 같음:
// - 5분봉 SAR(0.02, 0.2) 전환 봉 완성 시 진입. 매수: RSI(14) > 50 이고 완성 1시간봉 SAR 상승 (매도는 반대)
// - 손절 7.5$ / 익절 3$ 서버 설정, 포지션 1개
// - 금요일 22:30 이후 진입 안 함, 23:30 청산 / 매일 15:00~21:00 진입 안 함 (서버시간)
// - 오늘 완성 1시간봉 SAR 전환 3번 이상이면 진입 안 함
// - 5분봉 ATR(14) 2.5$ 미만, 또는 최근 20일 5분봉 ATR 중앙값의 2배 초과면 진입 안 함
// - 과열 거르기: 신호 봉 종가가 뚫은 SAR 값에서 ATR x 0.8 넘게 갔거나 새 SAR 점에서 ATR x 2.8 넘게 떨어져 있으면 진입 안 함
// - 보유 중 5분봉 SAR이 반대로 전환되면(완성 봉) 시장가 청산 (반대 콩 청산, 2026-10-21 추가)
// - 승률 쉬기(최근 200건 71.4% 미만이면 가상 진입만)는 꺼 둠 (M_WINRATE_PAUSE). 승률은 기록만
// - 계좌 손실 한도 50% (최고 잔고 대비 평가금)
// lot: SAR 단타 = 그날 시작 잔고 1,000$당 LotPer1000 고정
//      2마틴 (sar_martin_trader.py) = 지면 다음 lot은 이번 묶음 누적 손실을 익절 3$로 딱 되찾는 크기 (2마틴까지, 시작 lot의 5배·MartinMaxLot까지),
//      묶음 합계가 0 이상이면 시작 lot으로, 오늘 2연속 손절이면 그날 쉼, 서버 날짜가 바뀌면 초기화
// 상태: MQL4\Files\<상태 파일> (승률 기록, 가상 포지션, 최고 잔고)

#define M_SL_USD 7.5
#define M_TP_USD 3.0
#define M_FRIDAY_NO_ENTRY 1350       // 22:30
#define M_FRIDAY_CLOSE 1410          // 23:30
#define M_NO_ENTRY_FROM 900          // 15:00
#define M_NO_ENTRY_UNTIL 1260        // 21:00
#define M_MAX_H1_FLIPS 3
#define M_MIN_ATR 2.5
#define M_MAX_ATR_MULT 2.0
#define M_ATR_REF_BARS 5520          // 20일 x 276
#define M_ATR_REF_MIN 2000
#define M_MAX_EXT 0.8
#define M_MAX_RISK 2.8
#define M_WINRATE_PAUSE false
#define M_SAR_EXIT true
#define M_WINRATE_N 200
#define M_WINRATE_MIN 71.4
#define M_ACCOUNT_STOP_PCT 50.0
#define M_SLIP_USD 0.30
#define M_MARTIN_STEPS 2             // 2마틴: 0 = 시작 lot, 2 = 2마틴
#define M_STOP_STREAK 2              // 2마틴: 오늘 이만큼 연속 손절이면 그날 쉼
#define M_MAX_LOT_MULT 5.0           // 2마틴: 시작 lot의 이 배수까지

// 최근 M_ATR_REF_BARS개 완성 5분봉 ATR(14) 중앙값. 기록이 부족하면 0
double M_AtrRef()
{
   datetime t[];
   double h[], l[], c[], a[];
   int n = LoadBars(PERIOD_M5, M_ATR_REF_BARS + 100, t, h, l, c);
   if (n < M_ATR_REF_MIN)
      return 0;
   Atr(h, l, c, n, 14, a);
   int k = MathMin(M_ATR_REF_BARS, n);
   double v[];
   ArrayResize(v, k);
   for (int i = 0; i < k; i++)
      v[i] = a[n - k + i];
   ArraySort(v);
   return k % 2 == 1 ? v[k / 2] : (v[k / 2 - 1] + v[k / 2]) / 2;
}

// SAR 단타 신호 + 공통 필터 (SAR 단타·2마틴·헤지 마틴이 같이 씀, demo_trader.compute_signal + 진입 필터와 같음)
// 반환: 신호 방향 (필터에 걸리면 0이고 why에 이유), flipDir = 방금 완성된 5분봉 SAR 전환 방향 (전환 없으면 0)
// nightOff: 15:00~21:00 진입 안 함 적용 여부 (헤지 마틴은 끔)
int M_Signal(datetime now, datetime bar, string tag, bool nightOff, string &why, int &flipDir)
{
   why = "";
   flipDir = 0;
   datetime t[], ht[];
   double h[], l[], c[], hh[], hl[], hc[], r[], a[];
   bool up[], hup[];
   int n = LoadBars(PERIOD_M5, 600, t, h, l, c);
   int hn = LoadBars(PERIOD_H1, 400, ht, hh, hl, hc);
   if (n < 50 || hn < 50 || t[n - 1] != bar)
   {
      why = "봉 부족";
      return 0;
   }
   Psar(h, l, n, up);
   Rsi(c, n, 14, r);
   if (up[n - 1] != up[n - 2])
      flipDir = up[n - 1] ? 1 : -1;
   // 신호봉이 끝난 시각까지 완성된 1시간봉만
   int hd = hn;
   while (hd > 0 && ht[hd - 1] + 3600 > bar + 300)
      hd--;
   if (hd < 50)
   {
      why = "1시간봉 부족";
      return 0;
   }
   Psar(hh, hl, hd, hup);
   int hdir = hup[hd - 1] ? 1 : -1;
   int side = 0;
   if (flipDir != 0 && (r[n - 1] - 50) * flipDir > 0 && hdir == flipDir)
      side = flipDir;
   string desc = StringFormat("[%s] 봉 %s 종가 %.2f: SAR %s%s RSI %.1f H1 SAR %s", tag, TimeToString(bar, TIME_DATE | TIME_MINUTES),
                              c[n - 1], up[n - 1] ? "상승" : "하락", flipDir != 0 ? "(전환)" : "", r[n - 1], hdir == 1 ? "상승" : "하락");
   if (side == 0)
   {
      Print(desc);
      why = "신호 없음";
      return 0;
   }
   Print(desc, " -> 신호 ", side == 1 ? "매수" : "매도");
   int mins = Minutes(now);
   if (TimeDayOfWeek(now) == 5 && mins >= M_FRIDAY_NO_ENTRY)
      why = "금요일 마감 전이라 신규 진입 안 함";
   else if (nightOff && mins >= M_NO_ENTRY_FROM && mins < M_NO_ENTRY_UNTIL)
      why = "미국장 제외 시간(15:00~21:00)이라 신규 진입 안 함";
   if (why == "")
   {
      // 오늘 완성된 1시간봉 SAR 전환 수 (1시간봉 전체 기준, regime_switch와 같음)
      bool hall[];
      Psar(hh, hl, hn, hall);
      int flips = 0;
      datetime d0 = DayStart(now);
      for (int i = 1; i < hn; i++)
         if (hall[i] != hall[i - 1] && DayStart(ht[i]) == d0)
            flips++;
      Atr(h, l, c, n, 14, a);
      double atr = a[n - 1];
      if (flips >= M_MAX_H1_FLIPS)
         why = StringFormat("오늘 1시간봉 SAR 전환 %d번 (횡보)이라 신규 진입 안 함", flips);
      else if (atr < M_MIN_ATR)
         why = StringFormat("5분봉 ATR %.2f$ < %g$ (조용한 시장)이라 신규 진입 안 함", atr, M_MIN_ATR);
      else
      {
         double ref = M_AtrRef();
         if (ref > 0 && atr > M_MAX_ATR_MULT * ref)
            why = StringFormat("5분봉 ATR %.2f$ > 20일 중앙값 %.2f$ x %g (급변동)이라 신규 진입 안 함", atr, ref, M_MAX_ATR_MULT);
         else
         {
            bool up2[];
            double sv[], tg[];
            PsarLevels(h, l, n, up2, sv, tg);
            double ext = (c[n - 1] - tg[n - 1]) * side / atr, risk = (c[n - 1] - sv[n - 1]) * side / atr;
            if (ext > M_MAX_EXT || risk > M_MAX_RISK)
               why = StringFormat("과열 (뚫은 SAR에서 %.2f ATR, 새 SAR 점까지 %.2f ATR)이라 신규 진입 안 함", ext, risk);
         }
      }
   }
   if (why != "")
   {
      Print("[", tag, "] ", why);
      return 0;
   }
   return side;
}

class CSar : public CStrat
{
public:
   string   tag, st;
   int      magic;
   bool     martin;
   int      ticket;
   datetime lastBar, lastCloseTry, riskNotified;
   bool     fridayClose, sarClose;
   string   wr;                      // 최근 결과 '1'(익절) / '0'(손절)
   int      vside;                   // 가상 포지션 방향
   double   ventry;
   double   openLot, openPx;
   int      openSide;
   double   chain;                   // 2마틴: 이번 묶음 누적 손익 $ (MartinState가 계산)

   void Setup(string panelName, string file, int mg, bool isMartin, bool on)
   {
      name = panelName;
      tag = panelName;
      st = file;
      magic = mg;
      martin = isMartin;
      defOn = on;
      ticket = -1;
      lastBar = 0;
      lastCloseTry = 0;
      riskNotified = 0;
      chain = 0;
   }

   int Magic() { return magic; }

   string Desc()
   {
      string lots = martin ? StringFormat("지면 다음 lot은 진 금액을 익절 %g$로 되찾는 크기 (%d마틴까지), 오늘 %d연속 손절이면 그날 쉼, 매일 초기화",
                                          M_TP_USD, M_MARTIN_STEPS, M_STOP_STREAK)
                           : StringFormat("고정 lot (1,000$당 %g lot, 마틴 없음)", LotPer1000);
      return StringFormat("5분봉 SAR 전환 + RSI50 + 1시간봉 SAR, 익절 %g$ / 손절 %g$, %s, "
                          "5분봉 ATR %g$ 미만 쉼, ATR 20일 중앙값 %g배 초과 쉼, 과열 거르기(%g / %g ATR), 5분봉 SAR 반대 전환이면 청산, 승률 쉬기 꺼짐",
                          M_TP_USD, M_SL_USD, lots, M_MIN_ATR, M_MAX_ATR_MULT, M_MAX_EXT, M_MAX_RISK);
   }

   double BaseLot() { return FloorLot((AccountBalance() - TodayTradePnl()) / 1000 * LotPer1000); }

   // 2마틴: 오늘(서버 날짜) 이 전략이 청산한 거래로 단계·오늘 쉼 여부 (chain도 갱신). SAR 단타는 늘 (0, false)
   int MartinState(bool &stopped)
   {
      stopped = false;
      chain = 0;
      if (!martin)
         return 0;
      datetime d0 = DayStart(TimeCurrent());
      int    n = 0;
      datetime ct[];
      double   pf[];
      for (int i = OrdersHistoryTotal() - 1; i >= 0; i--)
         if (OrderSelect(i, SELECT_BY_POS, MODE_HISTORY) && OrderType() <= OP_SELL && OrderMagicNumber() == magic
             && OrderSymbol() == Symbol() && OrderCloseTime() >= d0)
         {
            ArrayResize(ct, n + 1);
            ArrayResize(pf, n + 1);
            ct[n] = OrderCloseTime();
            pf[n] = OrderProfit() + OrderSwap() + OrderCommission();
            n++;
         }
      // 청산 시각 순으로 (같은 초면 이력 순서 그대로)
      for (int i = 1; i < n; i++)
         for (int j = i; j > 0 && ct[j] < ct[j - 1]; j--)
         {
            datetime tt = ct[j];
            ct[j] = ct[j - 1];
            ct[j - 1] = tt;
            double pp = pf[j];
            pf[j] = pf[j - 1];
            pf[j - 1] = pp;
         }
      int level = 0, streak = 0;
      for (int k = 0; k < n; k++)
      {
         double p = pf[k];
         streak = p < 0 ? streak + 1 : 0;
         chain += p;
         if (chain >= 0 || level >= M_MARTIN_STEPS)
         {
            chain = 0;
            level = 0;
         }
         else
            level++;
      }
      stopped = streak >= M_STOP_STREAK;
      return level;
   }

   double LotFor(int level, double base)
   {
      double step = MarketInfo(Symbol(), MODE_LOTSTEP);
      double lot = base;
      if (martin)
      {
         if (level > 0)
            lot = MathMax(base, MathCeil(-chain / (M_TP_USD * Contract()) / step - 1e-9) * step);
         lot = MathMin(lot, base * M_MAX_LOT_MULT);
      }
      lot = MathRound(MathMin(lot, MartinMaxLot) / step) * step;
      return NormalizeDouble(MathMin(MathMax(lot, MarketInfo(Symbol(), MODE_MINLOT)), MarketInfo(Symbol(), MODE_MAXLOT)), 2);
   }

   string LevelText(int level)
   {
      return level == 0 ? "시작 lot" : StringFormat("%d마틴 (묶음 %+.2f$ 회수)", level, chain);
   }

   //---- 승률 쉬기
   double WinRate(int &n)
   {
      n = StringLen(wr);
      int w = 0;
      for (int i = 0; i < n; i++)
         if (StringGetCharacter(wr, i) == '1')
            w++;
      return n > 0 ? w * 100.0 / n : 0;
   }

   bool IsPaused()
   {
      if (!M_WINRATE_PAUSE)
         return false;
      int n;
      double r = WinRate(n);
      return n >= M_WINRATE_N && r < M_WINRATE_MIN;
   }

   void RecordResult(bool win, string how)
   {
      bool was = IsPaused();
      wr += win ? "1" : "0";
      if (StringLen(wr) > M_WINRATE_N)
         wr = StringSubstr(wr, StringLen(wr) - M_WINRATE_N);
      StSet(st, "wr", wr);
      int n;
      double r = WinRate(n);
      Print(StringFormat("[%s] 승률 기록 (%s %s): 최근 %d건 %.1f%%", tag, how, win ? "익절" : "손절", n, r));
      bool paused = IsPaused();
      if (paused != was)
      {
         if (paused)
            Send(tag, StringFormat("⏸ 최근 %d건 승률 %.1f%% < %g%% -> 실제 진입을 쉬고 가상 진입만 기록합니다.\n"
                                   "승률이 %g%% 이상으로 돌아오면 자동으로 다시 시작합니다.", n, r, M_WINRATE_MIN, M_WINRATE_MIN));
         else
            Send(tag, StringFormat("▶️ 최근 %d건 승률 %.1f%% -> 실제 진입을 다시 시작합니다.", n, r));
      }
   }

   void SaveVirtual()
   {
      StSet(st, "vside", IntegerToString(vside));
      StSet(st, "ventry", DoubleToString(ventry, Digits));
   }

   // 가상 포지션이 익절/손절 가격에 닿았는지 (매수는 Bid, 매도는 Ask 기준)
   void CheckVirtual()
   {
      if (vside == 0)
         return;
      RefreshRates();
      double px = vside == 1 ? Bid : Ask;
      double move = (px - ventry) * vside;
      if (move >= M_TP_USD || move <= -M_SL_USD)
      {
         Print(StringFormat("[%s] 가상 %s @ %.2f 종료 @ %.2f", tag, vside == 1 ? "매수" : "매도", ventry, px));
         vside = 0;
         SaveVirtual();
         RecordResult(move > 0, "가상");
      }
   }

   //---- 진입 / 청산
   void Place(int side, int level, double lot, datetime bar, double signalPx)
   {
      if (!MarginOk(tag, lot))
         return;
      double fill;
      int tk = OpenMarket(tag, side, lot, M_SL_USD, M_TP_USD, magic, StringFormat("SAR RSI50 H1 L%d", level), M_SLIP_USD, fill);
      if (tk < 0)
         return;
      ticket = tk;
      openLot = lot;
      openPx = fill;
      openSide = side;
      fridayClose = false;
      sarClose = false;
      Send(tag, StringFormat("%s 진입 %.2f lot (%s)\n가격 %.2f | SL %.2f | TP %.2f (신호봉 %s 종가 %.2f)\n"
                             "최대 손실 약 %.0f$ / 목표 이익 약 %.0f$\n%s\n%s",
                             side == 1 ? "🟢 매수" : "🔴 매도", lot, martin ? LevelText(level) : "고정 lot", fill, fill - side * M_SL_USD,
                             fill + side * M_TP_USD, TimeToString(bar, TIME_DATE | TIME_MINUTES), signalPx, M_SL_USD * lot * Contract(),
                             M_TP_USD * lot * Contract(), AccountLine(), SummaryLine(magic)));
   }

   void CheckClosed()
   {
      if (ticket < 0 || MyTicket(magic) == ticket)
         return;
      double profit, px;
      string reason;
      if (!ClosedInfo(ticket, profit, px, reason))
         return;
      if (fridayClose)
         reason = "FRIDAY";
      else if (sarClose)
         reason = "SAR";
      double bal = AccountBalance();
      double pct = bal - profit > 0 ? profit / (bal - profit) * 100 : 0;
      string icon = reason == "TP" ? "✅ 익절" : reason == "SL" ? "❌ 손절" : reason == "FRIDAY" ? "⏹ 금요일 청산" : reason == "SAR" ? "↩️ 반대 콩 청산" : "⏹ 청산";
      bool stopped;
      int level = MartinState(stopped);
      string nxt = stopped ? StringFormat("오늘 %d연속 손절 -> 오늘은 쉼 (내일 시작 lot부터)", M_STOP_STREAK)
                           : StringFormat("다음 진입 %s (%.2f lot)", martin ? LevelText(level) : "고정 lot", LotFor(level, BaseLot()));
      Send(tag, StringFormat("%s %s %.2f lot\n%.2f → %.2f\n손익 %+.2f$ (%+.2f%%)\n%s\n%s\n%s",
                             icon, openSide == 1 ? "BUY" : "SELL", openLot, openPx, px, profit, pct, nxt, AccountLine(), SummaryLine(magic)));
      RecordResult(profit > 0, "실제");
      ticket = -1;
   }

   void OnBar(datetime now, datetime bar)
   {
      int flipDir;
      string why;
      int side = M_Signal(now, bar, tag, true, why, flipDir);
      if (side == 0)
         return;
      datetime d0 = DayStart(now);
      bool stopped;
      int level = MartinState(stopped);
      if (MyTicket(magic) >= 0)
         why = "포지션 보유 중이라 신호 무시";
      else if (M_WINRATE_PAUSE && vside != 0)
         why = "가상 포지션 진행 중이라 신호 무시";
      else if (stopped)
         why = StringFormat("오늘 %d연속 손절로 쉬는 중이라 신호 무시", M_STOP_STREAK);
      if (why != "")
      {
         Print("[", tag, "] ", why);
         return;
      }
      string reason = RiskBlock(tag, st, M_ACCOUNT_STOP_PCT);
      if (reason != "")
      {
         Print("[", tag, "] 안전장치: ", reason, " -> 신규 진입 안 함");
         if (riskNotified != d0)
         {
            riskNotified = d0;
            Send(tag, "⛔ 안전장치: " + reason + "\n오늘은 신규 진입을 멈춥니다.");
         }
         return;
      }
      if (IsPaused())
      {
         RefreshRates();
         vside = side;
         ventry = side == 1 ? Ask : Bid;
         SaveVirtual();
         int k;
         double w = WinRate(k);
         Print(StringFormat("[%s] 승률 쉬기 중 (최근 %d건 %.1f%%) -> 가상 %s @ %.2f", tag, k, w, side == 1 ? "매수" : "매도", ventry));
         return;
      }
      Place(side, level, LotFor(level, BaseLot()), bar, iClose(NULL, PERIOD_M5, 1));
   }

   // 반대 콩 청산: 방금 완성된 5분봉 SAR 방향이 포지션과 반대면 시장가 청산
   void SarExit(datetime bar)
   {
      int tk = MyTicket(magic);
      if (!M_SAR_EXIT || tk < 0 || !OrderSelect(tk, SELECT_BY_TICKET))
         return;
      int pside = OrderType() == OP_BUY ? 1 : -1;
      datetime t[];
      double h[], l[], c[];
      bool up[];
      int n = LoadBars(PERIOD_M5, 600, t, h, l, c);
      if (n < 50 || t[n - 1] != bar)
         return;
      Psar(h, l, n, up);
      if ((up[n - 1] ? 1 : -1) == pside)
         return;
      Print(StringFormat("[%s] 5분봉 SAR %s 전환 -> 반대 콩 청산: 티켓 %d", tag, up[n - 1] ? "상승" : "하락", tk));
      sarClose = true;
      CloseTicket(tag, tk, M_SLIP_USD);
   }

   //---- Infinox_Bots.mq4에서 부름
   void Init()
   {
      wr = StGet(st, "wr", "");
      vside = (int)StringToInteger(StGet(st, "vside", "0"));
      ventry = StringToDouble(StGet(st, "ventry", "0"));
      lastBar = iTime(NULL, PERIOD_M5, 1);     // 켜진 뒤 새로 완성되는 봉부터 판단
      ticket = MyTicket(magic);
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
      int k;
      double w = WinRate(k);
      bool stopped;
      int level = MartinState(stopped);
      string lots = StringFormat("lot %.2f (오늘 시작 잔고 기준)", LotFor(0, BaseLot()));
      if (martin)
         lots = StringFormat("시작 lot %.2f, 지금 %s%s", LotFor(0, BaseLot()), LevelText(level), stopped ? " (오늘은 쉼)" : "");
      string s = StringFormat("전략: %s\n%s\n안전장치: 최대 lot %g, 계좌 손실 한도 %g%% (최근 %d건 승률 %.1f%%%s)",
                              Desc(), lots, MartinMaxLot, M_ACCOUNT_STOP_PCT, k, w, IsPaused() ? " (쉬는 중)" : "");
      string halted = StGet(st, "halted", "");
      if (halted != "")
         s += "\n🛑 계좌 손실 한도로 중단된 상태: " + halted;
      return s;
   }

   // 매초. on이면 신규 진입까지, 꺼져 있어도 이미 열린 포지션 관리(청산 기록, 금요일 청산)와 가상 포지션 확인은 계속
   void Timer(bool on)
   {
      datetime now = TimeCurrent();
      CheckClosed();
      CheckVirtual();
      if (TimeDayOfWeek(now) == 5 && Minutes(now) >= M_FRIDAY_CLOSE && TimeLocal() - lastCloseTry > 60)
      {
         int tk = MyTicket(magic);
         if (tk >= 0)
         {
            fridayClose = true;
            CloseTicket(tag, tk, M_SLIP_USD);
            lastCloseTry = TimeLocal();
         }
      }
      datetime bar = iTime(NULL, PERIOD_M5, 1);
      if (bar == 0 || bar == lastBar)
         return;
      lastBar = bar;
      SarExit(bar);
      CheckClosed();      // 반대 콩으로 닫혔으면 새 진입 전에 기록
      if (on)
         OnBar(now, bar);
   }
};
