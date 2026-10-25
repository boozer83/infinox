// 아시아 3마틴(asia_breakout_trader.py, 아시아 레인지 돌파 + 마틴 3단계) 전략 모듈. Infinox_Bots.mq4가 include. 시각은 모두 서버시간
// 종목마다 CAsia 하나 (금 = 차트 종목, XAUEUR = 같은 접미사를 붙인 다른 종목). 매직·위험·상태 파일을 따로 씀
// 1. 아시아 레인지 = 오늘 01:05~08:55 5분봉 최고가 / 최저가 (50봉 미만이면 그날 거래 없음)
// 2. 09:00~17:55에 시작한 5분봉 종가가 레인지 위 → 매수, 아래 → 매도. 그날 첫 돌파만 봄
// 3. 5일 모멘텀: 전일 종가 > 6거래일 전 종가면 매수만, 작으면 매도만
// 4. 기준폭 U = 0.5 x 일봉 ATR(14, 전일까지). 익절 = U, 손절 = 0.75 x U (v2). 23:00에 보유 포지션 청산
//    미국 휴일 조기 마감일(연방 공휴일·추수감사절 다음 날·12/24, 마지막 시세 19:44~21:44)은 19:30에 청산
// 5. 신호 강도 (v2): 돌파 봉 종가가 전일 고가 위(매수) / 전일 저가 아래(매도)면 가중치 1.5, 아니면 0.5
// 자금관리: lot = 사이클 시작 잔고 x 위험% x 가중치 x 2^(단계-1) / (U x 1 lot 1$당 손익), 결과 0 이하면 다음 단계, 최대 3단계 (1·2·4배)
// 상태: MQL4\Files\<상태 파일> (오늘 처리 여부, 마틴 단계, 사이클 시작 잔고, 최고 잔고)

#define A_RANGE_FROM 65              // 01:05
#define A_RANGE_UNTIL 540            // 09:00
#define A_ENTRY_UNTIL 1080           // 18:00
#define A_EXIT_AT 1380               // 23:00
#define A_EARLY_EXIT_AT 1170         // 19:30 (미국 휴일 조기 마감일, 가장 이른 마감 19:44)
#define A_MIN_RANGE_BARS 50
#define A_ATR_MULT 0.5
#define A_TP_RATIO 1.0
#define A_SL_RATIO 0.75
#define A_W_STRONG 1.5
#define A_W_WEAK 0.5
#define A_MOM_DAYS 5
#define A_MARTIN_STEPS 3
#define A_ACCOUNT_STOP_PCT 50.0
#define A_SLIP_USD 0.50

//---------------------------------------------------------------- 미국 휴일 조기 마감일
// y년 m월의 n번째 wday요일(0=일) 날짜. n이 0이면 마지막
datetime A_NthWeekday(int y, int m, int wday, int n)
{
   datetime first = StringToTime(StringFormat("%04d.%02d.01", y, m));
   if (n > 0)
      return first + (((wday - TimeDayOfWeek(first)) + 7) % 7 + (n - 1) * 7) * 86400;
   int ny = m == 12 ? y + 1 : y, nm = m == 12 ? 1 : m + 1;
   datetime last = StringToTime(StringFormat("%04d.%02d.01", ny, nm)) - 86400;
   return last - ((TimeDayOfWeek(last) - wday + 7) % 7) * 86400;
}

// 토요일이면 금요일, 일요일이면 월요일 (pandas nearest_workday)
datetime A_Nearest(int y, int m, int d)
{
   datetime t = StringToTime(StringFormat("%04d.%02d.%02d", y, m, d));
   int w = TimeDayOfWeek(t);
   return w == 6 ? t - 86400 : w == 0 ? t + 86400 : t;
}

// 금이 일찍 닫힌 미국 휴일 (asia_breakout_trader.early_close_day와 같음. 콜럼버스·재향군인의 날은 정상 마감)
bool A_EarlyCloseDay(datetime d)
{
   int y = TimeYear(d);
   datetime thanks = A_NthWeekday(y, 11, 4, 4);
   return d == A_NthWeekday(y, 1, 1, 3)          // 마틴 루터 킹
       || d == A_NthWeekday(y, 2, 1, 3)          // 대통령의 날
       || d == A_NthWeekday(y, 5, 1, 0)          // 현충일
       || d == A_Nearest(y, 6, 19)               // 준틴스
       || d == A_Nearest(y, 7, 4)                // 독립기념일
       || d == A_NthWeekday(y, 9, 1, 1)          // 노동절
       || d == thanks || d == thanks + 86400     // 추수감사절, 다음 날
       || (TimeMonth(d) == 12 && TimeDay(d) == 24);
}

int A_ExitAt(datetime day) { return A_EarlyCloseDay(day) ? A_EARLY_EXIT_AT : A_EXIT_AT; }

string A_Hm(int m) { return StringFormat("%02d:%02d", m / 60, m % 60); }

//---------------------------------------------------------------- 전략 (종목 하나)
class CAsia : public CStrat
{
public:
   string   sym, tag, st;
   int      magic;
   double   riskPct;
   bool     ok;                      // 종목을 쓸 수 있으면 true

   int      ticket;
   datetime lastBar, lastCloseTry;
   bool     timeExit;
   double   openLot, openPx;
   int      openSide;
   datetime featDay;
   double   datr, ph, pl;            // 일봉 ATR, 전일 고가 / 저가
   int      mom;

   void Setup(string s, string t, string file, int mg, double rp, bool on)
   {
      sym = s;
      tag = t;
      name = t;
      defOn = on;
      st = file;
      magic = mg;
      riskPct = rp;
      ticket = -1;
      lastBar = 0;
      lastCloseTry = 0;
      timeExit = false;
      featDay = 0;
      ok = SymbolSelect(sym, true) && MarketInfo(sym, MODE_BID) > 0;
      if (!ok)
         Print("[", tag, "] 종목 ", sym, "을 찾을 수 없어 이 전략은 쉽니다");
   }

   // 계좌 통화로 1 lot이 가격 1 움직일 때 손익 (금 100, XAUEUR은 유로 손익을 달러로 바꿔서)
   double Contract1()
   {
      if (sym == Symbol())
         return Contract();
      return MarketInfo(sym, MODE_TICKVALUE) / MarketInfo(sym, MODE_TICKSIZE);
   }

   string Desc()
   {
      return StringFormat("아시아 레인지(01:05~09:00) 돌파 09:00~18:00 + %d일 모멘텀, U=일봉ATR×%g, 익절 %gU / 손절 %gU, "
                          "전일 고저 돌파면 가중치 %g 아니면 %g, 23:00 청산 (미국 조기 마감일 19:30), 마틴 %d단계 (v2), 위험 %g%%",
                          A_MOM_DAYS, A_ATR_MULT, A_TP_RATIO, A_SL_RATIO, A_W_STRONG, A_W_WEAK, A_MARTIN_STEPS, riskPct);
   }

   int    Magic() { return magic; }
   string Sym() { return sym; }

   int Level() { return (int)StringToInteger(StGet(st, "level", "0")); }

   // 오늘 시작 시점에 알 수 있는 값: 일봉 ATR14(전일까지), 5일 모멘텀 방향
   void DailyFeatures(datetime today)
   {
      featDay = today;
      datr = 0;
      mom = 0;
      int s0 = 0;
      while (s0 < iBars(sym, PERIOD_D1) && iTime(sym, PERIOD_D1, s0) >= today)
         s0++;
      int n = MathMin(400, iBars(sym, PERIOD_D1) - s0);
      if (n < 30)
         return;
      double h[], l[], c[], a[];
      ArrayResize(h, n);
      ArrayResize(l, n);
      ArrayResize(c, n);
      for (int i = 0; i < n; i++)
      {
         int s = s0 + n - 1 - i;
         h[i] = iHigh(sym, PERIOD_D1, s);
         l[i] = iLow(sym, PERIOD_D1, s);
         c[i] = iClose(sym, PERIOD_D1, s);
      }
      Atr(h, l, c, n, 14, a);
      datr = a[n - 1];
      ph = h[n - 1];
      pl = l[n - 1];
      double d = c[n - 1] - c[n - 1 - A_MOM_DAYS];
      mom = d > 0 ? 1 : d < 0 ? -1 : 0;
   }

   // 사이클 시작 잔고(1단계 진입 때)는 상태 파일에 보관. 반환: 이번 lot, risk = 손절 시 잔고 대비 %
   double PlanLot(int level, double u, double w, bool save, double &risk)
   {
      double bal = AccountBalance();
      string saved = StGet(st, "cycle_bal", "");
      double cyc = level == 0 || saved == "" ? bal : StringToDouble(saved);
      if (save)
         StSet(st, "cycle_bal", DoubleToString(cyc, 2));
      double lot = FloorLotSym(sym, cyc * riskPct / 100 * w * MathPow(2, level) / (u * Contract1()));
      risk = bal > 0 ? lot * u * A_SL_RATIO * Contract1() / bal * 100 : 0;
      return lot;
   }

   void CheckClosed()
   {
      if (ticket < 0 || MyTicketSym(sym, magic) == ticket)
         return;
      double profit, px;
      string reason;
      if (!ClosedInfo(ticket, profit, px, reason))
         return;
      if (timeExit)
         reason = "TIME";
      int lvl = Level();
      int nxt = profit > 0 ? 0 : (lvl + 1 < A_MARTIN_STEPS ? lvl + 1 : 0);
      StSet(st, "level", IntegerToString(nxt));
      string icon = reason == "TP" ? "✅ 익절" : reason == "SL" ? "❌ 손절" : reason == "TIME" ? "⏹ 시간 청산" : "⏹ 청산";
      Send(tag, StringFormat("%s %s %.2f lot\n%.2f → %.2f\n손익 %+.2f$\n다음 진입: 마틴 %d/%d단계\n%s\n%s",
                             icon, openSide == 1 ? "BUY" : "SELL", openLot, openPx, px, profit, nxt + 1, A_MARTIN_STEPS,
                             AccountLine(), SummaryLine(magic, sym)));
      ticket = -1;
   }

   void OnBar(datetime now, datetime bar)
   {
      datetime today = DayStart(now);
      int bm = Minutes(bar);
      if (DayStart(bar) != today || bm < A_RANGE_UNTIL || bm >= A_ENTRY_UNTIL)
         return;
      string todayStr = TimeToString(today, TIME_DATE);
      if (StGet(st, "done_date", "") == todayStr)
         return;
      if (featDay != today)
         DailyFeatures(today);
      // 오늘 5분봉: 레인지 / 진입 구간 (shift가 클수록 오래된 봉)
      double hi = -1, lo = 1e9;
      int nr = 0, last = 0;
      for (int s = 1; s < iBars(sym, PERIOD_M5) && iTime(sym, PERIOD_M5, s) >= today; s++)
      {
         last = s;
         int m = Minutes(iTime(sym, PERIOD_M5, s));
         if (m >= A_RANGE_FROM && m < A_RANGE_UNTIL)
         {
            hi = MathMax(hi, iHigh(sym, PERIOD_M5, s));
            lo = MathMin(lo, iLow(sym, PERIOD_M5, s));
            nr++;
         }
      }
      if (nr < A_MIN_RANGE_BARS || datr <= 0)
      {
         Print("[", tag, "] 레인지 봉 부족 또는 일봉 ATR 없음 -> 오늘 거래 없음");
         StSet(st, "done_date", todayStr);
         return;
      }
      datetime bt = 0;
      int side = 0;
      double close = 0;
      for (int s = last; s >= 1 && side == 0; s--)
      {
         int m = Minutes(iTime(sym, PERIOD_M5, s));
         if (m < A_RANGE_UNTIL || m >= A_ENTRY_UNTIL)
            continue;
         double c = iClose(sym, PERIOD_M5, s);
         if (c > hi || c < lo)
         {
            side = c > hi ? 1 : -1;
            bt = iTime(sym, PERIOD_M5, s);
            close = c;
         }
      }
      if (side == 0)
         return;
      StSet(st, "done_date", todayStr);   // 첫 돌파가 나오면 결과와 상관없이 오늘은 끝
      double u = datr * A_ATR_MULT, slUsd = u * A_SL_RATIO, tpUsd = u * A_TP_RATIO;
      bool strong = side == 1 ? close > ph : close < pl;
      double w = strong ? A_W_STRONG : A_W_WEAK;
      string desc = StringFormat("[%s] 봉 %s 종가 %.2f %s 돌파 (레인지 %.2f~%.2f, %d봉), 5일 모멘텀 %+d, 일봉 ATR %.2f, "
                                 "전일 고/저 %.2f/%.2f -> %s 신호 x%g",
                                 tag, TimeToString(bt, TIME_DATE | TIME_MINUTES), close, side == 1 ? "상향" : "하향", lo, hi, nr,
                                 mom, datr, ph, pl, strong ? "강한" : "약한", w);
      if (bt != bar)
      {
         Print(desc, " -> 첫 돌파가 이미 지난 봉이라 (재시작 등) 오늘은 진입 안 함");
         return;
      }
      if (side != mom)
      {
         Print(desc, " -> 모멘텀과 반대라 오늘은 거래 없음");
         return;
      }
      if (MyTicketSym(sym, magic) >= 0)
      {
         Print(desc, " -> 포지션 보유 중이라 진입 안 함");
         return;
      }
      int lvl = Level();
      double risk;
      double lot = PlanLot(lvl, u, w, true, risk);
      string reason = RiskBlock(tag, st, A_ACCOUNT_STOP_PCT);
      if (reason == "" && lot > AsiaMaxLot)
         reason = StringFormat("진입 lot %.2f이 최대 lot %g을 넘음", lot, AsiaMaxLot);
      if (reason != "")
      {
         Print(desc, " -> 안전장치: ", reason);
         Send(tag, "⛔ 안전장치: " + reason + "\n오늘은 진입하지 않습니다.");
         return;
      }
      Print(desc, " -> 진입");
      if (!MarginOkSym(tag, sym, lot))
         return;
      double fill;
      int tk = OpenMarketSym(tag, sym, side, lot, slUsd, tpUsd, magic, StringFormat("asia brk L%d", lvl), A_SLIP_USD, fill);
      if (tk < 0)
         return;
      ticket = tk;
      openLot = lot;
      openPx = fill;
      openSide = side;
      timeExit = false;
      Send(tag, StringFormat("%s 진입 %.2f lot (마틴 %d/%d단계, %s 신호 x%g)\n레인지 %.2f ~ %.2f 돌파, 일봉 ATR %.2f\n가격 %.2f | SL %.2f | TP %.2f\n"
                             "최대 손실 약 %.0f$ (잔고의 %.2f%%) / 목표 이익 약 %.0f$\n%s\n%s",
                             side == 1 ? "🟢 매수" : "🔴 매도", lot, lvl + 1, A_MARTIN_STEPS, strong ? "강한" : "약한", w, lo, hi, datr,
                             fill, fill - side * slUsd, fill + side * tpUsd, slUsd * lot * Contract1(), risk,
                             tpUsd * lot * Contract1(), AccountLine(), SummaryLine(magic, sym)));
   }

   //---- Infinox_Bots.mq4에서 부름
   void Init()
   {
      if (!ok)
         return;
      lastBar = iTime(sym, PERIOD_M5, 1);
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
      DailyFeatures(DayStart(TimeCurrent()));
      int lvl = Level();
      string s = StringFormat("전략: %s\n종목 %s, 다음 진입 마틴 %d단계", Desc(), sym, lvl + 1);
      if (datr <= 0)
         return s + "\n일봉 데이터 부족";
      double u = datr * A_ATR_MULT;
      s += StringFormat("\n오늘 일봉 ATR %.2f → U %.2f (손절 %.2f / 익절 %.2f), 5일 모멘텀 %s", datr, u, u * A_SL_RATIO,
                        u * A_TP_RATIO, mom > 0 ? "+" : mom < 0 ? "-" : "0");
      for (int k = 0; k < 2; k++)
      {
         double w = k == 0 ? A_W_STRONG : A_W_WEAK, risk;
         double lot = PlanLot(lvl, u, w, false, risk);
         double aim = riskPct * w * A_SL_RATIO * MathPow(2, lvl);
         s += StringFormat("\n%d단계 %s 신호 %.2f lot (손절 시 잔고의 %.2f%%, 목표 %.2f%%%s)", lvl + 1, k == 0 ? "강한" : "약한", lot, risk,
                           aim, risk > aim * 1.3 ? " — 최소 lot 때문에 큼" : "");
      }
      if (A_EarlyCloseDay(DayStart(TimeCurrent())))
         s += "\n오늘은 미국 휴일 조기 마감일 -> " + A_Hm(A_EARLY_EXIT_AT) + "에 청산";
      return s;
   }

   // 꺼져 있어도 이미 열린 포지션 관리(청산 기록, 23:00 / 조기 마감일 19:30 청산)는 계속
   void Timer(bool on)
   {
      if (!ok)
         return;
      datetime now = TimeCurrent();
      CheckClosed();
      int at = A_ExitAt(DayStart(now));
      if (Minutes(now) >= at && TimeLocal() - lastCloseTry > 60)
      {
         int tk = MyTicketSym(sym, magic);
         if (tk >= 0)
         {
            timeExit = true;
            CloseTicket(tag, tk, A_SLIP_USD);
            lastCloseTry = TimeLocal();
         }
      }
      datetime bar = iTime(sym, PERIOD_M5, 1);
      if (bar == 0 || bar == lastBar)
         return;
      lastBar = bar;
      if (on)
         OnBar(now, bar);
   }
};
