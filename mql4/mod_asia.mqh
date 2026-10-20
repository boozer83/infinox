// 아시아 3마틴(asia_breakout_trader.py, 아시아 레인지 돌파 + 마틴 3단계) 전략 모듈. Infinox_Bots.mq4가 include. 시각은 모두 서버시간
// 1. 아시아 레인지 = 오늘 01:05~08:55 5분봉 최고가 / 최저가 (50봉 미만이면 그날 거래 없음)
// 2. 09:00~17:55에 시작한 5분봉 종가가 레인지 위 → 매수, 아래 → 매도. 그날 첫 돌파만 봄
// 3. 5일 모멘텀: 전일 종가 > 6거래일 전 종가면 매수만, 작으면 매도만
// 4. 기준폭 U = 0.5 x 일봉 ATR(14, 전일까지). 익절 = U, 손절 = 0.75 x U (v2). 23:00에 보유 포지션 청산
// 5. 신호 강도 (v2): 돌파 봉 종가가 전일 고가 위(매수) / 전일 저가 아래(매도)면 가중치 1.5, 아니면 0.5
// 자금관리: lot = 사이클 시작 잔고 x AsiaRiskPct% x 가중치 x 2^(단계-1) / (U x 100), 결과 0 이하면 다음 단계, 최대 3단계 (1·2·4배)
// 상태: MQL4\Files\infinox_asia_state.txt (오늘 처리 여부, 마틴 단계, 사이클 시작 잔고, 최고 잔고)

#define A_RANGE_FROM 65              // 01:05
#define A_RANGE_UNTIL 540            // 09:00
#define A_ENTRY_UNTIL 1080           // 18:00
#define A_EXIT_AT 1380               // 23:00
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

string   A_ST = "infinox_asia_state.txt";
string   a_tag = "아시아 3마틴";
int      a_ticket = -1;
datetime a_lastBar = 0;
datetime a_lastCloseTry = 0;
bool     a_timeExit = false;
double   a_openLot = 0, a_openPx = 0;
int      a_openSide = 0;
datetime a_featDay = 0;
double   a_datr = 0;
int      a_mom = 0;
double   a_ph = 0, a_pl = 0;         // 전일 고가 / 저가

string A_Desc()
{
   return StringFormat("아시아 레인지(01:05~09:00) 돌파 09:00~18:00 + %d일 모멘텀, U=일봉ATR×%g, 익절 %gU / 손절 %gU, "
                       "전일 고저 돌파면 가중치 %g 아니면 %g, 23:00 청산, 마틴 %d단계 (v2)",
                       A_MOM_DAYS, A_ATR_MULT, A_TP_RATIO, A_SL_RATIO, A_W_STRONG, A_W_WEAK, A_MARTIN_STEPS);
}

int A_Level() { return (int)StringToInteger(StGet(A_ST, "level", "0")); }

// 오늘 시작 시점에 알 수 있는 값: 일봉 ATR14(전일까지), 5일 모멘텀 방향
void A_DailyFeatures(datetime today)
{
   a_featDay = today;
   a_datr = 0;
   a_mom = 0;
   int s0 = 0;
   while (s0 < iBars(NULL, PERIOD_D1) && iTime(NULL, PERIOD_D1, s0) >= today)
      s0++;
   int n = MathMin(400, iBars(NULL, PERIOD_D1) - s0);
   if (n < 30)
      return;
   double h[], l[], c[], a[];
   ArrayResize(h, n);
   ArrayResize(l, n);
   ArrayResize(c, n);
   for (int i = 0; i < n; i++)
   {
      int s = s0 + n - 1 - i;
      h[i] = iHigh(NULL, PERIOD_D1, s);
      l[i] = iLow(NULL, PERIOD_D1, s);
      c[i] = iClose(NULL, PERIOD_D1, s);
   }
   Atr(h, l, c, n, 14, a);
   a_datr = a[n - 1];
   a_ph = h[n - 1];
   a_pl = l[n - 1];
   double d = c[n - 1] - c[n - 1 - A_MOM_DAYS];
   a_mom = d > 0 ? 1 : d < 0 ? -1 : 0;
}

// 사이클 시작 잔고(1단계 진입 때)는 상태 파일에 보관. 반환: 이번 lot, risk = 손절 시 잔고 대비 %
double A_PlanLot(int level, double u, double w, bool save, double &risk)
{
   double bal = AccountBalance();
   string saved = StGet(A_ST, "cycle_bal", "");
   double cyc = level == 0 || saved == "" ? bal : StringToDouble(saved);
   if (save)
      StSet(A_ST, "cycle_bal", DoubleToString(cyc, 2));
   double lot = FloorLot(cyc * AsiaRiskPct / 100 * w * MathPow(2, level) / (u * Contract()));
   risk = bal > 0 ? lot * u * A_SL_RATIO * Contract() / bal * 100 : 0;
   return lot;
}

void A_CheckClosed()
{
   if (a_ticket < 0 || MyTicket(MagicAsia) == a_ticket)
      return;
   double profit, px;
   string reason;
   if (!ClosedInfo(a_ticket, profit, px, reason))
      return;
   if (a_timeExit)
      reason = "TIME";
   int lvl = A_Level();
   int nxt = profit > 0 ? 0 : (lvl + 1 < A_MARTIN_STEPS ? lvl + 1 : 0);
   StSet(A_ST, "level", IntegerToString(nxt));
   string icon = reason == "TP" ? "✅ 익절" : reason == "SL" ? "❌ 손절" : reason == "TIME" ? "⏹ 23:00 청산" : "⏹ 청산";
   Send(a_tag, StringFormat("%s %s %.2f lot\n%.2f → %.2f\n손익 %+.2f$\n다음 진입: 마틴 %d/%d단계\n%s\n%s",
                            icon, a_openSide == 1 ? "BUY" : "SELL", a_openLot, a_openPx, px, profit, nxt + 1, A_MARTIN_STEPS,
                            AccountLine(), SummaryLine(MagicAsia)));
   a_ticket = -1;
}

void A_OnBar(datetime now, datetime bar)
{
   datetime today = DayStart(now);
   int bm = Minutes(bar);
   if (DayStart(bar) != today || bm < A_RANGE_UNTIL || bm >= A_ENTRY_UNTIL)
      return;
   string todayStr = TimeToString(today, TIME_DATE);
   if (StGet(A_ST, "done_date", "") == todayStr)
      return;
   if (a_featDay != today)
      A_DailyFeatures(today);
   // 오늘 5분봉: 레인지 / 진입 구간 (shift가 클수록 오래된 봉)
   double hi = -1, lo = 1e9;
   int nr = 0, last = 0;
   for (int s = 1; s < iBars(NULL, PERIOD_M5) && iTime(NULL, PERIOD_M5, s) >= today; s++)
   {
      last = s;
      int m = Minutes(iTime(NULL, PERIOD_M5, s));
      if (m >= A_RANGE_FROM && m < A_RANGE_UNTIL)
      {
         hi = MathMax(hi, iHigh(NULL, PERIOD_M5, s));
         lo = MathMin(lo, iLow(NULL, PERIOD_M5, s));
         nr++;
      }
   }
   if (nr < A_MIN_RANGE_BARS || a_datr <= 0)
   {
      Print("[아시아 3마틴] 레인지 봉 부족 또는 일봉 ATR 없음 -> 오늘 거래 없음");
      StSet(A_ST, "done_date", todayStr);
      return;
   }
   datetime bt = 0;
   int side = 0;
   double close = 0;
   for (int s = last; s >= 1 && side == 0; s--)
   {
      int m = Minutes(iTime(NULL, PERIOD_M5, s));
      if (m < A_RANGE_UNTIL || m >= A_ENTRY_UNTIL)
         continue;
      double c = iClose(NULL, PERIOD_M5, s);
      if (c > hi || c < lo)
      {
         side = c > hi ? 1 : -1;
         bt = iTime(NULL, PERIOD_M5, s);
         close = c;
      }
   }
   if (side == 0)
      return;
   StSet(A_ST, "done_date", todayStr);   // 첫 돌파가 나오면 결과와 상관없이 오늘은 끝
   double u = a_datr * A_ATR_MULT, slUsd = u * A_SL_RATIO, tpUsd = u * A_TP_RATIO;
   bool strong = side == 1 ? close > a_ph : close < a_pl;
   double w = strong ? A_W_STRONG : A_W_WEAK;
   string desc = StringFormat("[아시아 3마틴] 봉 %s 종가 %.2f %s 돌파 (레인지 %.2f~%.2f, %d봉), 5일 모멘텀 %+d, 일봉 ATR %.2f, "
                              "전일 고/저 %.2f/%.2f -> %s 신호 x%g",
                              TimeToString(bt, TIME_DATE | TIME_MINUTES), close, side == 1 ? "상향" : "하향", lo, hi, nr, a_mom, a_datr,
                              a_ph, a_pl, strong ? "강한" : "약한", w);
   if (bt != bar)
   {
      Print(desc, " -> 첫 돌파가 이미 지난 봉이라 (재시작 등) 오늘은 진입 안 함");
      return;
   }
   if (side != a_mom)
   {
      Print(desc, " -> 모멘텀과 반대라 오늘은 거래 없음");
      return;
   }
   if (MyTicket(MagicAsia) >= 0)
   {
      Print(desc, " -> 포지션 보유 중이라 진입 안 함");
      return;
   }
   int lvl = A_Level();
   double risk;
   double lot = A_PlanLot(lvl, u, w, true, risk);
   string reason = RiskBlock(a_tag, A_ST, A_ACCOUNT_STOP_PCT);
   if (reason == "" && lot > AsiaMaxLot)
      reason = StringFormat("진입 lot %.2f이 최대 lot %g을 넘음", lot, AsiaMaxLot);
   if (reason != "")
   {
      Print(desc, " -> 안전장치: ", reason);
      Send(a_tag, "⛔ 안전장치: " + reason + "\n오늘은 진입하지 않습니다.");
      return;
   }
   Print(desc, " -> 진입");
   if (!MarginOk(a_tag, lot))
      return;
   double fill;
   int tk = OpenMarket(a_tag, side, lot, slUsd, tpUsd, MagicAsia, StringFormat("asia brk L%d", lvl), A_SLIP_USD, fill);
   if (tk < 0)
      return;
   a_ticket = tk;
   a_openLot = lot;
   a_openPx = fill;
   a_openSide = side;
   a_timeExit = false;
   Send(a_tag, StringFormat("%s 진입 %.2f lot (마틴 %d/%d단계, %s 신호 x%g)\n레인지 %.2f ~ %.2f 돌파, 일봉 ATR %.2f\n가격 %.2f | SL %.2f | TP %.2f\n"
                            "최대 손실 약 %.0f$ (잔고의 %.2f%%) / 목표 이익 약 %.0f$\n%s\n%s",
                            side == 1 ? "🟢 매수" : "🔴 매도", lot, lvl + 1, A_MARTIN_STEPS, strong ? "강한" : "약한", w, lo, hi, a_datr,
                            fill, fill - side * slUsd, fill + side * tpUsd, slUsd * lot * Contract(), risk, tpUsd * lot * Contract(),
                            AccountLine(), SummaryLine(MagicAsia)));
}

//---------------------------------------------------------------- Infinox_Bots.mq4에서 부름
void A_Init()
{
   a_lastBar = iTime(NULL, PERIOD_M5, 1);
   a_ticket = MyTicket(MagicAsia);
   if (a_ticket >= 0 && OrderSelect(a_ticket, SELECT_BY_TICKET))
   {
      a_openLot = OrderLots();
      a_openPx = OrderOpenPrice();
      a_openSide = OrderType() == OP_BUY ? 1 : -1;
      Print("[아시아 3마틴] 기존 포지션 이어받음: 티켓 ", a_ticket);
   }
}

string A_StartText()
{
   A_DailyFeatures(DayStart(TimeCurrent()));
   int lvl = A_Level();
   string s = StringFormat("전략: %s\n다음 진입 마틴 %d단계", A_Desc(), lvl + 1);
   if (a_datr <= 0)
      return s + "\n일봉 데이터 부족";
   double u = a_datr * A_ATR_MULT;
   s += StringFormat("\n오늘 일봉 ATR %.2f$ → U %.2f$ (손절 %.2f$ / 익절 %.2f$), 5일 모멘텀 %s", a_datr, u, u * A_SL_RATIO,
                     u * A_TP_RATIO, a_mom > 0 ? "+" : a_mom < 0 ? "-" : "0");
   for (int k = 0; k < 2; k++)
   {
      double w = k == 0 ? A_W_STRONG : A_W_WEAK, risk;
      double lot = A_PlanLot(lvl, u, w, false, risk);
      double aim = AsiaRiskPct * w * A_SL_RATIO * MathPow(2, lvl);
      s += StringFormat("\n%d단계 %s 신호 %.2f lot (손절 시 잔고의 %.2f%%, 목표 %.2f%%%s)", lvl + 1, k == 0 ? "강한" : "약한", lot, risk,
                        aim, risk > aim * 1.3 ? " — 최소 lot 때문에 큼" : "");
   }
   return s;
}

// 꺼져 있어도 이미 열린 포지션 관리(청산 기록, 23:00 청산)는 계속
void A_Timer(bool on)
{
   datetime now = TimeCurrent();
   A_CheckClosed();
   if (Minutes(now) >= A_EXIT_AT && TimeLocal() - a_lastCloseTry > 60)
   {
      int tk = MyTicket(MagicAsia);
      if (tk >= 0)
      {
         a_timeExit = true;
         CloseTicket(a_tag, tk, A_SLIP_USD);
         a_lastCloseTry = TimeLocal();
      }
   }
   datetime bar = iTime(NULL, PERIOD_M5, 1);
   if (bar == 0 || bar == a_lastBar)
      return;
   a_lastBar = bar;
   if (on)
      A_OnBar(now, bar);
}
