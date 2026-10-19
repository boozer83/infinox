// SAR 단타(demo_trader.py, 이전 이름 "마틴") 전략 모듈. Infinox_Bots.mq4가 include
// 규칙은 demo_trader.py와 같음 (마틴 꺼짐 = 그날 시작 잔고 1,000$당 0.05 lot 고정):
// - 5분봉 SAR(0.02, 0.2) 전환 봉 완성 시 진입. 매수: RSI(14) > 50 이고 완성 1시간봉 SAR 상승 (매도는 반대)
// - 손절 7.5$ / 익절 3$ 서버 설정, 포지션 1개
// - 금요일 22:30 이후 진입 안 함, 23:30 청산 / 매일 15:00~21:00 진입 안 함 (서버시간)
// - 오늘 완성 1시간봉 SAR 전환 3번 이상이면 진입 안 함
// - 5분봉 ATR(14) 2.5$ 미만, 또는 최근 20일 5분봉 ATR 중앙값의 2배 초과면 진입 안 함
// - 최근 200건(실제+가상) 승률 71.4% 미만이면 가상 진입만
// - 계좌 손실 한도 50% (최고 잔고 대비 평가금)
// 상태: MQL4\Files\infinox_martin_state.txt (승률 기록, 가상 포지션, 최고 잔고)

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
#define M_WINRATE_N 200
#define M_WINRATE_MIN 71.4
#define M_ACCOUNT_STOP_PCT 50.0
#define M_SLIP_USD 0.30

string   M_ST = "infinox_martin_state.txt";
string   m_tag = "SAR 단타";
int      m_ticket = -1;
datetime m_lastBar = 0;
datetime m_lastCloseTry = 0;
datetime m_riskNotified = 0;
bool     m_fridayClose = false;
string   m_wr = "";                  // 최근 결과 '1'(익절) / '0'(손절)
int      m_vside = 0;                // 가상 포지션 방향
double   m_ventry = 0;
double   m_openLot = 0, m_openPx = 0;
int      m_openSide = 0;

string M_Desc()
{
   return StringFormat("5분봉 SAR 전환 + RSI50 + 1시간봉 SAR, 익절 %g$ / 손절 %g$, 고정 lot (1,000$당 %g lot, 마틴 없음), "
                       "5분봉 ATR %g$ 미만 쉼, ATR 20일 중앙값 %g배 초과 쉼, 최근 %d건 승률 %g%% 미만 쉼",
                       M_TP_USD, M_SL_USD, LotPer1000, M_MIN_ATR, M_MAX_ATR_MULT, M_WINRATE_N, M_WINRATE_MIN);
}

double M_BaseLot() { return FloorLot((AccountBalance() - TodayTradePnl()) / 1000 * LotPer1000); }

double M_LotFor(double base)
{
   double step = MarketInfo(Symbol(), MODE_LOTSTEP);
   double lot = MathRound(MathMin(base, MartinMaxLot) / step) * step;
   return NormalizeDouble(MathMin(MathMax(lot, MarketInfo(Symbol(), MODE_MINLOT)), MarketInfo(Symbol(), MODE_MAXLOT)), 2);
}

//---------------------------------------------------------------- 승률 쉬기
double M_WinRate(int &n)
{
   n = StringLen(m_wr);
   int w = 0;
   for (int i = 0; i < n; i++)
      if (StringGetCharacter(m_wr, i) == '1')
         w++;
   return n > 0 ? w * 100.0 / n : 0;
}

bool M_IsPaused()
{
   int n;
   double r = M_WinRate(n);
   return n >= M_WINRATE_N && r < M_WINRATE_MIN;
}

void M_RecordResult(bool win, string how)
{
   bool was = M_IsPaused();
   m_wr += win ? "1" : "0";
   if (StringLen(m_wr) > M_WINRATE_N)
      m_wr = StringSubstr(m_wr, StringLen(m_wr) - M_WINRATE_N);
   StSet(M_ST, "wr", m_wr);
   int n;
   double r = M_WinRate(n);
   Print(StringFormat("[%s] 승률 기록 (%s %s): 최근 %d건 %.1f%%", m_tag, how, win ? "익절" : "손절", n, r));
   bool paused = M_IsPaused();
   if (paused != was)
   {
      if (paused)
         Send(m_tag, StringFormat("⏸ 최근 %d건 승률 %.1f%% < %g%% -> 실제 진입을 쉬고 가상 진입만 기록합니다.\n"
                                  "승률이 %g%% 이상으로 돌아오면 자동으로 다시 시작합니다.", n, r, M_WINRATE_MIN, M_WINRATE_MIN));
      else
         Send(m_tag, StringFormat("▶️ 최근 %d건 승률 %.1f%% -> 실제 진입을 다시 시작합니다.", n, r));
   }
}

void M_SaveVirtual()
{
   StSet(M_ST, "vside", IntegerToString(m_vside));
   StSet(M_ST, "ventry", DoubleToString(m_ventry, Digits));
}

// 가상 포지션이 익절/손절 가격에 닿았는지 (매수는 Bid, 매도는 Ask 기준)
void M_CheckVirtual()
{
   if (m_vside == 0)
      return;
   RefreshRates();
   double px = m_vside == 1 ? Bid : Ask;
   double move = (px - m_ventry) * m_vside;
   if (move >= M_TP_USD || move <= -M_SL_USD)
   {
      Print(StringFormat("[%s] 가상 %s @ %.2f 종료 @ %.2f", m_tag, m_vside == 1 ? "매수" : "매도", m_ventry, px));
      m_vside = 0;
      M_SaveVirtual();
      M_RecordResult(move > 0, "가상");
   }
}

//---------------------------------------------------------------- 진입 / 청산
void M_Place(int side, double lot, datetime bar, double signalPx)
{
   if (!MarginOk(m_tag, lot))
      return;
   double fill;
   int tk = OpenMarket(m_tag, side, lot, M_SL_USD, M_TP_USD, MagicMartin, "SAR RSI50 H1 L0", M_SLIP_USD, fill);
   if (tk < 0)
      return;
   m_ticket = tk;
   m_openLot = lot;
   m_openPx = fill;
   m_openSide = side;
   m_fridayClose = false;
   Send(m_tag, StringFormat("%s 진입 %.2f lot (고정 lot)\n가격 %.2f | SL %.2f | TP %.2f (신호봉 %s 종가 %.2f)\n"
                            "최대 손실 약 %.0f$ / 목표 이익 약 %.0f$\n%s\n%s",
                            side == 1 ? "🟢 매수" : "🔴 매도", lot, fill, fill - side * M_SL_USD, fill + side * M_TP_USD,
                            TimeToString(bar, TIME_DATE | TIME_MINUTES), signalPx, M_SL_USD * lot * Contract(),
                            M_TP_USD * lot * Contract(), AccountLine(), SummaryLine(MagicMartin)));
}

void M_CheckClosed()
{
   if (m_ticket < 0 || MyTicket(MagicMartin) == m_ticket)
      return;
   double profit, px;
   string reason;
   if (!ClosedInfo(m_ticket, profit, px, reason))
      return;
   if (m_fridayClose)
      reason = "FRIDAY";
   double bal = AccountBalance();
   double pct = bal - profit > 0 ? profit / (bal - profit) * 100 : 0;
   string icon = reason == "TP" ? "✅ 익절" : reason == "SL" ? "❌ 손절" : reason == "FRIDAY" ? "⏹ 금요일 청산" : "⏹ 청산";
   Send(m_tag, StringFormat("%s %s %.2f lot\n%.2f → %.2f\n손익 %+.2f$ (%+.2f%%)\n다음 진입 고정 lot (%.2f lot)\n%s\n%s",
                            icon, m_openSide == 1 ? "BUY" : "SELL", m_openLot, m_openPx, px, profit, pct, M_LotFor(M_BaseLot()),
                            AccountLine(), SummaryLine(MagicMartin)));
   M_RecordResult(profit > 0, "실제");
   m_ticket = -1;
}

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

void M_OnBar(datetime now, datetime bar)
{
   datetime t[], ht[];
   double h[], l[], c[], hh[], hl[], hc[], r[], a[];
   bool up[], hup[];
   int n = LoadBars(PERIOD_M5, 600, t, h, l, c);
   int hn = LoadBars(PERIOD_H1, 400, ht, hh, hl, hc);
   if (n < 50 || hn < 50 || t[n - 1] != bar)
      return;
   Psar(h, l, n, up);
   Rsi(c, n, 14, r);
   // 신호봉이 끝난 시각까지 완성된 1시간봉만
   int hd = hn;
   while (hd > 0 && ht[hd - 1] + 3600 > bar + 300)
      hd--;
   if (hd < 50)
      return;
   Psar(hh, hl, hd, hup);
   int hdir = hup[hd - 1] ? 1 : -1;
   bool flip = up[n - 1] != up[n - 2];
   int side = 0;
   if (flip)
   {
      int s = up[n - 1] ? 1 : -1;
      if ((r[n - 1] - 50) * s > 0 && hdir == s)
         side = s;
   }
   string desc = StringFormat("[%s] 봉 %s 종가 %.2f: SAR %s%s RSI %.1f H1 SAR %s", m_tag, TimeToString(bar, TIME_DATE | TIME_MINUTES),
                              c[n - 1], up[n - 1] ? "상승" : "하락", flip ? "(전환)" : "", r[n - 1], hdir == 1 ? "상승" : "하락");
   if (side == 0)
   {
      Print(desc);
      return;
   }
   Print(desc, " -> 신호 ", side == 1 ? "매수" : "매도");

   int mins = Minutes(now);
   bool friday = TimeDayOfWeek(now) == 5;
   if (MyTicket(MagicMartin) >= 0)
   {
      Print("[SAR 단타] 포지션 보유 중이라 신호 무시");
      return;
   }
   if (m_vside != 0)
   {
      Print("[SAR 단타] 가상 포지션 진행 중이라 신호 무시");
      return;
   }
   if (friday && mins >= M_FRIDAY_NO_ENTRY)
   {
      Print("[SAR 단타] 금요일 마감 전이라 신규 진입 안 함");
      return;
   }
   if (mins >= M_NO_ENTRY_FROM && mins < M_NO_ENTRY_UNTIL)
   {
      Print("[SAR 단타] 미국장 제외 시간(15:00~21:00)이라 신규 진입 안 함");
      return;
   }
   // 오늘 완성된 1시간봉 SAR 전환 수 (1시간봉 전체 기준, regime_switch와 같음)
   bool hall[];
   Psar(hh, hl, hn, hall);
   int flips = 0;
   datetime d0 = DayStart(now);
   for (int i = 1; i < hn; i++)
      if (hall[i] != hall[i - 1] && DayStart(ht[i]) == d0)
         flips++;
   if (flips >= M_MAX_H1_FLIPS)
   {
      Print(StringFormat("[SAR 단타] 오늘 1시간봉 SAR 전환 %d번 (횡보)이라 신규 진입 안 함", flips));
      return;
   }
   Atr(h, l, c, n, 14, a);
   double atr = a[n - 1];
   if (atr < M_MIN_ATR)
   {
      Print(StringFormat("[SAR 단타] 5분봉 ATR %.2f$ < %g$ (조용한 시장)이라 신규 진입 안 함", atr, M_MIN_ATR));
      return;
   }
   double ref = M_AtrRef();
   if (ref > 0 && atr > M_MAX_ATR_MULT * ref)
   {
      Print(StringFormat("[SAR 단타] 5분봉 ATR %.2f$ > 20일 중앙값 %.2f$ x %g (급변동)이라 신규 진입 안 함", atr, ref, M_MAX_ATR_MULT));
      return;
   }
   string reason = RiskBlock(m_tag, M_ST, M_ACCOUNT_STOP_PCT);
   if (reason != "")
   {
      Print("[SAR 단타] 안전장치: ", reason, " -> 신규 진입 안 함");
      if (m_riskNotified != d0)
      {
         m_riskNotified = d0;
         Send(m_tag, "⛔ 안전장치: " + reason + "\n오늘은 신규 진입을 멈춥니다.");
      }
      return;
   }
   if (M_IsPaused())
   {
      RefreshRates();
      m_vside = side;
      m_ventry = side == 1 ? Ask : Bid;
      M_SaveVirtual();
      int k;
      double wr = M_WinRate(k);
      Print(StringFormat("[SAR 단타] 승률 쉬기 중 (최근 %d건 %.1f%%) -> 가상 %s @ %.2f", k, wr, side == 1 ? "매수" : "매도", m_ventry));
      return;
   }
   M_Place(side, M_LotFor(M_BaseLot()), bar, c[n - 1]);
}

//---------------------------------------------------------------- Infinox_Bots.mq4에서 부름
void M_Init()
{
   m_wr = StGet(M_ST, "wr", "");
   m_vside = (int)StringToInteger(StGet(M_ST, "vside", "0"));
   m_ventry = StringToDouble(StGet(M_ST, "ventry", "0"));
   m_lastBar = iTime(NULL, PERIOD_M5, 1);     // 켜진 뒤 새로 완성되는 봉부터 판단
   m_ticket = MyTicket(MagicMartin);
   if (m_ticket >= 0 && OrderSelect(m_ticket, SELECT_BY_TICKET))
   {
      m_openLot = OrderLots();
      m_openPx = OrderOpenPrice();
      m_openSide = OrderType() == OP_BUY ? 1 : -1;
      Print("[SAR 단타] 기존 포지션 이어받음: 티켓 ", m_ticket);
   }
}

string M_StartText()
{
   int k;
   double wr = M_WinRate(k);
   string s = StringFormat("전략: %s\nlot %.2f (오늘 시작 잔고 기준)\n안전장치: 최대 lot %g, 계좌 손실 한도 %g%%, 최근 %d건 승률 %.1f%%%s",
                           M_Desc(), M_LotFor(M_BaseLot()), MartinMaxLot, M_ACCOUNT_STOP_PCT, k, wr, M_IsPaused() ? " (쉬는 중)" : "");
   string halted = StGet(M_ST, "halted", "");
   if (halted != "")
      s += "\n🛑 계좌 손실 한도로 중단된 상태: " + halted;
   return s;
}

// 매초. on이면 신규 진입까지, 꺼져 있어도 이미 열린 포지션 관리(청산 기록, 금요일 청산)와 가상 포지션 확인은 계속
void M_Timer(bool on)
{
   datetime now = TimeCurrent();
   M_CheckClosed();
   M_CheckVirtual();
   if (TimeDayOfWeek(now) == 5 && Minutes(now) >= M_FRIDAY_CLOSE && TimeLocal() - m_lastCloseTry > 60)
   {
      int tk = MyTicket(MagicMartin);
      if (tk >= 0)
      {
         m_fridayClose = true;
         CloseTicket(m_tag, tk, M_SLIP_USD);
         m_lastCloseTry = TimeLocal();
      }
   }
   datetime bar = iTime(NULL, PERIOD_M5, 1);
   if (bar == 0 || bar == m_lastBar)
      return;
   m_lastBar = bar;
   if (on)
      M_OnBar(now, bar);
}
