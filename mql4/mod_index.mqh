// 나스닥·독일 지수 장중(index_trader.py) 전략 모듈. Infinox_Bots.mq4가 include. 종목마다 CIndex 하나
// NAS100 첫 30분 방향: 본장(뉴욕 09:30 = 서버 16:30) 첫 30분 1분봉 움직임(마지막 종가 - 첫 시가)이 ATR x 0.2 넘으면
//   그 방향으로 시가 30분 + 0.3초 뒤 시장가 진입, 손절 = 진입 순간 Bid - ATR (매수) / Ask + ATR (매도), MT4 SL로 걺
// GER40 ORB 30분 매도만: Xetra 09:00(서버시간으로 바꿈) 처음 30분 저가를 마감된 1분봉 종가가 처음 깨면 다음 분 + 0.3초 뒤 매도
//   (진입이 시가 + 3시간 전일 때만). 손절 = Bid가 범위 가운데에 닿으면 시장가 청산 (EA가 매초 확인) + 만일용 MT4 SL (가운데 + 스프레드 + 20pt)
// 청산: 마감 2초 전 시장가 (안 닫히면 다시). ATR = 전날까지 본장 고가-저가 14일 평균 (1분봉). 휴장일·미국 조기 마감 반영
// lot = 잔고의 0.25% 위험 / 1 lot 손실, 최소 lot 위험이 0.5% 넘으면 그날 건너뜀. 하루 1번 (오늘 이 매직으로 진입한 기록이 있으면 안 함)
// 서버시간 = 뉴욕 + 7시간 (MT5 파이썬 봇과 같은 서버 기준). 1분봉이 빠진 분은 그대로 (파이썬은 틱으로 채움 — MT4엔 틱 이력이 없음)
// 상태: MQL4\Files\<상태 파일> (그날 처리 여부, GER 범위)

#define X_FIRST_MIN 0.2
#define X_FIRST_STOP 1.0
#define X_ORB_MIN 30
#define X_ENTRY_UNTIL 180
#define X_ATR_DAYS 14
#define X_RISK_PCT 0.25
#define X_MAX_RISK_PCT 0.5
#define X_ENTRY_DELAY_MS 300
#define X_LATE_SEC 60
#define X_CLOSE_BEFORE 2
#define X_SAFETY_SL_PTS 20.0
#define X_SLIP_PTS 5.0

//---------------------------------------------------------------- 휴장일 (common/exchange_calendar.py와 같음, 2026~2027. 해가 바뀌기 전에 추가)
string X_US_HOLIDAYS = "2026.01.01 2026.01.19 2026.02.16 2026.04.03 2026.05.25 2026.06.19 2026.07.03 2026.09.07 2026.11.26 2026.12.25 "
                       "2027.01.01 2027.01.18 2027.02.15 2027.03.26 2027.05.31 2027.06.18 2027.07.05 2027.09.06 2027.11.25 2027.12.24";
string X_US_EARLY = "2026.11.27 2026.12.24 2027.11.26";

bool X_InList(string list, datetime d) { return StringFind(list, TimeToString(d, TIME_DATE)) >= 0; }

datetime X_Easter(int y)
{
   int a = y % 19, b = y / 100, c = y % 100, d = b / 4, e = b % 4, f = (b + 8) / 25, g = (b - f + 1) / 3;
   int h = (19 * a + b - d - g + 15) % 30, i = c / 4, k = c % 4, l = (32 + 2 * e + 2 * i - h - k) % 7, m = (a + 11 * h + 22 * l) / 451;
   int month = (h + l - 7 * m + 114) / 31, day = (h + l - 7 * m + 114) % 31 + 1;
   return StringToTime(StringFormat("%04d.%02d.%02d", y, month, day));
}

bool X_UsClosed(datetime d) { return TimeDayOfWeek(d) == 0 || TimeDayOfWeek(d) == 6 || X_InList(X_US_HOLIDAYS, d); }

bool X_DeClosed(datetime d)
{
   int w = TimeDayOfWeek(d), mo = TimeMonth(d), dy = TimeDay(d);
   if (w == 0 || w == 6)
      return true;
   if ((mo == 1 && dy == 1) || (mo == 5 && dy == 1) || (mo == 12 && (dy == 24 || dy == 25 || dy == 26 || dy == 31)))
      return true;
   int diff = (int)((d - X_Easter(TimeYear(d))) / 86400);
   return diff == -2 || diff == 1;
}

// 서머타임 (날짜 기준, 바뀌는 날은 일요일이라 평일 본장에는 상관없음)
bool X_NyDst(datetime d)
{
   int y = TimeYear(d);
   return d >= A_NthWeekday(y, 3, 0, 2) && d < A_NthWeekday(y, 11, 0, 1);
}

bool X_DeDst(datetime d)
{
   int y = TimeYear(d);
   return d >= A_NthWeekday(y, 3, 0, 0) && d < A_NthWeekday(y, 10, 0, 0);
}

//---------------------------------------------------------------- 전략 (종목 하나)
class CIndex : public CStrat
{
public:
   string   sym, tag, st, kind;      // kind: "NAS100" / "GER40"
   int      magic;
   bool     ok;
   datetime okTry;
   datetime lastTry;

   void Setup(string s, string k, string panelName, string file, int mg, bool on)
   {
      sym = s;
      kind = k;
      name = panelName;
      tag = panelName;
      st = file;
      magic = mg;
      defOn = on;
      lastTry = 0;
      okTry = TimeLocal();
      ok = UseSymbol(sym);
      if (!ok)
         Print("[", tag, "] 종목 ", sym, "을 찾을 수 없어 이 전략은 쉽니다");
   }

   int    Magic() { return magic; }
   string Sym() { return sym; }
   bool   Nas() { return kind == "NAS100"; }

   string Desc()
   {
      if (Nas())
         return StringFormat("%s 본장 첫 30분이 ATR x %g 넘게 움직이면 그 방향, 손절 ATR x %g, 마감 %d초 전 청산, 거래당 위험 %g%%",
                             sym, X_FIRST_MIN, X_FIRST_STOP, X_CLOSE_BEFORE, X_RISK_PCT);
      return StringFormat("%s 본장 처음 %d분 저가를 1분봉 종가가 처음 깨면 매도 (진입이 시가 + %d시간 전일 때), 손절 범위 가운데, 마감 %d초 전 청산, 거래당 위험 %g%%",
                          sym, X_ORB_MIN, X_ENTRY_UNTIL / 60, X_CLOSE_BEFORE, X_RISK_PCT);
   }

   // 그날 본장 (서버시간 시작, 끝). 휴장일이면 false
   bool Bounds(datetime d, datetime &so, datetime &sc)
   {
      if (Nas())
      {
         if (X_UsClosed(d))
            return false;
         so = d + (9 * 60 + 30) * 60 + 7 * 3600;
         sc = d + (X_InList(X_US_EARLY, d) ? 13 * 60 : 16 * 60) * 60 + 7 * 3600;
         return true;
      }
      if (X_DeClosed(d))
         return false;
      int off = (X_NyDst(d) ? -4 : -5) - (X_DeDst(d) ? 2 : 1) + 7;      // 프랑크푸르트 -> 서버
      so = d + 9 * 3600 + off * 3600;
      sc = d + (17 * 60 + 30) * 60 + off * 3600;
      return true;
   }

   // t0 <= 봉 시작 < t1 이고 이미 마감된 1분봉: 개수, 고가·저가, 첫 시가·마지막 종가, 첫 봉 시각
   int Range(datetime t0, datetime t1, double &hi, double &lo, double &op, double &cl, datetime &first)
   {
      hi = -1;
      lo = 1e18;
      int n = 0;
      int bars = iBars(sym, PERIOD_M1);
      int s = iBarShift(sym, PERIOD_M1, t1 - 1, false);
      if (s < 0)
         return 0;
      datetime now = TimeCurrent();
      for (; s < bars; s++)
      {
         datetime bt = iTime(sym, PERIOD_M1, s);
         if (bt >= t1 || bt + 60 > now)
            continue;
         if (bt < t0)
            break;
         if (n == 0)
            cl = iClose(sym, PERIOD_M1, s);
         hi = MathMax(hi, iHigh(sym, PERIOD_M1, s));
         lo = MathMin(lo, iLow(sym, PERIOD_M1, s));
         op = iOpen(sym, PERIOD_M1, s);
         first = bt;
         n++;
      }
      return n;
   }

   // 전날까지 본장 고가-저가 14일 평균 (0이면 계산 안 됨)
   double AtrOf(datetime d)
   {
      double sum = 0;
      int got = 0;
      for (int k = 1; k <= 60 && got < X_ATR_DAYS; k++)
      {
         datetime so, sc, first;
         if (!Bounds(d - k * 86400, so, sc))
            continue;
         double hi, lo, op, cl;
         if (Range(so, sc, hi, lo, op, cl, first) < 60)
            continue;
         sum += hi - lo;
         got++;
      }
      return got == X_ATR_DAYS ? sum / got : 0;
   }

   int Ticket() { return MyTicketSym(sym, magic); }

   // 오늘(서버 날짜) 이 매직으로 진입한 기록이 있으면 true (열린 것·닫힌 것 모두)
   bool EnteredToday(datetime d)
   {
      for (int i = OrdersTotal() - 1; i >= 0; i--)
         if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES) && OrderMagicNumber() == magic && OrderSymbol() == sym && OrderOpenTime() >= d)
            return true;
      for (int i = OrdersHistoryTotal() - 1; i >= 0; i--)
         if (OrderSelect(i, SELECT_BY_POS, MODE_HISTORY) && OrderType() <= OP_SELL && OrderMagicNumber() == magic && OrderSymbol() == sym
             && OrderOpenTime() >= d)
            return true;
      return false;
   }

   void Done(string day, string why)
   {
      StSet(st, "done_date", day);
      StSet(st, "done_why", why);
      Print("[", tag, "] 오늘 끝: ", why);
   }

   void Enter(string day, int side, double sl, string signal)
   {
      RefreshRates();
      double bid = MarketInfo(sym, MODE_BID), ask = MarketInfo(sym, MODE_ASK);
      double price = side > 0 ? ask : bid, dist = MathAbs(price - sl);
      double minDist = MarketInfo(sym, MODE_STOPLEVEL) * MarketInfo(sym, MODE_POINT);
      if ((side > 0 && sl >= bid - minDist) || (side < 0 && sl <= ask + minDist))
      {
         Done(day, StringFormat("손절가 %.2f가 현재가 쪽으로 너무 가까움", sl));
         Send(tag, "⚠️ 진입 건너뜀: 손절가가 너무 가까움");
         return;
      }
      double per = dist * PerLot(sym), bal = AccountBalance();
      double step = MarketInfo(sym, MODE_LOTSTEP), minLot = MarketInfo(sym, MODE_MINLOT);
      double lot = MathFloor(bal * X_RISK_PCT / 100 / per / step + 1e-9) * step;
      lot = MathMin(MathMax(lot, minLot), MathMin(MarketInfo(sym, MODE_MAXLOT), 1.0));
      lot = NormalizeDouble(lot, 2);
      if (per <= 0 || lot * per > bal * X_MAX_RISK_PCT / 100)
      {
         Done(day, StringFormat("최소 lot 위험 %.2f$가 %g%% 넘음", lot * per, X_MAX_RISK_PCT));
         Send(tag, StringFormat("⏭ 진입 건너뜀: 최소 lot 위험 %.2f$가 잔고의 %g%% 넘음", lot * per, X_MAX_RISK_PCT));
         return;
      }
      Done(day, "진입");                   // 주문 결과와 상관없이 오늘은 끝 (다시 보내지 않게)
      if (!MarginOkSym(tag, sym, lot))
         return;
      double fill;
      int tk = OpenMarketSym(tag, sym, side, lot, 0, 0, magic, "IDX " + kind, X_SLIP_PTS, fill);
      if (tk < 0)
         return;
      int digits = (int)MarketInfo(sym, MODE_DIGITS);
      bool slOk = false;
      for (int i = 0; i < 5 && !slOk; i++)
      {
         slOk = OrderModify(tk, fill, NormalizeDouble(sl, digits), 0, 0, clrNONE);
         if (!slOk)
            Sleep(300);
      }
      if (!slOk)
         Send(tag, StringFormat("❗ 손절 설정 실패 (티켓 %d, 오류 %d). MT4에서 직접 확인하세요.", tk, GetLastError()));
      Send(tag, StringFormat("%s %s %.2f lot @ %.2f (요청 %.2f, 스프레드 %.2f)\n%s\n손절 %.2f (위험 약 %.2f$), 마감 전 청산\n%s",
                             side > 0 ? "📈" : "📉", side > 0 ? "매수" : "매도", lot, fill, price, ask - bid, signal, sl,
                             MathAbs(fill - sl) * lot * PerLot(sym), AccountLine()));
   }

   void ReportClosed()
   {
      int last = (int)StringToInteger(StGet(st, "ticket", "-1"));
      if (last < 0)
         return;
      double profit, px;
      string reason;
      if (!ClosedInfo(last, profit, px, reason))
         return;
      string why = StGet(st, "closing", "");
      if (why != "")
         reason = why;
      Send(tag, StringFormat("%s 청산 (%s) @ %.2f: %+.2fpt, %+.2f$\n%s\n%s", profit > 0 ? "✅" : "❌", reason, px,
                             OrderSelect(last, SELECT_BY_TICKET) ? (px - OrderOpenPrice()) * (OrderType() == OP_BUY ? 1 : -1) : 0,
                             profit, AccountLine(), SummaryLine(magic, sym)));
      StSet(st, "ticket", "-1");
      StSet(st, "closing", "");
   }

   //---- Infinox_Bots.mq4에서 부름
   void Init()
   {
      if (!ok)
         return;
      int tk = Ticket();
      if (tk >= 0)
      {
         StSet(st, "ticket", IntegerToString(tk));
         Print("[", tag, "] 기존 포지션 이어받음: 티켓 ", tk);
      }
   }

   string StartText()
   {
      if (!ok)
         return StringFormat("⚠️ 종목 %s을 찾을 수 없어 이 전략은 쉽니다 (MT4 시장 감시 창에 종목이 있는지 확인)", sym);
      datetime d = DayStart(TimeCurrent()), so, sc;
      string s = "전략: " + Desc();
      if (Bounds(d, so, sc))
         s += StringFormat("\n오늘 본장 %s~%s (서버), ATR %.1f", TimeToString(so, TIME_MINUTES), TimeToString(sc, TIME_MINUTES), AtrOf(d));
      else
         s += "\n오늘 휴장";
      return s;
   }

   // 꺼져 있어도 열린 포지션 관리(GER 가운데 손절, 마감 청산)는 계속. 진입은 on일 때만
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
      datetime now = TimeCurrent(), d = DayStart(now), so = 0, sc = 0;
      string day = TimeToString(d, TIME_DATE);
      bool open = Bounds(d, so, sc);
      ReportClosed();
      // 1) 열린 포지션: GER 가운데 손절, 마감·지난 날 청산
      int tk = Ticket();
      if (tk >= 0 && OrderSelect(tk, SELECT_BY_TICKET))
      {
         StSet(st, "ticket", IntegerToString(tk));
         datetime pday = DayStart(OrderOpenTime());
         string why = "";
         if (!open || pday != d || now >= sc - X_CLOSE_BEFORE)
            why = pday == d ? "마감 청산" : "지난 날 포지션 청산";
         else if (!Nas())
         {
            double mid = StringToDouble(StGet(st, "mid", "0"));
            if (mid <= 0 || StGet(st, "mid_date", "") != TimeToString(pday, TIME_DATE))
            {
               datetime pso, psc, first;
               double hi, lo, op, cl;
               if (Bounds(pday, pso, psc) && Range(pso, pso + X_ORB_MIN * 60, hi, lo, op, cl, first) >= 20)
               {
                  mid = (hi + lo) / 2;
                  StSet(st, "mid", DoubleToString(mid, 2));
                  StSet(st, "mid_date", TimeToString(pday, TIME_DATE));
               }
               else
                  why = "범위 복원 실패, 정리";
            }
            if (why == "" && mid > 0 && MarketInfo(sym, MODE_BID) >= mid)
               why = "SL";
         }
         if (why != "" && TimeLocal() - lastTry >= 1)
         {
            lastTry = TimeLocal();
            StSet(st, "closing", why);
            CloseTicket(tag, tk, X_SLIP_PTS);
         }
         return;
      }
      if (!on || !open || now < so || now >= sc || StGet(st, "done_date", "") == day)
         return;
      if (EnteredToday(d))
      {
         Done(day, "오늘 이미 진입함 (주문 기록)");
         return;
      }
      double atr = StGet(st, "atr_date", "") == day ? StringToDouble(StGet(st, "atr", "0")) : 0;
      if (atr <= 0)
      {
         atr = AtrOf(d);
         if (atr <= 0)
         {
            Done(day, "ATR 없음");
            Send(tag, "⚠️ ATR 계산할 1분봉이 부족해서 오늘은 쉽니다");
            return;
         }
         StSet(st, "atr_date", day);
         StSet(st, "atr", DoubleToString(atr, 2));
      }
      double hi, lo, op, cl;
      datetime first;
      // 2) 신호
      if (Nas())
      {
         datetime planned = so + 30 * 60;           // + 0.3초 (MT4 시각은 초 단위라 1초 타이머의 다음 틱에 들어감)
         if (now < planned)
            return;
         if (now - planned > X_LATE_SEC)
         {
            Done(day, StringFormat("늦게 켜짐 (%d초)", (int)(now - planned)));
            return;
         }
         int n = Range(so, planned, hi, lo, op, cl, first);
         if (n == 0 || first > so + 5 * 60)
         {
            Done(day, "첫 30분 봉 없음");
            return;
         }
         double mv = cl - op;
         if (MathAbs(mv) < X_FIRST_MIN * atr)
         {
            Done(day, StringFormat("움직임 %+.1f < ATR %.0f x %g", mv, atr, X_FIRST_MIN));
            return;
         }
         int side = mv > 0 ? 1 : -1;
         RefreshRates();
         double sl = side > 0 ? MarketInfo(sym, MODE_BID) - X_FIRST_STOP * atr : MarketInfo(sym, MODE_ASK) + X_FIRST_STOP * atr;
         Enter(day, side, sl, StringFormat("첫 30분 %+.1fpt (ATR %.0f)", mv, atr));
         return;
      }
      datetime rngEnd = so + X_ORB_MIN * 60, cur = now - now % 60;
      if (cur < rngEnd + 60)
         return;
      if (StGet(st, "range_date", "") != day)
      {
         if (Range(so, rngEnd, hi, lo, op, cl, first) < 20)
         {
            Done(day, "범위 봉 부족");
            return;
         }
         StSet(st, "range_date", day);
         StSet(st, "hi", DoubleToString(hi, 2));
         StSet(st, "lo", DoubleToString(lo, 2));
         StSet(st, "mid", DoubleToString((hi + lo) / 2, 2));
         StSet(st, "mid_date", day);
      }
      double rlo = StringToDouble(StGet(st, "lo", "0")), rhi = StringToDouble(StGet(st, "hi", "0")), mid = StringToDouble(StGet(st, "mid", "0"));
      // 범위 이후 마감된 1분봉을 오래된 것부터: 종가가 처음 저가 밑인 봉
      datetime brk = 0;
      int s0 = iBarShift(sym, PERIOD_M1, rngEnd, false);
      for (int s = s0; s >= 1; s--)
      {
         datetime bt = iTime(sym, PERIOD_M1, s);
         if (bt < rngEnd || bt + 60 > now)
            continue;
         if (iClose(sym, PERIOD_M1, s) < rlo)
         {
            brk = bt;
            break;
         }
      }
      if (brk == 0)
      {
         if (cur >= so + X_ENTRY_UNTIL * 60)
            Done(day, "3시간 안에 돌파 없음");
         return;
      }
      datetime planned = brk + 60;
      if (planned >= so + X_ENTRY_UNTIL * 60)
         Done(day, "3시간 안에 돌파 없음");
      else if (now - planned > X_LATE_SEC)
         Done(day, StringFormat("돌파(%s 봉)를 늦게 봄 (%d초)", TimeToString(brk, TIME_MINUTES), (int)(now - planned)));
      else if (now >= planned)
      {
         RefreshRates();
         double spread = MarketInfo(sym, MODE_ASK) - MarketInfo(sym, MODE_BID);
         Enter(day, -1, mid + spread + X_SAFETY_SL_PTS,
               StringFormat("30분 저가 %.1f 깸 (%s 봉, 범위 %.1fpt)", rlo, TimeToString(brk, TIME_MINUTES), rhi - rlo));
      }
   }
};
