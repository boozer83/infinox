// 헤지 마틴(hedge_martin_trader.py) 전략 모듈. Infinox_Bots.mq4가 include
// - 사이클 시작: SAR 단타와 같은 신호 봉(5분봉 SAR 전환 + RSI50 + 1시간봉 SAR + ATR/과열 필터, 밤에도 시작)이 완성되면
//   매수·매도를 같은 lot(L0)으로 동시 진입. 중요 지표 앞뒤 H_NEWS_MIN분이면 시작 안 함
//   (지표 시각은 파이썬 서프라이즈 봇이 공용 폴더 Common\Files\infinox_news_times.txt에 써 줌. 없으면 거르지 않음)
//   L0 = 사이클 시작 잔고 1,000달러당 HedgeLotPer1000 lot (0.01 단위 내림, 최소 0.01)
// - 모든 포지션은 진입가 ± H_TP_USD에 서버 익절, 손절 없음 (진 쪽은 들고 감)
// - 사이클 중 5분봉 SAR이 전환될 때마다 그 방향에 열린 포지션이 없으면 다음 단계 lot(L0 x 2^단계)으로 1개 진입 (최대 H_MAX_LEVEL)
// - 사이클 합계(이번 사이클 확정 손익 + 평가손익)가 시작 잔고의 +H_TARGET%면 전부 청산, -H_LIMIT%면 전부 청산
//   청산은 포지션이 0개가 될 때까지 다시 하고, 그 뒤에 사이클을 끝냄
// - 주말에도 보유. 신호 봉 마감 H_LATE_SEC초 넘게 지난 신호로는 시작·추가 안 함
// 상태: MQL4\Files\infinox_hedge_state.txt (사이클 시작 시각·잔고·단계·청산 중)

#define H_TP_USD 5.0
#define H_TARGET 0.01
#define H_LIMIT 0.30
#define H_MAX_LEVEL 4
#define H_NEWS_MIN 30
#define H_LATE_SEC 60
#define H_RETRY_SEC 5
#define H_SLIP_USD 0.30
#define H_NEWS_FILE "infinox_news_times.txt"

class CHedge : public CStrat
{
public:
   string   tag, st;
   int      magic;
   datetime lastBar, lastRetry, newsLoaded;
   datetime news[];
   bool     newsWarned;

   void Setup(string panelName, string file, int mg, bool on)
   {
      name = panelName;
      tag = panelName;
      st = file;
      magic = mg;
      defOn = on;
      lastBar = 0;
      lastRetry = 0;
      newsLoaded = 0;
      newsWarned = false;
   }

   int Magic() { return magic; }

   string Desc()
   {
      return StringFormat("SAR 단타 신호에 매수·매도 동시 진입, 각자 익절 %g$ / 손절 없음, SAR 전환마다 2배 추가 (최대 %d단계), "
                          "사이클 +%g%% 청산 / -%g%% 손절, 뉴스 앞뒤 %d분 시작 안 함, 밤에도 시작",
                          H_TP_USD, H_MAX_LEVEL, H_TARGET * 100, H_LIMIT * 100, H_NEWS_MIN);
   }

   double Lot0(double bal) { return MathMax(0.01, MathFloor(bal / 1000 * HedgeLotPer1000 * 100 + 1e-9) / 100); }

   bool     InCycle() { return StGet(st, "since", "") != ""; }
   datetime Since() { return (datetime)StringToInteger(StGet(st, "since", "0")); }
   double   StartBal() { return StringToDouble(StGet(st, "start_bal", "0")); }
   int      Level() { return (int)StringToInteger(StGet(st, "level", "0")); }
   double   L0() { return StringToDouble(StGet(st, "l0", "0.01")); }

   void Clear()
   {
      StSet(st, "since", "");
      StSet(st, "closing", "");
      StSet(st, "level", "0");
   }

   // 이 전략의 열린 포지션: 개수, 매수·매도 있는지, 평가손익 합
   int Open(bool &hasBuy, bool &hasSell, double &floating)
   {
      hasBuy = false;
      hasSell = false;
      floating = 0;
      int n = 0;
      for (int i = OrdersTotal() - 1; i >= 0; i--)
         if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES) && OrderSymbol() == Symbol() && OrderMagicNumber() == magic && OrderType() <= OP_SELL)
         {
            n++;
            if (OrderType() == OP_BUY)
               hasBuy = true;
            else
               hasSell = true;
            floating += OrderProfit() + OrderSwap() + OrderCommission();
         }
      return n;
   }

   // since 뒤로 청산된 이 전략의 손익
   double Realized(datetime since)
   {
      double s = 0;
      for (int i = OrdersHistoryTotal() - 1; i >= 0; i--)
         if (OrderSelect(i, SELECT_BY_POS, MODE_HISTORY) && OrderType() <= OP_SELL && OrderMagicNumber() == magic
             && OrderSymbol() == Symbol() && OrderCloseTime() >= since)
            s += OrderProfit() + OrderSwap() + OrderCommission();
      return s;
   }

   void CloseAll()
   {
      for (int i = OrdersTotal() - 1; i >= 0; i--)
         if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES) && OrderSymbol() == Symbol() && OrderMagicNumber() == magic && OrderType() <= OP_SELL)
            CloseTicket(tag, OrderTicket(), H_SLIP_USD);
   }

   void EndCycle(string why)
   {
      double total = Realized(Since()), bal0 = StartBal();
      string icon = why == "target" || why == "tp" ? "✅" : why == "limit" ? "🛑" : "⚠️";
      string msg = why == "target" ? StringFormat("사이클 목표 +%g%% 도달", H_TARGET * 100)
                 : why == "limit" ? StringFormat("사이클 손실 한도 -%g%%", H_LIMIT * 100)
                 : why == "tp" ? "포지션 모두 익절" : "최대 단계 다 쓰고 모두 익절";
      Send(tag, StringFormat("%s 사이클 끝: %s\n합계 %+.2f$ (시작 잔고 %.2f의 %+.2f%%), 최대 단계 L%d\n%s",
                             icon, msg, total, bal0, bal0 > 0 ? total / bal0 * 100 : 0, Level(), AccountLine()));
      Clear();
   }

   //---- 뉴스 시각 (서버시간, 파이썬이 써 줌)
   void LoadNews()
   {
      newsLoaded = TimeLocal();
      ArrayResize(news, 0);
      int h = FileOpen(H_NEWS_FILE, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
      if (h == INVALID_HANDLE)
      {
         if (!newsWarned)
            Print("[", tag, "] 뉴스 시각 파일 없음 (Common\\Files\\", H_NEWS_FILE, ") -> 뉴스로 거르지 않음");
         newsWarned = true;
         return;
      }
      int n = 0;
      while (!FileIsEnding(h))
      {
         string line = StringTrimRight(StringTrimLeft(FileReadString(h)));
         if (StringLen(line) < 16)
            continue;
         ArrayResize(news, n + 1);
         news[n++] = StringToTime(line);
      }
      FileClose(h);
   }

   bool NearNews(datetime t)
   {
      if (TimeLocal() - newsLoaded > 600)
         LoadNews();
      for (int i = 0; i < ArraySize(news); i++)
         if (MathAbs((double)(news[i] - t)) <= H_NEWS_MIN * 60)
            return true;
      return false;
   }

   bool Stale(datetime bar, string &why)
   {
      int late = (int)(TimeCurrent() - (bar + 300));
      if (late > H_LATE_SEC)
      {
         why = StringFormat("신호 봉 마감 %d초 뒤 (재시작·멈춤)", late);
         return true;
      }
      return false;
   }

   bool OpenPos(int side, double lot, int level)
   {
      if (!MarginOk(tag, lot))
         return false;
      double fill;
      int tk = OpenMarket(tag, side, NormalizeDouble(lot, 2), 0, H_TP_USD, magic, StringFormat("HM L%d", level), H_SLIP_USD, fill);
      if (tk < 0)
         return false;
      Print(StringFormat("[%s] 진입 %s %.2f lot L%d @ %.2f", tag, side == 1 ? "BUY" : "SELL", lot, level, fill));
      return true;
   }

   //---- Infinox_Bots.mq4에서 부름
   void Init()
   {
      lastBar = iTime(NULL, PERIOD_M5, 1);
      if (InCycle())
         Print(StringFormat("[%s] 진행 중 사이클 이어받음: L%d", tag, Level()));
   }

   string StartText()
   {
      string s = StringFormat("전략: %s\n시작 lot %.2f (1,000$당 %g)", Desc(), Lot0(AccountBalance()), HedgeLotPer1000);
      if (InCycle())
         s += StringFormat("\n진행 중 사이클 이어받음: L%d", Level());
      return s;
   }

   // 꺼져 있어도 진행 중인 사이클 관리(목표·한도 청산, SAR 전환 추가)는 계속. 새 사이클 시작만 on일 때
   void Timer(bool on)
   {
      bool hasBuy, hasSell;
      double floating;
      int n = Open(hasBuy, hasSell, floating);
      string closing = StGet(st, "closing", "");
      if (closing != "")
      {
         if (n > 0)
         {
            if (TimeLocal() - lastRetry >= H_RETRY_SEC)
            {
               lastRetry = TimeLocal();
               CloseAll();
            }
            return;
         }
         EndCycle(closing);
      }
      else if (InCycle())
      {
         double real = Realized(Since()), bal0 = StartBal();
         double total = real + floating;
         if (n > 0 && (total >= H_TARGET * bal0 || total <= -H_LIMIT * bal0))
         {
            StSet(st, "closing", total > 0 ? "target" : "limit");
            lastRetry = TimeLocal();
            CloseAll();
            return;
         }
         if (n == 0 && (real >= H_TARGET * bal0 || Level() >= H_MAX_LEVEL))
            EndCycle(real >= H_TARGET * bal0 ? "tp" : "max");
      }
      datetime bar = iTime(NULL, PERIOD_M5, 1);
      if (bar == 0 || bar == lastBar)
         return;
      lastBar = bar;
      string why;
      if (!InCycle())
      {
         if (!on)
            return;
         int flipDir;
         int side = M_Signal(TimeCurrent(), bar, tag, false, why, flipDir);
         if (side == 0)
            return;
         if (NearNews(bar + 300))
            why = StringFormat("뉴스 앞뒤 %d분", H_NEWS_MIN);
         else
            Stale(bar, why);
         if (why != "")
         {
            Print("[", tag, "] 신호 있지만 시작 안 함: ", why);
            return;
         }
         double bal = AccountBalance(), l0 = Lot0(bal);
         StSet(st, "since", IntegerToString((long)(TimeCurrent() - 60)));
         StSet(st, "start_bal", DoubleToString(bal, 2));
         StSet(st, "level", "0");
         StSet(st, "l0", DoubleToString(l0, 2));
         StSet(st, "closing", "");
         OpenPos(1, l0, 0);
         OpenPos(-1, l0, 0);
         Send(tag, StringFormat("🔀 사이클 시작: 매수·매도 %.2f lot 동시 진입 (SAR 단타 신호 %s)\n목표 +%.2f$ / 한도 -%.2f$\n%s",
                                l0, side == 1 ? "매수" : "매도", H_TARGET * bal, H_LIMIT * bal, AccountLine()));
         return;
      }
      // 사이클 중: 방금 완성된 5분봉 SAR 전환 방향에 포지션이 없으면 다음 단계 추가
      datetime t[];
      double h[], l[], c[];
      bool up[];
      int m = LoadBars(PERIOD_M5, 600, t, h, l, c);
      if (m < 50 || t[m - 1] != bar)
         return;
      Psar(h, l, m, up);
      int flip = up[m - 1] != up[m - 2] ? (up[m - 1] ? 1 : -1) : 0;
      int level = Level();
      if (flip == 0 || level >= H_MAX_LEVEL || (flip == 1 ? hasBuy : hasSell))
         return;
      if (Stale(bar, why))
      {
         Print("[", tag, "] SAR 전환 추가 안 함: ", why);
         return;
      }
      level++;
      StSet(st, "level", IntegerToString(level));
      double lot = NormalizeDouble(L0() * MathPow(2, level), 2);
      if (OpenPos(flip, lot, level))
         Send(tag, StringFormat("%s 추가 %.2f lot (L%d, SAR 전환)\n%s", flip == 1 ? "🟢 매수" : "🔴 매도", lot, level, AccountLine()));
   }
};
