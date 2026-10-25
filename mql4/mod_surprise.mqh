// 뉴스 서프라이즈(news_surprise_paper.py --live와 같은 규칙) 전략 모듈. Infinox_Bots.mq4가 include
// MT4에는 경제지표 실제치를 받는 기능이 없어서, 파이썬 데모 서프라이즈 봇이 판단하는 순간 공용 폴더
// Common\Files\infinox_surprise_signals.csv에 한 줄을 씀: 발표, 방향, S, 등급(base = [기본] / watch = [관찰]), 판단 시각(UTC ms), 발표 시각(UTC ms), 대표 지표
// - EA는 0.1초마다 파일을 보고, [기본](|S| >= 2) 신호가 판단 뒤 S_MAX_AGE_MS 안이면 시장가 진입 (늦게 본 신호는 건너뜀)
// - lot = 잔고 1,000$당 SurpriseLotPer1000 (최소 0.01), 만일용 손절 S_SAFETY_SL$
// - 청산 = 큰이익추적 5/2.5/15분: 발표 15분 안에 +5$면 최고점 -2.5$ 추적 (발표 30분 직전까지), 아니면 15분에 청산.
//   추적이 켜지면 MT4 손절도 최고점 -2.5$로 올림 (EA가 멈춰도 지킴)
// 상태: MQL4\Files\infinox_surprise_state.txt (읽은 줄 수, 포지션별 추적 상태)

#define S_FILE "infinox_surprise_signals.csv"
#define S_MAX_AGE_MS 5000
#define S_SAFETY_SL 30.0
#define S_TRAIL_ON 5.0
#define S_TRAIL_GAP 2.5
#define S_ARM_MIN 15
#define S_HOLD_SEC 1799              // 발표 30분 직전
#define S_SL_STEP 0.5
#define S_SLIP_USD 1.0
#define S_MAX_OPEN 8

class CSurprise : public CStrat
{
public:
   string   tag, st;
   int      magic;
   int      seen;                    // 파일에서 읽은 줄 수
   int      tk[S_MAX_OPEN], side[S_MAX_OPEN];
   double   e0[S_MAX_OPEN], peak[S_MAX_OPEN], sl[S_MAX_OPEN];
   long     rel[S_MAX_OPEN];          // 발표 시각 UTC 초
   bool     armed[S_MAX_OPEN];
   string   why[S_MAX_OPEN];
   uint     lastTry[S_MAX_OPEN];
   bool     fileWarned;

   void Setup(string panelName, string file, int mg, bool on)
   {
      name = panelName;
      tag = panelName;
      st = file;
      magic = mg;
      defOn = on;
      seen = 0;
      fileWarned = false;
      for (int i = 0; i < S_MAX_OPEN; i++)
         tk[i] = -1;
   }

   int Magic() { return magic; }

   string Desc()
   {
      return StringFormat("파이썬 서프라이즈 봇의 [기본] 신호(|S| >= 2)를 받아 시장가 진입 (판단 뒤 %d초 안), 1,000$당 %g lot, 만일용 손절 %g$, "
                          "큰이익추적 %g/%g/%d분 (발표 30분 직전 청산)", S_MAX_AGE_MS / 1000, SurpriseLotPer1000, S_SAFETY_SL,
                          S_TRAIL_ON, S_TRAIL_GAP, S_ARM_MIN);
   }

   void Save(int i)
   {
      string key = "t" + IntegerToString(i);
      if (tk[i] < 0)
      {
         StSet(st, key, "");
         return;
      }
      StSet(st, key, StringFormat("%d|%d|%.2f|%I64d|%d|%.2f|%.2f|%s", tk[i], side[i], e0[i], rel[i], armed[i] ? 1 : 0, peak[i], sl[i], why[i]));
   }

   void Load(int i)
   {
      tk[i] = -1;
      string v = StGet(st, "t" + IntegerToString(i), "");
      string f[];
      if (StringSplit(v, '|', f) < 8)
         return;
      tk[i] = (int)StringToInteger(f[0]);
      side[i] = (int)StringToInteger(f[1]);
      e0[i] = StringToDouble(f[2]);
      rel[i] = StringToInteger(f[3]);
      armed[i] = f[4] == "1";
      peak[i] = StringToDouble(f[5]);
      sl[i] = StringToDouble(f[6]);
      why[i] = f[7];
      lastTry[i] = 0;
   }

   long NowMs() { return (long)TimeGMT() * 1000; }

   // 새로 붙은 줄: 발표,방향,S,등급,판단 UTC ms,발표 UTC ms,대표 지표
   void ReadSignals(bool on)
   {
      int h = FileOpen(S_FILE, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON | FILE_SHARE_READ | FILE_SHARE_WRITE);
      if (h == INVALID_HANDLE)
      {
         if (!fileWarned)
            Print("[", tag, "] 신호 파일 없음 (Common\\Files\\", S_FILE, ") -> 파이썬 데모 서프라이즈 봇이 켜져 있어야 함");
         fileWarned = true;
         return;
      }
      int n = 0;
      while (!FileIsEnding(h))
      {
         string line = FileReadString(h);
         if (line == "")
            continue;
         n++;
         if (n <= seen)
            continue;
         seen = n;
         StSet(st, "seen", IntegerToString(seen));
         string f[];
         if (StringSplit(line, ',', f) < 7)
            continue;
         int sd = (int)StringToInteger(f[1]);
         double s = StringToDouble(f[2]);
         long decided = StringToInteger(f[4]), relMs = StringToInteger(f[5]);
         long age = NowMs() - decided;
         string head = StringFormat("[%s] 신호 %s %s S %+.2f (%s, %s)", tag, f[0], sd > 0 ? "매수" : "매도", s, f[3], f[6]);
         if (f[3] != "base")
            Print(head, " -> 관찰 등급이라 진입 안 함");
         else if (!on)
            Print(head, " -> 꺼져 있어 진입 안 함");
         else if (age > S_MAX_AGE_MS || age < -S_MAX_AGE_MS)
            Print(head, StringFormat(" -> 판단 뒤 %.1f초라 진입 안 함", age / 1000.0));
         else
            Enter(sd, s, relMs / 1000, f[0], f[6]);
      }
      FileClose(h);
   }

   void Enter(int sd, double s, long relSec, string release, string key)
   {
      int slot = -1;
      for (int i = 0; i < S_MAX_OPEN && slot < 0; i++)
         if (tk[i] < 0)
            slot = i;
      if (slot < 0)
         return;
      double lot = MathMax(0.01, MathFloor(AccountBalance() / 1000 * SurpriseLotPer1000 * 100 + 1e-9) / 100);
      lot = FloorLot(lot);
      if (!MarginOk(tag, lot))
         return;
      double fill;
      int t = OpenMarket(tag, sd, lot, S_SAFETY_SL, 0, magic, StringFormat("SP %+.1f", s), S_SLIP_USD, fill);
      if (t < 0)
         return;
      tk[slot] = t;
      side[slot] = sd;
      e0[slot] = fill;
      rel[slot] = relSec;
      armed[slot] = false;
      peak[slot] = 0;
      sl[slot] = NormalizeDouble(fill - sd * S_SAFETY_SL, Digits);
      why[slot] = "";
      lastTry[slot] = 0;
      Save(slot);
      Send(tag, StringFormat("%s %s %.2f lot @ %.2f (S %+.2f, %s 발표, %s)\n손절 %.2f (만일용), 발표 %d분 안에 +%g$면 최고점 -%g$ 추적, 아니면 %d분에 청산\n%s",
                             sd > 0 ? "📈" : "📉", sd > 0 ? "매수" : "매도", lot, fill, s, release, key, sl[slot], S_ARM_MIN,
                             S_TRAIL_ON, S_TRAIL_GAP, S_ARM_MIN, AccountLine()));
   }

   void Manage()
   {
      long now = (long)TimeGMT();
      for (int i = 0; i < S_MAX_OPEN; i++)
      {
         if (tk[i] < 0)
            continue;
         if (!OrderSelect(tk[i], SELECT_BY_TICKET))
            continue;
         if (OrderCloseTime() > 0)                       // 닫힘 -> 알림
         {
            double profit = OrderProfit() + OrderSwap() + OrderCommission(), px = OrderClosePrice();
            string r = why[i] != "" ? why[i] : (OrderStopLoss() > 0 && MathAbs(px - OrderStopLoss()) < 0.5 ? "MT4 손절" : "청산");
            Send(tag, StringFormat("%s 서프라이즈 청산 (%s) @ %.2f: %+.2f$/oz, %+.2f$\n%s\n%s", profit > 0 ? "✅" : "❌", r, px,
                                   (px - e0[i]) * side[i], profit, AccountLine(), SummaryLine(magic)));
            tk[i] = -1;
            Save(i);
            continue;
         }
         RefreshRates();
         double pnl = side[i] > 0 ? Bid - e0[i] : e0[i] - Ask;
         bool changed = false;
         long armEnd = rel[i] + S_ARM_MIN * 60;
         if (!armed[i])
         {
            if (pnl >= S_TRAIL_ON && now <= armEnd)
            {
               armed[i] = true;
               peak[i] = pnl;
               changed = true;
            }
            else if (now >= armEnd && why[i] == "")
            {
               why[i] = StringFormat("%d분 (+%g 못 감)", S_ARM_MIN, S_TRAIL_ON);
               changed = true;
            }
         }
         if (armed[i])
         {
            if (pnl > peak[i])
            {
               peak[i] = pnl;
               changed = true;
            }
            if (pnl <= peak[i] - S_TRAIL_GAP)
            {
               if (why[i] == "")
               {
                  why[i] = StringFormat("추적 (최고 +%.2f)", peak[i]);
                  changed = true;
               }
            }
            else
            {
               double nsl = NormalizeDouble(e0[i] + (peak[i] - S_TRAIL_GAP) * side[i], Digits);
               if ((nsl - sl[i]) * side[i] >= S_SL_STEP && OrderModify(tk[i], OrderOpenPrice(), nsl, OrderTakeProfit(), 0, clrNONE))
               {
                  sl[i] = nsl;
                  changed = true;
               }
            }
         }
         if (now >= rel[i] + S_HOLD_SEC && why[i] == "")
         {
            why[i] = "30분";
            changed = true;
         }
         if (changed)
            Save(i);
         if (why[i] != "" && GetTickCount() - lastTry[i] >= 3000)
         {
            lastTry[i] = GetTickCount();
            CloseTicket(tag, tk[i], S_SLIP_USD);
         }
      }
   }

   //---- Infinox_Bots.mq4에서 부름
   void Init()
   {
      for (int i = 0; i < S_MAX_OPEN; i++)
         Load(i);
      // 처음 켤 때는 지금까지 쌓인 줄을 건너뜀 (지난 신호로 들어가지 않게)
      string v = StGet(st, "seen", "");
      if (v == "")
      {
         int n = 0, h = FileOpen(S_FILE, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON | FILE_SHARE_READ | FILE_SHARE_WRITE);
         if (h != INVALID_HANDLE)
         {
            while (!FileIsEnding(h))
               if (FileReadString(h) != "")
                  n++;
            FileClose(h);
         }
         seen = n;
         StSet(st, "seen", IntegerToString(seen));
      }
      else
         seen = (int)StringToInteger(v);
   }

   string StartText() { return "전략: " + Desc(); }

   void Fast(bool on)
   {
      ReadSignals(on);
      Manage();
   }
};
