// 파이썬 따라하기 전략 모듈 (2026-10-26 사용자 요청). Infinox_Bots.mq4가 include
// 판단은 파이썬 데모 봇 한 곳에서만 하고, MT4는 공용 폴더 Common\Files\infinox_bridge.csv로 받은 진입·청산 신호를 따라 함
// (common/signal_bridge.py). MT4·MT5 서버 봉 가격 차이로 신호가 갈리지 않게, 전략 로직을 MT4에 다시 짜지 않으려고.
// 한 줄: 시각(UTC ms),전략,open/close,방향,손절 거리,익절 거리,손절 절대 가격,1000$당 lot,데모 티켓,E (E = 줄 끝 표시)
// - open: 보낸 뒤 F_MAX_AGE_MS 안에 본 신호만, lot = 1000$당 lot x MT4 잔고 / 1000 (최소 lot 아래면 최소 lot, FollowMaxLot까지),
//   SL/TP는 MT4 체결가 기준 거리 (손절 절대 가격이 있으면 그 가격). 주문 설명에 데모 티켓을 넣음 ("F SAR 123456")
//   같은 데모 티켓으로 이미 연(또는 닫은) MT4 주문이 있으면 다시 안 들어감 (재시작 때 같은 줄을 또 읽어도)
// - close: 그 데모 티켓을 주문 설명에 가진 MT4 포지션만 청산 (데모가 닫은 포지션의 짝만. MT4가 안 따라 들어간 건 할 게 없음)
//   닫힐 때까지 3초마다 다시, 대기 목록은 상태 파일에 저장 (재시작해도 이어짐). 같은 전략 여러 포지션(헤지)도 티켓별로 따로
// - 서버 SL/TP로 닫히는 건 MT4 자기 서버에서. 파이썬이 멈추면 신규 진입은 없고 열린 포지션은 SL/TP로만 관리
// 상태: MQL4\Files\<상태 파일> (최고 잔고, 청산 대기 데모 티켓 목록). 통로 읽기는 mod_bridge.mqh

#define F_FILE "infinox_bridge.csv"
#define F_STATE "infinox_bridge_state.txt"
#define F_MAX_AGE_MS 5000
#define F_CLOSE_RETRY_SEC 3
#define F_ACCOUNT_STOP_PCT 50.0

class CFollow : public CStrat
{
public:
   string   sym, tag, st, id;
   int      magic;
   double   slip;                    // 허용 미끄러짐 (가격)
   bool     ok;                      // 종목을 쓸 수 있으면 true
   datetime okTry;
   int      known[];                 // 알고 있는 열린 티켓 (닫히면 알림)
   string   closing;                 // 청산 대기 데모 티켓들 ";ref;ref;" (상태 파일 close_refs)
   bool     dirty;                   // closing을 아직 상태 파일에 못 씀
   bool     openUnsure;              // 마지막 Open() 스캔 중 주문 조회 실패가 있었음
   datetime lastCloseTry, lastRisk;

   void Setup(string s, string panelName, string file, int mg, string bridgeId, double slipPx, bool on)
   {
      sym = s;
      name = panelName;
      tag = panelName;
      st = file;
      magic = mg;
      id = bridgeId;
      slip = slipPx;
      defOn = on;
      closing = "";
      dirty = false;
      lastCloseTry = 0;
      lastRisk = 0;
      ArrayResize(known, 0);
      okTry = TimeLocal();
      ok = UseSymbol(sym);
      if (!ok)
         Print("[", tag, "] 종목 ", sym, "을 찾을 수 없어 이 전략은 쉽니다");
   }

   int    Magic() { return magic; }
   string Sym() { return sym; }
   string BridgeId() { return id; }

   string Comment(string ref) { return "F " + id + " " + ref; }

   // 주문 설명이 이 ref의 것인지 (정확히 같은 번호만: 123이 1234에 걸리지 않게. 서버가 뒤에 " [sl]" 등을 붙인 경우는 인정)
   bool IsRef(string cmt, string ref)
   {
      string c = Comment(ref);
      int n = StringLen(c);
      if (StringFind(cmt, c) != 0)
         return false;
      if (StringLen(cmt) == n)
         return true;
      ushort ch = StringGetCharacter(cmt, n);
      return ch == ' ' || ch == '[';
   }

   // 이 전략의 열린 시장가 주문 티켓들
   // (주문 조회가 하나라도 실패하면 openUnsure = true: 목록이 빠졌을 수 있음)
   int Open(int &t[])
   {
      ArrayResize(t, 0);
      openUnsure = false;
      int n = 0;
      for (int i = OrdersTotal() - 1; i >= 0; i--)
      {
         if (!OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
         {
            openUnsure = true;
            continue;
         }
         if (OrderSymbol() == sym && OrderMagicNumber() == magic && OrderType() <= OP_SELL)
         {
            ArrayResize(t, n + 1);
            t[n++] = OrderTicket();
         }
      }
      return n;
   }

   // 데모 티켓 ref를 주문 설명에 가진 이 전략 주문이 있는지 (열린 것 + history=true면 닫힌 것까지)
   // 반환: 1 있음, 0 없음, -1 모름 (주문 조회가 중간에 실패해서 확실하지 않음)
   int HasRef(string ref, bool history)
   {
      bool unsure = false;
      for (int i = OrdersTotal() - 1; i >= 0; i--)
      {
         if (!OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
         {
            unsure = true;
            continue;
         }
         if (OrderMagicNumber() == magic && IsRef(OrderComment(), ref))
            return 1;
      }
      if (history)
         for (int i = OrdersHistoryTotal() - 1; i >= 0; i--)
         {
            if (!OrderSelect(i, SELECT_BY_POS, MODE_HISTORY))
            {
               unsure = true;
               continue;
            }
            if (OrderMagicNumber() == magic && IsRef(OrderComment(), ref))
               return 1;
         }
      return unsure ? -1 : 0;
   }

   void Remember(int tk)
   {
      int n = ArraySize(known);
      for (int i = 0; i < n; i++)
         if (known[i] == tk)
            return;
      ArrayResize(known, n + 1);
      known[n] = tk;
   }

   // 알고 있던 티켓이 닫혔으면 알림 (SL/TP든 청산 신호든)
   void CheckClosed()
   {
      int n = ArraySize(known), keep = 0;
      for (int i = 0; i < n; i++)
      {
         double profit, px;
         string reason;
         if (!ClosedInfo(known[i], profit, px, reason))
         {
            known[keep++] = known[i];                  // 아직 열림, 또는 조회 실패 (다음에 다시)
            continue;
         }
         int side = OrderType() == OP_BUY ? 1 : -1;
         double op = OrderOpenPrice(), lots = OrderLots();
         string icon = reason == "TP" ? "✅ 익절" : reason == "SL" ? "❌ 손절" : "⏹ 청산 (파이썬 신호)";
         Send(tag, StringFormat("%s %s %.2f lot\n%.2f → %.2f\n손익 %+.2f$\n%s\n%s", icon, side == 1 ? "BUY" : "SELL", lots, op, px,
                                profit, AccountLine(), SummaryLine(magic, sym)));
      }
      ArrayResize(known, keep);
   }

   bool Fresh()
   {
      if (!IsConnected() || !TickFresh())
         return false;
      return sym == Symbol() || TimeCurrent() - (datetime)MarketInfo(sym, MODE_TIME) <= TICK_FRESH_SEC;
   }

   void Enter(int side, double slD, double tpD, double slAbs, double lot1000, long ageMs, string ref)
   {
      string why = "";
      double step = MarketInfo(sym, MODE_LOTSTEP), minLot = MarketInfo(sym, MODE_MINLOT);
      int lotDg = 0;                                   // lot step의 소수 자리 (0.25면 2, 0.1이면 1)
      while (lotDg < 8 && MathAbs(step * MathPow(10, lotDg) - MathRound(step * MathPow(10, lotDg))) > 1e-7)
         lotDg++;
      double raw = lot1000 * AccountBalance() / 1000;
      // 상한(FollowMaxLot·종목 최대)까지 자른 뒤 step으로 내림 (step의 정수배가 되게), 최소 lot 아래면 최소 lot
      double cap = MathMin(FollowMaxLot, MarketInfo(sym, MODE_MAXLOT));
      double lot = step > 0 ? MathFloor(MathMin(raw, cap) / step + 1e-9) * step : 0;
      bool forced = lot < minLot;
      lot = NormalizeDouble(MathMax(lot, minLot), lotDg);
      if (ageMs > F_MAX_AGE_MS || ageMs < -F_MAX_AGE_MS)
         why = StringFormat("보낸 뒤 %.1f초에 봄 (늦은 신호)", ageMs / 1000.0);
      else if (minLot > cap + 1e-9)
         why = StringFormat("최소 lot %g이 최대 lot %g보다 큼 (FollowMaxLot 확인)", minLot, cap);
      else if (ref == "" || ref == "0")
         why = "데모 티켓 없음";
      else if (HasRef(ref, true) != 0)
         why = "이 신호로 이미 주문함 (다시 읽은 줄) 또는 주문 조회 실패로 확인 못 함";
      else if (lot1000 <= 0 || step <= 0)
         why = "lot 비율 없음";
      else if (!Fresh())
         why = StringFormat("시세 멈춤 / 연결 끊김 (틱 %d초 넘게 없음)", TICK_FRESH_SEC);
      else
         why = RiskBlock(tag, st, F_ACCOUNT_STOP_PCT);
      if (why != "")
      {
         Print("[", tag, "] 파이썬 ", side == 1 ? "매수" : "매도", " 신호 (티켓 ", ref, ") -> 진입 안 함: ", why);
         return;
      }
      if (!MarginOkSym(tag, sym, lot))
         return;
      double fill;
      int tk = OpenMarketSym(tag, sym, side, lot, slAbs > 0 ? 0 : slD, tpD, magic, Comment(ref), slip, fill, slAbs);
      if (tk < 0)
         return;
      Remember(tk);
      int dg = (int)MarketInfo(sym, MODE_DIGITS);
      double sl = slAbs > 0 ? slAbs : slD > 0 ? fill - side * slD : 0, tp = tpD > 0 ? fill + side * tpD : 0;
      Send(tag, StringFormat("%s 진입 %s lot (파이썬 신호, 1,000$당 %.4f lot)\n가격 %s | SL %s | TP %s%s%s\n%s\n%s",
                             side == 1 ? "🟢 매수" : "🔴 매도", DoubleToString(lot, lotDg), lot1000, DoubleToString(fill, dg),
                             sl > 0 ? DoubleToString(sl, dg) : "없음", tp > 0 ? DoubleToString(tp, dg) : "없음",
                             sl > 0 ? StringFormat("\n손절 시 약 %.0f$", MathAbs(fill - sl) * lot * PerLot(sym)) : "",
                             forced ? "\n⚠️ 최소 lot이라 파이썬 비율보다 큼" : "", AccountLine(), SummaryLine(magic, sym)));
   }

   // 청산 대기 목록의 데모 티켓마다 그 짝 MT4 포지션을 청산. 짝이 더는 열려 있지 않으면 목록에서 뺌
   void DoClose()
   {
      if (closing == "" && !dirty)
         return;
      lastCloseTry = TimeLocal();
      string refs[], keep = "";
      int n = StringSplit(closing, ';', refs);
      for (int r = 0; r < n; r++)
      {
         if (refs[r] == "")
            continue;
         int t[], m = Open(t);
         bool open = openUnsure;                       // 목록 조회가 불완전하면 짝이 빠졌을 수 있으니 남김
         for (int i = 0; i < m; i++)
         {
            if (!OrderSelect(t[i], SELECT_BY_TICKET))
            {
               open = true;                            // 조회 실패: 짝일 수도 있으니 목록에 남김
               continue;
            }
            if (IsRef(OrderComment(), refs[r]))
            {
               open = true;
               Remember(t[i]);
               CloseTicket(tag, t[i], slip);
            }
         }
         if (open)
            keep += refs[r] + ";";                     // 닫혔는지는 다음 재시도 때 다시 확인
      }
      keep = keep == "" ? "" : ";" + keep;
      if (keep != closing)
      {
         closing = keep;
         dirty = true;
      }
      if (dirty && StSet(st, "close_refs", closing))   // 저장 실패면 다음 재시도 때 다시 저장
         dirty = false;
   }

   bool OnBridge(bool on, string action, int side, double slD, double tpD, double slAbs, double lot1000, long sigMs, string ref)
   {
      if (!ok)
         return true;
      long ageMs = (long)TimeGMT() * 1000 - sigMs;
      if (action == "open")
      {
         if (side != 1 && side != -1)
            return true;
         if (!on)
         {
            Print("[", tag, "] 파이썬 신호 (티켓 ", ref, ") 왔지만 꺼져 있어 진입 안 함");
            return true;
         }
         Enter(side, slD, tpD, slAbs, lot1000, ageMs, ref);
         return true;
      }
      if (action != "close")
         return true;
      // 꺼져 있어도 이미 연 포지션은 파이썬 청산을 따라 감. 짝이 없으면(MT4가 안 들어갔던 신호) 할 일 없음
      int has = ref == "" || ref == "0" ? 0 : HasRef(ref, false);
      if (has < 0)
         return false;                                 // 주문 조회 실패로 짝이 있는지 모름: 다음에 같은 줄부터 다시
      if (has == 0)
      {
         Print("[", tag, "] 파이썬 청산 신호 (티켓 ", ref, "): MT4에 짝 포지션 없음");
         return true;
      }
      if (StringFind(closing, ";" + ref + ";") < 0)
      {
         string next = (closing == "" ? ";" : closing) + ref + ";";
         if (!StSet(st, "close_refs", next))
         {
            // 대기 목록을 못 남기면 이 줄을 아직 안 읽은 것으로 (통로 위치를 안 넘겨 재시작해도 다시 읽음). 청산은 일단 시도
            Print("[", tag, "] 청산 대기 저장 실패 -> 이 신호를 다음에 다시 처리");
            int t[], m = Open(t);
            for (int i = 0; i < m; i++)
               if (OrderSelect(t[i], SELECT_BY_TICKET) && IsRef(OrderComment(), ref))
                  CloseTicket(tag, t[i], slip);
            return false;
         }
         closing = next;
         dirty = false;
      }
      Print("[", tag, "] 파이썬 청산 신호 (티켓 ", ref, ") -> 짝 포지션 청산");
      DoClose();
      return true;
   }

   string Desc()
   {
      return StringFormat("파이썬 데모 봇(%s) 진입·청산 신호를 따라 함 (보낸 뒤 %d초 안), lot = 파이썬 1,000$당 lot x MT4 잔고, "
                          "SL/TP는 MT4 체결가 기준, 최대 lot %g", id, F_MAX_AGE_MS / 1000, FollowMaxLot);
   }

   //---- Infinox_Bots.mq4에서 부름
   void Init()
   {
      if (!ok)
         return;
      int t[], n = Open(t);
      for (int i = 0; i < n; i++)
         Remember(t[i]);
      closing = StGet(st, "close_refs", "");
      if (n > 0)
         Print("[", tag, "] 기존 포지션 ", n, "개 이어받음");
   }

   string StartText()
   {
      if (!ok)
         return StringFormat("⚠️ 종목 %s을 찾을 수 없어 이 전략은 쉽니다 (MT4 시장 감시 창에 종목이 있는지 확인)", sym);
      return "전략: " + Desc();
   }

   // 꺼져 있어도 열린 포지션 관리(청산 알림, 대기 중인 청산 재시도)는 계속
   void Timer(bool on)
   {
      if (!ok)
      {
         if (TimeLocal() - okTry >= 60)
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
      int t[], n = Open(t);
      for (int i = 0; i < n; i++)
         Remember(t[i]);
      CheckClosed();
      if (on && n > 0 && TimeLocal() - lastRisk >= 5)
      {
         lastRisk = TimeLocal();
         RiskBlock(tag, st, F_ACCOUNT_STOP_PCT);    // 계좌 손실 한도 감시 (닿으면 halted가 남아 신규 진입 중단)
      }
      if ((closing != "" || dirty) && TimeLocal() - lastCloseTry >= F_CLOSE_RETRY_SEC)
         DoClose();
   }
};
