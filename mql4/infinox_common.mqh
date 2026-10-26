// Infinox_Bots EA 공통 코드 (전략 모듈 mod_*.mqh가 같이 씀)
// - 텔레그램 전송 (MQL4\Files\infinox_telegram.txt: 1줄 토큰, 2줄 chat_id)
// - 상태 파일 (MQL4\Files\<이름>.txt, key=value 줄) -> EA를 껐다 켜도 이어짐
// - 지표: 파이썬 indicators.py와 같은 계산 (SAR, RSI·ATR은 Wilder ewm)
// - 주문 도우미: 이 EA의 매직 넘버 포지션만 다룸

string g_token = "";
string g_chat = "";
int    g_login = 0;

//---------------------------------------------------------------- 텔레그램
void TgInit()
{
   int h = FileOpen("infinox_telegram.txt", FILE_READ | FILE_TXT | FILE_ANSI);
   if (h == INVALID_HANDLE)
   {
      Print("텔레그램 설정 파일 없음 (MQL4\\Files\\infinox_telegram.txt) -> 알림 꺼짐");
      return;
   }
   g_token = StringTrimRight(StringTrimLeft(FileReadString(h)));
   g_chat = StringTrimRight(StringTrimLeft(FileReadString(h)));
   FileClose(h);
}

string UrlEncode(string s)
{
   uchar b[];
   int n = StringToCharArray(s, b, 0, WHOLE_ARRAY, CP_UTF8) - 1;   // 마지막 0 제외
   string out = "";
   for (int i = 0; i < n; i++)
   {
      uchar c = b[i];
      if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '~')
         out += CharToString(c);
      else
         out += StringFormat("%%%02X", c);
   }
   return out;
}

void Send(string tag, string text)
{
   string msg = "[MT4 · " + tag + "] " + text;   // 파이썬 봇은 [데모 · ...] / [실계좌 · ...]
   Print(msg);
   if (g_token == "" || IsTesting())
      return;
   string body = "chat_id=" + g_chat + "&text=" + UrlEncode(msg);
   char data[], res[];
   string rh;
   int n = StringToCharArray(body, data, 0, WHOLE_ARRAY, CP_UTF8) - 1;
   ArrayResize(data, n);
   ResetLastError();
   int code = WebRequest("POST", "https://api.telegram.org/bot" + g_token + "/sendMessage",
                         "Content-Type: application/x-www-form-urlencoded\r\n", 5000, data, res, rh);
   if (code != 200)
      Print("텔레그램 전송 실패 ", code, " 오류 ", GetLastError(),
            " (도구 > 옵션 > 전문가 조언자에서 'WebRequest 허용 URL'에 https://api.telegram.org 추가 필요)");
}

//---------------------------------------------------------------- 상태 파일
string StGet(string file, string key, string def)
{
   int h = FileOpen(file, FILE_READ | FILE_TXT | FILE_ANSI);
   if (h == INVALID_HANDLE)
      return def;
   string v = def;
   while (!FileIsEnding(h))
   {
      string line = FileReadString(h);
      int p = StringFind(line, "=");
      if (p > 0 && StringSubstr(line, 0, p) == key)
         v = StringSubstr(line, p + 1);
   }
   FileClose(h);
   return v;
}

// 상태 파일 통째로 쓰기: 임시 파일에 다 쓴 뒤 바꿔치기 (쓰다가 꺼져도 예전 파일이 온전히 남음). 성공하면 true
bool StWrite(string file, string &lines[], int n)
{
   string tmp = file + ".tmp";
   int h = FileOpen(tmp, FILE_WRITE | FILE_TXT | FILE_ANSI);
   if (h == INVALID_HANDLE)
   {
      Print("상태 파일 저장 실패: ", tmp, " 오류 ", GetLastError());
      return false;
   }
   bool ok = true;
   for (int i = 0; i < n && ok; i++)
      ok = FileWriteString(h, lines[i] + "\r\n") > 0;
   FileFlush(h);
   FileClose(h);
   // 다시 읽어서 줄 하나하나 같은지 확인 (일부만 써진 파일로 바꾸지 않게)
   if (ok)
   {
      h = FileOpen(tmp, FILE_READ | FILE_TXT | FILE_ANSI);
      ok = h != INVALID_HANDLE;
      int k = 0;
      while (ok && !FileIsEnding(h))
      {
         string line = FileReadString(h);
         if (line == "")
            continue;
         ok = k < n && line == lines[k];
         k++;
      }
      if (h != INVALID_HANDLE)
         FileClose(h);
      ok = ok && k == n;
   }
   if (!ok)
   {
      Print("상태 파일 저장 실패 (기록 확인 안 됨): ", tmp, " 오류 ", GetLastError());
      return false;
   }
   if (!FileMove(tmp, 0, file, FILE_REWRITE))
   {
      Print("상태 파일 바꾸기 실패: ", file, " 오류 ", GetLastError());
      return false;
   }
   return true;
}

// 키 두 개를 파일 한 번 바꿔치기로 같이 저장 (중간에 꺼져도 둘 중 하나만 바뀌지 않게, k2가 ""면 하나만). 성공하면 true
// 파일이 있는데 못 읽으면 저장 안 함 (새 키만 남고 나머지가 지워지지 않게)
bool StSet2(string file, string k1, string v1, string k2, string v2)
{
   string lines[];
   int n = 0;
   bool f1 = false, f2 = false;
   int h = FileOpen(file, FILE_READ | FILE_TXT | FILE_ANSI);
   if (h == INVALID_HANDLE && FileIsExist(file))
   {
      Print("상태 파일 읽기 실패: ", file, " 오류 ", GetLastError(), " -> 저장 안 함 (다른 키가 지워지지 않게)");
      return false;
   }
   if (h != INVALID_HANDLE)
   {
      while (!FileIsEnding(h))
      {
         string line = FileReadString(h);
         if (line == "")
            continue;
         int p = StringFind(line, "=");
         string k = p > 0 ? StringSubstr(line, 0, p) : "";
         if (k == k1)
         {
            line = k1 + "=" + v1;
            f1 = true;
         }
         else if (k2 != "" && k == k2)
         {
            line = k2 + "=" + v2;
            f2 = true;
         }
         ArrayResize(lines, n + 1);
         lines[n++] = line;
      }
      FileClose(h);
   }
   if (!f1)
   {
      ArrayResize(lines, n + 1);
      lines[n++] = k1 + "=" + v1;
   }
   if (!f2 && k2 != "")
   {
      ArrayResize(lines, n + 1);
      lines[n++] = k2 + "=" + v2;
   }
   return StWrite(file, lines, n);
}

bool StSet(string file, string key, string val) { return StSet2(file, key, val, "", ""); }

//---------------------------------------------------------------- 전략 공통 모양
// 전략 하나 = CStrat를 이은 클래스 하나 (mod_*.mqh). Infinox_Bots.mq4는 목록에 넣기만 하면 패널·알림·타이머가 붙음
// Timer: 1초마다 (on = 패널에서 켜짐. 꺼져 있어도 열린 포지션 관리는 계속), Fast: 0.1초마다 (빠른 진입이 필요한 전략만)
class CStrat
{
public:
   string name;                      // 패널·알림 이름
   bool   defOn;                     // 처음 켤 때 선택돼 있는지 (패널 파일이 없을 때)
   virtual void   Init() {}
   virtual void   Timer(bool on) {}
   virtual void   Fast(bool on) {}
   virtual string StartText() { return ""; }
   virtual int    Magic() { return 0; }
   virtual string Sym() { return Symbol(); }
   // 파이썬 따라하기 (mod_follow.mqh): 통로의 전략 이름, 신호 처리 (따라하기가 아닌 전략은 "" / 아무것도 안 함)
   virtual string BridgeId() { return ""; }
   // 반환: 이 줄을 다 처리했으면 true (false면 통로 읽기를 거기서 멈추고 다음에 같은 줄부터 다시)
   virtual bool   OnBridge(bool on, string action, int side, double slD, double tpD, double slAbs, double lot1000, long sigMs, string ref) { return true; }
};

//---------------------------------------------------------------- 봉 / 지표
// sym 종목의 완성된 봉(shift 1부터) count개를 오래된 것부터 배열에 담음. 담은 개수 반환
int LoadBarsSym(string sym, int tf, int count, datetime &t[], double &h[], double &l[], double &c[])
{
   int n = MathMin(count, iBars(sym, tf) - 1);
   if (n <= 0)
      return 0;
   ArrayResize(t, n);
   ArrayResize(h, n);
   ArrayResize(l, n);
   ArrayResize(c, n);
   for (int i = 0; i < n; i++)
   {
      int s = n - i;
      t[i] = iTime(sym, tf, s);
      h[i] = iHigh(sym, tf, s);
      l[i] = iLow(sym, tf, s);
      c[i] = iClose(sym, tf, s);
   }
   return n;
}

int LoadBars(int tf, int count, datetime &t[], double &h[], double &l[], double &c[])
{
   return LoadBarsSym(NULL, tf, count, t, h, l, c);
}

// 파라볼릭 SAR(0.02, 0.2): 각 봉이 상승 추세면 true (indicators.psar와 같음)
void Psar(const double &h[], const double &l[], int n, bool &up[])
{
   double sv[], tg[];
   PsarLevels(h, l, n, up, sv, tg);
}

// SAR + 각 봉 끝의 SAR 점(sv), 그 봉에서 뚫어야 했던 값(tg) (indicators.psar_levels와 같음)
void PsarLevels(const double &h[], const double &l[], int n, bool &up[], double &sv[], double &tg[])
{
   double step = 0.02, mx = 0.2;
   ArrayResize(up, n);
   ArrayResize(sv, n);
   ArrayResize(tg, n);
   up[0] = true;
   sv[0] = l[0];
   tg[0] = 0;
   double sar = l[0], ep = h[0], af = step;
   for (int i = 1; i < n; i++)
   {
      double s = sar + af * (ep - sar);
      double l2 = i > 1 ? l[i - 2] : l[i - 1];
      double h2 = i > 1 ? h[i - 2] : h[i - 1];
      if (up[i - 1])
      {
         s = MathMin(s, MathMin(l[i - 1], l2));
         tg[i] = s;
         if (l[i] < s)
         {
            up[i] = false;
            s = ep;
            ep = l[i];
            af = step;
         }
         else
         {
            up[i] = true;
            if (h[i] > ep)
            {
               ep = h[i];
               af = MathMin(af + step, mx);
            }
         }
      }
      else
      {
         s = MathMax(s, MathMax(h[i - 1], h2));
         tg[i] = s;
         if (h[i] > s)
         {
            up[i] = true;
            s = ep;
            ep = h[i];
            af = step;
         }
         else
         {
            up[i] = false;
            if (l[i] < ep)
            {
               ep = l[i];
               af = MathMin(af + step, mx);
            }
         }
      }
      sar = s;
      sv[i] = s;
   }
}

// RSI (Wilder ewm, indicators.rsi와 같음). out[0]은 계산 안 됨
void Rsi(const double &c[], int n, int period, double &out[])
{
   ArrayResize(out, n);
   double a = 1.0 / period, g = 0, d = 0;
   out[0] = 50;
   for (int i = 1; i < n; i++)
   {
      double x = c[i] - c[i - 1];
      double gu = MathMax(x, 0), gd = MathMax(-x, 0);
      if (i == 1)
      {
         g = gu;
         d = gd;
      }
      else
      {
         g = (1 - a) * g + a * gu;
         d = (1 - a) * d + a * gd;
      }
      out[i] = d == 0 ? 100 : 100 - 100 / (1 + g / d);
   }
}

// ATR (Wilder ewm, 첫 봉 TR = 고가 - 저가, indicators.atr과 같음)
void Atr(const double &h[], const double &l[], const double &c[], int n, int period, double &out[])
{
   ArrayResize(out, n);
   double a = 1.0 / period;
   out[0] = h[0] - l[0];
   for (int i = 1; i < n; i++)
   {
      double tr = MathMax(h[i] - l[i], MathMax(MathAbs(h[i] - c[i - 1]), MathAbs(l[i] - c[i - 1])));
      out[i] = (1 - a) * out[i - 1] + a * tr;
   }
}

int Minutes(datetime t) { return TimeHour(t) * 60 + TimeMinute(t); }
datetime DayStart(datetime t) { return t - t % 86400; }

//---------------------------------------------------------------- 계좌 / lot
bool AccountOk()
{
   if (AccountNumber() == 0)
      return false;
   if (g_login == 0)
      g_login = AccountNumber();
   if (AccountNumber() != g_login)
   {
      Send("Infinox", StringFormat("로그인 계좌가 바뀌었습니다 (%d -> %d). 안전을 위해 EA를 멈춥니다.", g_login, AccountNumber()));
      ExpertRemove();
      return false;
   }
   return true;
}

string AccountKind() { return IsDemo() ? "데모" : "실계좌"; }

string AccountLine()
{
   return StringFormat("잔고 %.2f | 평가 %.2f %s", AccountBalance(), AccountEquity(), AccountCurrency());
}

double Contract() { return MarketInfo(Symbol(), MODE_LOTSIZE); }

// 계좌 통화로 1 lot이 가격 1 움직일 때 손익 (금 100, 이더리움 1, 유로 손익은 달러로 바꿔서)
double PerLot(string sym)
{
   if (sym == Symbol())
      return Contract();
   double ts = MarketInfo(sym, MODE_TICKSIZE);
   return ts > 0 ? MarketInfo(sym, MODE_TICKVALUE) / ts : 0;
}

// 종목을 시장 감시에 넣고 종목 정보가 있으면 true (주말엔 시세가 없어도 됨. 못 찾으면 전략이 1분마다 다시 확인)
bool UseSymbol(string sym) { return sym != "" && SymbolSelect(sym, true) && MarketInfo(sym, MODE_TICKSIZE) > 0; }

// 0.01 단위 내림, 최소/최대 lot 안으로
double FloorLotSym(string sym, double x)
{
   double step = MarketInfo(sym, MODE_LOTSTEP);
   double lot = MathFloor(x / step + 1e-9) * step;
   lot = MathMin(MathMax(lot, MarketInfo(sym, MODE_MINLOT)), MarketInfo(sym, MODE_MAXLOT));
   return NormalizeDouble(lot, 2);
}

double FloorLot(double x) { return FloorLotSym(Symbol(), x); }

// 오늘(서버 날짜) 청산된 매매 손익 합 (모든 EA·종목, 입출금 제외)
double TodayTradePnl()
{
   datetime d0 = DayStart(TimeCurrent());
   double s = 0;
   for (int i = OrdersHistoryTotal() - 1; i >= 0; i--)
      if (OrderSelect(i, SELECT_BY_POS, MODE_HISTORY) && OrderType() <= OP_SELL && OrderCloseTime() >= d0)
         s += OrderProfit() + OrderSwap() + OrderCommission();
   return s;
}

// start 시점 잔고 = 현재 잔고 - start 이후 모든 잔고 변동 (거래 + 입출금 6). 크레딧(7, 보너스)은 잔고에 안 들어가서 뺌
double BalanceAt(datetime start)
{
   double s = 0;
   for (int i = OrdersHistoryTotal() - 1; i >= 0; i--)
      if (OrderSelect(i, SELECT_BY_POS, MODE_HISTORY) && (OrderType() <= OP_SELL || OrderType() == 6) && OrderCloseTime() >= start)
         s += OrderProfit() + OrderSwap() + OrderCommission();
   return AccountBalance() - s;
}

// start ~ end 입금 합계 (출금 제외). 수익률 기준 = 시작 잔고 + 입금 (입금 때문에 %가 부풀지 않게)
double Deposits(datetime start, datetime end)
{
   double s = 0;
   for (int i = OrdersHistoryTotal() - 1; i >= 0; i--)
      if (OrderSelect(i, SELECT_BY_POS, MODE_HISTORY) && OrderType() == 6 && OrderProfit() > 0
          && OrderCloseTime() >= start && OrderCloseTime() < end)
         s += OrderProfit();
   return s;
}

string PctLine(string name, datetime start, int magic, string sym)
{
   double pnl = 0;
   for (int i = OrdersHistoryTotal() - 1; i >= 0; i--)
      if (OrderSelect(i, SELECT_BY_POS, MODE_HISTORY) && OrderType() <= OP_SELL && OrderMagicNumber() == magic
          && OrderSymbol() == sym && OrderCloseTime() >= start)
         pnl += OrderProfit() + OrderSwap() + OrderCommission();
   double b0 = BalanceAt(start) + Deposits(start, TimeCurrent() + 86400);
   return StringFormat("%s %+.2f%% (%+.2f$)", name, b0 > 0 ? pnl / b0 * 100 : 0.0, pnl);
}

// 오늘 / 이번주 / 이번달 수익률 (reports.summary_line과 같은 모양). sym을 안 주면 차트 종목
string SummaryLine(int magic, string sym = "")
{
   if (sym == "")
      sym = Symbol();
   datetime now = TimeCurrent(), d0 = DayStart(now);
   datetime w0 = d0 - ((TimeDayOfWeek(d0) + 6) % 7) * 86400;
   datetime m0 = StringToTime(StringFormat("%04d.%02d.01", TimeYear(now), TimeMonth(now)));
   return PctLine("오늘", d0, magic, sym) + "\n" + PctLine("이번주", w0, magic, sym) + "\n" + PctLine("이번달", m0, magic, sym);
}

// 계좌 손실 한도: 평가금이 최고 잔고보다 stopPct% 이상 줄면 멈춤 (상태 파일에 halted로 남음). 막아야 하면 이유, 아니면 ""
string RiskBlock(string tag, string file, double stopPct)
{
   string halted = StGet(file, "halted", "");
   if (halted != "")
      return "계좌 손실 한도로 중단된 상태 (" + halted + "). 확인 후 MQL4\\Files\\" + file + "에서 halted 줄을 지우면 다시 시작";
   if (AccountBalance() <= 0 || AccountEquity() <= 0)        // 계좌 정보를 아직 못 받은 순간 (0으로 읽힘): 판단 안 함
      return "";
   double old = StringToDouble(StGet(file, "peak", "0"));
   double peak = MathMax(old, AccountBalance());
   if (peak != old)
      StSet(file, "peak", DoubleToString(peak, 2));
   if (AccountEquity() <= peak * (1 - stopPct / 100))
   {
      halted = StringFormat("%s 평가금 %.2f / 최고 잔고 %.2f", TimeToString(TimeCurrent(), TIME_DATE | TIME_MINUTES), AccountEquity(), peak);
      StSet(file, "halted", halted);
      Send(tag, StringFormat("🛑 계좌 손실 한도(%g%%) 도달 -> 신규 진입을 중단합니다.\n%s\n확인 후 MQL4\\Files\\%s에서 halted 줄을 지우세요.",
                        stopPct, halted, file));
      return "계좌 손실 한도 도달 (" + halted + ")";
   }
   return "";
}

//---------------------------------------------------------------- 주문
int MyTicketSym(string sym, int magic)
{
   for (int i = OrdersTotal() - 1; i >= 0; i--)
      if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES) && OrderSymbol() == sym && OrderMagicNumber() == magic
          && OrderType() <= OP_SELL)
         return OrderTicket();
   return -1;
}

int MyTicket(int magic) { return MyTicketSym(Symbol(), magic); }

bool MarginOkSym(string tag, string sym, double lot)
{
   double need = MarketInfo(sym, MODE_MARGINREQUIRED) * lot;
   double free = AccountFreeMargin();
   if (need > free * 0.9)
   {
      Send(tag, StringFormat("⚠️ 증거금 부족으로 진입 안 함\n필요 %.2f / 여유 %.2f (%.2f lot)", need, free, lot));
      return false;
   }
   return true;
}

bool MarginOk(string tag, double lot) { return MarginOkSym(tag, Symbol(), lot); }

//---------------------------------------------------------------- 시세 신선도
// 차트 종목 마지막 틱 시각이 바뀐 로컬 시각. OnTimer가 0.1초마다 TickWatch()를 불러 갱신
// EA를 켠 뒤 처음 본 틱은 언제 온 것인지 모르니(캐시) 신선하다고 보지 않음. 그 뒤 틱 시각이 바뀌어야 신선
datetime g_tickT = 0, g_tickLocal = 0;
#define TICK_FRESH_SEC 30            // 진입 전 틱이 이 안에 들어왔어야 함 (demo_trader.TICK_FRESH_SEC)
#define ENTRY_LATE_SEC 60            // 신호 봉 마감보다 이만큼 늦으면 진입 안 함 (demo_trader.ENTRY_LATE_SEC)

void TickWatch()
{
   datetime t = (datetime)MarketInfo(Symbol(), MODE_TIME);
   if (t != g_tickT)
   {
      if (g_tickT > 0)
         g_tickLocal = TimeLocal();
      g_tickT = t;
   }
}

// 서버에 연결돼 있고 차트 종목 틱이 TICK_FRESH_SEC 안에 들어왔으면 true
bool TickFresh()
{
   TickWatch();
   return IsConnected() && g_tickLocal > 0 && TimeLocal() - g_tickLocal <= TICK_FRESH_SEC;
}

//---------------------------------------------------------------- 서버 SL/TP 지키기
// 진입하자마자(SL/TP 걸기 전에) MQL4\Files\infinox_protect.txt에 "티켓=sl|tp|tag"를 남기고, 걸리면 지움.
// 못 걸었으면 ProtectTick()이 5초마다 다시 건다. 이미 그 가격을 넘었으면 시장가 청산 (demo_trader.protect와 같음). 재시작해도 이어짐
#define PROTECT_FILE "infinox_protect.txt"
#define PROTECT_SEC 5
datetime g_protectTry = 0;

bool SetStops(int tk, double sl, double tp, int tries)
{
   if (!OrderSelect(tk, SELECT_BY_TICKET))
      return false;
   for (int i = 0; i < tries; i++)
   {
      if (OrderModify(tk, OrderOpenPrice(), sl, tp, 0, clrNONE))
         return true;
      if (i + 1 < tries)
         Sleep(300);
   }
   return false;
}

// sl/tp는 가격 (0이면 없음). 목록에 먼저 넣고 바로 5번 시도, 걸리면 목록에서 뺌.
// 반환: 서버에 걸렸거나 복구 목록에 저장됐으면 true (어느 쪽도 아니면 false -> 주문 중 기록을 지우면 안 됨)
bool Protect(string tag, int tk, double sl, double tp)
{
   if (sl <= 0 && tp <= 0)
      return true;
   bool saved = StSet(PROTECT_FILE, IntegerToString(tk), StringFormat("%.5f|%.5f|%s", sl, tp, tag));
   if (SetStops(tk, sl, tp, 5))
   {
      if (saved)
         StSet(PROTECT_FILE, IntegerToString(tk), "");
      return true;
   }
   int err = GetLastError();
   Send(tag, saved ? StringFormat("❗ 손절/익절 설정 실패 (티켓 %d, 오류 %d) -> 5초마다 다시 걸고, 이미 넘은 가격이면 청산합니다", tk, err)
                   : StringFormat("❗ 손절/익절 설정 실패 (티켓 %d, 오류 %d), 복구 목록 저장도 실패 -> 1분마다 포지션 대사로 다시 확인합니다", tk, err));
   return saved;
}

void ProtectTick()
{
   if (TimeLocal() - g_protectTry < PROTECT_SEC)
      return;
   g_protectTry = TimeLocal();
   int h = FileOpen(PROTECT_FILE, FILE_READ | FILE_TXT | FILE_ANSI);
   if (h == INVALID_HANDLE)
      return;
   string lines[];
   int n = 0;
   while (!FileIsEnding(h))
   {
      string line = FileReadString(h);
      if (StringFind(line, "=") > 0 && StringFind(line, "|") > 0)
      {
         ArrayResize(lines, n + 1);
         lines[n++] = line;
      }
   }
   FileClose(h);
   for (int i = 0; i < n; i++)
   {
      int p = StringFind(lines[i], "=");
      string key = StringSubstr(lines[i], 0, p), f[];
      if (StringGetCharacter(key, 0) == 'm')
         continue;                                  // "m<매직>" = 주문 보내는 중 기록 (ReconcileStops가 씀)
      if (StringSplit(StringSubstr(lines[i], p + 1), '|', f) < 3)
         continue;
      int tk = (int)StringToInteger(key);
      double sl = StringToDouble(f[0]), tp = StringToDouble(f[1]);
      string tag = f[2];
      if (!OrderSelect(tk, SELECT_BY_TICKET))
         continue;                                  // 조회 실패: 지우지 않고 다음에 다시
      if (OrderCloseTime() > 0)
      {
         StSet(PROTECT_FILE, key, "");              // 닫힘 확인: 목록에서 뺌
         continue;
      }
      if ((sl <= 0 || OrderStopLoss() > 0) && (tp <= 0 || OrderTakeProfit() > 0))
      {
         StSet(PROTECT_FILE, key, "");              // 이미 걸려 있음 (전략이 따로 옮긴 경우 포함)
         continue;
      }
      int side = OrderType() == OP_BUY ? 1 : -1;
      string sym = OrderSymbol();
      RefreshRates();
      double bid = MarketInfo(sym, MODE_BID), ask = MarketInfo(sym, MODE_ASK);
      // 호가가 비었거나(0) 뒤집혔거나 30초 넘게 멈춘 종목이면 판단 보류 (잘못 청산하지 않게)
      // 시세 전체가 멈추면 TimeCurrent도 같이 멈추니 차트 종목 틱의 로컬 경과(TickFresh)와 연결도 같이 봄
      if (bid <= 0 || ask < bid || !TickFresh() || TimeCurrent() - (datetime)MarketInfo(sym, MODE_TIME) > TICK_FRESH_SEC)
         continue;
      double px = side == 1 ? bid : ask;
      double nsl = OrderStopLoss() > 0 ? OrderStopLoss() : sl, ntp = OrderTakeProfit() > 0 ? OrderTakeProfit() : tp;
      if (nsl > 0 && (px - nsl) * side <= 0)
      {
         Print("[", tag, "] SL 없는 포지션 ", tk, ": 이미 손절선 ", nsl, " 너머 -> 청산");
         CloseTicket(tag, tk, 1.0);
         continue;
      }
      if (ntp > 0 && (ntp - px) * side <= 0)
      {
         Print("[", tag, "] TP 없는 포지션 ", tk, ": 이미 익절선 ", ntp, " 너머 -> 청산");
         CloseTicket(tag, tk, 1.0);
         continue;
      }
      if (SetStops(tk, nsl, ntp, 1))
      {
         StSet(PROTECT_FILE, key, "");
         Send(tag, StringFormat("✅ 티켓 %d 손절/익절 다시 걸었어요 (SL %.2f / TP %.2f)", tk, nsl, ntp));
      }
   }
}

// 시장가 진입 후 체결가 기준으로 손절/익절(가격 거리, 0이면 없음)을 서버에 설정 (실패하면 Protect가 계속 다시 검). 티켓 또는 -1
// slAbs: 손절을 거리 대신 절대 가격으로 (지수. 체결 미끄러짐과 상관없이 그 가격)
int OpenMarketSym(string tag, string sym, int side, double lot, double slUsd, double tpUsd, int magic, string comment,
                  double slipUsd, double &fill, double slAbs = 0)
{
   // 주문 보내기 전에 "m<매직>=손절|익절 거리|tag|손절이 절대 가격이면 1"을 남김: 체결 직후 꺼져 티켓 기록이 없어도
   // 켤 때·1분마다 ReconcileStops가 이 값으로 다시 검. 티켓 기록 또는 서버 설정이 확인된 뒤에만 지움
   string mk = "m" + IntegerToString(magic);
   bool pend = slUsd > 0 || tpUsd > 0 || slAbs > 0;
   if (pend && !StSet(PROTECT_FILE, mk, StringFormat("%.5f|%.5f|%s|%d", slAbs > 0 ? slAbs : slUsd, tpUsd, tag, slAbs > 0 ? 1 : 0)))
   {
      Send(tag, "❗ 복구 기록을 저장하지 못해 주문을 보내지 않았어요 (MQL4 Files 폴더 쓰기 확인)");
      return -1;
   }
   RefreshRates();
   int digits = (int)MarketInfo(sym, MODE_DIGITS);
   int slip = (int)MathRound(slipUsd / MarketInfo(sym, MODE_POINT));
   double price = side == 1 ? MarketInfo(sym, MODE_ASK) : MarketInfo(sym, MODE_BID);
   int tk = OrderSend(sym, side == 1 ? OP_BUY : OP_SELL, lot, price, slip, 0, 0, comment, magic, 0,
                      side == 1 ? clrDodgerBlue : clrRed);
   if (tk < 0)
   {
      Send(tag, StringFormat("❗ 주문 실패 (%s %.2f lot): 오류 %d", side == 1 ? "매수" : "매도", lot, GetLastError()));
      if (pend)
         StSet(PROTECT_FILE, mk, "");
      return -1;
   }
   bool known = false;
   for (int i = 0; i < 3 && !known; i++)
   {
      known = OrderSelect(tk, SELECT_BY_TICKET);
      if (!known)
         Sleep(200);
   }
   fill = known ? OrderOpenPrice() : price;
   if (!known)
   {
      // 체결가를 모르면 요청가로 보호선을 정하지 않음: 주문 중 기록을 남겨 1분 대사가 실제 체결가로 검
      Send(tag, StringFormat("❗ 티켓 %d 체결가 조회 실패 -> 1분마다 실제 체결가 기준으로 손절/익절 복구를 시도합니다 (진입 알림의 가격은 요청가)", tk));
      return tk;
   }
   double sl = slAbs > 0 ? NormalizeDouble(slAbs, digits) : slUsd > 0 ? NormalizeDouble(fill - side * slUsd, digits) : 0;
   double tp = tpUsd > 0 ? NormalizeDouble(fill + side * tpUsd, digits) : 0;
   if (Protect(tag, tk, sl, tp) && pend)
      StSet(PROTECT_FILE, mk, "");
   return tk;
}

int OpenMarket(string tag, int side, double lot, double slUsd, double tpUsd, int magic, string comment, double slipUsd, double &fill)
{
   return OpenMarketSym(tag, Symbol(), side, lot, slUsd, tpUsd, magic, comment, slipUsd, fill);
}

// 청산 실패 알림은 티켓마다 처음 한 번 + 그 뒤 1분에 한 번만 (재시도는 호출하는 쪽이 계속)
#define CLOSE_WARN_SLOTS 32
int      g_cwTk[CLOSE_WARN_SLOTS];
datetime g_cwAt[CLOSE_WARN_SLOTS];

bool CloseWarnDue(int tk)
{
   int free = -1, old = 0;
   for (int i = 0; i < CLOSE_WARN_SLOTS; i++)
   {
      if (g_cwTk[i] == tk)
      {
         if (TimeLocal() - g_cwAt[i] < 60)
            return false;
         g_cwAt[i] = TimeLocal();
         return true;
      }
      if (g_cwTk[i] == 0 && free < 0)
         free = i;
      if (g_cwAt[i] < g_cwAt[old])
         old = i;
   }
   int k = free >= 0 ? free : old;
   g_cwTk[k] = tk;
   g_cwAt[k] = TimeLocal();
   return true;
}

bool CloseTicket(string tag, int tk, double slipUsd)
{
   if (!OrderSelect(tk, SELECT_BY_TICKET) || OrderCloseTime() > 0)
      return false;
   RefreshRates();
   string sym = OrderSymbol();
   double px = OrderType() == OP_BUY ? MarketInfo(sym, MODE_BID) : MarketInfo(sym, MODE_ASK);
   bool ok = OrderClose(tk, OrderLots(), px, (int)MathRound(slipUsd / MarketInfo(sym, MODE_POINT)), clrYellow);
   if (!ok)
   {
      int err = GetLastError();
      Print("[", tag, "] 청산 실패: 티켓 ", tk, " 오류 ", err);
      if (CloseWarnDue(tk))
         Send(tag, StringFormat("❗ 청산 실패: 티켓 %d 오류 %d (닫힐 때까지 다시 시도, 알림은 1분에 한 번)", tk, err));
   }
   return ok;
}

// 닫힌 주문이면 손익/청산가/이유(TP, SL, CLOSE) 채우고 true
bool ClosedInfo(int tk, double &profit, double &closePx, string &reason)
{
   if (!OrderSelect(tk, SELECT_BY_TICKET) || OrderCloseTime() == 0)
      return false;
   profit = OrderProfit() + OrderSwap() + OrderCommission();
   closePx = OrderClosePrice();
   string cm = OrderComment();
   reason = "CLOSE";
   if (StringFind(cm, "[sl]") >= 0)
      reason = "SL";
   else if (StringFind(cm, "[tp]") >= 0)
      reason = "TP";
   else if (OrderStopLoss() > 0 && MathAbs(closePx - OrderStopLoss()) < 0.5)
      reason = "SL";
   else if (OrderTakeProfit() > 0 && MathAbs(closePx - OrderTakeProfit()) < 0.5)
      reason = "TP";
   return true;
}

bool SymbolOk()
{
   if (StringFind(Symbol(), "XAUUSD") != 0)
   {
      Alert("XAUUSD 차트에 붙여 주세요 (지금 ", Symbol(), ")");
      return false;
   }
   return true;
}
