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

void StSet(string file, string key, string val)
{
   string lines[];
   int n = 0;
   bool found = false;
   int h = FileOpen(file, FILE_READ | FILE_TXT | FILE_ANSI);
   if (h != INVALID_HANDLE)
   {
      while (!FileIsEnding(h))
      {
         string line = FileReadString(h);
         if (line == "")
            continue;
         int p = StringFind(line, "=");
         if (p > 0 && StringSubstr(line, 0, p) == key)
         {
            line = key + "=" + val;
            found = true;
         }
         ArrayResize(lines, n + 1);
         lines[n++] = line;
      }
      FileClose(h);
   }
   if (!found)
   {
      ArrayResize(lines, n + 1);
      lines[n++] = key + "=" + val;
   }
   h = FileOpen(file, FILE_WRITE | FILE_TXT | FILE_ANSI);
   if (h == INVALID_HANDLE)
   {
      Print("상태 파일 저장 실패: ", file, " 오류 ", GetLastError());
      return;
   }
   for (int i = 0; i < n; i++)
      FileWriteString(h, lines[i] + "\r\n");
   FileClose(h);
}

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

// start 시점 잔고 = 현재 잔고 - start 이후 모든 잔고 변동 (입출금 포함)
double BalanceAt(datetime start)
{
   double s = 0;
   for (int i = OrdersHistoryTotal() - 1; i >= 0; i--)
      if (OrderSelect(i, SELECT_BY_POS, MODE_HISTORY) && OrderType() != OP_BUYLIMIT && OrderType() != OP_SELLLIMIT
          && OrderType() != OP_BUYSTOP && OrderType() != OP_SELLSTOP && OrderCloseTime() >= start)
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

// 시장가 진입 후 체결가 기준으로 손절/익절(가격 거리, 0이면 없음)을 서버에 설정. 티켓 또는 -1
int OpenMarketSym(string tag, string sym, int side, double lot, double slUsd, double tpUsd, int magic, string comment,
                  double slipUsd, double &fill)
{
   RefreshRates();
   int digits = (int)MarketInfo(sym, MODE_DIGITS);
   int slip = (int)MathRound(slipUsd / MarketInfo(sym, MODE_POINT));
   double price = side == 1 ? MarketInfo(sym, MODE_ASK) : MarketInfo(sym, MODE_BID);
   int tk = OrderSend(sym, side == 1 ? OP_BUY : OP_SELL, lot, price, slip, 0, 0, comment, magic, 0,
                      side == 1 ? clrDodgerBlue : clrRed);
   if (tk < 0)
   {
      Send(tag, StringFormat("❗ 주문 실패 (%s %.2f lot): 오류 %d", side == 1 ? "매수" : "매도", lot, GetLastError()));
      return -1;
   }
   fill = OrderSelect(tk, SELECT_BY_TICKET) ? OrderOpenPrice() : price;
   double sl = slUsd > 0 ? NormalizeDouble(fill - side * slUsd, digits) : 0;
   double tp = tpUsd > 0 ? NormalizeDouble(fill + side * tpUsd, digits) : 0;
   bool ok = false;
   for (int i = 0; i < 5 && !ok; i++)
   {
      ok = OrderModify(tk, fill, sl, tp, 0, clrNONE);
      if (!ok)
         Sleep(500);
   }
   if (!ok)
      Send(tag, StringFormat("❗ 손절/익절 설정 실패 (티켓 %d, 오류 %d). MT4에서 직접 확인하세요.", tk, GetLastError()));
   return tk;
}

int OpenMarket(string tag, int side, double lot, double slUsd, double tpUsd, int magic, string comment, double slipUsd, double &fill)
{
   return OpenMarketSym(tag, Symbol(), side, lot, slUsd, tpUsd, magic, comment, slipUsd, fill);
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
      Send(tag, StringFormat("❗ 청산 실패: 티켓 %d 오류 %d", tk, GetLastError()));
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
