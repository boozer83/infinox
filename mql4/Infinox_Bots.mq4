// Infinox 봇을 한 EA로 돌리는 MT4판. XAUUSD 차트 하나(아무 시간봉)에 붙임
// 전략 (파이썬 봇과 같은 규칙, 뉴스 양방향만 빼고 전부):
//   0 SAR 단타 / 1 H4 추세 금 / 2 아시아 3마틴 금 / 3 아시아 3마틴 XAUEUR / 4 단타 2마틴 / 5 H4 추세 이더리움 /
//   6 헤지 마틴 / 7 나스닥 첫30분 / 8 독일 ORB 매도 / 9 서프라이즈
//   새로 넣은 4~9는 처음엔 꺼진 채 (패널에서 체크해야 켜짐)
// 차트 패널:
// - 전략 체크박스와 전략별 오늘 / 이번주 / 이번달 수익률
// - 시작 버튼: 선택한 전략만 켬. 정지 버튼: 전부 끔. 켜져 있는 동안 선택을 바꾸면 바로 반영
// - 끄면 신규 진입만 멈춤. 이미 열린 포지션은 서버 SL/TP와 청산 규칙(금요일·23:00·추세 종료·마감·사이클 목표)으로 계속 관리
// 선택 / 시작 상태는 MQL4\Files\infinox_panel.txt에 저장 -> MT4를 껐다 켜도 그대로
// 새 전략 추가: mod_*.mqh에 CStrat를 이은 클래스를 만들고 아래 OnInit의 Add(...)에 한 줄 (패널·알림·타이머·리포트가 같이 붙음)
// 다른 종목(XAUEUR, ETHUSD, NAS100, GER40)은 입력값의 종목명을 씀 (빈칸이면 XAUEUR + 차트 종목 접미사). MT4 시장 감시에 있어야 함
// 서프라이즈·헤지 마틴의 뉴스 거르기는 파이썬 데모 서프라이즈 봇이 Common\Files에 써 주는 파일을 읽음 (mod_surprise.mqh, mod_hedge.mqh)
// 리포트: 한국시간 평일 21:00에 계좌 리포트 (mod_report.mqh, report_bot.py와 같음)
#property strict
#property description "Infinox 봇 10개 (SAR 단타·2마틴 / H4 추세 금·이더리움 / 아시아 3마틴 금·XAUEUR / 헤지 마틴 / 나스닥·독일 / 서프라이즈) + 켜고 끄는 패널"

input bool   AllowLive = true;       // 실계좌에서도 실행
input double LotPer1000 = 0.05;      // SAR 단타·2마틴: 그날 시작 잔고 1,000$당 시작 lot
input double MartinMaxLot = 1.0;     // SAR 단타·2마틴: 넘으면 이 lot으로 깎아서 진입
input double TrendRiskPct = 2.0;     // H4 추세: 거래당 위험 (잔고 %)
input double TrendMaxLot = 5.0;      // H4 추세: 최대 lot
input double AsiaRiskPct = 0.6;      // 아시아 3마틴 금: 1단계 위험 (잔고 %, XAUEUR와 같이 돌려서 1% -> 0.6%)
input double AsiaEurRiskPct = 0.6;   // 아시아 3마틴 XAUEUR: 1단계 위험 (잔고 %)
input double AsiaMaxLot = 1.0;       // 아시아 3마틴: 진입 lot이 이보다 크면 진입 안 함
input double HedgeLotPer1000 = 0.01; // 헤지 마틴: 사이클 시작 잔고 1,000$당 시작 lot
input double SurpriseLotPer1000 = 0.01; // 서프라이즈: 잔고 1,000$당 lot
input string AsiaEurSymbol = "";     // XAUEUR 종목명 (빈칸이면 XAUEUR + 차트 종목 접미사)
input string EthSymbol = "ETHUSD";   // 이더리움 종목명
input string NasSymbol = "NAS100";   // 나스닥 종목명
input string GerSymbol = "GER40";    // 독일 종목명
input int    MagicMartin = 20261017;
input int    MagicTrend = 20261018;
input int    MagicAsia = 20261019;
input int    MagicHedge = 20261020;
input int    MagicSarMartin = 20261022;
input int    MagicTrendEth = 20261023;
input int    MagicAsiaEur = 20261024;
input int    MagicNas = 20261025;
input int    MagicGer = 20261026;
input int    MagicSurprise = 20261027;

#include "infinox_common.mqh"
#include "mod_martin.mqh"
#include "mod_trend.mqh"
#include "mod_asia.mqh"
#include "mod_hedge.mqh"
#include "mod_index.mqh"
#include "mod_surprise.mqh"

#define PANEL_FILE "infinox_panel.txt"
#define PFX "ifx_"
#define FONT "Malgun Gothic"
#define MAX_STRAT 16

CSar      g_sar, g_sarMartin;
CTrend    g_trend, g_trendEth;
CAsia     g_asia, g_asiaEur;
CHedge    g_hedge;
CIndex    g_nas, g_ger;
CSurprise g_surprise;

CStrat  *g_s[MAX_STRAT];
int      N_STRAT = 0;
bool     g_sel[MAX_STRAT];
bool     g_master = false;
datetime g_lastPanel = 0;
uint     g_lastSec = 0;

void Add(CStrat *s)
{
   if (N_STRAT < MAX_STRAT)
      g_s[N_STRAT++] = s;
}

int    StratMagic(int i) { return g_s[i].Magic(); }
string StratSymbol(int i) { return g_s[i].Sym(); }
string StratName(int i) { return g_s[i].name; }
bool   IsOn(int i) { return g_master && g_sel[i]; }

#include "mod_report.mqh"

void SavePanel()
{
   for (int i = 0; i < N_STRAT; i++)
      StSet(PANEL_FILE, "sel" + IntegerToString(i), g_sel[i] ? "1" : "0");
   StSet(PANEL_FILE, "master", g_master ? "1" : "0");
}

void LoadPanel()
{
   for (int i = 0; i < N_STRAT; i++)
      g_sel[i] = StGet(PANEL_FILE, "sel" + IntegerToString(i), g_s[i].defOn ? "1" : "0") == "1";
   g_master = StGet(PANEL_FILE, "master", "0") == "1";
}

void NotifyStart(int i)
{
   string head = StringFormat("▶️ 시작 (%s %d)\n", AccountKind(), AccountNumber());
   Send(g_s[i].name, head + g_s[i].StartText() + "\n" + AccountLine());
}

void NotifyStop(int i)
{
   Send(g_s[i].name, "⏸ 정지 (신규 진입만 멈춤. 열린 포지션은 서버 SL/TP와 청산 규칙으로 계속 관리)");
}

//---------------------------------------------------------------- 패널
void Label(string name, int x, int y, string text, color clr, int size = 10)
{
   string n = PFX + name;
   if (ObjectFind(0, n) < 0)
   {
      ObjectCreate(0, n, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, n, OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, n, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, n, OBJPROP_HIDDEN, true);
   }
   ObjectSetInteger(0, n, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, n, OBJPROP_YDISTANCE, y);
   ObjectSetString(0, n, OBJPROP_TEXT, text);
   ObjectSetString(0, n, OBJPROP_FONT, FONT);
   ObjectSetInteger(0, n, OBJPROP_FONTSIZE, size);
   ObjectSetInteger(0, n, OBJPROP_COLOR, clr);
}

void Button(string name, int x, int y, int w, int h, string text, color bg, color fg = clrWhite, string font = FONT, int size = 10)
{
   string n = PFX + name;
   if (ObjectFind(0, n) < 0)
   {
      ObjectCreate(0, n, OBJ_BUTTON, 0, 0, 0);
      ObjectSetInteger(0, n, OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, n, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, n, OBJPROP_HIDDEN, true);
   }
   ObjectSetInteger(0, n, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, n, OBJPROP_YDISTANCE, y);
   ObjectSetInteger(0, n, OBJPROP_XSIZE, w);
   ObjectSetInteger(0, n, OBJPROP_YSIZE, h);
   ObjectSetString(0, n, OBJPROP_TEXT, text);
   ObjectSetString(0, n, OBJPROP_FONT, font);
   ObjectSetInteger(0, n, OBJPROP_FONTSIZE, size);
   ObjectSetInteger(0, n, OBJPROP_COLOR, fg);
   ObjectSetInteger(0, n, OBJPROP_BGCOLOR, bg);
   ObjectSetInteger(0, n, OBJPROP_BORDER_COLOR, clrDimGray);
   ObjectSetInteger(0, n, OBJPROP_STATE, false);
}

double PeriodPct(datetime start, int magic, string sym, double &pnl)
{
   pnl = 0;
   for (int i = OrdersHistoryTotal() - 1; i >= 0; i--)
      if (OrderSelect(i, SELECT_BY_POS, MODE_HISTORY) && OrderType() <= OP_SELL && OrderMagicNumber() == magic
          && OrderSymbol() == sym && OrderCloseTime() >= start)
         pnl += OrderProfit() + OrderSwap() + OrderCommission();
   double b0 = BalanceAt(start);
   return b0 > 0 ? pnl / b0 * 100 : 0;
}

string ReturnText(int k)
{
   int magic = StratMagic(k);
   string sym = StratSymbol(k);
   datetime now = TimeCurrent(), d0 = DayStart(now);
   datetime w0 = d0 - ((TimeDayOfWeek(d0) + 6) % 7) * 86400;
   datetime m0 = StringToTime(StringFormat("%04d.%02d.01", TimeYear(now), TimeMonth(now)));
   double pd, pw, pm;
   double rd = PeriodPct(d0, magic, sym, pd), rw = PeriodPct(w0, magic, sym, pw), rm = PeriodPct(m0, magic, sym, pm);
   return StringFormat("오늘 %+.2f%%   이번주 %+.2f%%   이번달 %+.2f%% (%+.0f$)", rd, rw, rm, pm);
}

void DrawPanel()
{
   int x0 = 10, y0 = 25, rowH = 28;
   string bg = PFX + "bg";
   if (ObjectFind(0, bg) < 0)
   {
      ObjectCreate(0, bg, OBJ_RECTANGLE_LABEL, 0, 0, 0);
      ObjectSetInteger(0, bg, OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, bg, OBJPROP_XDISTANCE, x0);
      ObjectSetInteger(0, bg, OBJPROP_YDISTANCE, y0);
      ObjectSetInteger(0, bg, OBJPROP_BGCOLOR, C'24,28,36');
      ObjectSetInteger(0, bg, OBJPROP_BORDER_TYPE, BORDER_FLAT);
      ObjectSetInteger(0, bg, OBJPROP_COLOR, clrDimGray);
      ObjectSetInteger(0, bg, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, bg, OBJPROP_HIDDEN, true);
   }
   ObjectSetInteger(0, bg, OBJPROP_XSIZE, 660);
   ObjectSetInteger(0, bg, OBJPROP_YSIZE, 48 + N_STRAT * rowH + 34);
   Label("title", x0 + 12, y0 + 8, StringFormat("Infinox 자동매매 · %s %d · 잔고 %.2f / 평가 %.2f %s",
                                                AccountKind(), AccountNumber(), AccountBalance(), AccountEquity(), AccountCurrency()),
         clrWhite, 10);
   for (int i = 0; i < N_STRAT; i++)
   {
      int y = y0 + 36 + i * rowH;
      // 체크박스: 흰 네모 안에 Wingdings 체크(252). 이름 글자를 눌러도 같이 바뀜
      Button("sel" + IntegerToString(i), x0 + 12, y + 2, 20, 20, g_sel[i] ? CharToString(252) : " ",
             clrWhite, C'20,110,220', "Wingdings", 13);
      Label("sel" + IntegerToString(i) + "n", x0 + 40, y + 3, g_s[i].name, g_sel[i] ? clrWhite : clrGray, 9);
      bool on = IsOn(i);
      Label("st" + IntegerToString(i), x0 + 210, y + 3, on ? "켜짐" : "꺼짐", on ? clrLime : clrGray, 9);
      Label("ret" + IntegerToString(i), x0 + 252, y + 3, ReturnText(i), clrSilver, 9);
   }
   int yb = y0 + 42 + N_STRAT * rowH;
   Button("master", x0 + 12, yb, 220, 30, g_master ? "■ 전체 정지" : "▶ 선택한 전략 시작",
          g_master ? C'170,40,40' : C'30,130,60');
   string hint = g_master ? "켜진 동안 전략 버튼을 누르면 바로 켜지고 꺼짐" : "전략을 고르고 시작을 누르세요";
   Label("hint", x0 + 244, yb + 8, hint, clrGray, 9);
   ChartRedraw();
}

void DeletePanel() { ObjectsDeleteAll(0, PFX); }

//---------------------------------------------------------------- MT4 이벤트
int OnInit()
{
   if (!SymbolOk())
      return INIT_FAILED;
   if (!IsDemo() && !AllowLive)
   {
      Alert("실계좌입니다. AllowLive를 true로 해야 실행됩니다.");
      return INIT_FAILED;
   }
   TgInit();
   g_login = AccountNumber();
   string sfx = StringSubstr(Symbol(), 6);
   string eur = AsiaEurSymbol != "" ? AsiaEurSymbol : "XAUEUR" + sfx;
   // 순서 = 패널 줄 번호 = 패널 파일의 sel 번호 (앞 4개는 예전 EA와 같은 번호라 선택 상태가 이어짐)
   N_STRAT = 0;
   g_sar.Setup("SAR 단타", "infinox_martin_state.txt", MagicMartin, false, true);
   g_trend.Setup(Symbol(), "H4 추세", "infinox_trend_state.txt", MagicTrend, true);
   g_asia.Setup(Symbol(), "아시아 3마틴", "infinox_asia_state.txt", MagicAsia, AsiaRiskPct, true);
   g_asiaEur.Setup(eur, "아시아 3마틴·XAUEUR", "infinox_asia_xaueur_state.txt", MagicAsiaEur, AsiaEurRiskPct, false);
   g_sarMartin.Setup("단타 2마틴", "infinox_sar_martin_state.txt", MagicSarMartin, true, false);
   g_trendEth.Setup(EthSymbol, "H4 추세·ETH", "infinox_trend_eth_state.txt", MagicTrendEth, false);
   g_hedge.Setup("헤지 마틴", "infinox_hedge_state.txt", MagicHedge, false);
   g_nas.Setup(NasSymbol, "NAS100", "나스닥 첫30분", "infinox_index_nas_state.txt", MagicNas, false);
   g_ger.Setup(GerSymbol, "GER40", "독일 ORB 매도", "infinox_index_ger_state.txt", MagicGer, false);
   g_surprise.Setup("서프라이즈", "infinox_surprise_state.txt", MagicSurprise, false);
   Add(GetPointer(g_sar));
   Add(GetPointer(g_trend));
   Add(GetPointer(g_asia));
   Add(GetPointer(g_asiaEur));
   Add(GetPointer(g_sarMartin));
   Add(GetPointer(g_trendEth));
   Add(GetPointer(g_hedge));
   Add(GetPointer(g_nas));
   Add(GetPointer(g_ger));
   Add(GetPointer(g_surprise));
   LoadPanel();
   for (int i = 0; i < N_STRAT; i++)
      g_s[i].Init();
   R_Init();
   if (g_master)
   {
      string list = "";
      for (int i = 0; i < N_STRAT; i++)
         if (g_sel[i])
            list += (list == "" ? "" : ", ") + g_s[i].name;
      Send("Infinox", StringFormat("▶️ EA 다시 켜짐 (%s %d): 실행 중 %s\n%s", AccountKind(), AccountNumber(),
                                       list == "" ? "없음" : list, AccountLine()));
   }
   if (!IsTradeAllowed())
      Send("Infinox", "⚠️ MT4의 '자동매매'가 꺼져 있거나 EA 설정에서 '실거래 허용'이 꺼져 있습니다. 켜야 주문이 들어갑니다.");
   DrawPanel();
   EventSetMillisecondTimer(100);      // 서프라이즈는 0.1초마다, 나머지는 1초마다
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   DeletePanel();
   if (g_master && (reason == REASON_REMOVE || reason == REASON_CLOSE))
      Send("Infinox", "⏸ EA가 차트에서 빠졌습니다. 신규 진입이 멈춥니다 (열린 포지션은 서버 SL/TP로만 관리)");
}

void OnTimer()
{
   if (!AccountOk())
      return;
   for (int i = 0; i < N_STRAT; i++)
      g_s[i].Fast(IsOn(i));
   if (GetTickCount() - g_lastSec < 1000)
      return;
   g_lastSec = GetTickCount();
   for (int i = 0; i < N_STRAT; i++)
      g_s[i].Timer(IsOn(i));
   R_Timer();
   if (TimeLocal() - g_lastPanel >= 5)
   {
      g_lastPanel = TimeLocal();
      DrawPanel();
   }
}

void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
   if (id != CHARTEVENT_OBJECT_CLICK || StringFind(sparam, PFX) != 0)
      return;
   string name = StringSubstr(sparam, StringLen(PFX));
   if (StringFind(name, "sel") == 0)
   {
      string num = StringSubstr(name, 3);
      if (StringFind(num, "n") > 0)
         num = StringSubstr(num, 0, StringFind(num, "n"));
      int i = (int)StringToInteger(num);
      if (i < 0 || i >= N_STRAT)
         return;
      g_sel[i] = !g_sel[i];
      SavePanel();
      if (g_master)
      {
         if (g_sel[i])
            NotifyStart(i);
         else
            NotifyStop(i);
      }
   }
   else if (name == "master")
   {
      if (!g_master)
      {
         string list = "";
         for (int i = 0; i < N_STRAT; i++)
            if (g_sel[i])
               list += (list == "" ? "" : ", ") + g_s[i].name;
         if (list == "")
            MessageBox("선택한 전략이 없어요. 전략 버튼을 먼저 눌러 주세요.", "Infinox", MB_OK);
         else if (MessageBox(StringFormat("%s %d에서 시작할까요?\n\n%s", AccountKind(), AccountNumber(), list),
                             "Infinox", MB_YESNO | MB_ICONQUESTION) == IDYES)
         {
            g_master = true;
            SavePanel();
            for (int i = 0; i < N_STRAT; i++)
               if (g_sel[i])
                  NotifyStart(i);
         }
      }
      else
      {
         g_master = false;
         SavePanel();
         for (int i = 0; i < N_STRAT; i++)
            if (g_sel[i])
               NotifyStop(i);
      }
   }
   DrawPanel();
}

void OnTick() {}
