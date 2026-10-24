// Infinox 봇 3개(SAR 단타 / H4 추세 / 아시아 3마틴)를 한 EA로 돌리는 MT4판. XAUUSD 차트 하나(아무 시간봉)에 붙임
// 차트 패널:
// - 전략 체크박스와 전략별 오늘 / 이번주 / 이번달 수익률
// - 시작 버튼: 선택한 전략만 켬. 정지 버튼: 전부 끔. 켜져 있는 동안 선택을 바꾸면 바로 반영
// - 끄면 신규 진입만 멈춤. 이미 열린 포지션은 서버 SL/TP와 청산 규칙(금요일·23:00·추세 종료)으로 계속 관리
// 선택 / 시작 상태는 MQL4\Files\infinox_panel.txt에 저장 -> MT4를 껐다 켜도 그대로
// 각 전략 규칙은 mod_martin.mqh / mod_trend.mqh / mod_asia.mqh (파이썬 봇과 같음)
// 리포트: 한국시간 평일 21:00에 계좌 리포트 (mod_report.mqh, report_bot.py와 같음)
#property strict
#property description "Infinox 봇 3개(SAR 단타 / H4 추세 / 아시아 3마틴) + 켜고 끄는 패널"

input bool   AllowLive = true;       // 실계좌에서도 실행
input double LotPer1000 = 0.05;      // SAR 단타: 그날 시작 잔고 1,000$당 lot
input double MartinMaxLot = 1.0;     // SAR 단타: 넘으면 이 lot으로 깎아서 진입
input double TrendRiskPct = 2.0;     // H4 추세: 거래당 위험 (잔고 %)
input double TrendMaxLot = 5.0;      // H4 추세: 최대 lot
input double AsiaRiskPct = 1.0;      // 아시아 3마틴: 1단계 위험 (잔고 %)
input double AsiaMaxLot = 1.0;       // 아시아 3마틴: 진입 lot이 이보다 크면 진입 안 함
input int    MagicMartin = 20261017;
input int    MagicTrend = 20261018;
input int    MagicAsia = 20261019;

#include "infinox_common.mqh"
#include "mod_martin.mqh"
#include "mod_trend.mqh"
#include "mod_asia.mqh"

#define PANEL_FILE "infinox_panel.txt"
#define PFX "ifx_"
#define FONT "Malgun Gothic"

string g_names[3] = {"SAR 단타", "H4 추세", "아시아 3마틴"};
bool   g_sel[3] = {true, true, true};
bool   g_master = false;
datetime g_lastPanel = 0;

int StratMagic(int i) { return i == 0 ? MagicMartin : i == 1 ? MagicTrend : MagicAsia; }
bool IsOn(int i) { return g_master && g_sel[i]; }

#include "mod_report.mqh"

void SavePanel()
{
   for (int i = 0; i < 3; i++)
      StSet(PANEL_FILE, "sel" + IntegerToString(i), g_sel[i] ? "1" : "0");
   StSet(PANEL_FILE, "master", g_master ? "1" : "0");
}

void LoadPanel()
{
   for (int i = 0; i < 3; i++)
      g_sel[i] = StGet(PANEL_FILE, "sel" + IntegerToString(i), "1") == "1";
   g_master = StGet(PANEL_FILE, "master", "0") == "1";
}

void NotifyStart(int i)
{
   string head = StringFormat("▶️ 시작 (%s %d)\n", AccountKind(), AccountNumber());
   string body = i == 0 ? M_StartText() : i == 1 ? T_StartText() : A_StartText();
   Send(g_names[i], head + body + "\n" + AccountLine());
}

void NotifyStop(int i)
{
   Send(g_names[i], "⏸ 정지 (신규 진입만 멈춤. 열린 포지션은 서버 SL/TP와 청산 규칙으로 계속 관리)");
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

double PeriodPct(datetime start, int magic, double &pnl)
{
   pnl = 0;
   for (int i = OrdersHistoryTotal() - 1; i >= 0; i--)
      if (OrderSelect(i, SELECT_BY_POS, MODE_HISTORY) && OrderType() <= OP_SELL && OrderMagicNumber() == magic
          && OrderSymbol() == Symbol() && OrderCloseTime() >= start)
         pnl += OrderProfit() + OrderSwap() + OrderCommission();
   double b0 = BalanceAt(start);
   return b0 > 0 ? pnl / b0 * 100 : 0;
}

string ReturnText(int magic)
{
   datetime now = TimeCurrent(), d0 = DayStart(now);
   datetime w0 = d0 - ((TimeDayOfWeek(d0) + 6) % 7) * 86400;
   datetime m0 = StringToTime(StringFormat("%04d.%02d.01", TimeYear(now), TimeMonth(now)));
   double pd, pw, pm;
   double rd = PeriodPct(d0, magic, pd), rw = PeriodPct(w0, magic, pw), rm = PeriodPct(m0, magic, pm);
   return StringFormat("오늘 %+.2f%%   이번주 %+.2f%%   이번달 %+.2f%% (%+.0f$)", rd, rw, rm, pm);
}

void DrawPanel()
{
   int x0 = 10, y0 = 25;
   string bg = PFX + "bg";
   if (ObjectFind(0, bg) < 0)
   {
      ObjectCreate(0, bg, OBJ_RECTANGLE_LABEL, 0, 0, 0);
      ObjectSetInteger(0, bg, OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, bg, OBJPROP_XDISTANCE, x0);
      ObjectSetInteger(0, bg, OBJPROP_YDISTANCE, y0);
      ObjectSetInteger(0, bg, OBJPROP_XSIZE, 620);
      ObjectSetInteger(0, bg, OBJPROP_YSIZE, 178);
      ObjectSetInteger(0, bg, OBJPROP_BGCOLOR, C'24,28,36');
      ObjectSetInteger(0, bg, OBJPROP_BORDER_TYPE, BORDER_FLAT);
      ObjectSetInteger(0, bg, OBJPROP_COLOR, clrDimGray);
      ObjectSetInteger(0, bg, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, bg, OBJPROP_HIDDEN, true);
   }
   Label("title", x0 + 12, y0 + 8, StringFormat("Infinox 자동매매 · %s %d · 잔고 %.2f / 평가 %.2f %s",
                                                AccountKind(), AccountNumber(), AccountBalance(), AccountEquity(), AccountCurrency()),
         clrWhite, 10);
   for (int i = 0; i < 3; i++)
   {
      int y = y0 + 38 + i * 32;
      // 체크박스: 흰 네모 안에 Wingdings 체크(252). 이름 글자를 눌러도 같이 바뀜
      Button("sel" + IntegerToString(i), x0 + 12, y + 2, 22, 22, g_sel[i] ? CharToString(252) : " ",
             clrWhite, C'20,110,220', "Wingdings", 14);
      Label("sel" + IntegerToString(i) + "n", x0 + 42, y + 4, g_names[i], g_sel[i] ? clrWhite : clrGray);
      bool on = IsOn(i);
      Label("st" + IntegerToString(i), x0 + 160, y + 4, on ? "켜짐" : "꺼짐", on ? clrLime : clrGray);
      Label("ret" + IntegerToString(i), x0 + 206, y + 4, ReturnText(StratMagic(i)), clrSilver, 9);
   }
   Button("master", x0 + 12, y0 + 138, 220, 30, g_master ? "■ 전체 정지" : "▶ 선택한 전략 시작",
          g_master ? C'170,40,40' : C'30,130,60');
   string hint = g_master ? "켜진 동안 전략 버튼을 누르면 바로 켜지고 꺼짐" : "전략을 고르고 시작을 누르세요";
   Label("hint", x0 + 244, y0 + 146, hint, clrGray, 9);
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
   LoadPanel();
   M_Init();
   T_Init();
   A_Init();
   R_Init();
   if (g_master)
   {
      string list = "";
      for (int i = 0; i < 3; i++)
         if (g_sel[i])
            list += (list == "" ? "" : ", ") + g_names[i];
      Send("Infinox", StringFormat("▶️ EA 다시 켜짐 (%s %d): 실행 중 %s\n%s", AccountKind(), AccountNumber(),
                                       list == "" ? "없음" : list, AccountLine()));
   }
   if (!IsTradeAllowed())
      Send("Infinox", "⚠️ MT4의 '자동매매'가 꺼져 있거나 EA 설정에서 '실거래 허용'이 꺼져 있습니다. 켜야 주문이 들어갑니다.");
   DrawPanel();
   EventSetTimer(1);
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
   M_Timer(IsOn(0));
   T_Timer(IsOn(1));
   A_Timer(IsOn(2));
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
      int i = (int)StringToInteger(StringSubstr(name, 3, 1));
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
         for (int i = 0; i < 3; i++)
            if (g_sel[i])
               list += (list == "" ? "" : ", ") + g_names[i];
         if (list == "")
            MessageBox("선택한 전략이 없어요. 전략 버튼을 먼저 눌러 주세요.", "Infinox", MB_OK);
         else if (MessageBox(StringFormat("%s %d에서 시작할까요?\n\n%s", AccountKind(), AccountNumber(), list),
                             "Infinox", MB_YESNO | MB_ICONQUESTION) == IDYES)
         {
            g_master = true;
            SavePanel();
            for (int i = 0; i < 3; i++)
               if (g_sel[i])
                  NotifyStart(i);
         }
      }
      else
      {
         g_master = false;
         SavePanel();
         for (int i = 0; i < 3; i++)
            if (g_sel[i])
               NotifyStop(i);
      }
   }
   DrawPanel();
}

void OnTick() {}
