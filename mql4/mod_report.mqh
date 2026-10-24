// 계좌 리포트 모듈 (report_bot.py와 같은 방식). Infinox_Bots.mq4가 include
// 한국시간 월~금 21:00에 메시지 하나로 전략별 손익:
// - 매일: 일일 = 지난 리포트(전 평일 21:00) 이후
// - 금요일: 주간 추가 = 지난 금요일 21:00 이후
// - 그 달 마지막 평일: 월간 추가 = 지난달 마지막 평일 21:00 이후
// 전략 on/off와 상관없이 보냄. 보낸 날짜는 MQL4\Files\infinox_report.txt (껐다 켜도 중복 전송 안 함)
// 한국시간은 PC 시계(TimeGMT) + 9시간, 서버시간은 TimeCurrent - TimeGMT 차이로 바꿈

#define R_FILE "infinox_report.txt"
#define R_HOUR 21

string R_WEEKDAYS = "일월화수목금토";
string r_last = "";                  // 마지막으로 보낸 한국 날짜

datetime R_KstNow() { return TimeGMT() + 9 * 3600; }
bool     R_Weekend(datetime d) { int w = TimeDayOfWeek(d); return w == 0 || w == 6; }

datetime R_PrevWeekday(datetime d0)
{
   d0 -= 86400;
   while (R_Weekend(d0))
      d0 -= 86400;
   return d0;
}

datetime R_LastWeekdayOfMonth(datetime d0)
{
   int y = TimeYear(d0), m = TimeMonth(d0) + 1;
   if (m > 12)
   {
      m = 1;
      y++;
   }
   datetime last = StringToTime(StringFormat("%04d.%02d.01", y, m)) - 86400;
   while (R_Weekend(last))
      last -= 86400;
   return last;
}

// 한국시간 start ~ end 전략별 손익 블록
string R_Section(string title, datetime start, datetime end, int off)
{
   datetime s = start - 9 * 3600 + off, e = end - 9 * 3600 + off;
   double b0 = BalanceAt(s), b1 = BalanceAt(e), dep = Deposits(s, e), base = b0 + dep;
   string out = StringFormat("%s (%s ~ %s)", title, TimeToString(start, TIME_DATE | TIME_MINUTES), TimeToString(end, TIME_DATE | TIME_MINUTES));
   double total = 0;
   for (int k = 0; k < N_STRAT; k++)
   {
      int magic = StratMagic(k), n = 0, w = 0;
      string sym = StratSymbol(k);
      double pnl = 0;
      for (int i = OrdersHistoryTotal() - 1; i >= 0; i--)
         if (OrderSelect(i, SELECT_BY_POS, MODE_HISTORY) && OrderType() <= OP_SELL && OrderMagicNumber() == magic
             && OrderSymbol() == sym && OrderCloseTime() >= s && OrderCloseTime() < e)
         {
            double p = OrderProfit() + OrderSwap() + OrderCommission();
            pnl += p;
            n++;
            if (p > 0)
               w++;
         }
      total += pnl;
      string trades = n > 0 ? StringFormat("%d건 (익절 %d / 손절 %d, 승률 %.0f%%)", n, w, n - w, w * 100.0 / n) : "거래 없음";
      out += StringFormat("\n• %s: %+.2f%% (%+.2f$) · %s", g_names[k], base > 0 ? pnl / base * 100 : 0.0, pnl, trades);
   }
   out += StringFormat("\n합계 %+.2f%% (%+.2f$)\n잔고 %.2f → %.2f", base > 0 ? total / base * 100 : 0.0, total, b0, b1);
   if (dep > 0)
      out += StringFormat(" (입금 +%.2f$ 포함, 수익률은 입금액까지 기준)", dep);
   return out;
}

string R_Text(datetime today, datetime now)
{
   int off = (int)MathRound((TimeCurrent() - TimeGMT()) / 3600.0) * 3600;
   string text = StringFormat("📊 %s %d 리포트 %02d/%02d(%s)\n\n", AccountKind(), AccountNumber(), TimeMonth(today), TimeDay(today),
                              StringSubstr(R_WEEKDAYS, TimeDayOfWeek(today), 1));
   text += R_Section("일일", R_PrevWeekday(today) + R_HOUR * 3600, now, off);
   if (TimeDayOfWeek(today) == 5)
      text += "\n\n" + R_Section("📅 주간", today - 7 * 86400 + R_HOUR * 3600, now, off);
   if (today == R_LastWeekdayOfMonth(today))
   {
      datetime prevEnd = R_LastWeekdayOfMonth(StringToTime(StringFormat("%04d.%02d.01", TimeYear(today), TimeMonth(today))) - 86400);
      text += "\n\n" + R_Section(StringFormat("🗓 월간 %04d-%02d", TimeYear(today), TimeMonth(today)), prevEnd + R_HOUR * 3600, now, off);
   }
   double floating = AccountEquity() - AccountBalance();
   if (MathAbs(floating) >= 0.01)
      text += StringFormat("\n\n열린 포지션 평가손익 %+.2f$ (위 손익에는 미포함)", floating);
   return text;
}

void R_Init()
{
   r_last = StGet(R_FILE, "last", "");
   if (r_last != "")
      return;
   // 첫 실행: 오늘 21:00이 이미 지났으면 오늘 것은 보낸 것으로 침
   datetime kst = R_KstNow(), d0 = kst - kst % 86400;
   r_last = TimeToString(TimeHour(kst) >= R_HOUR ? d0 : d0 - 86400, TIME_DATE);
   StSet(R_FILE, "last", r_last);
}

void R_Timer()
{
   datetime kst = R_KstNow(), today = kst - kst % 86400;
   if (R_Weekend(today) || TimeHour(kst) < R_HOUR)
      return;
   string d = TimeToString(today, TIME_DATE);
   if (r_last >= d)
      return;
   r_last = d;
   StSet(R_FILE, "last", d);
   Send("리포트", R_Text(today, kst - kst % 60));
}
