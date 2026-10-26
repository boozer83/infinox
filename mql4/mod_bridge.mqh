// 따라하기 통로 읽기 (mod_follow.mqh의 CFollow 전략들 공용). g_s / N_STRAT / IsOn을 써서 Infinox_Bots.mq4가 그 뒤에 include

//---------------------------------------------------------------- 통로 읽기 (모든 따라하기 전략 공용)
// 읽은 위치(바이트)를 저장해 두고 새로 붙은 줄만 처리. 처음 켤 때는 지금까지 쌓인 줄을 건너뜀 (지난 신호로 들어가지 않게)
long g_bridgePos = -1;
bool g_bridgeWarned = false;

void BridgeInit()
{
   string v = StGet(F_STATE, "pos", "");
   if (v != "")
   {
      g_bridgePos = StringToInteger(v);
      return;
   }
   int h = FileOpen(F_FILE, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON | FILE_SHARE_READ | FILE_SHARE_WRITE);
   g_bridgePos = 0;
   if (h != INVALID_HANDLE)
   {
      g_bridgePos = (long)FileSize(h);
      FileClose(h);
   }
   StSet(F_STATE, "pos", IntegerToString(g_bridgePos));
}

// 한 줄 처리: 전략 이름이 같은 따라하기 전략에 넘김. false = 아직 처리 못 함 (다음에 같은 줄부터 다시)
bool BridgeLine(string &f[])
{
   string sid = f[1];
   for (int k = 0; k < N_STRAT; k++)
      if (g_s[k].BridgeId() == sid)
         return g_s[k].OnBridge(IsOn(k), f[2], (int)StringToInteger(f[3]), StringToDouble(f[4]), StringToDouble(f[5]),
                                StringToDouble(f[6]), StringToDouble(f[7]), StringToInteger(f[0]), f[8]);
   return true;                                     // 모르는 전략 이름: 건너뜀
}

void BridgeRead()
{
   if (g_bridgePos < 0)
      BridgeInit();
   int h = FileOpen(F_FILE, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON | FILE_SHARE_READ | FILE_SHARE_WRITE);
   if (h == INVALID_HANDLE)
   {
      if (!g_bridgeWarned)
         Print("[통로] 신호 파일 없음 (Common\\Files\\", F_FILE, ") -> 파이썬 데모 봇이 신호를 보내면 생김");
      g_bridgeWarned = true;
      return;
   }
   long size = (long)FileSize(h);
   if (size < g_bridgePos)
      g_bridgePos = 0;                               // 파일을 새로 만든 경우
   if (size == g_bridgePos)
   {
      FileClose(h);
      return;
   }
   FileSeek(h, g_bridgePos, SEEK_SET);
   long start = g_bridgePos;
   while (!FileIsEnding(h))
   {
      string line = StringTrimRight(FileReadString(h));
      long at = (long)FileTell(h);
      if (line == "")
      {
         g_bridgePos = at;
         continue;
      }
      // 줄 끝 표시 E가 있는 완성된 줄만 (파이썬이 아직 쓰는 중인 마지막 줄은 다음에 처음부터 다시)
      string f[];
      if (StringSplit(line, ',', f) < 10 || f[9] != "E")
      {
         if (at >= size)
            break;
         Print("[통로] 잘못된 줄 건너뜀: ", line);
         g_bridgePos = at;
         continue;
      }
      if (!BridgeLine(f))
         break;                                     // 처리 못 한 줄: 위치를 안 넘기고 다음에 다시
      g_bridgePos = at;
   }
   FileClose(h);
   if (g_bridgePos != start)
      StSet(F_STATE, "pos", IntegerToString(g_bridgePos));
}
