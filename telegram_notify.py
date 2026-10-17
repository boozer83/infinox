"""텔레그램 알림

설정 (둘 중 하나):
- telegram.json 파일: {"token": "봇 토큰", "chat_id": 123456789}   (git에 올리지 않음)
- 환경변수: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID (파일보다 우선)

chat_id 찾기: 텔레그램에서 봇에게 아무 메시지(/start)를 보낸 뒤
  python telegram_notify.py --setup     # chat_id를 찾아 telegram.json에 저장
  python telegram_notify.py --test      # 테스트 메시지 전송

보내기는 별도 스레드에서 처리 -> 네트워크가 느리거나 끊겨도 매매 루프는 멈추지 않음.
"""
import argparse
import json
import logging
import os
import queue
import threading
import urllib.parse
import urllib.request
from pathlib import Path

CONFIG = Path(__file__).resolve().parent / "telegram.json"
log = logging.getLogger("demo")


def load_config():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {}
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or cfg.get("token")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or cfg.get("chat_id")
    return token, chat_id


def api(token, method, params=None, timeout=15):
    data = urllib.parse.urlencode(params).encode() if params else None
    with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/{method}", data=data, timeout=timeout) as r:
        return json.load(r)


class Notifier:
    """설정이 없으면 아무것도 보내지 않는 알림기"""

    def __init__(self):
        self.token, self.chat_id = load_config()
        self.enabled = bool(self.token and self.chat_id)
        self.q = queue.Queue()
        if self.enabled:
            threading.Thread(target=self._worker, daemon=True).start()

    def send(self, text):
        if self.enabled:
            self.q.put(text)

    def flush(self, timeout=10):
        """종료 전에 남은 메시지 전송 (최대 timeout초)"""
        if not self.enabled:
            return
        done = threading.Event()
        self.q.put(done)
        done.wait(timeout)

    def _worker(self):
        while True:
            item = self.q.get()
            if isinstance(item, threading.Event):
                item.set()
                continue
            for attempt in range(3):
                try:
                    api(self.token, "sendMessage", {"chat_id": self.chat_id, "text": item})
                    break
                except Exception as e:
                    if attempt == 2:
                        log.warning(f"텔레그램 전송 실패: {e}")


def setup():
    token, _ = load_config()
    if not token:
        raise SystemExit("토큰이 없습니다. telegram.json에 \"token\"을 넣거나 TELEGRAM_BOT_TOKEN을 설정하세요.")
    me = api(token, "getMe")["result"]
    chats = {}
    for u in api(token, "getUpdates")["result"]:
        m = u.get("message") or u.get("my_chat_member") or {}
        c = m.get("chat")
        if c:
            chats[c["id"]] = c.get("username") or c.get("title") or c.get("first_name")
    if not chats:
        raise SystemExit(f"@{me['username']} 봇에게 텔레그램에서 /start 를 먼저 보낸 뒤 다시 실행하세요.")
    chat_id = list(chats)[-1]
    cfg = json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {"token": token}
    cfg["chat_id"] = chat_id
    CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"chat_id {chat_id} ({chats[chat_id]}) 저장 완료 -> {CONFIG.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true", help="chat_id를 찾아 telegram.json에 저장")
    ap.add_argument("--test", action="store_true", help="테스트 메시지 전송")
    a = ap.parse_args()
    if a.setup:
        setup()
    if a.test:
        token, chat_id = load_config()
        if not (token and chat_id):
            raise SystemExit("token/chat_id 설정이 없습니다. 먼저 --setup을 실행하세요.")
        api(token, "sendMessage", {"chat_id": chat_id, "text": "✅ infinox 알림 테스트"})
        print("전송 완료")
    if not (a.setup or a.test):
        ap.print_help()


if __name__ == "__main__":
    main()
