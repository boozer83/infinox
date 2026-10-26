"""MT4 통로(common/signal_bridge.py) 동시 쓰기 시험 (2026-10-26 20:30 SAR 줄이 2마틴 줄에 덮인 사고 재현)
python backtests/bridge_race_test.py  -> 임시 폴더에서: 잠금 없이 / 잠금으로 6프로세스 x 300줄, 남의 조각 뒤 쓰기, 만료 진입"""
import multiprocessing as mp
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import signal_bridge as sb  # noqa: E402

N_PROC, N_LINE = 6, 300


def use(folder):
    sb.BRIDGE_FILE = Path(folder) / "bridge.csv"
    sb.LOCK_FILE = sb.BRIDGE_FILE.with_suffix(".lock")


def worker(folder, k, start, locked):
    use(folder)
    while time.time() < start:
        pass
    for i in range(N_LINE):
        ref = k * 100000 + i
        if locked:
            sb.send(f"S{k}", "open", 1, 7.5, 3.0, 0, 0.03, ref, ts_ms=int(time.time() * 1000))
            if sb._pending:
                sys.exit(2)
        else:
            with open(sb.BRIDGE_FILE, "a", encoding="ascii", newline="") as f:
                f.write(f"0,S{k},open,1,7.5,3.0,0,0.03,{ref},E\r\n")


def check(path):
    lines = path.read_bytes().split(b"\r\n")
    good = [l.split(b",") for l in lines if l.endswith(b",E")]
    refs = [int(f[8]) for f in good if len(f) == 10]
    want = {k * 100000 + i for k in range(N_PROC) for i in range(N_LINE)}
    return len(good), len(set(refs)), len(refs) - len(set(refs)), set(refs) == want


def race(locked):
    with tempfile.TemporaryDirectory() as d:
        start = time.time() + 1.5
        ps = [mp.Process(target=worker, args=(d, k, start, locked)) for k in range(N_PROC)]
        [p.start() for p in ps]
        [p.join() for p in ps]
        n, uniq, dup, full = check(Path(d) / "bridge.csv")
        codes = [p.exitcode for p in ps]
        print(f"{'잠금' if locked else '잠금 없음'}: 정상 줄 {n}/{N_PROC * N_LINE}, 고유 티켓 {uniq}, 중복 {dup}, 전부 있음 {full}, 종료 코드 {codes}")
        return full and dup == 0 and codes == [0] * N_PROC


def fragment():
    """다른 프로세스가 반쯤 쓴 줄을 남겼어도 다음 줄은 따로 된 줄로 들어가야 함"""
    with tempfile.TemporaryDirectory() as d:
        use(d)
        sb.BRIDGE_FILE.write_bytes(b"1,SAR,open,1,7.5")
        sb._pending.clear()
        sb.send("SARM", "open", 1, 7.5, 3.0, 0, 0.03, 42)
        lines = sb.BRIDGE_FILE.read_bytes().split(b"\r\n")
        ok = lines[0] == b"1,SAR,open,1,7.5" and lines[1].endswith(b",42,E") and lines[2] == b""
        print(f"남의 조각 뒤 쓰기: {lines} -> {'통과' if ok else '실패'}")
        return ok


def expired():
    """4초 지난 진입 신호는 쓰지 않음"""
    with tempfile.TemporaryDirectory() as d:
        use(d)
        sb._pending.clear()
        sb.send("SAR", "open", 1, 7.5, 3.0, 0, 0.03, 7, ts_ms=int(time.time() * 1000) - 5000)
        ok = not sb.BRIDGE_FILE.exists() and not sb._pending
        print(f"만료 진입 버림: {'통과' if ok else '실패'}")
        return ok


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    race(False)
    results = [race(True), fragment(), expired()]
    print("전체", "통과" if all(results) else "실패")
    sys.exit(0 if all(results) else 1)
