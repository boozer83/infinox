"""SAR 단타 + 틱 우세 (데모 관찰용, 2026-10-29 사용자 아이디어)

진입·청산은 demo_trader.py(SAR 단타)와 똑같고, 주문 직전에 직전 60초 동안 Bid가 오른 틱 비율을 봄
  (>= 0.6 매수 우세, <= 0.4 매도 우세, 그 사이는 우세 없음 -> 신호대로)
  --mode skip : 신호와 반대 우세면 쉬기
  --mode flip : 신호와 반대 우세면 우세 방향으로 진입. 그 거래는 반대 콩 청산 없이 손절·익절(금요일 청산)로만
백테스트 (backtests/sar_tick_dom.py, 2022~2026 실제 틱, 지금 봇 2,441건 58.6% +651oz 낙폭 166):
  skip 60초 0.6: 2,354건 59.4% +693oz 낙폭 151 / flip 60초 0.6: 2,417건 60.0% +679oz 낙폭 166
  비슷한 설정 9개 중 4~5개만 둘 다 좋아짐 (30초·120초 기준 0.55·0.52는 나쁨) -> 과최적화 의심, 그래서 데모로 관찰
데모 계좌 전용 (--live 안 받음), MT4 따라하기 안 보냄. magic skip 20261029 / flip 20261030
기록: logs/sar_dom_skip_log/, logs/sar_dom_flip_log/

사용법: python sar_dom_trader.py --mode skip
        python sar_dom_trader.py --mode flip
"""
import argparse
import sys

import demo_trader as dt

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["skip", "flip"], required=True)
args, rest = ap.parse_known_args()
if "--live" in rest:
    raise SystemExit("데모 관찰용이라 --live는 안 받음")
sys.argv = [sys.argv[0]] + rest

dt.DOM_MODE = args.mode
dt.MAGIC = 20261029 if args.mode == "skip" else 20261030
dt.BRIDGE_ID = None
dt.LOG_DIR = dt.LOG_DIR.parent / f"sar_dom_{args.mode}_log"
dt.TRADES_CSV, dt.STATE_JSON = dt.LOG_DIR / "trades.csv", dt.LOG_DIR / "state.json"
dt.RISK_JSON, dt.WINRATE_JSON = dt.LOG_DIR / "risk_state.json", dt.LOG_DIR / "winrate.json"
dt.STRATEGY = f"단타 틱우세 {'쉬기' if args.mode == 'skip' else '방향'}"
dt.STRATEGY_DESC = (f"SAR 단타와 같은 진입·청산 + 직전 {dt.DOM_SEC}초 틱 우세 (오른 틱 {dt.DOM_TH:.0%} 이상 매수 / "
                    f"{1 - dt.DOM_TH:.0%} 이하 매도 우세), 신호와 반대 우세면 "
                    + ("쉼" if args.mode == "skip" else "우세 방향으로 진입 (그 거래는 반대 콩 청산 없음)"))

if __name__ == "__main__":
    dt.main()
