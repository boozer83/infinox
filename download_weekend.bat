@echo off
rem Weekend tick download for other symbols (scheduled task InfinoxTickDownload)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
python -m backtests.download_symbol XAGUSD BTCUSD XPTUSD XETUSD US30 NAS100 SPX500 GER40 USOUSD EURUSD GBPUSD USDJPY GBPJPY EURJPY AUDUSD JPN225 UK100 UKOUSD XAUEUR XPDUSD > data\download_log.txt 2>&1
