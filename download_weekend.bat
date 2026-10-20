@echo off
rem Weekend tick download for other symbols (scheduled task InfinoxTickDownload)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
python -m backtests.download_symbol BTCUSD XAGUSD > data\download_log.txt 2>&1
