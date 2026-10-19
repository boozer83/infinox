@echo off
rem Asia breakout bot on the DEMO account (default terminal). Keep the minimized window open.
cd /d "%~dp0"
set MT5_TERMINAL_PATH=
set MT5_PORTABLE=
set MT5_SYMBOL=
start "asia_breakout demo" /min cmd /k python asia_breakout_trader.py
