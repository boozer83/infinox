@echo off
rem Start/restart the 3 DEMO bots (demo_trader, trend_trader, asia_breakout_trader)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bots.ps1" demo
pause
