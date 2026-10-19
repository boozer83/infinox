@echo off
rem Start/restart the 3 LIVE bots (C:\MT5_LIVE terminal, XAUUSD#, --live)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bots.ps1" live
pause
