@echo off
rem Start/restart all 6 bots (3 demo + 3 live)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bots.ps1" all
pause
