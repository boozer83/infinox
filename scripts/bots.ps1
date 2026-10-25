# Start (or restart) the 3 bots + the report bot on the demo and/or live account.
#   powershell -ExecutionPolicy Bypass -File scripts\bots.ps1 demo|live|all|status|stop-demo|stop-live
# demo: default MT5 terminal (demo account 100136645)
# live: portable terminal C:\MT5_LIVE, symbol XAUUSD#, every bot with --live
# Bots of the same set that are already running are stopped first, so nothing runs twice.
param([ValidateSet("demo", "live", "all", "status", "stop-demo", "stop-live")][string]$Which = "status")

$Dir = Split-Path $PSScriptRoot -Parent   # infinox 폴더 (이 파일은 scripts/ 안)
$Bots = @("demo_trader.py", "trend_trader.py", "asia_breakout_trader.py", "report_bot.py")
$DemoSet = @("hedge_martin_trader.py", "news_trader.py", "news_surprise_paper.py", "index_trader.py --symbol NAS100", "index_trader.py --symbol GER40", "sar_martin_trader.py", "trend_trader.py --symbol XETUSD", "asia_breakout_trader.py --symbol XAUEUR")   # extra bots on the demo account
$LiveExtra = @("asia_breakout_trader.py --symbol XAUEUR", "trend_trader.py --symbol XETUSD", "index_trader.py --symbol NAS100", "index_trader.py --symbol GER40", "news_surprise_paper.py")   # live account too (2026-10-25 user request; news_trader.py demo only, sar_martin / hedge have --live but on hold), on top of $Bots
$DemoTerminal = "C:\Program Files\Infinox MetaTrader 5 Terminal\terminal64.exe"
$LiveEnv = @{ MT5_TERMINAL_PATH = "C:\MT5_LIVE\terminal64.exe"; MT5_PORTABLE = "1"; MT5_SYMBOL = "XAUUSD#" }

function Get-Bots([bool]$Live) {
    Get-CimInstance Win32_Process -Filter "name like 'python%'" | Where-Object {
        $cl = $_.CommandLine
        $isBot = $false
        foreach ($b in $Bots + $DemoSet + $LiveExtra) { if ($cl -like "*$b*") { $isBot = $true } }
        $isBot -and (($cl -like "*--live*") -eq $Live)
    }
}

function Stop-Set([bool]$Live) {
    foreach ($p in Get-Bots $Live) {
        Write-Host "stop  PID $($p.ProcessId)  $($p.CommandLine)"
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

function Start-Set([bool]$Live) {
    Stop-Set $Live
    Start-Sleep 2
    foreach ($k in $LiveEnv.Keys) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
    if ($Live) { foreach ($k in $LiveEnv.Keys) { Set-Item "Env:$k" $LiveEnv[$k] } }
    # pin demo bots to the demo terminal; without a path mt5.initialize() may attach to the live one
    else { $env:MT5_TERMINAL_PATH = $DemoTerminal }
    $set = if ($Live) { $Bots + $LiveExtra } else { $Bots + $DemoSet }
    foreach ($b in $set) {
        $a = if ($Live) { "$b --live" } else { $b }
        $p = Start-Process python -ArgumentList $a -WorkingDirectory $Dir -WindowStyle Hidden -PassThru
        Write-Host "start PID $($p.Id)  $a"
    }
    foreach ($k in $LiveEnv.Keys) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
}

function Show-Status {
    foreach ($live in $false, $true) {
        Write-Host ($(if ($live) { "== live" } else { "== demo" }))
        foreach ($p in Get-Bots $live) { Write-Host "  PID $($p.ProcessId)  $($p.CommandLine)" }
    }
}

switch ($Which) {
    "demo" { Start-Set $false }
    "live" { Start-Set $true }
    "all" { Start-Set $false; Start-Set $true }
    "stop-demo" { Stop-Set $false }
    "stop-live" { Stop-Set $true }
}
Start-Sleep 3
Show-Status
