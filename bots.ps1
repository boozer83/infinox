# Start (or restart) the 3 bots + the report bot on the demo and/or live account.
#   powershell -ExecutionPolicy Bypass -File bots.ps1 demo|live|all|status|stop-demo|stop-live
# demo: default MT5 terminal (demo account 100136645)
# live: portable terminal C:\MT5_LIVE, symbol XAUUSD#, every bot with --live
# Bots of the same set that are already running are stopped first, so nothing runs twice.
param([ValidateSet("demo", "live", "all", "status", "stop-demo", "stop-live")][string]$Which = "status")

$Dir = $PSScriptRoot
$Bots = @("demo_trader.py", "trend_trader.py", "asia_breakout_trader.py", "report_bot.py")
$DemoOnly = @("hedge_martin_trader.py", "news_trader.py", "sar_martin_trader.py")   # demo account only (no --live)
$LiveEnv = @{ MT5_TERMINAL_PATH = "C:\MT5_LIVE\terminal64.exe"; MT5_PORTABLE = "1"; MT5_SYMBOL = "XAUUSD#" }

function Get-Bots([bool]$Live) {
    Get-CimInstance Win32_Process -Filter "name like 'python%'" | Where-Object {
        $cl = $_.CommandLine
        $isBot = $false
        foreach ($b in $Bots + $DemoOnly) { if ($cl -like "*$b*") { $isBot = $true } }
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
    $set = if ($Live) { $Bots } else { $Bots + $DemoOnly }
    foreach ($b in $set) {
        $a = if ($Live) { "$b --live" } else { $b }
        $p = Start-Process python -ArgumentList $a -WorkingDirectory $Dir -WindowStyle Minimized -PassThru
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
