@echo off
rem ============================================
rem  Train Center - stop entry (no pause)
rem  Usage: stop.bat            (port 8050)
rem         stop.bat 8060       (custom port)
rem ============================================
setlocal
set "TRAIN_PORT=8050"
if not "%1"=="" set "TRAIN_PORT=%1"
echo Stopping Train Center (port %TRAIN_PORT%) ...
powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort %TRAIN_PORT% -State Listen -ErrorAction SilentlyContinue; if ($c) { $c | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }; Write-Output 'Train Center stopped' } else { Write-Output 'Train Center is not running' }"
exit /b 0
