@echo off
rem ============================================================
rem  Train Center - stop (kills the process listening on port 8050)
rem  Also honored: environment variable TRAIN_PORT for custom port.
rem ============================================================
setlocal
if not defined TRAIN_PORT set "TRAIN_PORT=8050"
echo Stopping Train Center (port %TRAIN_PORT%) ...
powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort %TRAIN_PORT% -State Listen -ErrorAction SilentlyContinue; if ($c) { $c | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }; Write-Output 'Train Center stopped' } else { Write-Output 'Train Center is not running' }"
echo.
pause
