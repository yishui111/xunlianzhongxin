@echo off
rem ============================================================
rem  Train Center - voice conversion mode (RVC / huan_sheng)
rem  Double-click to start: the web page opens in your browser
rem  automatically (as soon as the service is ready) and all
rem  service / training logs stream into this window.
rem  Close this window (X) = stop the service and free GPU/RAM.
rem  Self-contained layout: runtime\py312, runtime\ffmpeg, rvc\,
rem  gptsovits\GPT-SoVITS must already be in place (see DEPLOY.md).
rem ============================================================
setlocal
set "ROOT=%~dp0"
if not defined TRAIN_MODE set "TRAIN_MODE=huan_sheng"
if not defined TRAIN_PORT set "TRAIN_PORT=8050"
if not defined TRAIN_AUTO_EXIT set "TRAIN_AUTO_EXIT=1"
set "PY=%ROOT%runtime\py312\python.exe"
title Train Center (RVC) port %TRAIN_PORT% - close this window to stop

rem ---- refuse to start a second instance when port is already used ----
powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort %TRAIN_PORT% -State Listen -ErrorAction SilentlyContinue; if ($c) { exit 0 } else { exit 1 }" >nul 2>&1
if %errorlevel% equ 0 (
    echo Train Center is already running on port %TRAIN_PORT%.
    echo Open http://127.0.0.1:%TRAIN_PORT%/ in your browser.
    echo If the page does not respond, close the old instance first with stop.bat.
    goto :END
)

rem ---- portable python present? ----
if not exist "%PY%" (
    echo [ERROR] Portable Python not found: %PY%
    echo Please follow DEPLOY.md to place the runtime and engines first.
    goto :END
)

rem ---- put bundled ffmpeg on PATH ----
set "PATH=%ROOT%runtime\ffmpeg\bin;%PATH%"

echo ============================================
echo   [HuanSheng Train Center] RVC voice-conversion training
echo   Web UI : http://127.0.0.1:%TRAIN_PORT%/  (opens automatically when ready)
echo   Mode   : %TRAIN_MODE%  - RVC only
echo   Stop   : close this window (X) = service + GPU/RAM released
echo            (or run stop.bat without closing this window)
echo   Logs   : service + training logs stream below in real time
echo ============================================

rem ---- open the web page once the health endpoint responds (max ~3 min wait) ----
start "" /b powershell -NoProfile -Command "$u='http://127.0.0.1:%TRAIN_PORT%/api/health'; $ok=$false; for($i=0;$i -lt 120;$i++){try{Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 1 | Out-Null; $ok=$true; break}catch{Start-Sleep -Milliseconds 500}}; if($ok){Start-Process 'http://127.0.0.1:%TRAIN_PORT%/'}" >nul 2>&1

"%PY%" "%ROOT%train_service\train_api.py"

echo.
echo Service stopped (window closed, training finished, or auto-exit).
echo You can close this window now.
:END
echo.
pause
