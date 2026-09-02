@echo off
rem ============================================================
rem  Train Center - voice conversion mode (RVC / huan_sheng)
rem  Starts the training web service on port 8050.
rem  Self-contained layout: runtime\py312, runtime\ffmpeg, rvc\,
rem  gptsovits\GPT-SoVITS must already be in place (see DEPLOY.md).
rem  Stop: run stop.bat.
rem ============================================================
setlocal
set "ROOT=%~dp0"
if not defined TRAIN_MODE set "TRAIN_MODE=huan_sheng"
if not defined TRAIN_PORT set "TRAIN_PORT=8050"
if not defined TRAIN_AUTO_EXIT set "TRAIN_AUTO_EXIT=1"
set "PY=%ROOT%runtime\py312\python.exe"

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
echo   Web UI   : http://127.0.0.1:%TRAIN_PORT%/
echo   Mode     : %TRAIN_MODE%  - RVC only
echo   Auto-exit after training: %TRAIN_AUTO_EXIT%
echo   Stop     : stop.bat
echo ============================================

rem ---- open the web page in the default browser once the service is up ----
start "" /b powershell -NoProfile -Command "Start-Sleep -Seconds 3; Start-Process 'http://127.0.0.1:%TRAIN_PORT%/'" >nul 2>&1

"%PY%" "%ROOT%train_service\train_api.py"
:END
echo.
pause
