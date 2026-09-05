@echo off
rem ============================================
rem  Train Center - unified start menu (ASCII)
rem  Entry point: choose training mode, then
rem  launch the service with the chosen
rem  TRAIN_MODE (same as the two launchers).
rem  The web page opens in your browser
rem  automatically once the service is ready;
rem  logs stream into this window.
rem  Close this window (X) = stop the service.
rem ============================================
setlocal
cd /d "%~dp0"
echo ============================================
echo   AI Voice Training Center - start menu
echo ============================================
echo   1 - Voice conversion training   (RVC)
echo   2 - Text-driven voice training  (GPT-SoVITS)
echo   0 - Cancel
echo ============================================
choice /c 120 /n /m "Select 1/2/0 : "
if errorlevel 3 goto :END
if errorlevel 2 goto :SET_WENZI
set "TRAIN_MODE=huan_sheng"
goto :RUN
:SET_WENZI
set "TRAIN_MODE=wen_zi"
:RUN
if not defined TRAIN_PORT set "TRAIN_PORT=8050"
if not defined TRAIN_AUTO_EXIT set "TRAIN_AUTO_EXIT=1"
set "ROOT=%~dp0"
set "PY=%ROOT%runtime\py312\python.exe"
if "%TRAIN_MODE%"=="wen_zi" (title Train Center (WenZi) port %TRAIN_PORT%) else (title Train Center (RVC) port %TRAIN_PORT%)

rem ---- refuse to start a second instance when port is already used ----
powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort %TRAIN_PORT% -State Listen -ErrorAction SilentlyContinue; if ($c) { exit 0 } else { exit 1 }" >nul 2>&1
if %errorlevel% equ 0 (
    echo Train Center is already running on port %TRAIN_PORT%.
    echo Open http://127.0.0.1:%TRAIN_PORT%/ in your browser.
    echo Close the old instance first with stop.bat.
    goto :END
)
if not exist "%PY%" (
    echo [ERROR] Portable Python not found: %PY%
    echo Please follow DEPLOY.md to place the runtime and engines first.
    goto :END
)
set "PATH=%ROOT%runtime\ffmpeg\bin;%PATH%"

echo ============================================
if "%TRAIN_MODE%"=="wen_zi" (
    echo   [WenziQudong Train Center] GPT-SoVITS TTS training
) else (
    echo   [HuanSheng Train Center] RVC voice-conversion training
)
echo   Web UI   : http://127.0.0.1:%TRAIN_PORT%/  (opens automatically when ready)
echo   Mode     : %TRAIN_MODE%
echo   Auto-exit after training: %TRAIN_AUTO_EXIT%
echo   Stop     : close this window (X) = service + GPU/RAM released
echo              (or run stop.bat without closing this window)
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
