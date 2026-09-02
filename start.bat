@echo off
rem ============================================
rem  Train Center - unified start menu (ASCII)
rem  Entry point: choose training mode, then
rem  launch the service with the chosen
rem  TRAIN_MODE (same as the two launchers).
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
echo   Web UI   : http://127.0.0.1:%TRAIN_PORT%/
echo   Mode     : %TRAIN_MODE%
echo   Auto-exit after training: %TRAIN_AUTO_EXIT%
echo ============================================

start "" /b powershell -NoProfile -Command "Start-Sleep -Seconds 3; Start-Process 'http://127.0.0.1:%TRAIN_PORT%/'" >nul 2>&1
"%PY%" "%ROOT%train_service\train_api.py"
:END
echo.
pause
