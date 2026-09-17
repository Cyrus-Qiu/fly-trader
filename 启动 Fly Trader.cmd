@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\fly-trader.exe" (
    echo [ERROR] .venv\Scripts\fly-trader.exe was not found.
    echo Please create the virtual environment and install the project first.
    pause
    exit /b 1
)

echo Starting Fly Trader...
echo Dashboard: http://127.0.0.1:8787/
echo Press Ctrl+C to stop.
echo.

".venv\Scripts\fly-trader.exe" --source both %*
set "FLY_TRADER_EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%FLY_TRADER_EXIT_CODE%"=="0" (
    echo Fly Trader exited with code %FLY_TRADER_EXIT_CODE%.
) else (
    echo Fly Trader has stopped.
)
pause
exit /b %FLY_TRADER_EXIT_CODE%
