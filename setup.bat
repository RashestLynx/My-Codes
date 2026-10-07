@echo off
rem Double-click this once: it fetches everything the upscaler needs into this folder (see
rem setup_windows.ps1). Run it again any time; what is already here is kept.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1"
echo.
pause
