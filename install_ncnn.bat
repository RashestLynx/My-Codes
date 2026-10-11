@echo off
setlocal EnableExtensions
rem Installs the current ncnn GPU engine that dvd_upscale.py uses for live action and CGI
rem (it stops the "vkQueueSubmit failed -4" GPU resets of realesrgan-ncnn-vulkan's own 2022
rem engine on current NVIDIA drivers). Double-click it once; it needs the internet.
rem --no-deps: ncnn itself needs only numpy (its other listed packages include opencv-python,
rem which can clash with the opencv-python-headless that face restoration uses).
cd /d "%~dp0"
set "PY=python"
if exist "%~dp0python\python.exe" set "PY=%~dp0python\python.exe"
echo Installing ncnn 1.0.20260526 and numpy for:
"%PY%" -c "import sys; print('  ' + sys.executable)"
if errorlevel 1 (
    echo.
    echo Python wasn't found. Install it from python.org ^(tick "Add python.exe to PATH"^),
    echo or run setup.bat.
    pause
    exit /b 1
)
"%PY%" -m pip install --no-deps ncnn==1.0.20260526 numpy
echo.
"%PY%" -c "import ncnn, numpy; n = ncnn.get_gpu_count(); print('ncnn', ncnn.__version__, 'works. GPUs:', ', '.join(ncnn.get_gpu_info(i).device_name() for i in range(n)) or 'none found')"
if errorlevel 1 (
    echo.
    echo The install didn't work: see the messages above.
) else (
    echo.
    echo Done. dvd_upscale.py now runs live action and CGI on it ^(its "Upscaler:" line says
    echo "run by the current ncnn"^).
)
pause
