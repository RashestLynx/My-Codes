@echo off
setlocal EnableExtensions
rem Upscales movies to 1080p, with face restoration. Either:
rem   - drag movies (or a folder of movies) onto this file, or
rem   - put movies in the Movies folder next to it and double-click it.
rem Finished movies go into a "1080p Upscale" folder next to the originals. It can be stopped
rem (close the window or Ctrl+C) and picks up where it left off when started again.
rem
rem Options for every movie: --faces restores faces (live action and home video; anime and 3D
rem animation are left alone). Add more after it, e.g.  --faces 0.8 --stabilize
set "OPTIONS=--faces"

set "HERE=%~dp0"
set "PY=%HERE%python\python.exe"
if not exist "%PY%" (
    echo The upscaler isn't set up yet: double-click setup.bat first.
    pause
    exit /b 1
)
rem Preview.bat: a 60-second test of each movie instead of the whole thing
if /i "%~1"=="--preview" (
    set "OPTIONS=%OPTIONS% --test 60"
    shift
)
if "%~1"=="" goto movies_folder

:next
if "%~1"=="" goto done
if exist "%~1\*" (
    rem a folder: every movie in it
    "%PY%" "%HERE%dvd_upscale.py" --all "%~f1." %OPTIONS%
) else (
    rem one movie: the result goes into "1080p Upscale" next to it
    pushd "%~dp1"
    "%PY%" "%HERE%dvd_upscale.py" "%~f1" %OPTIONS%
    popd
)
shift
goto next

:movies_folder
if not exist "%HERE%Movies\" mkdir "%HERE%Movies"
"%PY%" "%HERE%dvd_upscale.py" --all "%HERE%Movies" %OPTIONS%

:done
echo.
pause
