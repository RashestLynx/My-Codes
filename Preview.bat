@echo off
rem Like Upscale.bat (drag movies onto it, or double-click for the Movies folder), but only the
rem first 60 seconds of each movie: a quick look at the faces and the picture before a full run.
call "%~dp0Upscale.bat" --preview %*
