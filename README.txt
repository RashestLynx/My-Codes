DVD Upscaler - everything in one folder
=======================================

Upscales DVD and VHS rips to 1080p with Real-ESRGAN, then restores faces with GFPGAN, all
automatically: each piece of the movie is upscaled, its faces are restored, and it is encoded,
one after another, until the whole movie is done.

Setting up (once)
-----------------
1. Put these files together in one folder, e.g.  C:\DVD Upscaler :
     dvd_upscale.py, setup.bat, setup_windows.ps1, Upscale.bat, Preview.bat, README.txt
   and realesrgan-x2plus.zip (the live-action model), as it is: setup unpacks it.
2. Double-click setup.bat. It downloads everything else into this folder (about 1 GB):
     python\        a private Python, only used by the upscaler (nothing is installed on the PC)
     ffmpeg.exe, ffprobe.exe, realesrgan-ncnn-vulkan.exe, models\, face_models\
   At the end it checks that the face restoration runs on the graphics card.
   If Windows says "Windows protected your PC": More info -> Run anyway.

Upscaling
---------
- Put your movies (.mkv from MakeMKV, .mp4, .mpg, .avi ...) in the Movies folder and
  double-click Upscale.bat, or
- drag one or more movies, or a folder of movies, onto Upscale.bat.
Finished movies go into a "1080p Upscale" folder next to the originals: Movie.mkv becomes
"1080p Upscale\Movie 1080p.mkv". Close the window (or Ctrl+C) to stop; starting it again carries
on where it stopped.

Preview.bat works the same way but does only the first 60 seconds of each movie: a quick look
at the result before a run that takes hours.

Options
-------
Open Upscale.bat in Notepad and change the line  set "OPTIONS=--faces" , for example:
  --faces 0.8        stronger face restoration (default 0.6)
  --stabilize        steady a shaky camcorder video
  --hevc             smaller files (HEVC) for newer TVs and players
Remove --faces to upscale without face restoration. All options:
  python\python.exe dvd_upscale.py --help

Notes
-----
- Faces are restored only in live action and home videos; anime, cartoons and 3D animation are
  recognised and left alone.
- A movie started without --faces can't be finished with it (or the other way round): delete its
  "<movie>_work" folder to start it again.
- Move the whole folder anywhere, it keeps working. Run setup.bat again after moving it, or to
  repair a missing or damaged file.
