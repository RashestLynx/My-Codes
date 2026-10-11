# Train the upscaler on real DVD / Blu-ray pairs: one file, one command

`upscale_training.py` does everything: it makes aligned frame pairs from a movie you have on both discs,
fine-tunes Real-ESRGAN x2plus on them, and writes the model your upscaler loads.

Needs Python with `numpy opencv-python torch torchvision` (torch with CUDA), and `ffmpeg`/`ffprobe`
on the PATH. Put `RealESRGAN_x2plus.pth` next to `upscale_training.py`
(https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth).
The converter to the upscaler's format is built in, so `upscale_training.py` works on its own. Keep it in
the `training` folder next to `dvd_upscale.py`: it asks `dvd_upscale.py --analyze` what each movie is (anime,
cgi, live or vhs) and prepares the DVD frames with exactly the filters the upscaler uses for it.

```
python upscale_training.py --dvd "movie_dvd.mkv" --bluray "movie_bd.mkv"
```

1. **pairs** (a few hours): 3000 aligned DVD / Blu-ray frame pairs into `movie_dvd_training\pairs`
   (a folder next to the DVD file; `--work` puts it elsewhere)
2. **train** (about 2 hours on an RTX 3060): into `movie_dvd_training\run`
3. **export**: `ai-<kind>-x2.param` / `.bin` (`ai-anime-x2`, `ai-live-x2`, `ai-cgi-x2`, `ai-vhs-x2`) into the
   upscaler's `models` folder (found next to `realesrgan-ncnn-vulkan`; `--models` to point elsewhere). A model
   already there is replaced only when the new one scores higher on the held-out frames.

Then: `python dvd_upscale.py <movie> --trained` (it picks `ai-<kind>-x2` for the kind it detects).

## Several movies, one model per kind
`python upscale_training.py --all "D:\Movies"` finds every `Name_dvd.mkv` + `Name_bd.mkv` pair in that folder
and its `anime` / `live` / `cgi` / `vhs` folders, makes the pairs of each, then trains one model per kind on
all its movies. `--queue` does the same from `training_queue.txt` (one `--dvd ... --bluray ...` per line).
Movies added later are learned on top of the model in use. `--list` shows the plan and stops.

## Mixing a trained model with x2plus
Your models are x2plus with its weights fine-tuned, so the two can be mixed (Real-ESRGAN's network
interpolation): a dial between the original's look (safe on any movie) and yours (closer to your Blu-rays).

```
python upscale_training.py --mix 0.5 0.7 0.9 --type live --work "D:\Movies\ai-live-x2_training"
```

This writes `ai-live-x2-mix50`, `-mix70` and `-mix90` next to `ai-live-x2` in the models folder (as fast as any
model) and scores each one, along with x2plus and the trained model, on held-out frames. The training's own
movies favour the full model, so judge on a movie that wasn't in the training: make its pairs
(`--stages pairs`) and pass `--pairs-from` that pairs folder. Then: `python dvd_upscale.py <movie> --model
ai-live-x2-mix70 --scale 2`. `--mix-from FILE.pth` mixes another trained model; only `--arch rrdb` models mix.

## Stopping and carrying on
Ctrl+C any time, then run **the same command again**. Finished steps are skipped, the pairs and the
training both carry on from where they stopped. `--stages train,export` or `--stages export` runs
only some steps. `--fresh` starts the pairs over. A bigger `--iters` trains longer.

## Watching progress
PowerShell shows a live status with an ETA and the clock time it should finish. A page for a browser
or phone starts by itself (port 8643); the address is printed. `--web 0` turns it off. Both show the GPU
(watts, load, temperature, clocks, memory, fan) and the processor (load, busiest core, clock, RAM), now and
as peak and average over the training, plus a one-line verdict on what limits the speed. A summary of the
peaks and averages is printed when the training ends. CPU watts can't be read from Python: use HWiNFO.

## What to check
- The pairs step prints the offset between the discs along the movie (it can change by a minute or
  more); look at a few `pairs\lr` / `pairs\hr` pictures with the same name: same moment, `hr` sharper.
- The train step prints, on frames it never trains on, the PSNR of plain bicubic, of the starting
  model and of yours. Yours should climb past both. If it doesn't, the pairs are the problem.
- Out of memory: `--batch 4`, `--patch 64`, or `--checkpoint` (the batch is also halved by itself and
  training carries on). PAL disc: `--speed 1.0427`.
- `--perceptual 0.5 --gan 0.05` adds texture but can invent detail. Do the default first. `--gan` uses
  Real-ESRGAN's U-Net discriminator (`--disc patch` for the old one).
- `--arch compact` trains a small, fast network that upscales several times quicker, but it learns from zero:
  use `--iters 100000` or more and `--count 6000` or more pairs. The default (`--arch rrdb`) has the best quality.
- `--batch auto` picks the biggest batch that fits in the graphics card's memory (`--gpu-memory 0.9` leaves room
  for other programs). Use it when the GPU shows under 90% busy.
- Several NVIDIA GPUs: `--gpus all` (the default) splits every step across them, `--gpus 0,1` picks which.
  `--gpu-test` times each card alone and together and says what to use.
- `--lpips` (needs `pip install lpips`) adds LPIPS to the held-out scores next to PSNR and SSIM.
- One film teaches its own grain and grade; test on a movie that wasn't in the training.
  Try `--ai-blend 1.0` with the new model.
