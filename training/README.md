# Train the upscaler on real DVD / Blu-ray pairs: one file, one command

`upscale_training.py` does everything: it makes aligned frame pairs from a movie you have on both discs,
fine-tunes Real-ESRGAN x2plus on them, and writes the model your upscaler loads.

Needs Python with `numpy opencv-python torch torchvision` (torch with CUDA), and `ffmpeg`/`ffprobe`
on the PATH. Put `RealESRGAN_x2plus.pth` next to `upscale_training.py`
(https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth).
The converter to the upscaler's format is built in, so `upscale_training.py` works on its own.

```
python upscale_training.py --dvd "movie_dvd.mkv" --bluray "movie_bd.mkv"
```

1. **pairs** (a few hours): 3000 aligned DVD / Blu-ray frame pairs into `upscale_training_work\pairs`
2. **train** (about 2 hours on an RTX 3060): into `upscale_training_work\run`
3. **export**: `upscale-training-x2.param` / `.bin` into the upscaler's `models` folder (found next to
   `realesrgan-ncnn-vulkan`; `--models` to point elsewhere)

Then: `python dvd_upscale.py <movie> --model upscale-training-x2 --scale 2`

## Stopping and carrying on
Ctrl+C any time, then run **the same command again**. Finished steps are skipped, the pairs and the
training both carry on from where they stopped. `--stages train,export` or `--stages export` runs
only some steps. `--fresh` starts the pairs over. A bigger `--iters` trains longer.

## Watching progress
PowerShell shows one live line with an ETA and the clock time it should finish. A page for a browser
or phone starts by itself (port 8643); the address is printed. `--web 0` turns it off.

## What to check
- The pairs step prints the offset between the discs along the movie (it can change by a minute or
  more); look at a few `pairs\lr` / `pairs\hr` pictures with the same name: same moment, `hr` sharper.
- The train step prints, on frames it never trains on, the PSNR of plain bicubic, of the starting
  model and of yours. Yours should climb past both. If it doesn't, the pairs are the problem.
- Out of memory: `--batch 4`, `--patch 64`, or `--checkpoint`. PAL disc: `--speed 1.0427`.
- `--perceptual 0.5 --gan 0.05` adds texture but can invent detail. Do the default first.
- `--arch compact` trains a small, fast network that upscales several times quicker, but it learns from zero:
  use `--iters 100000` or more and `--count 6000` or more pairs. The default (`--arch rrdb`) has the best quality.
- The progress line and page also show GPU watts / load / temperature / clock, CPU load, and a one-line
  verdict on what limits the speed (data loading, heat, power limit, or the GPU itself). CPU watts can't be read
  from Python: use HWiNFO for those.
- Using more of the PC: the processor workers now scale with your core count (override with `--train-workers`,
  `--pair-workers`), the GPU memory layout is tensor-core friendly, and `--batch auto` picks the biggest batch (`--gpu-memory 0.9` to fill nearly all the card) that
  fits in the graphics card's memory. Use it when the monitor shows the GPU under 90% busy; if it is already
  ~100% busy, the GPU is maxed out and only a smaller job (`--iters`, `--arch compact`) will finish sooner.
- `--lpips` (needs `pip install lpips`) adds LPIPS to the held-out scores next to PSNR and SSIM.
- `--gan` uses Real-ESRGAN's U-Net discriminator (`--disc patch` for the old one). `--rotate` also turns patches
  by 90 degrees; it is off because a DVD's blur has a direction.
- One film teaches its own grain and grade; test on a movie that wasn't in the training.
  Live action and animation want separate models. Try `--ai-blend 1.0` with the new model.
