# Train the upscaler on real DVD / Blu-ray pairs

Three steps, one script each. Needs Python with `torch` (CUDA build), `torchvision`, `numpy`,
`opencv-python`, and `ffmpeg`/`ffprobe` on the PATH.

```
# 1. Aligned frame pairs from one movie you own on both discs (about 3000, ~10 GB of PNGs)
python make_pairs.py --dvd movie_dvd.mkv --bluray movie_bd.mkv --out pairs --count 3000

# 2. Fine-tune Real-ESRGAN x2plus on them (RTX 3060 laptop, 6 GB: ~1-2 s/step, 20000 steps)
python train.py --pairs pairs --out run1 --pretrained RealESRGAN_x2plus.pth

# 3. Convert for dvd_upscale.py (into the folder where realesrgan-x2plus.param/.bin live)
python export_ncnn.py run1/dvd2bd_latest.pth --name dvd2bd-x2 --models "C:\DVD Upscaler\models"
python dvd_upscale.py movie.mkv --model dvd2bd-x2 --scale 2 --test 60     # 60 s preview
```

Get `RealESRGAN_x2plus.pth` from
https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

## What to check

- **make_pairs** prints the offset between the discs from three places in the movie; they must agree
  (`+2.50 s`, `+2.50 s`, `+2.49 s`). It also prints how many candidates it rejected and why.
  Look at a few `pairs/lr` / `pairs/hr` pairs yourself. PAL discs: `--speed 1.0427`.
- **train** prints, on frames it never trains on, the PSNR of plain bicubic, of the starting model,
  and of your model every 1000 steps. If yours is not clearly above the starting model, the pairs are
  the problem (misalignment, different grade or cut), not the training.
- Out of memory: `--batch 4`, `--patch 64`, or `--checkpoint`.
- Stage 1 (L1 only, the default) is the safe one. `--perceptual 0.5 --gan 0.05` (start from the
  stage 1 result) adds texture and can invent detail.
- One film teaches its own grain and grade. A few different films, put together
  (`make_pairs.py --out pairs` once per film into one folder tree, then merge the `lr`/`hr`
  folders with renamed files), generalise better. Live action and animation want separate models.
- A model trained on real Blu-ray frames may deserve more of the AI result than the default
  blend: try `--ai-blend 1.0`.
