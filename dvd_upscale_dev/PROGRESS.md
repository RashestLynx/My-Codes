# dvd_upscale.py: development notes

These are the test tools and the state of the work. Test media, models and run output are not
committed. To recreate them:

```
./make_test_media.sh                 # synthetic clips (frame-number barcodes), ~2 min
./make_test_media.sh faces master    # + face models and test frames, + current ffmpeg build
```

## Done (on branch `claude/code-analysis-faults-qu27bv`)

| Commit | What |
| --- | --- |
| b5c0849 | the original script |
| e935466 | the first round of fixes |
| 5468d52 | AVI seeking: B-frame decoder delay |
| 51bca2d | `--gpu-jobs` (two upscalers per GPU), interpolated chroma, crash-safe resume |
| 9c13cad | meter: an upscaler's frames counted from its first second |
| a13c0ec | `--stabilize [normal\|strong]` (vid.stab, motion measured once, sliced per chunk) |
| (next)  | the user's changes, merged and refined (details below) |

The user's changes:
- compact models run 8 GPU threads by default; a failed chunk lowers this to 6, then 4;
- the helper-lane watchdog reads the upscaler log incrementally;
- NaN and inf are rejected for `--ai-blend`, `--smooth` and `--sharpen`;
- a "GPU settings" line is printed.

## Tests

- `python3 e2e.py ../dvd_upscale.py TAG [case...]` runs 14 cases: frame order, sync and chunk
  seams for NTSC/PAL film, VHS and AVI, plus the AI path with a fake upscaler and two GPUs.
  - Expected: every case has `breaks 0`, except `ntsc_film_bff`, which has one repeated first
    frame (it was there before these changes).
  - VHS AVI captures are offset by -33 ms (mpeg4) and -67 ms (h264): this is the B-frame delay,
    and it is the same as before.
  - Run it on ffmpeg 6.1 and on master: `PATH=dl/ffmpeg-master-latest-linux64-gpl/bin:$PATH`.
- `simgpu/run.sh` and `simgpu/runR.sh`: `--gpu-jobs` 1 vs 2 on a simulated GPU, plus the retire
  case.
- `failesrgan/`: an upscaler that fails once on the chunks in `FAIL_CHUNKS` (retry path).
- `stabrun.py NAME ZOOM [args]` with `stabtest.py`: the jitter left after `--stabilize`, seams,
  sync.
- `filtertest.py`: filter-graph checks.
- `barcode.py`: decodes the frame number from each frame.

## Face recovery (in progress)

- `faces_core.py` holds the working code:
  - YuNet detection (on frames at most 1280 px);
  - 5-landmark alignment to the FFHQ 512 template;
  - GFPGAN 1.4 or CodeFormer as ONNX (CodeFormer also takes the `weight` input, i.e. fidelity);
  - a soft-mask paste back.
- Two passes per folder:
  1. Detect every frame, link faces into tracks, smooth the landmarks (±2 frames), fill
     1–2-frame gaps, fade in and out over 4 frames, and drop tracks shorter than 5 frames.
  2. Restore and blend.
- Size gate by eye distance in the upscaled frame:
  - under 14 px: skipped;
  - 14–20 px: ramps up;
  - over 110 px: tapers off (the model's crop is only 512 px).
- `evalfaces.py LAYOUT VARIANT...` runs on the `closeup` and `medium` layouts.
  - Variants: `none`, `gfpgan:S`, `codeformer:F:S`.
  - It compares each variant with the real HD face (PSNR, SSIM, sharpness, extra flicker, and
    SFace identity similarity; 0.36 or more means the same person).

Planned integration:
- `--faces [strength]`, opt-in, for live and VHS only (cartoons have no real faces);
- after each chunk's upscale, on `tmp/out`, in a subprocess per chunk so an onnxruntime crash
  can't take the run down;
- needs `pip install onnxruntime` (or onnxruntime-directml / -gpu), `opencv-python-headless` and
  `numpy`, plus the models in a `face_models` folder next to the script;
- if anything is missing, the run stops with what to install and where to get the models.

## Downloads

- YuNet: https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
- SFace (evaluation only): https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
- GFPGAN 1.4: https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx
- CodeFormer: https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/codeformer.onnx
- Face test video: https://github.com/facefusion/facefusion-assets/releases/download/examples-3.0.0/target-1080p.mp4
- Real-ESRGAN ncnn (Linux): https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip
  - With apt `mesa-vulkan-drivers vulkan-tools`, it runs on the CPU (lavapipe).
- ffmpeg master: https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz
