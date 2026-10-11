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
  - Expected: every case has `breaks 0` (`ntsc_film_bff` had one repeated first frame until
    `repeatfields` was left out for discs with no soft pulldown).
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

## Face recovery (`--faces`, in dvd_upscale.py)

- The face code is in `dvd_upscale.py` itself (`FaceRestorer`, `face_tracks`, `face_plan`,
  `restore_faces`); `faces_core.py` was its first prototype (older rules: full from 10 px,
  top_k 50, plain averaging, simpler chunk edges; it no longer gives the same frames and no
  tool uses it).
  - YuNet detection (on frames at most 1280 px);
  - 5-landmark alignment to the FFHQ 512 template;
  - GFPGAN 1.4 or CodeFormer as ONNX (CodeFormer also takes the `weight` input, i.e. fidelity);
  - a soft-mask paste back over only the face's part of the frame.
- Two passes per chunk:
  1. Detect every frame (YuNet top_k 5000: 50 lost faces in crowds), link faces into tracks,
     fill 1–2-frame gaps, steady the landmarks with a straight-line fit over ±2 frames (slid
     inward at an open chunk edge), fade in and out over 4 frames, and drop tracks shorter than
     5 frames.
     - A track of 3+ detections that touches the chunk's first or last frame, or misses only
       that frame (held there if its face box still looks the same: 8x8 average difference
       under 12), is open there: no fade.
  2. Restore and blend. The paste is made up for `--ai-blend` (mask × S / ai_blend, at most 1),
     so S is the face's share of the final picture, at most ai_blend.
- Size gate by eye distance in source pixels (upscaled eye distance / scale): under 7 skipped,
  7–16 ramps up (full from 16), over 90 tapers off.
- How it runs: `--faces [S]` (default 0.6), `--face-model gfpgan|codeformer`, `--face-models DIR`
  (default `face_models` next to the script). Live and VHS only (anime/CGI: a NOTE, turned off;
  `--fast` or `--ai-blend 0`: turned off). At the start of `Chunk.finish`, a worker process
  (`dvd_upscale.py --faces-worker ...`) writes the changed frames into `tmp/faces`; once it
  exits 0 they are moved over `tmp/out`. One worker at a time (`FACE_LOCK`), one retry (on the
  processor, `--cpu`), killed on Ctrl+C (and at exit; it also stops when the main run's pipe to
  it closes) or after 10 minutes without a frame found or a face restored. A start-up check
  (`--faces-worker --check`, 10-minute limit) reports missing, broken or too old packages,
  missing or damaged model files, and the provider used.
- `evalfaces.py LAYOUT VARIANT...` (imports dvd_upscale.py) on the `closeup`, `medium`, `small`
  and `tiny` layouts (source eye distance about 65/36/17/14 px).
  - Variants: `none`, `gfpgan:S`, `codeformer:F:S`.
  - It compares each variant with the real HD face (PSNR, SSIM, sharpness, extra flicker, and
    SFace identity similarity; 0.36 or more means the same person).
  - The 7/16 ramp came from the `small` and `tiny` runs: at 14 px, 0.6 drew more detail than
    the real face had; from 17 px, 0.6 matched it (see the comment at `FACE_MIN_EYES` in
    dvd_upscale.py; the result tables, `faces/eval*.txt`, are git-ignored).
- `facerun.py LAYOUT [S]`: the whole script with and without `--faces` on `faces/LAYOUT/sd.mpg`
  (3 chunks of 20 frames, the bicubic stand-in upscaler); the difference inside and outside the
  face, and at the chunk seams against inside the chunks. `REUSE=1` measures the last runs again.
  - medium, 0.6, CPU: 57 frames both (the source's 60 come out as 57 with or without --faces:
    the clip is taken for 23.976 fps), face difference 2.67 (2.16 before the paste was made up
    for --ai-blend 0.75), outside 0.04 at most; extra flicker at the seams 0.19 / 0.02 against a
    median of 0.28 inside the chunks (no seam pulse).

## Downloads

- YuNet: https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
- SFace (evaluation only): https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
- GFPGAN 1.4: https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx
- CodeFormer: https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/codeformer.onnx
- Face test video: https://github.com/facefusion/facefusion-assets/releases/download/examples-3.0.0/target-1080p.mp4
- Real-ESRGAN ncnn (Linux): https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip
  - With apt `mesa-vulkan-drivers vulkan-tools`, it runs on the CPU (lavapipe).
- ffmpeg master: https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz
