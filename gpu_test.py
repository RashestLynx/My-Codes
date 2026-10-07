#!/usr/bin/env python3
"""GPU test for dvd_upscale.py: runs realesrgan-x2plus with the current ncnn (pip install ncnn,
2026) instead of realesrgan-ncnn-vulkan.exe (ncnn from April 2022), on your GPU, for 90 s each:
whole frames, then 200-pixel tiles (the .exe's default). It reports the speed and whether the
GPU was reset (vkQueueSubmit failed -4).

    python -m pip install --no-deps ncnn numpy
    python gpu_test.py ["CGI\\COCO.mkv"]

(--no-deps: ncnn itself needs only numpy; without it pip also installs opencv-python, which
can clash with the opencv-python-headless that --faces uses)

Put it next to dvd_upscale.py (the models folder next to realesrgan-ncnn-vulkan.exe is used).
With a movie, a frame from it is used (ffmpeg needed), else a test picture.
"""
import os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SECS = 90


def frame(movie):
    """852x480 RGB frame (the size dvd_upscale.py gives the upscaler for a 16:9 DVD)"""
    import numpy as np
    src = (["-ss", "1200", "-i", movie] if movie else
           ["-f", "lavfi", "-i", "testsrc2=s=852x480:d=1"])
    try:
        r = subprocess.run(["ffmpeg", "-v", "error", *src, "-frames:v", "1", "-vf", "scale=852:480",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
        if len(r.stdout) == 852 * 480 * 3:
            return np.frombuffer(r.stdout, np.uint8).reshape(480, 852, 3)
    except OSError:
        pass
    rng = np.random.default_rng(1)          # no ffmpeg: a smooth random picture
    small = rng.integers(0, 255, (30, 54, 3)).astype(np.float32)
    return np.kron(small, np.ones((16, 16, 1)))[:480, :852].astype(np.uint8)


def run(tile, movie):
    import numpy as np, ncnn
    models = HERE / "models"
    net = ncnn.Net()
    net.opt.use_vulkan_compute = True
    net.opt.use_fp16_packed = net.opt.use_fp16_storage = True
    net.opt.use_fp16_arithmetic = False
    gpu = ncnn.get_default_gpu_index()
    net.set_vulkan_device(gpu)
    net.load_param(str(models / "realesrgan-x2plus.param"))
    net.load_model(str(models / "realesrgan-x2plus.bin"))
    img, pad, scale = frame(movie), 10, 2
    h, w, _ = img.shape
    tile = tile or max(w, h)
    src = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect").astype(np.float32) / 255
    src = np.ascontiguousarray(src.transpose(2, 0, 1))
    start, frames, first = time.time(), 0, None
    while time.time() - start < SECS:
        t0 = time.time()
        for y0 in range(0, h, tile):
            for x0 in range(0, w, tile):
                th, tw = min(tile, h - y0), min(tile, w - x0)
                arr = np.ascontiguousarray(src[:, y0:y0 + th + 2 * pad, x0:x0 + tw + 2 * pad])
                ex = net.create_extractor()
                m = ncnn.Mat(arr)
                ex.input("data", m)
                ret, out = ex.extract("output")
                if ret:
                    print(f"  FAILED after {frames} frames ({time.time() - start:.0f} s): "
                          f"ncnn extract returned {ret}", flush=True)
                    return 1
                np.array(out)
        frames += 1
        first = first or time.time() - t0
        print(f"\r  {frames} frames, {frames / (time.time() - start):.2f} frames/s", end="",
              flush=True)
    secs = time.time() - start
    print(f"\n  OK: {frames} frames in {secs:.0f} s, {frames / secs:.2f} frames/s "
          f"(first frame {first:.1f} s, with the start-up)", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--one":
        sys.exit(run(int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None))
    try:
        import numpy, ncnn
    except ImportError:
        sys.exit("First: python -m pip install --no-deps ncnn numpy")
    if not (HERE / "models" / "realesrgan-x2plus.param").exists():
        sys.exit(f"realesrgan-x2plus.param not found in {HERE / 'models'}")
    print(f"ncnn {ncnn.__version__}; GPUs: " + ", ".join(
        f"{i} = {ncnn.get_gpu_info(i).device_name()}" for i in range(ncnn.get_gpu_count()))
        + f"; using {ncnn.get_default_gpu_index()}")
    movie = sys.argv[1] if len(sys.argv) > 1 else None
    results = {}
    for tile, name in ((0, "whole frames"), (200, "200-pixel tiles")):
        print(f"\nx2plus, {name}, {SECS} s:", flush=True)
        # (a process of its own: after a GPU reset the next test starts on a fresh device)
        rc = subprocess.run([sys.executable, __file__, "--one", str(tile)]
                            + ([movie] if movie else [])).returncode
        results[name] = "OK" if rc == 0 else "FAILED"
        if rc:
            time.sleep(15)          # (the graphics driver restarting)
    print("\nResults: " + ", ".join(f"{k}: {v}" for k, v in results.items()))
    print("(send a screenshot of this window)")
