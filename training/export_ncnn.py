"""Turn a model trained by train.py into the .param/.bin files dvd_upscale.py loads.

usage:
  python export_ncnn.py run1/dvd2bd_latest.pth --name dvd2bd-x2 --models "C:\\DVD Upscaler\\models"

Then upscale with it:
  python dvd_upscale.py movie.mkv --model dvd2bd-x2 --scale 2

It uses ../dvd_upscale_dev/make_x2plus_ncnn.py (the same converter that makes realesrgan-x2plus),
which reads the weights back strictly: a wrong or damaged file stops here, not in the middle of a movie.
Tip: the pipeline blends the AI result with a plain upscale (--ai-blend 0.75 for live action). A model
trained on real Blu-ray frames can take more of it:  --ai-blend 1.0  (compare with a --test 60 run).
"""
import argparse, shutil, subprocess, sys, tempfile
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("pth", help="dvd2bd_latest.pth or dvd2bd_best.pth from train.py")
p.add_argument("--name", default="dvd2bd-x2", help="model name to give dvd_upscale.py (default dvd2bd-x2)")
p.add_argument("--models", default=".", help="the upscaler's models folder (default: here)")
a = p.parse_args()

conv = Path(__file__).resolve().parent.parent / "dvd_upscale_dev" / "make_x2plus_ncnn.py"
if not conv.exists():
    sys.exit(f"{conv} not found: run this from a full checkout of the repository")
dest = Path(a.models)
dest.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    r = subprocess.run([sys.executable, str(conv), a.pth, tmp])
    if r.returncode:
        sys.exit("conversion failed")
    for ext in ("param", "bin"):
        shutil.copy(Path(tmp) / f"realesrgan-x2plus.{ext}", dest / f"{a.name}.{ext}")
print(f"wrote {dest / (a.name + '.param')} and {dest / (a.name + '.bin')}")
print(f"use it:  python dvd_upscale.py <movie> --model {a.name} --scale 2")
