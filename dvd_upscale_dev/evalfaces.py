"""Compare face-restoration variants against the real (HD) faces.
usage: python3 evalfaces.py LAYOUT VARIANT...   variant: none | gfpgan:STRENGTH | codeformer:FIDELITY:STRENGTH
For each variant: copies LAYOUT/up (the 2x upscaled DVD frames) to LAYOUT/v_<variant>, restores it,
then measures in the face box of the real frames (LAYOUT/gt): PSNR, SSIM, sharpness (variance of
the Laplacian; the real face's value for reference), extra flicker (frame-to-frame change not in
the real video), identity (SFace cosine similarity to the real face; same person above 0.36)."""
import importlib.util, os, shutil, sys, time
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
# the face code of dvd_upscale.py itself (one implementation; faces_core.py was its prototype)
sys.dont_write_bytecode = True          # (no __pycache__ in the repository's root)
_spec = importlib.util.spec_from_file_location("dvd_upscale",
                                               os.path.join(HERE, "..", "dvd_upscale.py"))
du = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(du)
DATA = os.path.join(HERE, "faces")             # made by make_test_media.sh
MODELS = os.path.join(DATA, "face_models")
layout, variants = sys.argv[1], sys.argv[2:]
gt_dir = os.path.join(DATA, layout, "gt")
names = sorted(os.listdir(gt_dir))
gt = [cv2.imread(os.path.join(gt_dir, n)) for n in names]

det = cv2.FaceDetectorYN.create(os.path.join(MODELS, "face_detection_yunet_2023mar.onnx"), "",
                                (gt[0].shape[1], gt[0].shape[0]), 0.7, 0.3, 5)
rec = cv2.FaceRecognizerSF.create(os.path.join(MODELS, "face_recognition_sface_2021dec.onnx"), "")
gt_faces = [det.detect(g)[1][0] for g in gt]           # the one face of each real frame
boxes = []
for f in gt_faces:
    x, y, w, h = f[:4]
    boxes.append((int(x + 0.1 * w), int(y + 0.1 * h), int(x + 0.9 * w), int(y + 0.9 * h)))
gt_feat = [rec.feature(rec.alignCrop(g, f)) for g, f in zip(gt, gt_faces)]


def ssim(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = cv2.GaussianBlur(a, (11, 11), 1.5), cv2.GaussianBlur(b, (11, 11), 1.5)
    s_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a ** 2
    s_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b ** 2
    s_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    return float((((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) /
                  ((mu_a ** 2 + mu_b ** 2 + c1) * (s_a + s_b + c2))).mean())


def crop(img, b):
    return img[b[1]:b[3], b[0]:b[2]]


def measure(frames):
    psnr, ss, sharp, flick, ident = [], [], [], [], []
    for i, (img, g, b) in enumerate(zip(frames, gt, boxes)):
        c, cg = crop(img, b), crop(g, b)
        psnr.append(cv2.PSNR(c, cg))
        ss.append(ssim(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), cv2.cvtColor(cg, cv2.COLOR_BGR2GRAY)))
        sharp.append(cv2.Laplacian(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        if i:
            pb = boxes[i - 1]
            bb = (max(b[0], pb[0]), max(b[1], pb[1]), min(b[2], pb[2]), min(b[3], pb[3]))
            d = crop(img, bb).astype(np.float32) - crop(frames[i - 1], bb).astype(np.float32)
            dg = crop(g, bb).astype(np.float32) - crop(gt[i - 1], bb).astype(np.float32)
            flick.append(float(np.abs(d - dg).mean()))
        found = det.detect(img)[1]
        if found is not None:
            ident.append(float(rec.match(rec.feature(rec.alignCrop(img, found[0])), gt_feat[i],
                                         cv2.FaceRecognizerSF_FR_COSINE)))
    return dict(psnr=np.mean(psnr), ssim=np.mean(ss), sharp=np.mean(sharp), flicker=np.mean(flick),
                identity=np.mean(ident) if ident else float("nan"), ident_found=len(ident))


ref = measure(gt)
print(f"[{layout}] real face: sharpness {ref['sharp']:.0f}, eye distance "
      f"{du.face_eye_dist(gt_faces[0][4:14].reshape(5, 2)):.0f} px")
for v in variants:
    out = os.path.join(DATA, layout, "v_" + v.replace(":", "_"))
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(os.path.join(DATA, layout, "up"), out)
    t0 = time.time()
    if v != "none":
        kind, *nums = v.split(":")
        # (in place; the "up" frames are 2x upscales of the DVD frames)
        if kind == "gfpgan":
            n, _, prov = du.restore_faces(out, MODELS, "gfpgan", float(nums[0]), scale=2)
        else:
            n, _, prov = du.restore_faces(out, MODELS, "codeformer", float(nums[1]),
                                          float(nums[0]), scale=2)
    took = time.time() - t0
    frames = [cv2.imread(os.path.join(out, x)) for x in names]
    m = measure(frames)
    print(f"[{layout}] {v:16} PSNR {m['psnr']:.2f}  SSIM {m['ssim']:.3f}  sharpness {m['sharp']:5.0f}"
          f"  extra flicker {m['flicker']:.2f}  identity {m['identity']:.3f} ({m['ident_found']}/60)"
          f"  {took:.0f} s", flush=True)
