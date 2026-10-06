"""Face restoration for upscaled video frames (GFPGAN / CodeFormer as ONNX, YuNet detector).

restore_folder(folder, models_dir, model, strength, fidelity, providers): every PNG in folder,
in name order (= frame order), restored in place. Two passes: detect every frame's faces, link
them into tracks across frames and smooth their landmarks (steady alignment = less flicker), then
restore and blend each face back with a soft mask, fading in/out where a track starts/ends.
"""
import os
import numpy as np
import cv2
import onnxruntime as ort

# the 5-point face layout of the 512x512 crops GFPGAN and CodeFormer were trained on (FFHQ):
# eye (image left), eye (image right), nose tip, mouth corner (image left), mouth corner (right)
TEMPLATE = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                     [201.26117, 371.41043], [313.08905, 371.15118]], np.float32)
EYES = 125.92                     # eye distance in the template
MIN_EYES, FULL_EYES, MAX_EYES = 14, 20, 110   # faces smaller: skipped; larger: tapered off


def soft_mask(size=512, blur=0.3):
    """1 inside, falling smoothly to 0 at the crop's edges (the restorer's hair/background
    stays out)."""
    amount = int(size * 0.5 * blur)
    area = max(amount // 2, 1)
    m = np.ones((size, size), np.float32)
    m[:area, :] = m[-area:, :] = m[:, :area] = m[:, -area:] = 0
    return cv2.GaussianBlur(m, (0, 0), amount * 0.25)


class Restorer:
    def __init__(self, models_dir, model="gfpgan", fidelity=0.7, providers=None):
        avail = ort.get_available_providers()
        providers = [p for p in (providers or ["CUDAExecutionProvider", "DmlExecutionProvider",
                                               "CPUExecutionProvider"]) if p in avail]
        name = {"gfpgan": "gfpgan_1.4.onnx", "codeformer": "codeformer.onnx"}[model]
        self.sess = ort.InferenceSession(os.path.join(models_dir, name), providers=providers)
        self.provider = self.sess.get_providers()[0]
        self.inputs = [i.name for i in self.sess.get_inputs()]
        self.fidelity = fidelity
        self.det_path = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")
        self.det = None
        self.mask = soft_mask()

    def detect(self, img):
        """[(5x2 landmarks, score)] for the faces in a BGR frame."""
        h, w = img.shape[:2]
        s = min(1.0, 1280 / max(w, h))        # (detection on a frame of at most 1280 px)
        small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) \
            if s < 1 else img
        if self.det is None:
            self.det = cv2.FaceDetectorYN.create(self.det_path, "", (small.shape[1], small.shape[0]),
                                                 0.7, 0.3, 50)
        self.det.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.det.detect(small)
        if faces is None:
            return []
        return [(f[4:14].reshape(5, 2) / s, float(f[14])) for f in faces]

    def restore_crop(self, crop):
        x = crop[:, :, ::-1].astype(np.float32) / 255.0
        x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]
        feed = {self.inputs[0]: x}
        if "weight" in self.inputs:
            feed["weight"] = np.array(self.fidelity, dtype=np.float64)
        y = self.sess.run(None, feed)[0][0]
        y = (np.clip(y, -1, 1) + 1) / 2
        return (y.transpose(1, 2, 0) * 255).round().astype(np.uint8)[:, :, ::-1]

    def paste(self, img, pts, strength):
        """Restore the face at landmarks pts and blend it into img (in place)."""
        M = cv2.estimateAffinePartial2D(pts.astype(np.float32), TEMPLATE, method=cv2.LMEDS)[0]
        if M is None:
            return
        crop = cv2.warpAffine(img, M, (512, 512), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))
        out = self.restore_crop(crop)
        IM = cv2.invertAffineTransform(M)
        h, w = img.shape[:2]
        # only the face's area of the frame
        corners = np.array([[0, 0, 1], [512, 0, 1], [0, 512, 1], [512, 512, 1]], np.float32) @ IM.T
        x0, y0 = np.floor(corners.min(0)).astype(int)
        x1, y1 = np.ceil(corners.max(0)).astype(int)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return
        IMs = IM.copy()
        IMs[:, 2] -= (x0, y0)
        size = (x1 - x0, y1 - y0)
        face = cv2.warpAffine(out, IMs, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        m = cv2.warpAffine(self.mask, IMs, size, flags=cv2.INTER_LINEAR)[:, :, None] * strength
        region = img[y0:y1, x0:x1].astype(np.float32)
        img[y0:y1, x0:x1] = np.clip(region * (1 - m) + face.astype(np.float32) * m + 0.5,
                                    0, 255).astype(np.uint8)


def eye_dist(pts):
    return float(np.linalg.norm(pts[1] - pts[0]))


def tracks_of(dets):
    """Link per-frame detections into tracks: [[(frame, landmarks)], ...]. A face continues a
    track when its eye centre is within one eye distance of where the track was 1-2 frames ago."""
    tracks, live = [], []                     # live: indexes into tracks
    for f, faces in enumerate(dets):
        used, still = set(), []
        for t in live:
            lf, lp = tracks[t][-1]
            if f - lf > 2:
                continue
            best, bd = None, None
            for k, (pts, _) in enumerate(faces):
                if k in used:
                    continue
                d = float(np.linalg.norm(pts[:2].mean(0) - lp[:2].mean(0)))
                if d < eye_dist(lp) and (bd is None or d < bd):
                    best, bd = k, d
            if best is not None:
                used.add(best)
                tracks[t].append((f, faces[best][0]))
            still.append(t)
        for k, (pts, _) in enumerate(faces):
            if k not in used:
                tracks.append([(f, pts)])
                still.append(len(tracks) - 1)
        live = [t for t in still if f - tracks[t][-1][0] <= 2]
    return tracks


def plan(dets, n, strength, smooth=2, fade=4, min_len=5):
    """Per frame: [(landmarks, strength)], from steadied tracks."""
    out = [[] for _ in range(n)]
    for tr in tracks_of(dets):
        if len(tr) < min_len:                  # a face seen for a moment: likely a false one
            continue
        frames = [f for f, _ in tr]
        pts = {f: p for f, p in tr}
        # fill gaps of a frame or two (a missed detection) by interpolation
        for f in range(frames[0], frames[-1] + 1):
            if f not in pts:
                a = max(g for g in frames if g < f)
                b = min(g for g in frames if g > f)
                pts[f] = pts[a] + (pts[b] - pts[a]) * (f - a) / (b - a)
        span = range(frames[0], frames[-1] + 1)
        for f in span:
            win = [pts[g] for g in range(f - smooth, f + smooth + 1) if g in pts]
            p = sum(win) / len(win)
            d = eye_dist(p)
            if d < MIN_EYES:
                continue
            size = min(1.0, (d - MIN_EYES) / (FULL_EYES - MIN_EYES))
            size *= 1.0 if d <= MAX_EYES else max(0.0, 1 - (d - MAX_EYES) / (EYES - MAX_EYES))
            edge = min(1.0, (f - frames[0] + 1) / fade, (frames[-1] - f + 1) / fade)
            s = strength * size * edge
            if s > 0.01:
                out[f].append((p, s))
    return out


def restore_folder(folder, models_dir, model="gfpgan", strength=0.6, fidelity=0.7,
                   providers=None, progress=None):
    r = Restorer(models_dir, model, fidelity, providers)
    names = sorted(x for x in os.listdir(folder) if x.endswith(".png"))
    dets = [r.detect(cv2.imread(os.path.join(folder, x))) for x in names]
    todo = plan(dets, len(names), strength)
    done = 0
    for i, (name, faces) in enumerate(zip(names, todo)):
        if faces:
            path = os.path.join(folder, name)
            img = cv2.imread(path)
            for pts, s in faces:
                r.paste(img, pts, s)
            cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            done += len(faces)
        if progress:
            progress(i + 1, len(names))
    return done, r.provider
