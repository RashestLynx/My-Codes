import subprocess, types, sys
import dvd_upscale as d

def A(**kw):
    base = dict(type="live", mode="progressive", src_hd=False, pal=False, combed=False, dar=None,
                denoise="2:1.5:6:5", dvd_trim=0, fps="24000/1001", out_w=1440, height=1080,
                fast=False, matrix="bt601-6-525", smooth=4, sharpen=0.3, hevc=False,
                vhs_dedup=None, parity="tff", vhs_h=480, vhs_fin=d.Fraction(30000,1001),
                vhs_crop="crop=704:480:8:0", chroma_shift="cbh=-3:crh=-3:cbv=-1:crv=-1",
                mask=(8,8,2,12), vhs_sw=640, vhs_sh=480, vhs_trim=0)
    base.update(kw); return types.SimpleNamespace(**base)

def tryvf(name, vf, src="testsrc2=s=720x480:r=30000/1001", n=40):
    try:
        r = subprocess.run(["ffmpeg","-v","error","-nostdin","-f","lavfi","-i",src,"-vf",vf,"-frames:v",str(n),"-f","null","-"],capture_output=True,text=True,timeout=30)
    except subprocess.TimeoutExpired:
        print("HANG "+name); return
    print(("OK  " if r.returncode==0 and not r.stderr.strip() else "ERR ")+name, r.stderr.strip()[:300])

cases = {
 "dvd progressive": A(),
 "dvd telecine": A(mode="telecine"),
 "dvd telecine warm": A(mode="telecine", dvd_trim=8),
 "dvd interlaced": A(mode="interlaced", fps="30000/1001"),
 "dvd combed": A(combed=True, fps="30000/1001"),
 "dvd dar 16/9": A(dar="16/9"),
 "hd src": A(src_hd=True),
 "pal telecine": A(mode="telecine", pal=True, fps="25"),
 "vhs telecine": A(type="vhs", mode="telecine", vhs_trim=8),
 "vhs interlaced": A(type="vhs", mode="interlaced", fps="60000/1001", vhs_trim=12),
 "vhs progressive": A(type="vhs", mode="progressive", fps="30000/1001"),
 "vhs dedup": A(type="vhs", mode="interlaced", vhs_dedup="fps=30000/1001", fps="60000/1001"),
}
for k, a in cases.items():
    tryvf("prefilter "+k, d.prefilter(a))
for k, a in {"post sd h264": A(), "post hevc": A(hevc=True), "post fast hd": A(fast=True, src_hd=True), "post nosmooth": A(smooth=0)}.items():
    tryvf(k, d.postfilter(a))
tryvf("VHS_CADENCE_VF", d.VHS_CADENCE_VF)
tryvf("vhs fieldmatch 486", d.prefilter(A(type="vhs", mode="telecine", vhs_h=486, vhs_crop="crop=704:480:8:4")), src="testsrc2=s=720x486:r=30000/1001")
tryvf("vhs PAL telecine", d.prefilter(A(type="vhs", mode="telecine", vhs_h=576, vhs_fin=d.Fraction(25), fps="25", vhs_crop="crop=704:576:8:0", vhs_sw=768, vhs_sh=576)), src="testsrc2=s=720x576:r=25")
