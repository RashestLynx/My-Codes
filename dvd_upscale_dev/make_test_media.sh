#!/bin/bash
# Makes the test media the tools here use (none of it is committed; see .gitignore).
#   ./make_test_media.sh          synthetic DVD/VHS/AVI clips with frame-number barcodes, the
#                                 shaky clip for --stabilize
#   ./make_test_media.sh faces    + the face-restoration models and the face test frames
#                                 (downloads ~550 MB; needs: pip install onnxruntime
#                                 opencv-python-headless numpy)
#   ./make_test_media.sh master   + a current ffmpeg build in dl/ (e2e runs on both)
# Needs ffmpeg/ffprobe (5.1+) and python3 on PATH.
set -euo pipefail
DEV=$(cd "$(dirname "$0")" && pwd); cd "$DEV"
E=$(python3 -c "import barcode; print(barcode.lum_expr())")     # frame number as 12 bars
ff() { ffmpeg -v error -nostdin -y "$@"; }
bars() {  # bars SIZE RATE SECONDS [extra filters]: black/white barcode video
  echo "color=c=black:s=$1:r=$2:d=$3,format=yuv420p,geq=lum='$E':cb=128:cr=128${4:+,$4}"
}

mkdir -p sync tc pal avi
# NTSC progressive DVD; cut.mpg is a byte cut from its middle (starts mid-GOP, timestamps ~25 s)
ff -f lavfi -i "$(bars 720x480 30000/1001 120 setsar=8/9)" -f lavfi -i "sine=f=440:d=120" \
   -c:v mpeg2video -b:v 4M -g 15 -bf 2 -c:a ac3 -f dvd sync/long.mpg
python3 - <<'PY'
size = __import__("os").path.getsize("sync/long.mpg")
start, length = int(size * 0.206) // 2048 * 2048, int(size * 0.298) // 2048 * 2048
with open("sync/long.mpg", "rb") as f, open("sync/cut.mpg", "wb") as o:
    f.seek(start); o.write(f.read(length))
PY
# NTSC film with 3:2 pulldown, top and bottom field first
for top in 1 0; do
  name=$([ $top = 1 ] && echo tc || echo tcb); field=$([ $top = 1 ] && echo t || echo b)
  ff -f lavfi -i "$(bars 720x480 24000/1001 30 "setsar=8/9,telecine=first_field=$field:pattern=23")" \
     -f lavfi -i "sine=f=440:d=30" -c:v mpeg2video -b:v 8M -flags +ilme+ildct -top $top -g 15 \
     -bf 2 -c:a ac3 -f dvd tc/$name.mpg
done
# PAL 25 fps film: interlaced-coded TFF/BFF and progressive
G=$(bars 720x576 25 16 setsar=16/15)
ff -f lavfi -i "$G" -f lavfi -i "sine=f=440:d=16" -c:v mpeg2video -b:v 6M -g 12 -bf 2 \
   -flags +ilme+ildct -top 1 -c:a ac3 -f dvd pal/bc_tff.mpg
ff -f lavfi -i "$G" -f lavfi -i "sine=f=440:d=16" -c:v mpeg2video -b:v 6M -g 12 -bf 2 \
   -flags +ilme+ildct -top 0 -c:a ac3 -f dvd pal/bc_bff.mpg
ff -f lavfi -i "$G" -f lavfi -i "sine=f=440:d=16" -c:v mpeg2video -b:v 6M -g 12 -bf 2 \
   -c:a ac3 -f dvd pal/bc_prog.mpg
# capture-card style AVIs with B-frames (the decoder delay the script corrects for)
ff -f lavfi -i "$(bars 720x480 30000/1001 10)" -c:v mpeg4 -bf 2 -q:v 3 avi/cap.avi
ff -f lavfi -i "$(bars 720x480 30000/1001 4)" -c:v libx264 -bf 2 -crf 18 -pix_fmt yuv420p \
   avi/h264cap.avi
G=$(bars 720x480 30000/1001 8)
ff -f lavfi -i "$G" -f lavfi -i "sine=f=440:d=8" -c:v mpeg4 -bf 2 -q:v 3 -c:a pcm_s16le \
   avi/av_mpeg4.avi
ff -f lavfi -i "$G" -f lavfi -i "sine=f=440:d=8" -c:v libx264 -bf 2 -crf 18 -pix_fmt yuv420p \
   -c:a pcm_s16le avi/av_h264.avi
# shaky camcorder clip: a still picture with a barcode and a green square, moved by a
# known shake plus a slow pan (stabtest.py measures what is left of the shake)
ff -f lavfi -i "mandelbrot=s=880x620:start_scale=1.5:end_scale=1.5:maxiter=200" -frames:v 1 \
   canvas.png
ff -loop 1 -framerate 30000/1001 -i canvas.png \
   -f lavfi -i "$(bars 480x40 30000/1001 12)" -f lavfi -i "color=c=0x00FF00:s=24x24:r=30000/1001" \
   -f lavfi -i "sine=f=440:d=12" -filter_complex \
   "[0:v]format=yuv420p[c];[c][1:v]overlay=200:420:shortest=1[c2];[c2][2:v]overlay=430:200:shortest=1,crop=720:480:x='60+0.6*n+14*sin(n*0.9)+6*sin(n*2.3+1)':y='60+11*sin(n*1.3+0.5)+5*cos(n*3.1)',setsar=8/9,format=yuv420p[v]" \
   -map "[v]" -map 3:a -t 12 -c:v mpeg2video -b:v 8M -g 15 -bf 2 -c:a ac3 -f dvd shaky.mpg
echo "synthetic media: done"

get() {  # get URL FILE
  [ -s "$2" ] || { mkdir -p "$(dirname "$2")"; curl -fsSL -m 900 -o "$2.part" "$1" && mv "$2.part" "$2"; }
}
for what in "$@"; do
  case $what in
  faces)
    M=faces/face_models
    get https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx $M/face_detection_yunet_2023mar.onnx
    get https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx $M/face_recognition_sface_2021dec.onnx
    get https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx $M/gfpgan_1.4.onnx
    get https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/codeformer.onnx $M/codeformer.onnx
    get https://github.com/facefusion/facefusion-assets/releases/download/examples-3.0.0/target-1080p.mp4 faces/target-1080p.mp4
    # 60 frames of a real face: gt = the HD original at 1280x960 (the "2x upscale" size), sd =
    # the same shrunk to a DVD-sized 640x480 frame (MPEG-2), up = sd upscaled 2x. closeup: the
    # face fills the frame; medium, small, tiny: the shot smaller in the middle
    # (eye distance in the 640x480 frame: closeup 65 px, medium 33, small 20, tiny 13)
    for layout in closeup medium small tiny; do
      d=faces/$layout; rm -rf $d; mkdir -p $d/gt $d/sd $d/up
      case $layout in
      closeup) pad="scale=640:338:flags=area,pad=640:480:0:71"
               gtpad="scale=1280:676:flags=area,pad=1280:960:0:142" ;;
      medium)  pad="scale=320:169:flags=area,pad=640:480:160:155"
               gtpad="scale=640:338:flags=area,pad=1280:960:320:310" ;;
      small)   pad="scale=192:101:flags=area,pad=640:480:224:189"
               gtpad="scale=384:202:flags=area,pad=1280:960:448:378" ;;
      tiny)    pad="scale=128:68:flags=area,pad=640:480:256:206"
               gtpad="scale=256:136:flags=area,pad=1280:960:512:412" ;;
      esac
      ff -ss 1 -i faces/target-1080p.mp4 -frames:v 60 -vf "$gtpad" $d/gt/%06d.png
      ff -ss 1 -i faces/target-1080p.mp4 -frames:v 60 -vf "$pad,format=yuv420p" -c:v mpeg2video \
         -b:v 4M -g 12 -bf 2 $d/sd.mpg
      ff -i $d/sd.mpg $d/sd/%06d.png
      if [ -n "${ESRGAN:-}" ]; then   # the real upscaler (e.g. on lavapipe): what --faces gets
        "$ESRGAN" -i $d/sd -o $d/up -n realesrgan-x2plus -s 2 -f png
      else
        ff -i $d/sd/%06d.png -vf "scale=iw*2:ih*2:flags=bicubic" $d/up/%06d.png
      fi
    done
    echo "faces: done" ;;
  master)
    get https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz dl/ffmpeg.tar.xz
    tar -xf dl/ffmpeg.tar.xz -C dl
    echo "ffmpeg master: PATH=$DEV/dl/ffmpeg-master-latest-linux64-gpl/bin:\$PATH" ;;
  esac
done
