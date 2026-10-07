#!/bin/bash
# --gpu-jobs 1 vs 2 on the simulated GPU (60-frame chunks), and 2 on a GPU that slows down with
# two upscalers (the second should be retired). Needs sync/long.mpg (make_test_media.sh).
DEV=$(cd "$(dirname "$0")/.." && pwd); cd "$DEV/sync" || exit 1
export SIM_DIR=$DEV/simgpu/state SIM_G=${SIM_G:-0.04} SIM_S=${SIM_S:-2}
run() { name=$1; shift; rm -rf sim_$name; mkdir sim_$name; s=$(date +%s); timeout 1500 python3 "$DEV/../dvd_upscale.py" long.mpg sim_$name/out.mkv --cpu --type live --mode progressive --chunk-frames ${CHUNK:-60} --height 240 --esrgan "$DEV/simgpu/realesrgan-ncnn-vulkan" --work sim_$name/work "$@" > sim_$name/log.txt 2>&1; echo "$name rc=$? wall=$(( $(date +%s)-s ))s | $(grep -E 'Upscaled|NOTE' sim_$name/log.txt | tr '\n' ' ')"; }
run one --gpu-jobs 1
run two --gpu-jobs 2
SIM_PENALTY=2.5 run slow_two --gpu-jobs 2
