#!/bin/bash
# run.sh with long (360-frame) chunks and a slower start-up: the retire decision mid-chunk
SIM_S=3 CHUNK=360 exec "$(dirname "$0")/run.sh"
