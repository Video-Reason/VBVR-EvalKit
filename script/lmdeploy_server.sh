#!/bin/bash

mkdir -p logs
LOG_FILE="logs/lmdeploy_server.log"
SCORE_LOG_FILE="logs/score.log"


CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server OpenGVLab/InternVL3-8B \
--chat-template internvl2_5 \
--server-port 23333 \
--tp 1 > $LOG_FILE 2>&1 & # takes 30GB vram.
# redirect to log


SERVER_PID=$!     # record background process PID
echo "Server PID: $SERVER_PID"

echo "Waiting for server..."
while ! nc -z localhost 23333; do
  sleep 1
done
echo "Server is ready."

python examples/score_videos.py internvl > $SCORE_LOG_FILE 2>&1

echo "Stopping server..."
kill $SERVER_PID