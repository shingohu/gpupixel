#!/usr/bin/env bash
# Download BlazeFace models used by face_detection_tflite into third_party/models
set -e
mkdir -p third_party/models
BASE="https://raw.githubusercontent.com/hugocornellier/face_detection_tflite/main/assets/models"
FILES=( 
  "face_detection_back.tflite"
  "face_detection_front.tflite"
  "face_detection_short_range.tflite"
  "face_detection_full_range.tflite"
)
for f in "${FILES[@]}"; do
  url="$BASE/$f"
  out="third_party/models/$f"
  if [ -f "$out" ]; then
    echo "$out already exists, skipping"
    continue
  fi
  echo "Downloading $url -> $out"
  curl -L -o "$out" "$url"
done

echo "Models downloaded to third_party/models/"
