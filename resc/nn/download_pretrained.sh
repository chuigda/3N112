#!/usr/bin/env bash

BASE_URL="https://raw.githubusercontent.com/club-doki7/sdkfz/refs/heads/master"

FILES=(
  "weights_L1_784x300.bin"
  "weights_L2_300x100.bin"
  "weights_L3_100x10.bin"
  "biases_L1_784x300.bin"
  "biases_L2_300x100.bin"
  "biases_L3_100x10.bin"
)

for file in "${FILES[@]}"; do
  echo "Downloading $file..."
  wget "$BASE_URL/$file" -O "$file"
done
