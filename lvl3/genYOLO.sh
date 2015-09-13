#!/usr/bin/env sh
for filename in models/*.yml; do
  echo "$filename"
  python genFinal.py $filename
done
python genYOLO.py
