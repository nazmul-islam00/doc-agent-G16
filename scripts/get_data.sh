#!/usr/bin/env bash
# A1 — fetch or recreate the scanned corpus into data/raw/
set -euo pipefail

readonly source_url="https://old.bansdoc.gov.bd/iaeabooks/IAEA%20Books/543%20%E0%A6%B6%E0%A6%B8%E0%A7%8D%E0%A6%AF%E0%A7%87%E0%A6%B0%20%E0%A6%B0%E0%A7%8B%E0%A6%97-%E0%A6%B9%E0%A6%BE%E0%A6%B8%E0%A6%BE%E0%A6%A8%20%E0%A6%86%E0%A6%B6%E0%A6%B0%E0%A6%BE%E0%A6%89%E0%A6%9C%E0%A7%8D%E0%A6%9C%E0%A6%BE%E0%A6%AE%E0%A6%BE%E0%A6%A8.pdf?fbclid=IwY2xjawTCl2tleHRuA2FlbQIxMQBzcnRjBmFwcF9pZAEwAAEeL1HZMmHUMjpsEdo6Q6_IwkkxAdQgtptBV36r-g5bzhL2LQAJrDrCJeMbz-g_aem_HuLEUtMAounrRfRT5S55MA"
readonly pdf_path="data/interim/krishipath.pdf"
readonly image_dir="data/raw/krishipath"

mkdir -p "$(dirname "$pdf_path")" "$image_dir"

if [[ ! -s "$pdf_path" ]]; then
  echo "Downloading corpus PDF..."
  curl --fail --location --retry 3 --output "$pdf_path" "$source_url"
fi

if ! command -v gs >/dev/null 2>&1; then
  echo "Error: Ghostscript 'gs' is required to render the PDF." >&2
  exit 1
fi

echo "Rendering scanned pages into $image_dir..."
gs -q -dNOPAUSE -dBATCH -sDEVICE=jpeg -r150 -dJPEGQ=92 \
  -sOutputFile="$image_dir/page-%04d.jpg" "$pdf_path"

count=$(find "$image_dir" -type f -name '*.jpg' | wc -l)
if [[ "$count" -eq 0 ]]; then
  echo "Error: PDF rendering produced no page images." >&2
  exit 1
fi

echo "Prepared $count scanned page images in $image_dir."
