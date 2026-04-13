"""
convert_images.py

Convert all .gif images in a directory to .png (RGB, white background).
Updates annotations.csv in the same output directory to point to the new
.png filenames instead of .gif.

Why PNG?
  - GIF uses an 8-bit palette (256 colours).  Converting to PNG RGB gives
    PIL / PyTorch / torchvision a consistent 3-channel tensor with no
    palette-dithering artefacts.
  - PNG is losslessly compressed, so no quality is lost.
  - Most training frameworks (torchvision, HuggingFace datasets) handle PNG
    natively; GIF support is inconsistent.

Usage:
    py convert_images.py [--input DIR] [--delete-gifs]

Defaults:
    --input  training_data   (expects a sub-folder called "images/")
    --delete-gifs  False     (keep original GIFs unless flag is passed)

Output:
    training_data/images/<name>.png   (alongside or replacing the .gif)
    training_data/annotations.csv     (image_file column updated to .png)
"""

import argparse
import csv
import os
import sys
from pathlib import Path

from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gif_to_png(gif_path: Path, png_path: Path) -> bool:
    """
    Convert a single GIF to a 24-bit RGB PNG with a white background.
    Returns True on success.
    """
    try:
        img = Image.open(gif_path)
        # Seek to frame 0 (some GIFs are multi-frame)
        img.seek(0)
        # Convert palette / RGBA to RGB, compositing onto white
        if img.mode in ("P", "RGBA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            bg.paste(img, mask=img.split()[3] if img.mode == "RGBA" else None)
            img = bg
        else:
            img = img.convert("RGB")
        img.save(png_path, format="PNG", optimize=False)
        return True
    except Exception as e:
        print(f"  ERROR converting {gif_path.name}: {e}", file=sys.stderr)
        return False


def update_annotations(annotations_path: Path) -> int:
    """
    Rewrite the image_file column in annotations.csv: foo.gif → foo.png.
    Returns the number of rows updated.
    """
    if not annotations_path.exists():
        return 0

    rows = []
    with open(annotations_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row.get("image_file", "").endswith(".gif"):
                row["image_file"] = row["image_file"][:-4] + ".png"
            rows.append(row)

    with open(annotations_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=str, default="training_data",
        help="Root directory containing images/ sub-folder (default: training_data)"
    )
    parser.add_argument(
        "--delete-gifs", action="store_true", default=False,
        help="Delete original .gif files after successful conversion"
    )
    args = parser.parse_args()

    root       = Path(args.input)
    images_dir = root / "images"

    if not images_dir.is_dir():
        print(f"ERROR: {images_dir} not found.", file=sys.stderr)
        sys.exit(1)

    gifs = sorted(images_dir.glob("*.gif"))
    if not gifs:
        print("No .gif files found — nothing to do.")
        sys.exit(0)

    print(f"Found {len(gifs):,} GIF files in {images_dir}")
    print(f"Converting to PNG (white background, RGB)...\n")

    ok_count   = 0
    fail_count = 0

    for i, gif_path in enumerate(gifs, 1):
        png_path = gif_path.with_suffix(".png")

        if png_path.exists():
            print(f"[{i:>4}/{len(gifs)}] {gif_path.name} — already converted, skipping")
            ok_count += 1
            continue

        success = gif_to_png(gif_path, png_path)
        if success:
            print(f"[{i:>4}/{len(gifs)}] {gif_path.name} → {png_path.name}")
            ok_count += 1
            if args.delete_gifs:
                gif_path.unlink()
        else:
            fail_count += 1

    # Update annotations CSV
    annotations_path = root / "annotations.csv"
    updated = update_annotations(annotations_path)
    if updated:
        print(f"\nUpdated {updated} rows in {annotations_path} (.gif → .png)")
    else:
        print(f"\n(No annotations.csv found or no rows to update at {annotations_path})")

    print(f"\nDone.  Converted: {ok_count}  |  Failed: {fail_count}")
    if fail_count:
        print(f"  {fail_count} files could not be converted — check stderr output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
