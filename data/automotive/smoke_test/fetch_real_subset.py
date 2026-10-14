"""
TrendLens — Stanford Cars Real Smoke-Test Subset Downloader
Downloads 45 authentic Stanford Cars images matching the curated
SMOKE_TEST_MANIFEST covering all 6 body types.

Source: Stanford Cars Dataset (Krause et al., ICCV 2013)
Mirror: https://raw.githubusercontent.com/jhpohovey/StanfordCars-Dataset/main/stanford_cars/
"""

import sys
from pathlib import Path
import urllib.request
import scipy.io as sio
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.automotive_adapter import SMOKE_TEST_MANIFEST

IMAGES_DIR = PROJECT_ROOT / "data" / "automotive" / "smoke_test" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

CARS_ANNOS_PATH = PROJECT_ROOT / "data" / "automotive" / "smoke_test" / "cars_annos.mat"
TRAIN_ANNOS_PATH = PROJECT_ROOT / "data" / "automotive" / "smoke_test" / "cars_train_annos.mat"


def fetch_stanford_cars_images():
    print("=" * 60)
    print("TrendLens: Downloading 45 Real Stanford Cars Smoke-Test Images")
    print("=" * 60)

    # 1. Ensure annotation files exist
    if not CARS_ANNOS_PATH.is_file():
        print("Downloading cars_annos.mat...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/jhpohovey/StanfordCars-Dataset/main/stanford_cars/cars_annos.mat",
            CARS_ANNOS_PATH
        )

    if not TRAIN_ANNOS_PATH.is_file():
        print("Downloading cars_train_annos.mat...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/jhpohovey/StanfordCars-Dataset/main/stanford_cars/devkit/cars_train_annos.mat",
            TRAIN_ANNOS_PATH
        )

    # 2. Load classes and training images
    cars_mat = sio.loadmat(CARS_ANNOS_PATH, squeeze_me=True)
    class_names = [str(c) for c in cars_mat["class_names"]]
    class_name_to_id = {name: i + 1 for i, name in enumerate(class_names)}

    train_mat = sio.loadmat(TRAIN_ANNOS_PATH, squeeze_me=True)
    train_annos = train_mat["annotations"]

    # Build mapping from class_id -> list of fnames
    class_to_fnames = {}
    for entry in train_annos:
        # entry format: (bbox_x1, bbox_y1, bbox_x2, bbox_y2, class_id, fname)
        cid = int(entry[4])
        fname = str(entry[5])
        if cid not in class_to_fnames:
            class_to_fnames[cid] = []
        class_to_fnames[cid].append(fname)

    # 3. Download images for each manifest entry
    base_url = "https://raw.githubusercontent.com/jhpohovey/StanfordCars-Dataset/main/stanford_cars/cars_train/"
    downloaded_count = 0

    print(f"\nProcessing {len(SMOKE_TEST_MANIFEST)} curated smoke-test vehicles...")
    for idx, (label, target_filename) in enumerate(SMOKE_TEST_MANIFEST, start=1):
        target_path = IMAGES_DIR / target_filename
        if target_path.is_file() and target_path.stat().st_size > 1000:
            print(f"[{idx:02d}/45] Exists: {target_filename}")
            downloaded_count += 1
            continue

        cid = class_name_to_id.get(label)
        if not cid or cid not in class_to_fnames:
            print(f"[{idx:02d}/45] ERROR: Class '{label}' not found in Stanford Cars dataset!")
            continue

        source_fname = class_to_fnames[cid][0]
        img_url = f"{base_url}{source_fname}"

        try:
            temp_path, _ = urllib.request.urlretrieve(img_url)
            # Validate and save with standard JPEG format
            with Image.open(temp_path) as img:
                img = img.convert("RGB")
                img.save(target_path, "JPEG", quality=95)
            downloaded_count += 1
            print(f"[{idx:02d}/45] Downloaded: {target_filename} <- {source_fname} ({label})")
        except Exception as e:
            print(f"[{idx:02d}/45] Failed to download {source_fname}: {e}")

    print("\n" + "=" * 60)
    print(f"Successfully downloaded {downloaded_count} / {len(SMOKE_TEST_MANIFEST)} Stanford Cars images.")
    print(f"Saved to: {IMAGES_DIR}")
    print("=" * 60)
    return downloaded_count


if __name__ == "__main__":
    count = fetch_stanford_cars_images()
    if count < len(SMOKE_TEST_MANIFEST):
        sys.exit(1)
