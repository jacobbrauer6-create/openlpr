"""
scripts/prepare_dataset.py
--------------------------
Download, clean, annotate, and version licence plate datasets
for domestic (US) and international plates.

Usage:
    python scripts/prepare_dataset.py --regions us eu asia --split 80/10/10
    python scripts/prepare_dataset.py --regions all --augment --version v1.2
"""

import argparse
import hashlib
import json
import os
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    "us": {
        "ufpr_alpr": {
            "url": "https://web.inf.ufpr.br/vri/databases/ufpr-alpr/",
            "license": "research_only",
            "approx_count": 4500,
            "notes": "Brazilian + US plates, bounding box + char annotations",
        },
        "openalpr_benchmark": {
            "url": "https://github.com/openalpr/benchmarks",
            "license": "apache2",
            "approx_count": 900,
        },
        "roboflow_lp": {
            "url": "https://universe.roboflow.com/search?q=license+plate+detection",
            "license": "varies",
            "approx_count": 10000,
            "notes": "Multiple US-focused datasets; requires free API key",
        },
        "synthetic_us": {
            "url": "generated",
            "license": "public_domain",
            "notes": "Script-generated plates via Pillow — see generate_synthetic.py",
        },
    },
    "eu": {
        "eu_lp_dataset": {
            "url": "https://github.com/RobertLucian/license-plate-dataset",
            "license": "mit",
            "approx_count": 1500,
            "countries": ["DE", "FR", "IT", "ES", "NL", "PL", "RO", "CZ"],
        },
        "aolp": {
            "url": "http://aolpr.ntust.edu.tw/lab/",
            "license": "research_only",
            "approx_count": 3000,
            "notes": "Access control, traffic speed, parking; needs institutional agreement",
        },
    },
    "asia": {
        "ccpd": {
            "url": "https://github.com/detectRecog/CCPD",
            "license": "research_only",
            "approx_count": 200000,
            "notes": "Chinese plate dataset — largest available; weather/tilt variants",
        },
        "clpd": {
            "url": "https://github.com/wulabthu/CLPD",
            "license": "research_only",
            "approx_count": 1200,
            "notes": "Chinese LP detection benchmark",
        },
        "kplatech": {
            "url": "https://github.com/qjadud1994/CRNN-Keras",
            "license": "mit",
            "notes": "Korean plates, small subset; full dataset via author request",
        },
    },
    "latam": {
        "ssig_segplate": {
            "url": "https://github.com/raysonlaroca/ufpr-alpr-dataset",
            "license": "research_only",
            "countries": ["BR"],
            "approx_count": 2000,
        },
    },
}

AUGMENTATION_PARAMS = {
    "brightness_range": (0.5, 1.5),
    "contrast_range": (0.7, 1.3),
    "blur_kernels": [0, 3, 5],           # 0 = no blur
    "noise_stddev": [0, 5, 15],           # Gaussian noise
    "rotation_degrees": (-5, 5),
    "perspective_strength": 0.05,
    "weather_effects": ["none", "rain", "fog", "glare"],
    "occlusion_prob": 0.15,
    "night_mode_prob": 0.2,
    "augment_factor": 4,                 # each real image → N augmented copies
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PlateAnnotation:
    """Single plate annotation in YOLO + character format."""
    image_path: str
    bbox: Tuple[float, float, float, float]  # cx, cy, w, h (normalised)
    text: str                                  # e.g. "ABC1234"
    country: str
    region: Optional[str] = None
    confidence: float = 1.0
    source_dataset: str = ""
    split: str = "train"                       # train | val | test

    def to_yolo_line(self) -> str:
        cx, cy, w, h = self.bbox
        return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "bbox": list(self.bbox),
            "text": self.text,
            "country": self.country,
            "region": self.region,
            "confidence": self.confidence,
            "source": self.source_dataset,
            "split": self.split,
        }


@dataclass
class DatasetStats:
    total_images: int = 0
    by_country: dict = field(default_factory=dict)
    by_split: dict = field(default_factory=dict)
    avg_plate_area_ratio: float = 0.0
    duplicate_hashes_removed: int = 0
    low_quality_removed: int = 0
    augmented_added: int = 0

    def report(self) -> str:
        lines = [
            f"Total images : {self.total_images:,}",
            f"Train/Val/Test: {self.by_split.get('train',0):,} / "
            f"{self.by_split.get('val',0):,} / {self.by_split.get('test',0):,}",
            f"Countries    : {len(self.by_country)}",
            f"Duplicates   : -{self.duplicate_hashes_removed:,}",
            f"Low quality  : -{self.low_quality_removed:,}",
            f"Augmented +  : +{self.augmented_added:,}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core pipeline steps
# ---------------------------------------------------------------------------

class DatasetPipeline:
    """
    Full dataset preparation pipeline:
        1. Inventory raw data
        2. Deduplicate by image hash
        3. Quality filter (blurriness, min plate size)
        4. Normalise annotations to YOLO format
        5. Augment training split
        6. Create train/val/test splits
        7. Write DVC-compatible manifests
    """

    def __init__(self, raw_dir: Path, output_dir: Path, config: dict):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.config = config
        self.stats = DatasetStats()
        self.annotations: List[PlateAnnotation] = []

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (output_dir / "images" / "test").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / "test").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load raw annotations (supports YOLO, VOC-XML, JSON)
    # ------------------------------------------------------------------

    def load_annotations(self, source_dir: Path, country: str, fmt: str = "yolo"):
        """Parse raw annotations from a source directory."""
        image_files = list(source_dir.glob("**/*.jpg")) + list(source_dir.glob("**/*.png"))
        print(f"  Loading {len(image_files)} images from {source_dir.name} ({country})")

        for img_path in tqdm(image_files, desc=f"  Parsing {country}"):
            ann = self._parse_annotation(img_path, country, fmt)
            if ann:
                self.annotations.append(ann)

        self.stats.total_images = len(self.annotations)

    def _parse_annotation(self, img_path: Path, country: str, fmt: str) -> Optional[PlateAnnotation]:
        """Parse a single annotation file."""
        if fmt == "yolo":
            label_path = img_path.parent.parent / "labels" / img_path.with_suffix(".txt").name
            if not label_path.exists():
                return None
            with open(label_path) as f:
                line = f.readline().strip().split()
            if len(line) < 5:
                return None
            _, cx, cy, w, h = map(float, line[:5])
            text = line[5] if len(line) > 5 else "UNKNOWN"
            return PlateAnnotation(
                image_path=str(img_path),
                bbox=(cx, cy, w, h),
                text=text,
                country=country,
                source_dataset=img_path.parent.name,
            )
        # Extend here for VOC XML, COCO JSON, etc.
        return None

    # ------------------------------------------------------------------
    # Step 2: Deduplication via perceptual hash
    # ------------------------------------------------------------------

    def deduplicate(self):
        """Remove exact and near-duplicate images."""
        seen_hashes = set()
        unique = []
        removed = 0

        for ann in tqdm(self.annotations, desc="  Deduplicating"):
            img = cv2.imread(ann.image_path)
            if img is None:
                removed += 1
                continue
            # Use 8x8 average hash (fast, robust to minor JPEG artefacts)
            small = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            h = hashlib.md5(gray.tobytes()).hexdigest()
            if h in seen_hashes:
                removed += 1
            else:
                seen_hashes.add(h)
                unique.append(ann)

        self.stats.duplicate_hashes_removed = removed
        self.annotations = unique
        print(f"  Removed {removed} duplicates → {len(unique)} remain")

    # ------------------------------------------------------------------
    # Step 3: Quality filter
    # ------------------------------------------------------------------

    def quality_filter(self, min_plate_area: float = 0.002, max_blur: float = 100.0):
        """
        Filter out:
        - Plates that are too small (plate area / image area < min_plate_area)
        - Images that are too blurry (Laplacian variance < max_blur)
        """
        good = []
        removed = 0

        for ann in tqdm(self.annotations, desc="  Quality filtering"):
            img = cv2.imread(ann.image_path)
            if img is None:
                removed += 1
                continue

            # Blurriness
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score < max_blur:
                removed += 1
                continue

            # Plate area check
            _, _, w, h = ann.bbox
            if w * h < min_plate_area:
                removed += 1
                continue

            good.append(ann)

        self.stats.low_quality_removed = removed
        self.annotations = good
        print(f"  Removed {removed} low-quality → {len(good)} remain")

    # ------------------------------------------------------------------
    # Step 4: Augmentation
    # ------------------------------------------------------------------

    def augment(self, factor: int = 4):
        """Apply augmentations to training images."""
        aug_params = self.config.get("augmentation", AUGMENTATION_PARAMS)
        new_annotations = []

        train_anns = [a for a in self.annotations if a.split == "train"]
        print(f"  Augmenting {len(train_anns)} training images × {factor}")

        for ann in tqdm(train_anns, desc="  Augmenting"):
            img = cv2.imread(ann.image_path)
            if img is None:
                continue

            for i in range(factor):
                aug_img = self._apply_augmentation(img, aug_params)
                stem = Path(ann.image_path).stem
                out_path = self.output_dir / "images" / "train" / f"{stem}_aug{i}.jpg"
                cv2.imwrite(str(out_path), aug_img)

                new_ann = PlateAnnotation(
                    image_path=str(out_path),
                    bbox=ann.bbox,
                    text=ann.text,
                    country=ann.country,
                    source_dataset=ann.source_dataset + "_aug",
                    split="train",
                )
                new_annotations.append(new_ann)

        self.stats.augmented_added = len(new_annotations)
        self.annotations.extend(new_annotations)

    def _apply_augmentation(self, img: np.ndarray, params: dict) -> np.ndarray:
        """Apply random combination of augmentations."""
        result = img.copy()

        # Brightness / contrast
        alpha = random.uniform(*params["contrast_range"])
        beta = random.uniform(*params["brightness_range"]) * 30 - 15
        result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)

        # Gaussian blur
        k = random.choice(params["blur_kernels"])
        if k > 0:
            result = cv2.GaussianBlur(result, (k, k), 0)

        # Gaussian noise
        std = random.choice(params["noise_stddev"])
        if std > 0:
            noise = np.random.normal(0, std, result.shape).astype(np.int16)
            result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Rotation
        angle = random.uniform(*params["rotation_degrees"])
        h, w = result.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # Night mode simulation
        if random.random() < params.get("night_mode_prob", 0.2):
            result = (result * 0.3).astype(np.uint8)
            # Add random bright spots (headlights)
            for _ in range(random.randint(0, 3)):
                cx, cy = random.randint(0, w), random.randint(0, h)
                cv2.circle(result, (cx, cy), random.randint(10, 40),
                           (200, 200, 200), -1)

        return result

    # ------------------------------------------------------------------
    # Step 5: Train/Val/Test split
    # ------------------------------------------------------------------

    def split(self, ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
        """Assign split labels and copy files to output directories."""
        train_r, val_r, _ = ratios
        indices = list(range(len(self.annotations)))
        random.shuffle(indices)

        n = len(indices)
        train_end = int(n * train_r)
        val_end = int(n * (train_r + val_r))

        for i, idx in enumerate(indices):
            if i < train_end:
                self.annotations[idx].split = "train"
            elif i < val_end:
                self.annotations[idx].split = "val"
            else:
                self.annotations[idx].split = "test"

        self.stats.by_split = {
            "train": train_end,
            "val": val_end - train_end,
            "test": n - val_end,
        }

    # ------------------------------------------------------------------
    # Step 6: Write output files
    # ------------------------------------------------------------------

    def write(self):
        """Write YOLO labels, metadata JSON, and DVC manifest."""
        manifest = []

        for ann in tqdm(self.annotations, desc="  Writing dataset"):
            # Copy image
            dst_img = self.output_dir / "images" / ann.split / Path(ann.image_path).name
            if not dst_img.exists() and Path(ann.image_path).exists():
                shutil.copy2(ann.image_path, dst_img)

            # Write YOLO label
            dst_label = self.output_dir / "labels" / ann.split / (Path(ann.image_path).stem + ".txt")
            with open(dst_label, "w") as f:
                f.write(ann.to_yolo_line() + "\n")

            manifest.append(ann.to_dict())

        # Dataset YAML (used by training scripts)
        dataset_yaml = {
            "path": str(self.output_dir),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": 1,
            "names": ["licence_plate"],
        }
        with open(self.output_dir / "dataset.yaml", "w") as f:
            yaml.dump(dataset_yaml, f)

        # Full annotation manifest (for MLflow + drift detection)
        with open(self.output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # Country distribution
        country_dist = {}
        for ann in self.annotations:
            country_dist[ann.country] = country_dist.get(ann.country, 0) + 1
        self.stats.by_country = country_dist

        with open(self.output_dir / "stats.json", "w") as f:
            json.dump({
                "total": self.stats.total_images,
                "by_split": self.stats.by_split,
                "by_country": self.stats.by_country,
                "duplicates_removed": self.stats.duplicate_hashes_removed,
                "low_quality_removed": self.stats.low_quality_removed,
                "augmented_added": self.stats.augmented_added,
            }, f, indent=2)

        print("\n" + "=" * 50)
        print("Dataset ready!")
        print(self.stats.report())
        print(f"Output: {self.output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Prepare LPR dataset")
    p.add_argument("--regions", nargs="+", default=["us"],
                   choices=["us", "eu", "asia", "latam", "all"],
                   help="Which regional datasets to include")
    p.add_argument("--raw-dir", default="data/raw", help="Raw dataset root")
    p.add_argument("--synthetic", default="data/raw/synthetic",
                   help="Path to synthetic dataset folder generated by generate_synthetic.py "
                        "(default: data/raw/synthetic). Pass --no-synthetic to skip.")
    p.add_argument("--no-synthetic", dest="synthetic", action="store_false",
                   help="Skip the synthetic dataset even if the folder exists")
    p.add_argument("--output-dir", default="data/processed", help="Processed output")
    p.add_argument("--split", default="80/10/10",
                   help="Train/val/test split ratio (default: 80/10/10)")
    p.add_argument("--augment", action="store_true", help="Apply augmentations")
    p.add_argument("--augment-factor", type=int, default=4,
                   help="Augmentation multiplier per image")
    p.add_argument("--version", default="v1.0", help="Dataset version tag")
    p.add_argument("--config", default="configs/data.yaml", help="Dataset config YAML")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Parse split ratios
    parts = [float(x) for x in args.split.split("/")]
    total = sum(parts)
    ratios = tuple(p / total for p in parts)

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir) / args.version

    # Load config
    config = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)

    pipeline = DatasetPipeline(raw_dir, output_dir, config)

    # -----------------------------------------------------------------------
    # Load synthetic data first (generated by generate_synthetic.py)
    # -----------------------------------------------------------------------
    if args.synthetic:
        synthetic_dir = Path(args.synthetic)
        if synthetic_dir.exists():
            pipeline.load_annotations(synthetic_dir, country="SYNTHETIC", fmt="yolo")
            print(f"  Loaded synthetic data from {synthetic_dir}")
        else:
            print(f"  [skip] Synthetic dir not found: {synthetic_dir}")
            print(f"         Run: python scripts/generate_synthetic.py --count 5000 --regions us eu asia")

    # -----------------------------------------------------------------------
    # Load real regional datasets (downloaded separately)
    # -----------------------------------------------------------------------
    regions = list(DATASET_REGISTRY.keys()) if "all" in args.regions else args.regions
    for region in regions:
        region_dir = raw_dir / region
        if region_dir.exists():
            pipeline.load_annotations(region_dir, country=region.upper())
        else:
            print(f"  [skip] {region_dir} not found — download first or mount dataset")

    if not pipeline.annotations:
        print("\nNo annotations loaded. To get started:")
        print("  1. Generate synthetic data:")
        print("       python scripts/generate_synthetic.py --count 5000 --regions us eu asia")
        print("  2. Then re-run this script:")
        print("       python scripts/prepare_dataset.py --regions us eu asia --split 80/10/10")
        print("\n  Or download real datasets:")
        for region in regions:
            for name, info in DATASET_REGISTRY.get(region, {}).items():
                print(f"  [{region}] {name}: {info['url']}")
        return

    print("\n[1/5] Deduplicating...")
    pipeline.deduplicate()

    print("\n[2/5] Quality filtering...")
    pipeline.quality_filter()

    print("\n[3/5] Splitting...")
    pipeline.split(ratios=ratios)

    if args.augment:
        print("\n[4/5] Augmenting...")
        pipeline.augment(factor=args.augment_factor)
    else:
        print("\n[4/5] Skipping augmentation (pass --augment to enable)")

    print("\n[5/5] Writing output...")
    pipeline.write()

    print(f"\nDataset version {args.version} ready.")
    print("Next step: python scripts/train.py --backbone resnet50 "
          f"--data {output_dir}/dataset.yaml --track-energy")


if __name__ == "__main__":
    main()
