#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a CN/EN mixed dataset into ImageNet (ImageFolder) format with splits.
This version:
- Merges "窗贴贴画" and "窗贴年画" as hard negatives of "中国剪纸"
- Renames the slug of 中国剪纸 from "paper_cut" to "cn_paper_cut"

Outputs:
  dst/
    train/<class_slug>/*.jpg
    val/<class_slug>/*.jpg
    test/<class_slug>/*.jpg
    metadata.csv
    class_index.json
    README_imagenet_conversion.md

Usage:
  python convert_to_imagenet.py --src /path/to/src --dst /path/to/dst --split 0.7 0.15 0.15 --seed 42 --mode copy
Modes:
  copy (default) | hardlink | symlink
"""
import argparse, os, re, shutil, random, json, csv
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Chinese -> slug mapping
SLUG_MAP = {
    "狗": "dog",
    "狼": "wolf",
    "篮球": "basketball",

    "猫": "cat",
    "猞猁": "lynx",
    "飞机": "airplane",

    "自行车": "bicycle",
    "电瓶车": "e_bike",
    "电动车": "e_bike",
    "卡车": "truck",

    # Bins
    "中国垃圾桶": "cn_trash_bins",
    "中国式垃圾分类桶": "cn_trash_bins",
    "中国式垃圾分类桶（四色）": "cn_trash_bins",
    "中国式四色垃圾桶": "cn_trash_bins",  # alias requested earlier
    "国外垃圾桶": "foreign_bins",
    "公共垃圾桶": "foreign_bins",
    "建筑": "building",

    # Paper cut & stickers
    "中国剪纸": "cn_paper_cut",           # renamed slug
    "剪纸": "cn_paper_cut",
    "窗贴贴画": "window_sticker",
    "窗贴年画": "window_sticker",
    "窗贴": "window_sticker",
    "贴画": "window_sticker",
    "年画": "window_sticker",
    "汉堡": "hamburger",
}

# Positive-hard-random pairing (by Chinese names)
PAIRING_ZH = {
    "狗": {"hard": ["狼"], "random": ["篮球"]},
    "猫": {"hard": ["猞猁"], "random": ["飞机"]},
    "自行车": {"hard": ["电瓶车"], "random": ["卡车"]},
    "中国垃圾桶": {"hard": ["国外垃圾桶"], "random": ["建筑"]},
    "中国式垃圾分类桶": {"hard": ["国外垃圾桶"], "random": ["建筑"]},
    "中国式垃圾分类桶（四色）": {"hard": ["国外垃圾桶"], "random": ["建筑"]},
    "中国式四色垃圾桶": {"hard": ["国外垃圾桶"], "random": ["建筑"]},
    # Merge both sticker variants as hard for 中国剪纸
    "中国剪纸": {"hard": ["窗贴贴画", "窗贴年画"], "random": ["汉堡"]},
}

# Which are positives (Chinese names)
POSITIVES_ZH = {"狗", "猫", "自行车", "中国垃圾桶", "中国剪纸", "中国式垃圾分类桶", "中国式垃圾分类桶（四色）", "中国式四色垃圾桶"}

# Parse difficulty from folder names like "猫（常规）", "猫(细粒度)"
DIFF_PAT = re.compile(r'^(?P<name>.+?)\s*[\(（]?(?P<diff>常规|细粒度)?[\)）]?$')

def slugify_zh(name: str) -> str:
    name = name.strip()
    if name in SLUG_MAP:
        return SLUG_MAP[name]
    # Fallback ascii-safe slug
    safe = (
        name.replace("（", "(").replace("）", ")")
            .replace(" ", "_").replace("/", "_").replace("\\", "_")
    )
    return safe

def parse_folder_name(name: str):
    """Return (base_class_zh, difficulty or None)."""
    m = DIFF_PAT.match(name.strip())
    if not m:
        return name.strip(), None
    base = m.group("name").strip()
    diff = m.group("diff")
    if diff:
        diff = "regular" if diff == "常规" else "fine"
    return base, diff

def gather_images(src: Path):
    """Collect image paths with class and difficulty parsed from folder names."""
    records = []
    for group in sorted([p for p in src.iterdir() if p.is_dir()]):
        for cls_dir in sorted([p for p in group.iterdir() if p.is_dir()]):
            base_zh, diff = parse_folder_name(cls_dir.name)
            for img in cls_dir.rglob("*"):
                if img.suffix.lower() in IMG_EXTS and img.is_file():
                    records.append({
                        "abs_path": str(img.resolve()),
                        "group": group.name,
                        "class_zh": base_zh,
                        "difficulty": diff or "regular"
                    })
    return records

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def transfer_file(src: Path, dst: Path, mode="copy"):
    ensure_dir(dst.parent)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
    elif mode == "symlink":
        if dst.exists():
            dst.unlink()
        os.symlink(src, dst)
    else:
        raise ValueError("mode must be copy|hardlink|symlink")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source root (current structure)")
    ap.add_argument("--dst", required=True, help="destination root (ImageNet format)")
    ap.add_argument("--split", nargs=3, type=float, default=[0.7, 0.15, 0.15], help="train/val/test ratios")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["copy","hardlink","symlink"], default="copy")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    ensure_dir(dst)

    records = gather_images(src)
    if not records:
        raise SystemExit(f"No images found under: {src}")

    # Build class set and map to slugs
    class_zh_set = sorted(set(r["class_zh"] for r in records))
    class_slug_map = {zh: slugify_zh(zh) for zh in class_zh_set}

    # Class index (sorted by slug for stability)
    slugs_sorted = sorted(set(class_slug_map.values()))
    class_index = {slug: i for i, slug in enumerate(slugs_sorted)}

    # Per-image semantics
    meta = []
    by_class = {}
    for r in records:
        zh = r["class_zh"]
        slug = class_slug_map[zh]
        is_positive = 1 if zh in POSITIVES_ZH else 0

        if is_positive:
            pair_zh = zh
            neg_type = ""
        else:
            pair_zh = zh
            neg_type = ""
            # check mapping to a positive via pairing
            found = False
            for pos_zh, conf in PAIRING_ZH.items():
                if zh in conf.get("hard", []):
                    pair_zh = pos_zh; neg_type = "hard"; found = True; break
                if zh in conf.get("random", []):
                    pair_zh = pos_zh; neg_type = "random"; found = True; break
            if not found:
                pair_zh = zh

        pair_slug = slugify_zh(pair_zh)

        item = {
            "abs_path": r["abs_path"],
            "class_zh": zh,
            "class_slug": slug,
            "is_positive": is_positive,
            "pair_class_zh": pair_zh,
            "pair_class_slug": pair_slug,
            "neg_type": neg_type,
            "difficulty": r["difficulty"],
        }
        meta.append(item)
        by_class.setdefault(slug, []).append(item)

    # Stratified split by class
    rng = random.Random(args.seed)
    for slug, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * args.split[0])
        n_val = int(n * args.split[1])
        for i, it in enumerate(items):
            if i < n_train:
                it["split"] = "train"
            elif i < n_train + n_val:
                it["split"] = "val"
            else:
                it["split"] = "test"

    # Transfer files
    for it in meta:
        src_path = Path(it["abs_path"])
        dst_path = dst / it["split"] / it["class_slug"] / src_path.name
        transfer_file(src_path, dst_path, mode=args.mode)
        it["rel_path"] = str(dst_path.relative_to(dst))

    # Save metadata CSV
    fields = ["rel_path","split","class_slug","class_zh","is_positive","pair_class_slug","pair_class_zh","neg_type","difficulty"]
    with open(dst / "metadata.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for it in meta: w.writerow({k: it.get(k, "") for k in fields})

    # Save class index
    with open(dst / "class_index.json", "w", encoding="utf-8") as f:
        json.dump(class_index, f, ensure_ascii=False, indent=2)

    # README
    readme = f"""# ImageNet (ImageFolder) Conversion

This dataset was converted from a CN/EN mixed structure.

- 中国剪纸 slug: **cn_paper_cut**
- Hard negatives for 中国剪纸: **窗贴贴画**, **窗贴年画**
- Chinese bins slugs: **cn_trash_bins** (aliases supported)

Structure:
dst/
  train/{{class_name}}/...
  val/{{class_name}}/...
  test/{{class_name}}/...
  metadata.csv
  class_index.json

Example (PyTorch):
```python
from torchvision import datasets, transforms
root = r"{dst.as_posix()}"
train_ds = datasets.ImageFolder(root + "/train", transform=transforms.ToTensor())
print(train_ds.class_to_idx)
```
"""
    with open(dst / "README_imagenet_conversion.md", "w", encoding="utf-8") as f:
        f.write(readme)

    print(f"Done. Converted to ImageNet format at: {dst}")
    print(f"- metadata: {dst/'metadata.csv'}")
    print(f"- class_index.json: {dst/'class_index.json'}")

if __name__ == "__main__":
    main()
