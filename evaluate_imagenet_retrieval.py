# Create a fresh, end-to-end script per the user's latest spec.
from pathlib import Path

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate zero-shot text->image retrieval on an ImageNet-format dataset
for four backbones, using F1 as the primary metric.

Backbones (select with --backbone):
  - openai-clip   : OpenAI CLIP (e.g., ViT-B/32), English prompts only
  - siglip        : google/siglip-*, English prompts only
  - chinese-clip  : OFA-Sys/chinese-clip-*, Single Chinese prompt per class
  - taiyi-clip    : IDEA-CCNL/Taiyi-CLIP-*,  Single Chinese prompt per class

Protocol:
  - When --split val: sweep thresholds & topK per class, save the best hyperparams.
  - When --split test: auto-load the saved hyperparams from default locations
    (unless user provides --fixed_threshold_csv / --fixed_topk_csv to override).

Default locations (relative to CWD, i.e., wd):
  cache_dir  = ./eval_cache
  out_dir    = ./eval_runs
  threshold file (auto-saved from val):  {out_dir}/best_thresholds_{tag}_val.csv
  topK file (auto-saved from val):       {out_dir}/best_topk_{tag}_val.csv

Notes:
  - We L2-normalize both text and image embeddings; similarity is dot-product=cosine.
  - For OpenMP issues on Windows, we set env flags to allow duplicate libiomp (safe here).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse, json, warnings
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="Corrupt EXIF data.")
warnings.filterwarnings("ignore", message="Palette images with Transparency")

# -------------------- Prompt templates --------------------
# Known classes can use curated prompts; otherwise fallback rules apply
PROMPTS_EN = {
    "cat": [
        "a photo of a domestic cat"
    ],
    "dog": [
        "a photo of a domestic dog"
    ],
    "bicycle": [
        "a photo of a bicycle"
    ],
    "cn_trash_bins": [
        "a Chinese trash-sorting bin with four categories"
    ],
    "cn_paper_cut": [
        "Chinese paper-cut decoration on a window"
    ],
}

PROMPT_ZH_SINGLE = {
    "cat": "猫，一只家猫的照片",
    "dog": "狗，一只家犬的照片",
    "bicycle": "自行车的照片",
    "cn_trash_bins": "中国式垃圾分类桶，四色分类",
    "cn_paper_cut": "中国剪纸，窗花装饰",
}

def english_fallback(slug: str) -> List[str]:
    text = slug.replace("_", " ")
    return [f"a photo of {text}"]

def chinese_fallback(slug: str) -> str:
    # 简单回退：直接使用 slug（如果是 cn_* 则已有映射；否则中文用户可自行补充）
    return slug

# -------------------- Dataset --------------------
def load_dataset(root: Path, split: str):
    from torchvision.transforms import InterpolationMode
    preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])
    ds = datasets.ImageFolder(str(root / split), transform=preprocess)
    return ds

# -------------------- Backbone wrappers --------------------
class Backbone:
    def __init__(self, device: str):
        self.device = device
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    def encode_texts(self, prompts: List[str]) -> torch.Tensor:
        raise NotImplementedError
    @property
    def cache_tag(self) -> str:
        return "backbone"

class OpenAIClip(Backbone):
    def __init__(self, model_name: str, device: str):
        super().__init__(device)
        import clip
        self.model, _ = clip.load(model_name, device=device, jit=False)
        self.tokenize = clip.tokenize
        self.model.eval()
        self._tag = f"openai-clip_{model_name.replace('/', '-')}"

    def encode_images(self, images):
        with torch.no_grad():
            feats = self.model.encode_image(images.to(self.device))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats

    def encode_texts(self, prompts):
        with torch.no_grad():
            toks = self.tokenize(prompts).to(self.device)
            feats = self.model.encode_text(toks)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            mean = feats.mean(dim=0, keepdim=True)
            return mean / mean.norm(dim=-1, keepdim=True)

    @property
    def cache_tag(self):
        return self._tag

class TaiyiClip(Backbone):
    def __init__(self, model_name: str, device: str):
        super().__init__(device)
        from transformers import AutoProcessor, CLIPModel
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = CLIPModel.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
        self.model.eval()
        self._tag = f"taiyi-clip_{model_name.split('/')[-1]}"

    def encode_images(self, images):
        from torchvision.transforms.functional import to_pil_image
        imgs = [to_pil_image(img.cpu()) for img in images]
        inputs = self.processor(images=imgs, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats

    def encode_texts(self, prompts):
        # prompts should contain ONE Chinese sentence
        text = prompts if isinstance(prompts, list) else [prompts]
        inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            mean = feats.mean(dim=0, keepdim=True)
            return mean / mean.norm(dim=-1, keepdim=True)

    @property
    def cache_tag(self):
        return self._tag

class ChineseClip(Backbone):
    def __init__(self, model_name: str, device: str):
        super().__init__(device)
        try:
            from transformers import ChineseCLIPModel, ChineseCLIPProcessor
            self.processor = ChineseCLIPProcessor.from_pretrained(model_name)
            self.model = ChineseCLIPModel.from_pretrained(model_name).to(device)
            self.model.eval()
        except Exception:
            from transformers import AutoProcessor, AutoModel
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.model.eval()
        self._tag = f"chinese-clip_{model_name.split('/')[-1]}"

    def encode_images(self, images):
        from torchvision.transforms.functional import to_pil_image
        imgs = [to_pil_image(img.cpu()) for img in images]
        inputs = self.processor(images=imgs, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            if hasattr(self.model, "get_image_features"):
                feats = self.model.get_image_features(**inputs)
            else:
                out = self.model(**inputs)
                feats = out.image_embeds if hasattr(out, "image_embeds") else out.last_hidden_state.mean(dim=1)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats

    def encode_texts(self, prompts):
        text = prompts if isinstance(prompts, list) else [prompts]
        inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            if hasattr(self.model, "get_text_features"):
                feats = self.model.get_text_features(**inputs)
            else:
                out = self.model(**inputs)
                feats = out.text_embeds if hasattr(out, "text_embeds") else out.last_hidden_state.mean(dim=1)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            mean = feats.mean(dim=0, keepdim=True)
            return mean / mean.norm(dim=-1, keepdim=True)

    @property
    def cache_tag(self):
        return self._tag

class SigLIP(Backbone):
    def __init__(self, model_name: str, device: str):
        super().__init__(device)
        from transformers import AutoProcessor, SiglipModel
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = SiglipModel.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
        self.model.eval()
        self._tag = f"siglip_{model_name.split('/')[-1]}"

    def encode_images(self, images):
        from torchvision.transforms.functional import to_pil_image
        imgs = [to_pil_image(img.cpu()) for img in images]
        inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats

    def encode_texts(self, prompts):
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            mean = feats.mean(dim=0, keepdim=True)
            return mean / mean.norm(dim=-1, keepdim=True)

    @property
    def cache_tag(self):
        return self._tag

def build_backbone(backbone: str, model_name: str, device: str) -> Backbone:
    b = backbone.lower()
    if b == "openai-clip":
        return OpenAIClip(model_name or "ViT-B/32", device)
    elif b == "siglip":
        return SigLIP(model_name or "google/siglip-base-patch16-224", device)
    elif b == "chinese-clip":
        return ChineseClip(model_name or "OFA-Sys/chinese-clip-vit-b-16", device)
    elif b == "taiyi-clip":
        return TaiyiClip(model_name or "IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese", device)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

# -------------------- Prompt selection by backbone --------------------
def get_prompts(backbone_name: str, slug: str) -> List[str]:
    b = backbone_name.lower()
    if b in ("openai-clip", "siglip"):
        if slug in PROMPTS_EN:
            return PROMPTS_EN[slug]
        return english_fallback(slug)
    else:  # chinese backbones: single sentence
        zh = PROMPT_ZH_SINGLE.get(slug, chinese_fallback(slug))
        return [zh]

# -------------------- Caching --------------------
def build_image_index(backbone: Backbone, ds, cache_dir: Path, split: str):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_prefix = f"{backbone.cache_tag}_{split}"
    embed_file = cache_dir / f"{cache_prefix}_image_embeds.pt"
    index_file = cache_dir / f"{cache_prefix}_image_index.json"
    if embed_file.exists() and index_file.exists():
        img_embeds = torch.load(embed_file, map_location="cpu")
        index = json.loads(index_file.read_text(encoding="utf-8"))
        return img_embeds, index

    img_embeds = []
    index = []
    loader = torch.utils.data.DataLoader(list(range(len(ds))), batch_size=64, shuffle=False, num_workers=0)
    with torch.no_grad():
        for idx_batch in tqdm(loader, desc=f"Embedding images ({backbone.cache_tag})"):
            imgs = torch.stack([ds[i][0] for i in idx_batch])
            labels = [ds[i][1] for i in idx_batch]
            paths = [ds.samples[i][0] for i in idx_batch]
            feats = backbone.encode_images(imgs)
            img_embeds.append(feats.cpu())
            for p, y in zip(paths, labels):
                index.append({"path": p, "class_idx": int(y)})
    img_embeds = torch.cat(img_embeds, dim=0)
    torch.save(img_embeds, embed_file)
    index_file.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return img_embeds, index

# -------------------- Metrics --------------------
def cosine_scores(text_feat: torch.Tensor, img_embeds: torch.Tensor) -> np.ndarray:
    return (img_embeds @ text_feat[0].cpu().numpy())

def metrics_from_scores(scores, y_true, threshold):
    # Normalize inputs
    scores = np.asarray(scores).ravel()
    y_true = np.asarray(y_true).ravel().astype(np.int32)

    # Basic sanity checks
    if scores.size != y_true.size:
        raise ValueError(f"Size mismatch: scores.size={scores.size}, y_true.size={y_true.size}")

    if not np.isfinite(scores).all() or np.isnan(scores).any():
        print("[WARN] scores contains NaN/inf — replacing NaN with -1.0 for safe comparisons")
        scores = np.nan_to_num(scores, nan=-1.0, posinf=1.0, neginf=-1.0)

    thr = float(threshold)

    # Predictions and counts
    y_pred = (scores >= thr).astype(np.int32)
    tp = int(np.logical_and(y_pred == 1, y_true == 1).sum())
    fp = int(np.logical_and(y_pred == 1, y_true == 0).sum())
    fn = int(np.logical_and(y_pred == 0, y_true == 1).sum())

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1, tp, fp, fn

def eval_threshold_sweep(class_slug, scores, y_true, out_csv, plot_dir=None):
    rows = []
    best = {"class": class_slug, "threshold": None, "precision": 0, "recall": 0, "f1": -1}

    # Adaptive threshold grid based on observed scores (handles small/negative similarities)
    scores_arr = np.asarray(scores).ravel()
    # sanitize
    if not np.isfinite(scores_arr).all() or np.isnan(scores_arr).any():
        print(f"[WARN] {class_slug}: scores contains NaN/inf — replacing NaN with -1.0")
        scores_arr = np.nan_to_num(scores_arr, nan=-1.0, posinf=1.0, neginf=-1.0)

    if scores_arr.size == 0:
        print(f"[WARN] {class_slug}: empty scores array")
        return best

    smin, smax = float(scores_arr.min()), float(scores_arr.max())
    # ensure a reasonable search window even when scores cluster near 0
    low = min(smin - 1e-6, -0.5)
    high = max(smax + 1e-6, 0.5)
    thr_list = np.linspace(low, high, 91)

    for thr in thr_list:
        p, r, f1, tp, fp, fn = metrics_from_scores(scores_arr, y_true, thr)
        rows.append({"class": class_slug, "threshold": round(thr, 6), "precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn})
        if f1 > best["f1"]:
            best = {"class": class_slug, "threshold": round(thr, 6), "precision": p, "recall": r, "f1": f1}

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, mode="a", index=False, header=not Path(out_csv).exists(), encoding="utf-8-sig")

    if plot_dir:
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.plot(df["recall"], df["precision"])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve - {class_slug}")
        plt.savefig(Path(plot_dir)/f"pr_curve_{class_slug}.png", dpi=160, bbox_inches="tight")
        plt.close()

    return best

def eval_topk(class_slug, scores, y_true, out_csv, K_list=(1,5,10,20,50,100)):
    order = np.argsort(-scores)
    rows = []
    best = {"class": class_slug, "K": None, "precision": 0, "recall": 0, "f1": -1}
    for K in K_list:
        idx = order[:K]
        y_pred = np.zeros_like(y_true)
        y_pred[idx] = 1
        tp = int((y_pred * y_true).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        rows.append({"class": class_slug, "K": K, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn})
        if f1 > best["f1"]:
            best = {"class": class_slug, "K": K, "precision": precision, "recall": recall, "f1": f1}
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, mode="a", index=False, header=not Path(out_csv).exists(), encoding="utf-8-sig")
    return best

# -------------------- Main --------------------
def main():
    '''
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="imagenet_dataset", help="Root containing train/val/test")
    ap.add_argument("--split", required=True, choices=["val","test"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--backbone", default="openai-clip", choices=["openai-clip","siglip","chinese-clip","taiyi-clip"])
    ap.add_argument("--model_name", default="", help="Optional override model name for the chosen backbone")
    ap.add_argument("--out_dir", default="eval_runs")
    ap.add_argument("--cache_dir", default="eval_cache")
    ap.add_argument("--plot_pr", action="store_true")
    ap.add_argument("--fixed_threshold_csv", default="", help="Optional manual path to thresholds CSV (val)")
    ap.add_argument("--fixed_topk_csv", default="", help="Optional manual path to topK CSV (val)")
    args = ap.parse_args()
    '''
    args = ["root", "imagenet_dataset", "split", "val", "device", "cuda:0", "backbone", "siglip", "model_name", "",
            "out_dir", "eval_runs", "cache_dir", "eval_cache", "plot_pr", "fixed_threshold_csv", "", "fixed_topk_csv", ""]
    args = argparse.Namespace(**{k: v for k, v in zip(args[::2], args[1::2])})
    wd = Path(".").resolve()
    root = wd / args.root
    out_dir = wd / args.out_dir
    cache_dir = wd / args.cache_dir
    out_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    backbone = build_backbone(args.backbone, args.model_name, device)
    tag = backbone.cache_tag

    # Dataset and cached embeddings
    ds = load_dataset(root, args.split)
    img_embeds, index = build_image_index(backbone, ds, cache_dir, args.split)
    img_embeds_np = img_embeds.numpy()
    img_paths = [Path(item["path"]).as_posix().replace("\\","/") for item in index]
    class_indices = [item["class_idx"] for item in index]
    slug_list = ds.classes  # folder names

    # Output files
    thr_metrics_csv = out_dir / f"metrics_threshold_{tag}_{args.split}.csv"
    topk_metrics_csv = out_dir / f"metrics_topk_{tag}_{args.split}.csv"
    best_thr_csv = out_dir / f"best_thresholds_{tag}_{args.split}.csv"
    best_k_csv = out_dir / f"best_topk_{tag}_{args.split}.csv"
    pr_plot_dir = out_dir / f"pr_plots_{tag}_{args.split}" if args.plot_pr else None

    # If test, auto-load val hyperparams unless user provides overrides
    thr_to_use = None
    k_to_use = None
    if args.split == "test":
        thr_file = Path(args.fixed_threshold_csv) if args.fixed_threshold_csv else (out_dir / f"best_thresholds_{tag}_val.csv")
        topk_file = Path(args.fixed_topk_csv) if args.fixed_topk_csv else (out_dir / f"best_topk_{tag}_val.csv")
        thr_to_use = pd.read_csv(thr_file) if thr_file.exists() else None
        k_to_use = pd.read_csv(topk_file) if topk_file.exists() else None
        if thr_to_use is None:
            print(f"[WARN] Threshold file not found: {thr_file}. Will sweep thresholds on TEST (may overfit).")
        if k_to_use is None:
            print(f"[WARN] TopK file not found: {topk_file}. Will sweep K on TEST (may overfit).")

    best_thr_rows, best_k_rows = [], []
    for class_idx, class_slug in enumerate(slug_list):
        # Text embedding
        prompts = get_prompts(args.backbone, class_slug)
        txt_feat = backbone.encode_texts(prompts)

        # Scores against all images
        scores = (img_embeds_np @ txt_feat[0].cpu().numpy())

        # Labels: positives = this class
        y_true = np.array([1 if ci == class_idx else 0 for ci in class_indices], dtype=np.int32)

        # Threshold path
        if args.split == "test" and thr_to_use is not None:
            row = thr_to_use[thr_to_use["class"] == class_slug]
            if len(row) == 1:
                thr = float(row["threshold"].values[0])
                p, r, f1, tp, fp, fn = metrics_from_scores(scores, y_true, thr)
                # append also to metrics csv (single line per class on test with fixed thr)
                pd.DataFrame([{"class": class_slug, "threshold": thr, "precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}]).to_csv(
                    thr_metrics_csv, mode="a", index=False, header=not thr_metrics_csv.exists(), encoding="utf-8-sig")
                best_thr = {"class": class_slug, "threshold": thr, "precision": p, "recall": r, "f1": f1}
            else:
                # fallback to sweep if class missing
                best_thr = eval_threshold_sweep(class_slug, scores, y_true, thr_metrics_csv, plot_dir=pr_plot_dir)
        else:
            best_thr = eval_threshold_sweep(class_slug, scores, y_true, thr_metrics_csv, plot_dir=pr_plot_dir)
        best_thr_rows.append(best_thr)

        # Top-K path
        if args.split == "test" and k_to_use is not None:
            rowk = k_to_use[k_to_use["class"] == class_slug]
            if len(rowk) == 1:
                K = int(rowk["K"].values[0])
                order = np.argsort(-scores)
                idx = order[:K]
                y_pred = np.zeros_like(y_true)
                y_pred[idx] = 1
                tp = int((y_pred * y_true).sum())
                fp = int(((y_pred == 1) & (y_true == 0)).sum())
                fn = int(((y_pred == 0) & (y_true == 1)).sum())
                precision = tp / (tp + fp + 1e-9)
                recall = tp / (tp + fn + 1e-9)
                f1 = 2 * precision * recall / (precision + recall + 1e-9)
                pd.DataFrame([{"class": class_slug, "K": K, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}]).to_csv(
                    topk_metrics_csv, mode="a", index=False, header=not topk_metrics_csv.exists(), encoding="utf-8-sig")
                best_k = {"class": class_slug, "K": K, "precision": precision, "recall": recall, "f1": f1}
            else:
                best_k = eval_topk(class_slug, scores, y_true, topk_metrics_csv)
        else:
            best_k = eval_topk(class_slug, scores, y_true, topk_metrics_csv)
        best_k_rows.append(best_k)

    # Save best for this split
    pd.DataFrame(best_thr_rows).to_csv(best_thr_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(best_k_rows).to_csv(best_k_csv, index=False, encoding="utf-8-sig")
    print("[OK] Results saved to:", out_dir)
    print("  -", best_thr_csv.name, "|", best_k_csv.name)
    if pr_plot_dir: print("  - PR plots:", pr_plot_dir)

if __name__ == "__main__":
    main()