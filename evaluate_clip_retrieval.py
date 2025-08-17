#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP Retrieval Evaluation: Threshold vs Top-K (RGB & warning-safe)
- Force images to RGB (avoid palette/EXIF warnings).
- Optional PR plots.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse, os, json, math, csv, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import clip
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# Suppress benign PIL EXIF warnings
warnings.filterwarnings("ignore", message="Corrupt EXIF data.")
warnings.filterwarnings("ignore", message="Palette images with Transparency")

PROMPTS = {
    "cat": [
        "a photo of a domestic cat",
        "a picture of a pet cat",
        "a small kitten",
        "a house cat on a couch",
        "猫，一只家猫的照片",
        "宠物猫的图片"
    ],
    "dog": [
        "a photo of a domestic dog",
        "a picture of a pet dog",
        "a dog walking in the street",
        "a golden retriever",
        "狗，一只家犬的照片",
        "宠物狗的图片"
    ],
    "bicycle": [
        "a photo of a bicycle",
        "a mountain bike on the street",
        "a road bike",
        "自行车的照片",
        "街道上的单车"
    ],
    "cn_trash_bins": [
        "a Chinese trash-sorting bin (four categories)",
        "Chinese style waste sorting bins",
        "中国式垃圾分类桶，四色分类",
        "小区里的垃圾分类桶"
    ],
    "cn_paper_cut": [
        "Chinese paper-cut decoration on window",
        "red Chinese paper cutting",
        "中国剪纸，窗花装饰",
        "红色的中国剪纸"
    ],
}

def load_dataset(root: Path, split: str):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Lambda(lambda im: im.convert("RGB")),  # ensure RGB
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])
    ds = datasets.ImageFolder(str(root / split), transform=preprocess)
    return ds

def build_image_index(model, ds, device, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    img_embeds = []
    index = []
    loader = torch.utils.data.DataLoader(list(range(len(ds))), batch_size=64, shuffle=False, num_workers=0)
    with torch.no_grad():
        for idx_batch in tqdm(loader, desc="Embedding images"):
            imgs = torch.stack([ds[i][0] for i in idx_batch])
            labels = torch.tensor([ds[i][1] for i in idx_batch])
            paths = [ds.samples[i][0] for i in idx_batch]
            imgs = imgs.to(device)
            feats = model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            img_embeds.append(feats.cpu())
            for p, y in zip(paths, labels.tolist()):
                index.append({"path": p, "class_idx": y})
    img_embeds = torch.cat(img_embeds, dim=0)
    torch.save(img_embeds, save_dir / "image_embeds.pt")
    with open(save_dir / "image_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    return img_embeds, index

def load_or_build_image_index(model, ds, device, save_dir: Path):
    embed_file = save_dir / "image_embeds.pt"
    index_file = save_dir / "image_index.json"
    if embed_file.exists() and index_file.exists():
        img_embeds = torch.load(embed_file, map_location="cpu")
        with open(index_file, "r", encoding="utf-8") as f:
            index = json.load(f)
        return img_embeds, index
    return build_image_index(model, ds, device, save_dir)

def text_embedding(model, device, prompts):
    with torch.no_grad():
        tokens = clip.tokenize(prompts).to(device)
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        mean_feat = feats.mean(dim=0, keepdim=True)
        mean_feat = mean_feat / mean_feat.norm(dim=-1, keepdim=True)
        return mean_feat

def cosine_scores(text_feat, img_embeds):
    return (img_embeds @ text_feat[0].cpu().numpy())

def metrics_from_scores(scores, y_true, threshold):
    y_pred = (scores >= threshold).astype(np.int32)
    tp = int((y_pred * y_true).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1, tp, fp, fn

def eval_threshold_sweep(class_slug, scores, y_true, out_csv, plot_dir=None):
    rows = []
    best = {"class": class_slug, "threshold": None, "precision": 0, "recall": 0, "f1": -1}
    for thr in np.linspace(0.05, 0.95, 91):
        p, r, f1, tp, fp, fn = metrics_from_scores(scores, y_true, thr)
        rows.append({"class": class_slug, "threshold": round(thr, 3), "precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn})
        if f1 > best["f1"]:
            best = {"class": class_slug, "threshold": round(thr,3), "precision": p, "recall": r, "f1": f1}
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="ViT-B/32")
    ap.add_argument("--plot_pr", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    save_dir = Path("./eval_clip")
    save_dir.mkdir(exist_ok=True)

    device = args.device if args.device != "cpu" else "cpu"
    device = device if torch.cuda.is_available() or device == "cpu" else "cpu"

    model, _ = clip.load(args.model, device=device, jit=False)
    model.eval()

    ds = load_dataset(root, args.split)
    img_embeds, index = load_or_build_image_index(model, ds, device, save_dir)

    meta_df = pd.read_csv(root/"metadata.csv")
    rel_to_pair = { (root / row.rel_path).as_posix().replace("\\","/"): (row.pair_class_slug, int(row.is_positive)) for _, row in meta_df.iterrows() }

    img_paths = [Path(item["path"]).as_posix().replace("\\","/") for item in index]
    img_embeds_np = img_embeds.numpy()

    classes_to_eval = sorted(PROMPTS.keys())
    best_thr_rows, best_k_rows = [], []
    for class_slug in classes_to_eval:
        prompts = PROMPTS[class_slug]
        txt_feat = text_embedding(model, device, prompts)
        scores = (img_embeds_np @ txt_feat[0].cpu().numpy())
        y_true = np.array([1 if (rel_to_pair.get(p, ("",0))[0] == class_slug and rel_to_pair.get(p, ("",0))[1]==1) else 0 for p in img_paths], dtype=np.int32)

        best_thr = eval_threshold_sweep(class_slug, scores, y_true, save_dir/"metrics_threshold.csv", plot_dir=(save_dir if args.plot_pr else None))
        best_k = eval_topk(class_slug, scores, y_true, save_dir/"metrics_topk.csv")

        best_thr_rows.append(best_thr)
        best_k_rows.append(best_k)

    pd.DataFrame(best_thr_rows).to_csv(save_dir/"best_threshold_per_class.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(best_k_rows).to_csv(save_dir/"best_k_per_class.csv", index=False, encoding="utf-8-sig")
    print("Done. Results saved under:", save_dir)

if __name__ == "__main__":
    main()
