import os
import time
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from models_nextgen import MSVMUNetNextGen
from models.unet import UNet
from metrics import compute_metrics
from dataset import get_dataloaders
from losses import HybridLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_BASE = os.path.join(BASE_DIR, "data", "PolyDataset")
if not os.path.isdir(DATA_BASE) and os.path.isdir("/root/autodl-tmp/polyp_mamba_project/data/PolyDataset"):
    DATA_BASE = "/root/autodl-tmp/polyp_mamba_project/data/PolyDataset"

RESULTS_DIR = os.path.join(BASE_DIR, "results")
if not os.path.isdir(RESULTS_DIR) and os.path.isdir("/root/autodl-tmp/polyp_mamba_project/results"):
    RESULTS_DIR = "/root/autodl-tmp/polyp_mamba_project/results"

class ExternalPolyDataset(Dataset):
    def __init__(self, ds_name, img_size=256):
        self.img_dir = os.path.join(DATA_BASE, ds_name, "images")
        self.mask_dir = os.path.join(DATA_BASE, ds_name, "masks")
        self.img_size = img_size
        self.names = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img_p = os.path.join(self.img_dir, name)
        mask_p = os.path.join(self.mask_dir, name)

        img = Image.open(img_p).convert("RGB")
        mask = Image.open(mask_p).convert("L")

        img = TF.resize(img, [self.img_size, self.img_size])
        mask = TF.resize(mask, [self.img_size, self.img_size], interpolation=T.InterpolationMode.NEAREST)

        img = TF.to_tensor(img)
        img = self.normalize(img)

        mask_np = (np.array(mask, dtype=np.float32) > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

        return {"image": img, "mask": mask_tensor, "name": name}

def train_unet_seed42_if_needed():
    unet_ckpt = os.path.join(RESULTS_DIR, "unet_seed_42", "best_model.pth")
    if os.path.exists(unet_ckpt):
        print(f"[FOUND] U-Net Seed 42 checkpoint exists: {unet_ckpt}")
        return unet_ckpt

    print("\n--- Training Standard U-Net on Seed 42 (60 Epochs, ~5.5 min) ---")
    os.makedirs(os.path.join(RESULTS_DIR, "unet_seed_42"), exist_ok=True)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=16, num_workers=4, img_size=256)
    model = UNet().to(device)
    criterion = HybridLoss(use_tv=False, use_sobel=False)
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-6)

    best_val_dice = 0.0
    for epoch in range(1, 61):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Val
        model.eval()
        val_dices = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                masks = batch["mask"].cpu().numpy()
                preds = (torch.sigmoid(model(imgs)) > 0.5).cpu().numpy()
                for i in range(len(imgs)):
                    inter = (preds[i, 0] * masks[i, 0]).sum()
                    union = preds[i, 0].sum() + masks[i, 0].sum()
                    d = 2.0 * inter / (union + 1e-7) if union > 0 else 1.0
                    val_dices.append(d)
        m_dice = np.mean(val_dices)
        if m_dice > best_val_dice:
            best_val_dice = m_dice
            torch.save(model.state_dict(), unet_ckpt)
        if epoch % 15 == 0 or epoch == 60:
            print(f"  U-Net Seed 42 Epoch [{epoch:02d}/60] Val Dice: {m_dice*100:.2f}% (Best: {best_val_dice*100:.2f}%)")

    return unet_ckpt

def evaluate_model_on_datasets(model, model_name):
    datasets = ["CVC-ClinicDB", "CVC-ColonDB", "ETIS"]
    results = []
    model.eval()

    for ds_name in datasets:
        ds = ExternalPolyDataset(ds_name)
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
        all_metrics = []

        with torch.no_grad():
            for batch in loader:
                imgs = batch["image"].to(device)
                masks = batch["mask"].cpu().numpy()
                preds = torch.sigmoid(model(imgs)).cpu().numpy()

                for i in range(len(imgs)):
                    m = compute_metrics(preds[i, 0], masks[i, 0])
                    all_metrics.append(m)

        res = {
            "dataset": ds_name,
            "sample_count": len(ds),
            "model": model_name,
            "dice": round(float(np.mean([m["dice"] for m in all_metrics])), 4),
            "iou": round(float(np.mean([m["iou"] for m in all_metrics])), 4),
            "hd95": round(float(np.mean([m["hd95"] for m in all_metrics])), 2),
            "sensitivity": round(float(np.mean([m["sensitivity"] for m in all_metrics])), 4),
            "specificity": round(float(np.mean([m["specificity"] for m in all_metrics])), 4),
        }
        results.append(res)
        print(f"  [{model_name} on {ds_name:<12}] Dice: {res['dice']*100:.2f}% | IoU: {res['iou']*100:.2f}% | HD95: {res['hd95']:.2f}px | Sens: {res['sensitivity']*100:.2f}% | Spec: {res['specificity']*100:.2f}%")

    return results

def run_unified_cross_eval():
    print("=" * 80)
    print("TASK 2: UNIFIED RE-EVALUATION ON 3 EXTERNAL DATASETS (CLINICDB, COLONDB, ETIS)")
    print("=" * 80)

    # 1. Evaluate Standard U-Net (Seed 42)
    unet_ckpt = train_unet_seed42_if_needed()
    unet_model = UNet().to(device)
    unet_model.load_state_dict(torch.load(unet_ckpt, map_location=device), strict=False)
    print("\n--- Evaluating Standard U-Net (Seed 42) across External Datasets ---")
    unet_res = evaluate_model_on_datasets(unet_model, "Standard U-Net")

    # 2. Evaluate MS-VMUNet-Lite (Seed 42)
    candidate_ckpts = [
        os.path.join(RESULTS_DIR, "step2_nextgen_full_seed_42", "best_model.pth"),
        os.path.join(RESULTS_DIR, "nextgen_full_seed_42", "best_model.pth"),
        os.path.join(BASE_DIR, "checkpoints", "nextgen_full_best_seed42.pth"),
        "checkpoints/nextgen_full_best_seed42.pth",
    ]
    lite_ckpt = None
    for ckpt in candidate_ckpts:
        if os.path.exists(ckpt):
            lite_ckpt = ckpt
            break
    if lite_ckpt is None:
        raise FileNotFoundError(f"MS-VMUNet Seed 42 checkpoint not found in: {candidate_ckpts}")
    lite_model = MSVMUNetNextGen(use_4way=True, use_msconv=True, skip_type="hfd").to(device)
    lite_model.load_state_dict(torch.load(lite_ckpt, map_location=device), strict=False)
    print("\n--- Evaluating MS-VMUNet-Lite (Seed 42) across External Datasets ---")
    lite_res = evaluate_model_on_datasets(lite_model, "MS-VMUNet-Lite [Ours]")

    all_data = unet_res + lite_res
    df = pd.DataFrame(all_data)
    out_csv = "cross_dataset_unified_results.csv"
    df.to_csv(out_csv, index=False)
    df.to_csv(os.path.join(RESULTS_DIR, "cross_dataset_unified_results.csv"), index=False)

    print("\n" + "=" * 80)
    print("CONSOLIDATED REPRODUCIBLE ZERO-SHOT GENERALIZATION SUMMARY (CSV SAVED)")
    print("=" * 80)
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_unified_cross_eval()
