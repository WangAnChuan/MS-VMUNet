import os
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from models.ms_vmunet import MSVMUNetNextGen as MSVMUNet
from models.unet import UNet
from metrics import compute_metrics
from dataset import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ExternalPolyDataset(Dataset):
    def __init__(self, data_root, ds_name, img_size=256):
        self.img_dir = os.path.join(data_root, ds_name, "images")
        self.mask_dir = os.path.join(data_root, ds_name, "masks")
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

def evaluate_checkpoint(args):
    print("=" * 80)
    print(f"EVALUATING CHECKPOINT: {args.checkpoint}")
    print(f"Model: {args.model} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)

    if args.model == "ms_vmunet":
        model = MSVMUNet(
            use_4way=not args.no_4way,
            use_msconv=not args.no_msconv,
            skip_type=args.skip_type
        ).to(device)
    else:
        model = UNet().to(device)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
    model.eval()

    # 1. Evaluate on Kvasir-SEG test set
    if args.eval_kvasir:
        print("\n--- Evaluating on Kvasir-SEG Held-Out Test Set (100 images) ---")
        _, _, test_loader = get_dataloaders(data_dir=args.kvasir_dir, batch_size=args.batch_size, img_size=args.img_size)
        all_metrics = []
        with torch.no_grad():
            for batch in test_loader:
                imgs = batch["image"].to(device)
                masks = batch["mask"].cpu().numpy()
                preds = torch.sigmoid(model(imgs)).cpu().numpy()
                for i in range(len(imgs)):
                    all_metrics.append(compute_metrics(preds[i, 0], masks[i, 0]))
        d = np.mean([m["dice"] for m in all_metrics]) * 100
        iou = np.mean([m["iou"] for m in all_metrics]) * 100
        hd = np.mean([m["hd95"] for m in all_metrics])
        sens = np.mean([m["sensitivity"] for m in all_metrics]) * 100
        spec = np.mean([m["specificity"] for m in all_metrics]) * 100
        print(f"  Test Dice: {d:.2f}% | IoU: {iou:.2f}% | HD95: {hd:.2f} px | Sens: {sens:.2f}% | Spec: {spec:.2f}%")

    # 2. Evaluate on Zero-Shot Multi-Center Datasets
    if args.eval_zeroshot:
        print("\n--- Evaluating Zero-Shot Generalization on 3 External Datasets ---")
        datasets = ["CVC-ClinicDB", "CVC-ColonDB", "ETIS"]
        for ds_name in datasets:
            ds = ExternalPolyDataset(data_root=args.external_dir, ds_name=ds_name, img_size=args.img_size)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
            all_metrics = []
            with torch.no_grad():
                for batch in loader:
                    imgs = batch["image"].to(device)
                    masks = batch["mask"].cpu().numpy()
                    preds = torch.sigmoid(model(imgs)).cpu().numpy()
                    for i in range(len(imgs)):
                        all_metrics.append(compute_metrics(preds[i, 0], masks[i, 0]))
            d = np.mean([m["dice"] for m in all_metrics]) * 100
            iou = np.mean([m["iou"] for m in all_metrics]) * 100
            hd = np.mean([m["hd95"] for m in all_metrics])
            sens = np.mean([m["sensitivity"] for m in all_metrics]) * 100
            spec = np.mean([m["specificity"] for m in all_metrics]) * 100
            print(f"  [{ds_name:<14} (N={len(ds)})] Dice: {d:.2f}% | IoU: {iou:.2f}% | HD95: {hd:.2f} px | Sens: {sens:.2f}% | Spec: {spec:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MS-VMUNet or UNet Checkpoints")
    parser.add_argument("--model", type=str, default="ms_vmunet", choices=["ms_vmunet", "unet"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth checkpoint")
    parser.add_argument("--eval_kvasir", action="store_true", default=True, help="Evaluate on Kvasir-SEG test set")
    parser.add_argument("--eval_zeroshot", action="store_true", default=True, help="Evaluate on external ClinicDB, ColonDB, ETIS")
    parser.add_argument("--kvasir_dir", type=str, default="data/Kvasir-SEG")
    parser.add_argument("--external_dir", type=str, default="data/PolyDataset")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--no_4way", action="store_true")
    parser.add_argument("--no_msconv", action="store_true")
    parser.add_argument("--skip_type", type=str, default="hfd")

    args = parser.parse_args()
    evaluate_checkpoint(args)
