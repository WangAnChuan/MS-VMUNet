import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_dataloaders
from models.ms_vmunet import MSVMUNetNextGen as MSVMUNet
from models.unet import UNet
from losses import HybridLoss
from metrics import compute_metrics

def evaluate(model, loader, device):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            preds = model(images)
            if isinstance(preds, tuple):
                preds = preds[0]
            preds_prob = torch.sigmoid(preds).cpu().numpy()
            masks_np = masks.cpu().numpy()
            for i in range(len(images)):
                p = preds_prob[i, 0]
                g = masks_np[i, 0]
                all_metrics.append(compute_metrics(p, g))
    return {k: float(np.mean([m[k] for m in all_metrics])) for k in ["dice", "iou", "sensitivity", "specificity", "hd95"]}

def train(args):
    # Set deterministic random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 80)
    print(f"TRAINING CONFIGURATION: Model={args.model} | Seed={args.seed} | Epochs={args.epochs}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )

    if args.model == "ms_vmunet":
        model = MSVMUNet(
            in_channels=3,
            num_classes=1,
            use_4way=not args.no_4way,
            use_msconv=not args.no_msconv,
            skip_type=args.skip_type
        ).to(device)
        criterion = HybridLoss(
            use_tv=not args.no_tv,
            tv_weight=args.tv_weight,
            use_sobel=not args.no_sobel,
            sobel_weight=args.sobel_weight
        )
    elif args.model == "unet":
        model = UNet(in_channels=3, num_classes=1).to(device)
        criterion = HybridLoss(use_tv=False, use_sobel=False)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_dice = 0.0
    best_ckpt = os.path.join(args.save_dir, "best_model.pth")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            preds = model(images)
            loss = criterion(preds, masks)
            loss = loss / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            epoch_loss += loss.item() * args.grad_accum

        scheduler.step()

        val_m = evaluate(model, val_loader, device)
        is_best = val_m["dice"] > best_val_dice
        if is_best:
            best_val_dice = val_m["dice"]
            torch.save(model.state_dict(), best_ckpt)

        if epoch % 10 == 0 or epoch == args.epochs or (is_best and epoch > 15):
            tag = " ★ (Best)" if is_best else ""
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] Loss: {epoch_loss/len(train_loader):.4f} | "
                  f"Val Dice: {val_m['dice']*100:.2f}% | Val HD95: {val_m['hd95']:.2f}px{tag}")

    print("\n--- Evaluating Best Checkpoint on Independent Test Set (100 Images) ---")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_m = evaluate(model, test_loader, device)

    elapsed_min = round((time.time() - start_time) / 60, 2)
    print("=" * 80)
    print(f"FINAL TEST SET RESULTS:")
    print(f"  Test Dice:        {test_m['dice']*100:.2f}%")
    print(f"  Test IoU:         {test_m['iou']*100:.2f}%")
    print(f"  Test HD95:        {test_m['hd95']:.2f} px")
    print(f"  Test Sensitivity: {test_m['sensitivity']*100:.2f}%")
    print(f"  Test Specificity: {test_m['specificity']*100:.2f}%")
    print(f"  Training Time:    {elapsed_min} min")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MS-VMUNet or UNet on Kvasir-SEG")
    parser.add_argument("--model", type=str, default="ms_vmunet", choices=["ms_vmunet", "unet"])
    parser.add_argument("--data_dir", type=str, default="data/Kvasir-SEG")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=256)
    # Architecture ablation switches
    parser.add_argument("--no_4way", action="store_true", help="Revert to 2-way horizontal scanning")
    parser.add_argument("--no_msconv", action="store_true", help="Revert to standard 3x3 depthwise convolution")
    parser.add_argument("--skip_type", type=str, default="hfd", choices=["hfd", "plain", "multiplicative_gate", "additive_raa"])
    parser.add_argument("--no_tv", action="store_true", help="Disable TV loss")
    parser.add_argument("--tv_weight", type=float, default=0.02)
    parser.add_argument("--no_sobel", action="store_true", help="Disable Sobel boundary gradient loss")
    parser.add_argument("--sobel_weight", type=float, default=0.4)

    args = parser.parse_args()
    train(args)
