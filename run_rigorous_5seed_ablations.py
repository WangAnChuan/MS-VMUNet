import os
import sys
import time
import json
import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from thop import profile

from dataset import get_dataloaders
from models_nextgen import MSVMUNetNextGen
from losses import HybridLoss
from metrics import compute_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
if not os.path.isdir(RESULTS_DIR) and os.path.isdir("/root/autodl-tmp/polyp_mamba_project/results"):
    RESULTS_DIR = "/root/autodl-tmp/polyp_mamba_project/results"
MASTER_CSV = os.path.join(RESULTS_DIR, "master_nextgen_experiments.csv")

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

def update_master_csv(report):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.exists(MASTER_CSV):
        df = pd.read_csv(MASTER_CSV)
        df = df[df["exp_name"] != report["exp_name"]]
        df = pd.concat([df, pd.DataFrame([report])], ignore_index=True)
    else:
        df = pd.DataFrame([report])
    df.to_csv(MASTER_CSV, index=False)
    print(f"[CSV UPDATED] {MASTER_CSV} (Total records: {len(df)})")

def train_and_eval_ablation(model_kwargs, loss_kwargs, exp_name, seed=42, epochs=40, batch_size=8, grad_accum=2, lr=2e-4):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    exp_dir = os.path.join(RESULTS_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    report_file = os.path.join(exp_dir, "report.json")

    if os.path.exists(report_file):
        print(f"\n[CACHE HIT] {exp_name} already completed! Skipping...")
        with open(report_file, "r") as f:
            res = json.load(f)
            update_master_csv(res)
            return res

    print("\n" + "=" * 75)
    print(f"RUNNING ABLATION: {exp_name}")
    print(f"Model Args: {model_kwargs}")
    print(f"Loss Args:  {loss_kwargs}")
    print(f"Seed: {seed} | Epochs: {epochs} | Batch: {batch_size} (Accum: {grad_accum})")
    print("=" * 75)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size, num_workers=4, img_size=256)
    model = MSVMUNetNextGen(**model_kwargs).to(device)

    # Calculate Params & FLOPs on temporary model instance to prevent hook pollution
    dummy_x = torch.randn(1, 3, 256, 256).to(device)
    try:
        m_tmp = MSVMUNetNextGen(**model_kwargs).to(device)
        flops, thop_p = profile(m_tmp, inputs=(dummy_x,), verbose=False)
        params_m = round(thop_p / 1e6, 2)
        flops_g = round(flops / 1e9, 2)
        del m_tmp
        torch.cuda.empty_cache()
    except Exception:
        params_m = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
        flops_g = 0.0

    criterion = HybridLoss(**loss_kwargs)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_dice = 0.0
    best_val_hd95 = 50.0
    best_ckpt = os.path.join(exp_dir, "best_model.pth")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            preds = model(images)
            loss = criterion(preds, masks)
            loss = loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            epoch_loss += loss.item() * grad_accum

        scheduler.step()

        val_m = evaluate(model, val_loader, device)
        is_best = val_m["dice"] > best_val_dice
        if is_best:
            best_val_dice = val_m["dice"]
            best_val_hd95 = val_m["hd95"]
            torch.save(model.state_dict(), best_ckpt)

        if epoch % 10 == 0 or epoch == epochs or (is_best and epoch > 15):
            tag = " ★ (Best)" if is_best else ""
            print(f"  [{exp_name}] Epoch [{epoch:02d}/{epochs:02d}] "
                  f"Train Loss: {epoch_loss/len(train_loader):.4f} | "
                  f"Val Dice: {val_m['dice']*100:.2f}% | Val HD95: {val_m['hd95']:.2f}px{tag}")

    # Evaluate on held-out test set
    model.load_state_dict(torch.load(best_ckpt), strict=False)
    test_m = evaluate(model, test_loader, device)

    elapsed_min = round((time.time() - start_time) / 60, 2)
    report = {
        "exp_name": exp_name,
        "seed": seed,
        "epochs": epochs,
        "params_m": params_m,
        "flops_g": flops_g,
        "train_time_min": elapsed_min,
        "best_val_dice": round(best_val_dice, 4),
        "best_val_hd95": round(best_val_hd95, 2),
        "test_dice": round(test_m["dice"], 4),
        "test_iou": round(test_m["iou"], 4),
        "test_hd95": round(test_m["hd95"], 2),
        "test_sensitivity": round(test_m["sensitivity"], 4),
        "test_specificity": round(test_m["specificity"], 4),
    }

    with open(report_file, "w") as f:
        json.dump(report, f, indent=4)

    update_master_csv(report)
    print(f"  [RESULT {exp_name}] Test Dice: {report['test_dice']*100:.2f}% | "
          f"IoU: {report['test_iou']*100:.2f}% | HD95: {report['test_hd95']:.2f}px | Time: {elapsed_min}m\n")
    return report

def run_all_5seed_ablations(epochs=40):
    print("\n" + "#" * 80)
    print("TASK 3: RIGOROUS 5-SEED SYSTEMATIC MULTI-MODULE ABLATION SUITE")
    print("Testing 6 Configurations across Seeds 100, 200, 300, 400 (Seed 42 completed)")
    print("#" * 80)

    configs = [
        # 1. 2-Way CS-SSM (w/o 4-Way)
        ("b2_wo_4way",
         dict(use_4way=False, use_msconv=True, skip_type="hfd"),
         dict(use_tv=True, use_sobel=True)),

        # 2. Single-Conv (w/o MS-Conv)
        ("b2_wo_msconv",
         dict(use_4way=True, use_msconv=False, skip_type="hfd"),
         dict(use_tv=True, use_sobel=True)),

        # 3. Plain Skip (w/o HFD-Skip)
        ("b2_wo_hfd",
         dict(use_4way=True, use_msconv=True, skip_type="plain"),
         dict(use_tv=True, use_sobel=True)),

        # 4. Multiplicative Gated Skip
        ("b5_skip_multiplicative_gate",
         dict(use_4way=True, use_msconv=True, skip_type="multiplicative_gate"),
         dict(use_tv=True, use_sobel=True)),

        # 5. w/o Sobel Loss
        ("b2_wo_sobel",
         dict(use_4way=True, use_msconv=True, skip_type="hfd"),
         dict(use_tv=True, use_sobel=False)),

        # 6. VM-UNet Slim Controlled (Param-aligned 1.85M with identical TV+Sobel losses)
        ("b6_vmunet_slim_controlled",
         dict(use_4way=True, use_msconv=False, skip_type="plain"),
         dict(use_tv=True, use_sobel=True)),
    ]

    seeds_to_run = [100, 200, 300, 400]

    for base_name, m_kwargs, l_kwargs in configs:
        # Also ensure Seed 42 is run if not already present
        for s in [42] + seeds_to_run:
            exp_n = f"{base_name}_seed_{s}"
            train_and_eval_ablation(
                model_kwargs=m_kwargs,
                loss_kwargs=l_kwargs,
                exp_name=exp_n,
                seed=s,
                epochs=epochs
            )

    print("\n" + "=" * 80)
    print("ALL 5-SEED ABLATIONS COMPLETED! GENERATING MASTER SUMMARY TABLE...")
    print("=" * 80)
    generate_ablation_5seed_summary()

def generate_ablation_5seed_summary():
    df = pd.read_csv(MASTER_CSV)
    
    config_groups = {
        "Full MS-VMUNet-Lite (Proposed)": "nextgen_full_seed_",
        "w/o 4-Way CS-SSM (2-Way)": "b2_wo_4way_seed_",
        "w/o MS-Conv (Single 3x3 Conv)": "b2_wo_msconv_seed_",
        "Plain Skip (Vanilla Concat)": "b2_wo_hfd_seed_",
        "Multiplicative Gated Skip": "b5_skip_multiplicative_gate_seed_",
        "w/o Sobel Boundary Loss": "b2_wo_sobel_seed_",
        "VM-UNet Slim (Param-Aligned)": "b6_vmunet_slim_controlled_seed_",
    }

    summary_rows = []
    full_dices = []
    full_hds = []

    # Get full model 5 seeds
    sub_full = df[df["exp_name"].str.contains(config_groups["Full MS-VMUNet-Lite (Proposed)"], na=False)]
    if not sub_full.empty:
        full_dices = (sub_full["test_dice"] * 100).tolist()
        full_hds = sub_full["test_hd95"].tolist()

    for label, prefix in config_groups.items():
        sub = df[df["exp_name"].str.contains(prefix, na=False)].sort_values("seed")
        if sub.empty:
            continue

        n_seeds = len(sub)
        dices = (sub["test_dice"] * 100).tolist()
        ious = (sub["test_iou"] * 100).tolist()
        hds = sub["test_hd95"].tolist()
        
        d_mean, d_std = np.mean(dices), np.std(dices, ddof=1) if n_seeds > 1 else 0.0
        iou_mean, iou_std = np.mean(ious), np.std(ious, ddof=1) if n_seeds > 1 else 0.0
        hd_mean, hd_std = np.mean(hds), np.std(hds, ddof=1) if n_seeds > 1 else 0.0

        p_m = sub["params_m"].iloc[0] if "params_m" in sub.columns else 1.85
        f_g = sub["flops_g"].iloc[0] if "flops_g" in sub.columns else 11.02

        # Paired t-test relative to full model if seeds match
        if len(dices) == len(full_dices) and len(dices) >= 2:
            t_d, p_d = stats.ttest_rel(dices, full_dices)
            t_h, p_h = stats.ttest_rel(hds, full_hds)
            p_d_str = f"{p_d:.4f}"
            p_h_str = f"{p_h:.4f}"
        else:
            p_d_str = "—"
            p_h_str = "—"

        summary_rows.append({
            "Configuration": label,
            "Seeds": n_seeds,
            "Params (M)": p_m,
            "FLOPs (G)": f_g,
            "Test Dice (%)": f"{d_mean:.2f} ± {d_std:.2f}",
            "Test IoU (%)": f"{iou_mean:.2f} ± {iou_std:.2f}",
            "HD95 (px) ↓": f"{hd_mean:.2f} ± {hd_std:.2f}",
            "p-val (Dice)": p_d_str,
            "p-val (HD95)": p_h_str
        })

    out_df = pd.DataFrame(summary_rows)
    out_csv = os.path.join(RESULTS_DIR, "rigorous_5seed_ablation_summary.csv")
    out_df.to_csv(out_csv, index=False)
    out_df.to_csv("rigorous_5seed_ablation_summary.csv", index=False)

    print("\n" + "=" * 90)
    print("5-SEED UNBIASED SYSTEMATIC ABLATION MATRIX (MEAN ± SAMPLE STD, ddof=1)")
    print("=" * 90)
    print(out_df.to_string(index=False))

if __name__ == "__main__":
    run_all_5seed_ablations(epochs=40)
