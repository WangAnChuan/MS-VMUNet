import time
import json
import torch
import pandas as pd
from thop import profile
from models_nextgen import MSVMUNetNextGen
from models.unet import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def benchmark_hardware():
    print("=" * 80)
    print("TASK 1: UNIFIED HARDWARE INFERENCE BENCHMARK (NVIDIA RTX 3090)")
    print("=" * 80)

    model_factories = {
        "Standard U-Net": lambda: UNet(),
        "MS-VMUNet-Lite [Ours]": lambda: MSVMUNetNextGen(use_4way=True, use_msconv=True, skip_type="hfd")
    }

    batch_sizes = [1, 4, 16]
    records = []

    for name, factory in model_factories.items():
        print(f"\n--- Profiling {name} ---")
        
        # 1. Measure Params & FLOPs on a temporary model
        m_tmp = factory().to(device).eval()
        dummy_1 = torch.randn(1, 3, 256, 256).to(device)
        try:
            flops, params = profile(m_tmp, inputs=(dummy_1,), verbose=False)
            params_m = round(params / 1e6, 2)
            flops_g = round(flops / 1e9, 2)
        except Exception:
            params_m = round(sum(p.numel() for p in m_tmp.parameters()) / 1e6, 2)
            flops_g = 0.0
        del m_tmp
        torch.cuda.empty_cache()

        # 2. Timing on a clean model without thop hooks
        model = factory().to(device).eval()

        for b in batch_sizes:
            x = torch.randn(b, 3, 256, 256).to(device)

            # Warmup (50 runs)
            for _ in range(50):
                with torch.no_grad():
                    _ = model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Timing (200 runs)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            t0 = time.perf_counter()
            for _ in range(200):
                with torch.no_grad():
                    _ = model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_total = time.perf_counter() - t0

            lat_ms = (t_total / 200.0) * 1000.0
            fps = (b * 200.0) / t_total
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

            rec = {
                "model": name,
                "batch_size": b,
                "params_m": params_m,
                "flops_g": flops_g,
                "latency_ms": round(lat_ms, 2),
                "fps": round(fps, 1),
                "vram_mb": round(vram_mb, 1)
            }
            records.append(rec)
            print(f"  Batch {b:02d}: Latency = {lat_ms:6.2f} ms | Throughput = {fps:6.1f} FPS | Peak VRAM = {vram_mb:6.1f} MB")
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    out_csv = "hardware_latency_benchmark.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[SAVED] Hardware benchmark results saved to: {out_csv}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    benchmark_hardware()
