#!/bin/bash
# ==============================================================================
# MS-VMUNet: One-Click Medical Dataset Downloader
# Downloads Kvasir-SEG (primary benchmark) and PolyDataset (ClinicDB, ColonDB, ETIS)
# ==============================================================================

set -e

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${DATA_DIR}"

echo "===================================================================="
echo ">>> Downloading Kvasir-SEG Benchmark (Primary Dataset) <<<"
echo "===================================================================="
if [ ! -d "Kvasir-SEG" ]; then
    curl -k -L -o kvasir-seg.zip "https://datasets.simula.no/downloads/kvasir-seg.zip"
    unzip -q kvasir-seg.zip
    rm -f kvasir-seg.zip
    echo "[SUCCESS] Kvasir-SEG extracted successfully."
else
    echo "[SKIPPED] Kvasir-SEG directory already exists."
fi

echo -e "\n===================================================================="
echo ">>> Cloning External PolyDataset (ClinicDB, ColonDB, ETIS) <<<"
echo "===================================================================="
if [ ! -d "PolyDataset" ]; then
    git clone --depth 1 https://github.com/Wangxlhaha/PolyDataset.git
    cd PolyDataset
    git lfs pull || true
    cd ..
    echo "[SUCCESS] PolyDataset cloned successfully."
else
    echo "[SKIPPED] PolyDataset directory already exists."
fi

echo -e "\n===================================================================="
echo ">>> Dataset Verification Summary <<<"
echo "===================================================================="
echo "Kvasir-SEG Images: $(ls -1 Kvasir-SEG/images 2>/dev/null | wc -l)"
echo "CVC-ClinicDB Images: $(ls -1 PolyDataset/CVC-ClinicDB/images 2>/dev/null | wc -l)"
echo "CVC-ColonDB Images: $(ls -1 PolyDataset/CVC-ColonDB/images 2>/dev/null | wc -l)"
echo "ETIS-Larib Images: $(ls -1 PolyDataset/ETIS/images 2>/dev/null | wc -l)"
echo "===================================================================="
