#!/usr/bin/env bash
# Downloads the UCSD Anomaly Detection Dataset (Ped1 + Ped2) from UCSD's
# Statistical Visual Computing Lab. ~50MB. Standard benchmark used across the
# video anomaly detection literature.
set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$DATA_DIR/raw"
URL="http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz"

mkdir -p "$RAW_DIR"
cd "$RAW_DIR"

if [ -d "UCSD_Anomaly_Dataset.v1p2" ]; then
    echo "Dataset already present at $RAW_DIR/UCSD_Anomaly_Dataset.v1p2 — skipping download."
    exit 0
fi

echo "Downloading UCSD Anomaly Dataset to $RAW_DIR ..."
curl -L -o UCSD_Anomaly_Dataset.tar.gz "$URL"

echo "Extracting..."
tar -xzf UCSD_Anomaly_Dataset.tar.gz
rm UCSD_Anomaly_Dataset.tar.gz

echo "Done. Data at $RAW_DIR/UCSD_Anomaly_Dataset.v1p2"
