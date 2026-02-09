# Run all 6 GNN experiments sequentially (PowerShell)

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "GNN Playground - Running All Experiments" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 1. Node Classification (Cora)
Write-Host ""
Write-Host "[1/6] Node Classification on Cora..." -ForegroundColor Green
Write-Host "----------------------------------------"
uv run python -m gnn_playground run --task node_classification --dataset cora --model gcn --epochs 50 --output-dir outputs/node_classification

# 2. Graph Classification (MUTAG)
Write-Host ""
Write-Host "[2/6] Graph Classification on MUTAG..." -ForegroundColor Green
Write-Host "----------------------------------------"
uv run python -m gnn_playground run --task graph_classification --dataset mutag --model gin --epochs 50 --output-dir outputs/graph_classification

# 3. Link Prediction (Email-Eu-core)
Write-Host ""
Write-Host "[3/6] Link Prediction on Email-Eu-core..." -ForegroundColor Green
Write-Host "----------------------------------------"
uv run python -m gnn_playground run --task link_prediction --dataset email_eu_core --model gcn --epochs 50 --output-dir outputs/link_prediction

# 4. Recommendation (MovieLens 100K)
Write-Host ""
Write-Host "[4/6] Recommendation on MovieLens 100K..." -ForegroundColor Green
Write-Host "----------------------------------------"
uv run python -m gnn_playground run --config src/gnn_playground/configs/recsys_movielens.yaml

# 5. Community Detection (Email-Eu-core)
Write-Host ""
Write-Host "[5/6] Community Detection on Email-Eu-core..." -ForegroundColor Green
Write-Host "----------------------------------------"
uv run python -m gnn_playground run --config src/gnn_playground/configs/community_detection_email.yaml

# 6. Fraud Detection (Elliptic) - requires manual dataset download
Write-Host ""
Write-Host "[6/6] Fraud Detection on Elliptic..." -ForegroundColor Green
Write-Host "----------------------------------------"
Write-Host "Note: Requires Elliptic dataset from Kaggle in data/elliptic/"
if (Test-Path "data/elliptic/elliptic_txs_features.csv") {
    uv run python -m gnn_playground run --config src/gnn_playground/configs/fraud_elliptic.yaml
} else {
    Write-Host "Skipping: Elliptic dataset not found. Download from:" -ForegroundColor Yellow
    Write-Host "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "All experiments complete!" -ForegroundColor Cyan
Write-Host "Results saved in outputs/" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
