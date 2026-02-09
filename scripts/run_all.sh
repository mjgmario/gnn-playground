#!/bin/bash
# Run all 6 GNN experiments sequentially

set -e  # Exit on error

echo "=========================================="
echo "GNN Playground - Running All Experiments"
echo "=========================================="

# 1. Node Classification (Cora)
echo ""
echo "[1/6] Node Classification on Cora..."
echo "----------------------------------------"
uv run python -m gnn_playground run --task node_classification --dataset cora --model gcn --epochs 50 --output-dir outputs/node_classification

# 2. Graph Classification (MUTAG)
echo ""
echo "[2/6] Graph Classification on MUTAG..."
echo "----------------------------------------"
uv run python -m gnn_playground run --task graph_classification --dataset mutag --model gin --epochs 50 --output-dir outputs/graph_classification

# 3. Link Prediction (Email-Eu-core)
echo ""
echo "[3/6] Link Prediction on Email-Eu-core..."
echo "----------------------------------------"
uv run python -m gnn_playground run --task link_prediction --dataset email_eu_core --model gcn --epochs 50 --output-dir outputs/link_prediction

# 4. Recommendation (MovieLens 100K)
echo ""
echo "[4/6] Recommendation on MovieLens 100K..."
echo "----------------------------------------"
uv run python -m gnn_playground run --config src/gnn_playground/configs/recsys_movielens.yaml

# 5. Community Detection (Email-Eu-core)
echo ""
echo "[5/6] Community Detection on Email-Eu-core..."
echo "----------------------------------------"
uv run python -m gnn_playground run --config src/gnn_playground/configs/community_detection_email.yaml

# 6. Fraud Detection (Elliptic) - requires manual dataset download
echo ""
echo "[6/6] Fraud Detection on Elliptic..."
echo "----------------------------------------"
echo "Note: Requires Elliptic dataset from Kaggle in data/elliptic/"
if [ -f "data/elliptic/elliptic_txs_features.csv" ]; then
    uv run python -m gnn_playground run --config src/gnn_playground/configs/fraud_elliptic.yaml
else
    echo "Skipping: Elliptic dataset not found. Download from:"
    echo "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set"
fi

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "Results saved in outputs/"
echo "=========================================="
