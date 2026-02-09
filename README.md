# GNN Playground

A modular Graph Neural Network playground featuring 6 classic graph learning tasks with a unified CLI, comprehensive testing, and reproducible experiments.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-292%20passed-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)

## What's Inside

| Task | Dataset | Models | Key Metrics |
|------|---------|--------|-------------|
| **Node Classification** | Cora, CiteSeer, PubMed | MLP, GCN, GraphSAGE, GAT | Accuracy, F1 |
| **Graph Classification** | MUTAG, PROTEINS | GCN, GIN | Accuracy, F1 |
| **Link Prediction** | Email-Eu-core | GCN/SAGE/GAT + DotProduct/MLP/Bilinear | AUC-ROC, AP |
| **Recommendation** | MovieLens 100K | MF, LightGCN | Recall@K, NDCG@K |
| **Community Detection** | Email-Eu-core | Louvain, Spectral, GNN+KMeans | Modularity, NMI |
| **Fraud Detection** | Elliptic | LogReg, GraphSAGE, GAT | PR-AUC, F1, Recall@Precision |

## Quickstart

```bash
# Install dependencies with UV
uv sync

# Run your first experiment
uv run python -m gnn_playground run --task node_classification --dataset cora --model gcn
```

## Usage

### Node Classification
```bash
# Single model
uv run python -m gnn_playground run --task node_classification --dataset cora --model gcn

# Compare multiple models with config
uv run python -m gnn_playground run --config src/gnn_playground/configs/node_classification_cora.yaml
```

### Graph Classification
```bash
uv run python -m gnn_playground run --task graph_classification --dataset mutag --model gin
```

### Link Prediction
```bash
uv run python -m gnn_playground run --task link_prediction --dataset email_eu_core --model gcn
```

### Recommendation
```bash
uv run python -m gnn_playground run --config src/gnn_playground/configs/recsys_movielens.yaml
```

### Community Detection
```bash
uv run python -m gnn_playground run --config src/gnn_playground/configs/community_detection_email.yaml
```

### Fraud Detection
```bash
uv run python -m gnn_playground run --config src/gnn_playground/configs/fraud_elliptic.yaml
```

## Repository Structure

```
gnn_playground/
├── src/gnn_playground/
│   ├── cli.py                 # Typer CLI application
│   ├── config.py              # Configuration management
│   ├── datasets/              # Dataset loaders
│   │   ├── planetoid.py       # Cora, CiteSeer, PubMed
│   │   ├── tudatasets.py      # MUTAG, PROTEINS
│   │   ├── snap_email.py      # Email-Eu-core
│   │   ├── movielens.py       # MovieLens 100K
│   │   └── elliptic.py        # Elliptic Bitcoin
│   ├── models/                # GNN architectures
│   │   ├── gcn.py             # Graph Convolutional Network
│   │   ├── graphsage.py       # GraphSAGE
│   │   ├── gat.py             # Graph Attention Network
│   │   ├── gin.py             # Graph Isomorphism Network
│   │   ├── gcn_graph.py       # GCN for graph classification
│   │   ├── lightgcn.py        # LightGCN for recommendations
│   │   ├── decoders.py        # Link prediction decoders
│   │   └── mlp.py             # MLP baseline
│   ├── tasks/                 # Task runners
│   │   ├── node_classification.py
│   │   ├── graph_classification.py
│   │   ├── link_prediction.py
│   │   ├── recsys.py
│   │   ├── community_detection.py
│   │   └── fraud.py
│   ├── training/              # Training utilities
│   │   ├── loops.py           # Train/eval loops for each task type
│   │   ├── metrics.py         # Metric computation
│   │   └── utils.py           # Seeds, early stopping, etc.
│   └── viz/                   # Visualization
│       ├── plots.py           # Training curves, ROC, PR, etc.
│       └── graph_viz.py       # Community visualization
│   └── configs/               # YAML configuration files
├── tests/                     # Comprehensive test suite (292 tests)
├── data/                      # Downloaded datasets (gitignored)
├── outputs/                   # Experiment outputs (gitignored)
└── scripts/                   # Utility scripts
```

## Benchmarks

Results on default configs (CPU, seed=42):

### Node Classification (Cora)
| Model | Accuracy | F1 |
|-------|----------|----|
| MLP | 0.580 | 0.565 |
| GCN | **0.811** | **0.805** |
| GraphSAGE | 0.805 | 0.797 |
| GAT | 0.799 | 0.797 |

### Graph Classification (MUTAG)
| Model | Accuracy | F1 |
|-------|----------|----|
| GIN | **0.684** | **0.593** |
| Graph GCN | 0.579 | 0.548 |

### Link Prediction (Email-Eu-core)
| Model + Decoder | AUC | AP |
|----------------|-----|----|
| GCN + Dot | 0.868 | 0.862 |
| GCN + MLP | 0.869 | 0.862 |
| GraphSAGE + Dot | 0.872 | 0.869 |
| GraphSAGE + MLP | **0.889** | **0.882** |

### Recommendation (MovieLens 100K)
| Model | Recall@20 | NDCG@20 |
|-------|-----------|---------|
| LightGCN | 0.148 | 0.058 |
| MF | **0.193** | **0.077** |

### Community Detection (Email-Eu-core)
| Method | Modularity | NMI |
|--------|------------|-----|
| Louvain | 0.232 | **0.559** |
| Spectral | **0.268** | 0.430 |
| GNN+KMeans | 0.032 | 0.257 |

### Fraud Detection (Elliptic)
| Model | PR-AUC | F1 | Recall@P=0.9 |
|-------|--------|----|--------------|
| GraphSAGE | **0.991** | **0.974** | 1.000 |
| GAT | 0.990 | 0.683 | 1.000 |

## Datasets

| Dataset | Task | Size | Auto-Download |
|---------|------|------|---------------|
| **Cora** | Node Classification | 2,708 nodes, 5,429 edges | Yes (PyG) |
| **CiteSeer** | Node Classification | 3,327 nodes, 4,732 edges | Yes (PyG) |
| **PubMed** | Node Classification | 19,717 nodes, 44,338 edges | Yes (PyG) |
| **MUTAG** | Graph Classification | 188 graphs | Yes (PyG) |
| **PROTEINS** | Graph Classification | 1,113 graphs | Yes (PyG) |
| **Email-Eu-core** | Link/Community | 1,005 nodes, 25,571 edges | Yes (SNAP) |
| **MovieLens 100K** | Recommendation | 943 users, 1,682 items | Yes (GroupLens) |
| **Elliptic** | Fraud Detection | 203K nodes, 234K edges | No (Kaggle) |

### Elliptic Dataset Setup

The Elliptic Bitcoin dataset requires manual download from Kaggle:
1. Visit: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
2. Download and extract to `data/elliptic/`
3. Ensure these files exist:
   - `data/elliptic/elliptic_txs_features.csv`
   - `data/elliptic/elliptic_txs_edgelist.csv`
   - `data/elliptic/elliptic_txs_classes.csv`

## Development

### Setup
```bash
# Install with dev dependencies
uv sync --extra dev

# Install pre-commit hooks
uv run pre-commit install
```

### Testing
```bash
# Run all tests
uv run pytest

# Run fast tests only (skip dataset downloads)
uv run pytest -m "not slow"

# Run with coverage report
uv run pytest --cov=src/gnn_playground --cov-report=html
```

### Code Quality
```bash
# Format code
uv run black src/ tests/ --line-length=120

# Lint
uv run ruff check src/ tests/ --fix

# Type check
uv run mypy src/
```

## Configuration

Tasks can be configured via YAML files or CLI arguments. CLI arguments override YAML values.

Example config (`src/gnn_playground/configs/node_classification_cora.yaml`):
```yaml
task: node_classification
dataset: cora
models: [mlp, gcn, graphsage, gat]
epochs: 200
lr: 0.01
hidden_dim: 64
weight_decay: 5.0e-4
patience: 20
output_dir: outputs/node_classification_cora
device: cpu
seed: 42
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) for the GNN framework
- Dataset providers: SNAP, GroupLens, Kaggle/Elliptic
