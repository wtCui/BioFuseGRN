# BioFuseGRN

**Improving Single-Cell Gene Regulatory Network Inference Through Adaptive Biological Knowledge Fusion**

[![Paper](https://img.shields.io/badge/Paper-IJMS-blue)](https://doi.org/10.3390/ijms272310567) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)

BioFuseGRN integrates gene family representations, promoter sequence embeddings and GO/pathway-derived functional similarities through mediator-guided attention. A hybrid GNN–Transformer encoder and Deep Graph Infomax (DGI) objective learn gene representations for directed TF–target link prediction. The associated paper reports the highest AUROC in 39 of 42 benchmark settings and the highest AUPRC in 40 of 42 settings.

This source-only repository contains a reconstruction of the published method and experimental utilities. No datasets, synthetic demo files, pretrained embeddings, model weights or experimental results are distributed. Users must supply their own input files. It is not the original archived experiment environment.

## Installation and quick start

Python 3.10+ / PyTorch 2.1+. Run commands from the repository root. CPU and CUDA are supported.

```bash
git clone https://github.com/wtCui/BioFuseGRN.git
cd BioFuseGRN
python -m venv .venv
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
python train.py --expression data/expression.csv --reference data/reference.csv --tfs data/tfs.txt --family data/family_embeddings.csv --promoter data/promoter_embeddings.csv --go data/go.csv --pathways data/pathways.csv --gene-count 500 --epochs 3 --device cpu --output runs/experiment
python infer.py --run runs/experiment --top-k 1000 --threshold 0.0 --exclude-observed
```

The `data/` paths are placeholders for your external input files, not files included in this repository. The three-epoch command is an optional smoke test using your own compatible data.

## Inputs

| Argument | File | Required columns |
|---|---|---|
| `--expression` | CSV, genes × cells | First column `gene`, other columns cells |
| `--reference` | CSV | `tf,target`, optional `label` (1: observed edge) |
| `--tfs` | TXT | One transcription factor per line |
| `--prior-edges` | CSV | `tf,target`, optional `label` |
| `--family` | CSV | First column `gene`, remaining columns pretrained Gene2Vec/family vector dimensions |
| `--promoter` | CSV | First column `gene`, remaining columns pretrained DNABERT vector dimensions |
| `--go` | CSV | `gene,term` |
| `--pathways` | CSV | `gene,pathway` |

Expression and annotation identifiers must match exactly. `--tfs` is optional when the regulators are fully represented by the first column of the reference network. `--prior-edges` is optional; training positive edges otherwise define the weakly supervised scaffold. `--gene-count 500` selects all TFs plus 500 highly variable non-TF genes; `--gene-count 1000` uses 1000. `--gene-count 0` retains all input genes.

## Pretrained feature preparation

```bash
pip install -r requirements-embeddings.txt
python scripts/encode_promoters.py --fasta promoters.fa --model YOUR_DNABERT_MODEL --output data/promoter_embeddings.csv --kmer 0
python scripts/build_family_prior.py --gene2vec data/pretrained_gene2vec.csv --families data/gene_families.csv --output data/family_embeddings.csv
```

For original DNABERT k-mer tokenization, set `--kmer 6` when the checkpoint requires 6-mer input. For a DNABERT-2 checkpoint accepting nucleotide strings directly, use `--kmer 0`. Promoter FASTA sequences and pretrained model weights must be provided externally.

## Training configuration

Defaults: Adam, learning rate 0.001, weight decay 0.0005, dropout 0.2, 200 epochs, early-stopping patience 20, hidden dimension 128, two GNN layers, four attention heads, DGI weight 0.8, GO and pathway weights 0.5 each, and top-k functional neighbors 10.

## Full training

```bash
python train.py --expression data/expression.csv --reference data/reference.csv --tfs data/tfs.txt --prior-edges data/prior_edges.csv --family data/family_embeddings.csv --promoter data/promoter_embeddings.csv --go data/go.csv --pathways data/pathways.csv --gene-count 500 --epochs 200 --seed 42 --output runs/experiment
```

Outputs: `best.pt`, `history.csv`, `metrics.json`, `train_pairs.csv`, `val_pairs.csv`, `test_pairs.csv`, `embeddings.npy`, `link_weight.npy`, `genes.csv`, `tfs.csv`. `infer.py` creates `predicted_edges.csv` with `tf,target,score`.

The positive reference edges are split 60%/20%/20%. Absent interactions are unlabeled operational negatives, not verified negatives. A sparse graph is defined by `--sparse-cutoff 0.1`; training negatives are limited to six per positive. Validation and test negatives are drawn from their respective 20% partitions of all unobserved candidate pairs. Validation and test candidate pairs are removed from the prior regulatory graph.

## Ablations

```bash
python train.py ... --disable-family
python train.py ... --disable-promoter
python train.py ... --disable-functional
python train.py ... --no-priors
python train.py ... --disable-mediator
python train.py ... --disable-dgi
python train.py ... --disable-prior-graph
python train.py ... --disable-transformer
```

The published manuscript specifies the main training settings. Graph symmetrization, normalization, residual connections, sparse cutoff and Transformer depth are implementation choices. Exact benchmark reproduction requires the original data and external pretrained embeddings.

## Experimental runner

```bash
python scripts/run_experiments.py --manifest data/manifest.csv --mode benchmark --gene-counts 500 1000 --seeds 1 2 3 4 5 6 7 8 9 10
python scripts/run_experiments.py --manifest data/manifest.csv --mode ablation --gene-counts 500
python scripts/run_experiments.py --manifest data/manifest.csv --mode sensitivity --gene-counts 500
python scripts/run_experiments.py --manifest data/manifest.csv --mode robustness --gene-counts 500
python scripts/run_experiments.py --manifest data/manifest.csv --mode fusion --gene-counts 500
python scripts/run_experiments.py --manifest data/manifest.csv --mode randomization --gene-counts 500
```

For the 42 BEELINE evaluation settings, supply a manifest containing the seven scRNA-seq datasets paired with the three reference-network types, and use both gene-count values. These datasets and external pretrained vectors are not bundled.

## Benchmark reproduction and analysis

Create your own `data/manifest.csv` with one row per dataset/reference-network combination. Required columns are `dataset,network,expression,reference`; optional path columns are `tfs,prior_edges,family,promoter,go,pathways`. No manifest or dataset is bundled. Use real data for hESC, hHEP, mDC, mESC, mHSC-E, mHSC-GM and mHSC-L, with STRING, non-cell-type-specific ChIP-seq and cell-type-specific ChIP-seq reference networks. Manifest paths are interpreted relative to the repository root (or may be absolute). The runner saves per-run results, `all_results.csv` and `summary.csv` (mean, standard deviation, minimum and maximum). Use a separate output directory for each experiment mode.

The robustness runner uses a fixed split seed and separate masking/model seeds and reports Jaccard similarity and masked-positive recovery. Validation F1 is used for threshold selection as an implementation choice. Standard attention and parameter-matched MLP are reconstructed comparator architectures. External baseline implementations and original real datasets are not bundled.

For statistical analysis, supply a CSV with columns `dataset,network,gene_count,seed,method,auroc,auprc`, including BioFuseGRN and external baseline results on matched splits:

```bash
python scripts/statistical_analysis.py --results paired_results.csv --output statistics.csv
```

For case studies, supply a trained human model and independent TF-target evidence with columns `tf,target`:

```bash
python scripts/case_study.py --run runs/experiment --validation evidence.csv --tfs SOX2 NANOG --top-k 20 --output case_study.csv
```

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Unit tests use small tensors constructed in memory to check algorithm behavior; no dataset files are required. GitHub Actions runs these tests and CLI argument checks. End-to-end training requires external data. Full real-data benchmarks and all experimental variants have not been independently revalidated in this release.

## Citation

If you use BioFuseGRN, please cite:

```bibtex
@article{cui2026biofusegrn,
  title = {Improving Single-Cell Gene Regulatory Network Inference Through Adaptive Biological Knowledge Fusion},
  author = {Cui, Wentao and Fu, Zhican and Hou, Runtong and Zhang, Jingcheng and Zhang, Xuejie and Cai, Junlan and Miao, Xiaoye and Chen, Yong and Tao, Tietuo and Lu, Peng},
  journal = {International Journal of Molecular Sciences},
  year = {2026},
  volume = {27},
  number = {23},
  pages = {10567},
  doi = {10.3390/ijms272310567}
}
```

## Licensing

The associated article is published under CC BY 4.0. A separate software license has not been designated in this package; the repository owner should select one before granting reuse rights. External data and pretrained model weights remain subject to their respective terms.
