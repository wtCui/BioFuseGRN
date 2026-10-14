# BioFuseGRN

Python 3.10+ / PyTorch 2.1+

```bash
pip install -r requirements.txt
python scripts/make_demo.py
python train.py --expression example/demo/expression.csv --reference example/demo/reference.csv --tfs example/demo/tfs.txt --family example/demo/family_embeddings.csv --promoter example/demo/promoter_embeddings.csv --go example/demo/go.csv --pathways example/demo/pathways.csv --gene-count 0 --epochs 10 --output runs/demo
python infer.py --run runs/demo --top-k 1000 --threshold 0.0 --exclude-observed
```

The demo embeddings are synthetic and are not outputs of pretrained biological models.

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

The manuscript specifies the overall architecture, loss equations, optimal results for two GNN layers, 128 hidden dimensions, four heads, and λ=0.8, but does not specify all implementation settings. Defaults for optimizer, normalization, dropout, graph symmetrization, sparse cutoff, functional top-k, and Transformer depth are implementation choices rather than verified original settings. The manuscript's reported benchmarks cannot be reproduced without the associated real expression, network, annotation, and pretrained embedding inputs.

## Experimental runner

```bash
python scripts/run_experiments.py --manifest example/manifest.csv --mode benchmark --gene-counts 500 1000 --seeds 1 2 3 4 5
python scripts/run_experiments.py --manifest example/manifest.csv --mode ablation --gene-counts 500
python scripts/run_experiments.py --manifest example/manifest.csv --mode sensitivity --gene-counts 500
python scripts/run_experiments.py --manifest example/manifest.csv --mode robustness --gene-counts 500
```

For the 42 BEELINE evaluation settings, supply a manifest containing the seven scRNA-seq datasets paired with the three reference-network types, and use both gene-count values. These datasets and external pretrained vectors are not bundled.
