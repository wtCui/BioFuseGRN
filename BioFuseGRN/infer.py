import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--output')
    p.add_argument('--top-k', type=int, default=1000)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--exclude-observed', action='store_true')
    args = p.parse_args()
    run = Path(args.run)
    embeddings = np.load(run / 'embeddings.npy')
    weight = np.load(run / 'link_weight.npy')
    genes = pd.read_csv(run / 'genes.csv')['gene'].astype(str).to_numpy()
    tfs = pd.read_csv(run / 'tfs.csv')['tf'].astype(str).to_numpy()
    indexes = {g: i for i, g in enumerate(genes)}
    regulator_ids = np.array([indexes[tf] for tf in tfs], dtype=int)
    logits = (embeddings[regulator_ids] @ weight) @ embeddings.T
    logits = np.clip(logits, -80, 80)
    probabilities = 1 / (1 + np.exp(-logits))
    probabilities[np.arange(len(tfs)), regulator_ids] = -np.inf
    if args.exclude_observed:
        observed = pd.read_csv(run / 'train_pairs.csv')
        observed = observed.loc[observed['label'].eq(1)]
        tf_lookup = {tf: i for i, tf in enumerate(tfs)}
        for tf, target in zip(observed['tf'], observed['target']):
            probabilities[tf_lookup[tf], indexes[target]] = -np.inf
    flat = probabilities.ravel()
    valid = np.flatnonzero(flat >= args.threshold)
    if args.top_k > 0 and len(valid) > args.top_k:
        selected = np.argpartition(flat[valid], -args.top_k)[-args.top_k:]
        valid = valid[selected]
    valid = valid[np.argsort(-flat[valid], kind='stable')]
    row, col = np.unravel_index(valid, probabilities.shape)
    result = pd.DataFrame({'tf': tfs[row], 'target': genes[col], 'score': flat[valid]})
    output = Path(args.output) if args.output else run / 'predicted_edges.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f'{len(result)} predictions written to {output}')


if __name__ == '__main__':
    main()
