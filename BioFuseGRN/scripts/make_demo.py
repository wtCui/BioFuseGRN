import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='example/demo')
    p.add_argument('--genes', type=int, default=64)
    p.add_argument('--cells', type=int, default=40)
    p.add_argument('--tfs', type=int, default=10)
    p.add_argument('--seed', type=int, default=9)
    args = p.parse_args()
    if not 3 <= args.tfs < args.genes:
        raise ValueError('tfs must be >= 3 and smaller than genes')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    genes = np.array([f'Gene_{i:04d}' for i in range(args.genes)])
    latent = rng.normal(size=(args.genes, 8))
    expression = latent @ rng.normal(size=(8, args.cells)) + rng.normal(scale=0.7, size=(args.genes, args.cells))
    frame = pd.DataFrame(expression, index=genes, columns=[f'Cell_{j:03d}' for j in range(args.cells)])
    frame.index.name = 'gene'
    frame.to_csv(output / 'expression.csv')
    tfs = genes[:args.tfs]
    pd.Series(tfs).to_csv(output / 'tfs.txt', index=False, header=False)
    pairs = np.stack(np.meshgrid(np.arange(args.tfs), np.arange(args.genes), indexing='ij'), axis=-1).reshape(-1, 2)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    similarities = (latent[pairs[:, 0]] * latent[pairs[:, 1]]).sum(-1)
    probabilities = 1 / (1 + np.exp(-0.6 * similarities))
    selected = pairs[rng.random(len(pairs)) < np.maximum(0.01, probabilities * 0.18)]
    pd.DataFrame({'tf': genes[selected[:, 0]], 'target': genes[selected[:, 1]]}).to_csv(output / 'reference.csv', index=False)
    family_embeddings = latent @ rng.normal(size=(8, 16)) + rng.normal(scale=0.1, size=(args.genes, 16))
    promoter_embeddings = latent @ rng.normal(size=(8, 16)) + rng.normal(scale=0.1, size=(args.genes, 16))
    for filename, values in [('family_embeddings.csv', family_embeddings), ('promoter_embeddings.csv', promoter_embeddings)]:
        t = pd.DataFrame(values, index=genes, columns=[f'd{i}' for i in range(values.shape[1])])
        t.index.name = 'gene'
        t.to_csv(output / filename)
    categories = np.argmax(latent[:, :6], axis=1)
    go = [(g, f'GO:{categories[i]:07d}') for i, g in enumerate(genes)]
    pw = [(g, f'PW:{int(categories[i] // 2)}') for i, g in enumerate(genes)]
    pd.DataFrame(go, columns=['gene', 'term']).to_csv(output / 'go.csv', index=False)
    pd.DataFrame(pw, columns=['gene', 'pathway']).to_csv(output / 'pathways.csv', index=False)
    print(str(output))


if __name__ == '__main__':
    main()
