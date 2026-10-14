import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gene2vec', required=True)
    p.add_argument('--families', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--blend', type=float, default=0.5)
    args = p.parse_args()
    if not 0 <= args.blend <= 1:
        raise ValueError('--blend must be between 0 and 1')
    vectors = pd.read_csv(args.gene2vec, index_col=0)
    vectors.index = vectors.index.astype(str)
    families = pd.read_csv(args.families, dtype=str)
    if not {'gene', 'family'} <= set(families.columns):
        raise ValueError('families CSV must contain gene and family columns')
    families = families[['gene', 'family']].dropna().drop_duplicates('gene').set_index('gene')
    joined = families.join(vectors, how='left')
    feature_cols = list(vectors.columns)
    centroids = joined.groupby('family')[feature_cols].mean()
    results = joined[['family']].join(centroids, on='family', rsuffix='_centroid')
    centroids_values = results[[f'{c}_centroid' for c in feature_cols]].to_numpy(dtype=np.float32)
    individual = joined[feature_cols].to_numpy(dtype=np.float32)
    only_centroid = ~np.isfinite(individual).all(axis=1)
    individual[only_centroid] = centroids_values[only_centroid]
    result = args.blend * individual + (1 - args.blend) * centroids_values
    output = pd.DataFrame(result, index=joined.index, columns=feature_cols)
    not_assigned = vectors.loc[~vectors.index.isin(output.index)]
    output = pd.concat([output, not_assigned])
    output.index.name = 'gene'
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output)


if __name__ == '__main__':
    main()
