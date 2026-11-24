import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gene2vec', required=True)
    p.add_argument('--families', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    vectors = pd.read_csv(args.gene2vec, index_col=0)
    vectors.index = vectors.index.astype(str)
    families = pd.read_csv(args.families, dtype=str)
    if not {'gene', 'family'} <= set(families.columns):
        raise ValueError('families CSV must contain gene and family columns')
    families = families[['gene', 'family']].dropna().drop_duplicates('gene').set_index('gene')
    # Section 4.3.1: retain the pretrained individual vector; missing genes are zero-filled.
    output = vectors.reindex(vectors.index.union(families.index)).fillna(0)
    output.index.name = 'gene'
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output)


if __name__ == '__main__':
    main()
