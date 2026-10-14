import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd


FIELDS = ['expression', 'reference', 'tfs', 'prior_edges', 'family', 'promoter', 'go', 'pathways']


def variants(mode):
    if mode == 'ablation':
        configs = {'full': [], 'no_priors': ['--no-priors'], 'no_mediator': ['--disable-mediator'],
                   'no_dgi': ['--disable-dgi'], 'no_graph': ['--disable-prior-graph'],
                   'no_transformer': ['--disable-transformer'], 'family_only': ['--disable-promoter', '--disable-functional'],
                   'promoter_only': ['--disable-family', '--disable-functional'],
                   'functional_only': ['--disable-family', '--disable-promoter'],
                   'family_promoter': ['--disable-functional'],
                   'family_functional': ['--disable-promoter'],
                   'promoter_functional': ['--disable-family']}
        return list(configs.items())
    if mode == 'sensitivity':
        result = [('full', [])]
        for name, options in [('gnn_layers', [1, 2, 3, 4]), ('hidden_dim', [32, 64, 128, 256]),
                              ('heads', [1, 2, 4, 8]), ('lambda_dgi', [0, 0.2, 0.5, 0.8, 1.0])]:
            for value in options:
                result.append((f'{name}_{value}', ['--' + name.replace('_', '-'), str(value)]))
        return result
    if mode == 'robustness':
        return [('clean', [])] + [(f'mask_{ratio}', ['--mask-positive-ratio', str(ratio)])
                                for ratio in [0.02, 0.05, 0.07, 0.10, 0.12, 0.15]]
    return [('full', [])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--mode', choices=['benchmark', 'ablation', 'sensitivity', 'robustness'], default='benchmark')
    p.add_argument('--output', default='runs/batch')
    p.add_argument('--gene-counts', nargs='+', type=int, default=[500, 1000])
    p.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--extra-args', nargs=argparse.REMAINDER, default=[])
    args = p.parse_args()
    manifest = pd.read_csv(args.manifest).fillna('')
    if not {'dataset', 'network', 'expression', 'reference'} <= set(manifest.columns):
        raise ValueError('manifest requires dataset, network, expression, reference')
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    rows = []
    repeat_seeds = list(range(10)) if args.mode == 'robustness' and args.seeds == [1, 2, 3, 4, 5] else args.seeds
    for _, experiment in manifest.iterrows():
        for gene_count, (variant, flags), seed in itertools.product(args.gene_counts, variants(args.mode), repeat_seeds):
            run_dir = output / str(experiment['dataset']) / str(experiment['network']) / str(gene_count) / variant / str(seed)
            cmd = [sys.executable, str(root / 'train.py'), '--gene-count', str(gene_count),
                   '--seed', str(seed), '--mask-seed', str(seed), '--epochs', str(args.epochs),
                   '--output', str(run_dir)]
            for key in FIELDS:
                value = str(experiment.get(key, ''))
                if value:
                    cmd.extend(['--' + key.replace('_', '-'), value])
            cmd.extend(flags + args.extra_args)
            if args.dry_run:
                print(' '.join(cmd))
                continue
            result = subprocess.run(cmd, cwd=root)
            if result.returncode:
                raise RuntimeError('Failed: ' + ' '.join(cmd))
            report = json.loads((run_dir / 'metrics.json').read_text())
            row = {'dataset': experiment['dataset'], 'network': experiment['network'],
                   'gene_count': gene_count, 'variant': variant, 'seed': seed,
                   'auroc': report['metrics']['test']['auroc'], 'auprc': report['metrics']['test']['auprc'],
                   'masked_positive_recovery': report['masked_positive_recovery']}
            rows.append(row)
            output.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(output / 'all_results.csv', index=False)
    if rows:
        table = pd.DataFrame(rows)
        aggregated = table.groupby(['dataset', 'network', 'gene_count', 'variant'], as_index=False)[['auroc', 'auprc']].agg(['mean', 'std'])
        aggregated.to_csv(output / 'summary.csv')


if __name__ == '__main__':
    main()
