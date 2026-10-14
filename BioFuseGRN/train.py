import argparse
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from biofusegrn import BioFuseGRN
from biofusegrn.data import (
    read_expression, read_tfs, load_reference, select_genes, standardize_expression,
    load_feature_table, build_functional_edges, map_edges, split_pairs,
    build_scaffold, undirected_edges, save_splits
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--expression', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--tfs')
    p.add_argument('--prior-edges')
    p.add_argument('--family')
    p.add_argument('--promoter')
    p.add_argument('--go')
    p.add_argument('--pathways')
    p.add_argument('--output', default='runs/biofusegrn')
    p.add_argument('--gene-count', type=int, default=500)
    p.add_argument('--log1p', action='store_true')
    p.add_argument('--hidden-dim', type=int, default=128)
    p.add_argument('--gnn-layers', type=int, default=2)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--transformer-layers', type=int, default=1)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--beta-go', type=float, default=0.5)
    p.add_argument('--beta-pathway', type=float, default=0.5)
    p.add_argument('--lambda-dgi', type=float, default=0.8)
    p.add_argument('--learning-rate', type=float, default=0.001)
    p.add_argument('--weight-decay', type=float, default=0.0001)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=25)
    p.add_argument('--negative-ratio', type=int, default=6)
    p.add_argument('--mask-positive-ratio', type=float, default=0.0)
    p.add_argument('--mask-seed', type=int, default=0)
    p.add_argument('--sparse-cutoff', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--no-priors', action='store_true')
    p.add_argument('--disable-family', action='store_true')
    p.add_argument('--disable-promoter', action='store_true')
    p.add_argument('--disable-functional', action='store_true')
    p.add_argument('--disable-mediator', action='store_true')
    p.add_argument('--disable-dgi', action='store_true')
    p.add_argument('--disable-prior-graph', action='store_true')
    p.add_argument('--disable-transformer', action='store_true')
    return p.parse_args()


def metrics(labels, logits):
    predictions = torch.sigmoid(logits).detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    return {'auroc': float(roc_auc_score(y, predictions)),
            'auprc': float(average_precision_score(y, predictions))}


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else
                          'cpu' if args.device == 'auto' else args.device)
    reference = load_reference(args.reference)
    gene_ids, expression = read_expression(args.expression, args.log1p)
    tfs = read_tfs(args.tfs, reference)
    gene_ids, expression, _ = select_genes(gene_ids, expression, tfs, args.gene_count)
    expression = torch.from_numpy(standardize_expression(expression)).to(device)
    tf_indices = np.flatnonzero(np.isin(gene_ids, list(tfs)))
    positives = map_edges(reference, gene_ids, tfs)
    splits, pos_parts, density = split_pairs(positives, tf_indices, len(gene_ids), args.seed,
                                            args.sparse_cutoff, args.negative_ratio)
    if not 0 <= args.mask_positive_ratio < 1:
        raise ValueError('--mask-positive-ratio must be in [0,1)')
    mask_rng = np.random.default_rng(args.mask_seed)
    train_pairs, train_labels = splits['train']
    train_pos_indices = torch.nonzero(train_labels == 1, as_tuple=True)[0].numpy()
    num_mask = int(round(len(train_pos_indices) * args.mask_positive_ratio))
    selected_indices = mask_rng.choice(train_pos_indices, size=num_mask, replace=False)
    masked_pairs = train_pairs[selected_indices].numpy()
    train_labels[selected_indices] = 0
    unmasked_positive = train_pairs[train_labels == 1].numpy()
    withheld = np.concatenate([splits['val'][0].numpy(), splits['test'][0].numpy(), masked_pairs], axis=0)
    if args.disable_prior_graph:
        graph = undirected_edges(np.empty((0, 2), dtype=np.int64), len(gene_ids))
    else:
        graph = build_scaffold(unmasked_positive, args.prior_edges, gene_ids, tfs, withheld)
    family_path = None if args.no_priors or args.disable_family else args.family
    promoter_path = None if args.no_priors or args.disable_promoter else args.promoter
    go_path = None if args.no_priors or args.disable_functional else args.go
    pw_path = None if args.no_priors or args.disable_functional else args.pathways
    family, family_coverage = load_feature_table(family_path, gene_ids)
    promoter, promoter_coverage = load_feature_table(promoter_path, gene_ids)
    fsg = build_functional_edges(gene_ids, go_path, pw_path, args.top_k, args.beta_go, args.beta_pathway)
    functional_enabled = fsg is not None
    model = BioFuseGRN(
        expression_dim=expression.shape[1], num_genes=len(gene_ids),
        family_dim=family.shape[1] if family is not None else None,
        promoter_dim=promoter.shape[1] if promoter is not None else None,
        use_functional=functional_enabled, hidden_dim=args.hidden_dim,
        gnn_layers=args.gnn_layers, attention_heads=args.heads,
        transformer_layers=args.transformer_layers, dropout=args.dropout,
        use_mediator=not args.disable_mediator,
        use_transformer=not args.disable_transformer,
        use_graph=not args.disable_prior_graph
    ).to(device)
    graph = graph.to(device)
    fsg = fsg.to(device) if fsg is not None else None
    family = family.to(device) if family is not None else None
    promoter = promoter.to(device) if promoter is not None else None
    splits = {k: (p.to(device), y.to(device)) for k, (p, y) in splits.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_ap = float('-inf')
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(expression, graph, fsg, family, promoter, splits['train'][0],
                    contrastive=not args.disable_dgi and args.lambda_dgi > 0)
        grn_loss = F.binary_cross_entropy_with_logits(out['logits'], splits['train'][1])
        dgi_loss = out.get('dgi_loss', torch.zeros((), device=device))
        loss = grn_loss + args.lambda_dgi * dgi_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            h = model(expression, graph, fsg, family, promoter, contrastive=False)['embeddings']
            val_scores = metrics(splits['val'][1], model.link_logits(h, splits['val'][0]))
        row = {'epoch': epoch, 'loss': float(loss.item()), 'grn_loss': float(grn_loss.item()),
               'dgi_loss': float(dgi_loss.item()), 'val_auroc': val_scores['auroc'],
               'val_auprc': val_scores['auprc']}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_scores['auprc'] > best_ap:
            best_ap = val_scores['auprc']
            stale = 0
            torch.save({'state_dict': model.state_dict(), 'model_args': {
                'expression_dim': expression.shape[1], 'num_genes': len(gene_ids),
                'family_dim': family.shape[1] if family is not None else None,
                'promoter_dim': promoter.shape[1] if promoter is not None else None,
                'use_functional': functional_enabled, 'hidden_dim': args.hidden_dim,
                'gnn_layers': args.gnn_layers, 'attention_heads': args.heads,
                'transformer_layers': args.transformer_layers, 'dropout': args.dropout,
                'use_mediator': not args.disable_mediator,
                'use_transformer': not args.disable_transformer,
                'use_graph': not args.disable_prior_graph},
                'gene_ids': gene_ids.tolist(), 'tf_ids': gene_ids[tf_indices].tolist(),
                'epoch': epoch, 'seed': args.seed}, output / 'best.pt')
        else:
            stale += 1
        if stale >= args.patience:
            break
    checkpoint = torch.load(output / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.no_grad():
        h = model(expression, graph, fsg, family, promoter, contrastive=False)['embeddings']
        final_metrics = {}
        for name in ['train', 'val', 'test']:
            pairs, labels = splits[name]
            final_metrics[name] = metrics(labels, model.link_logits(h, pairs))
        np.save(output / 'embeddings.npy', h.cpu().numpy())
        np.save(output / 'link_weight.npy', model.link_weight.detach().cpu().numpy())
    if len(masked_pairs):
        with torch.no_grad():
            masked_tensor = torch.from_numpy(masked_pairs).to(device)
            masked_scores = torch.sigmoid(model.link_logits(h, masked_tensor)).cpu().numpy()
        pd.DataFrame({'tf': gene_ids[masked_pairs[:, 0]], 'target': gene_ids[masked_pairs[:, 1]],
                      'score': masked_scores}).to_csv(output / 'masked_positive_recovery.csv', index=False)
    pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
    pd.Series(gene_ids, name='gene').to_csv(output / 'genes.csv', index=False)
    pd.Series(gene_ids[tf_indices], name='tf').to_csv(output / 'tfs.csv', index=False)
    save_splits({k: (p.cpu(), y.cpu()) for k, (p, y) in splits.items()}, gene_ids, output)
    report = {'best_epoch': checkpoint['epoch'], 'density': density, 'num_genes': len(gene_ids),
              'num_tfs': len(tf_indices), 'num_positive_edges': len(positives),
              'family_coverage': family_coverage, 'promoter_coverage': promoter_coverage,
              'metrics': final_metrics, 'masked_positive_recovery': float((masked_scores >= 0.5).mean()) if len(masked_pairs) else None,
              'arguments': vars(args)}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({'best_epoch': checkpoint['epoch'], 'metrics': final_metrics}, indent=2), flush=True)


if __name__ == '__main__':
    main()
