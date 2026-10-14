from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def read_expression(path, log1p=False):
    table = pd.read_csv(path, index_col=0)
    table.index = table.index.astype(str)
    if not table.index.is_unique:
        raise ValueError('Expression matrix contains duplicate gene IDs')
    values = table.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Expression matrix has missing or non-finite values')
    if log1p:
        if (values < 0).any():
            raise ValueError('Cannot apply log1p to negative expression values')
        values = np.log1p(values)
    return table.index.to_numpy(), values


def read_tfs(path, reference):
    if path:
        with open(path) as handle:
            names = [line.strip().split(',')[0].split('\t')[0] for line in handle if line.strip()]
        return set(names) - {'tf', 'TF', 'gene'}
    return set(reference['tf'].astype(str))


def load_reference(path):
    edges = pd.read_csv(path, dtype=str)
    if not {'tf', 'target'} <= set(edges.columns):
        raise ValueError('Reference CSV must contain tf and target columns')
    if 'label' in edges.columns:
        edges = edges.loc[edges['label'].astype(int).eq(1)]
    return edges[['tf', 'target']].dropna().drop_duplicates()


def select_genes(gene_ids, expression, tfs, count):
    valid_tfs = set(gene_ids) & set(tfs)
    if not valid_tfs:
        raise ValueError('No transcription factors overlap the expression matrix')
    if count <= 0:
        idx = np.arange(len(gene_ids))
    else:
        tf_mask = np.isin(gene_ids, list(valid_tfs))
        tf_idx = np.flatnonzero(tf_mask)
        rest_idx = np.flatnonzero(~tf_mask)
        variances = np.var(expression[rest_idx], axis=1)
        selected = rest_idx[np.argsort(-variances, kind='stable')[:count]]
        idx = np.sort(np.concatenate([tf_idx, selected]))
    return gene_ids[idx], expression[idx], idx


def standardize_expression(expression):
    means = expression.mean(axis=1, keepdims=True)
    standard = expression.std(axis=1, keepdims=True)
    return ((expression - means) / np.maximum(standard, 1e-6)).astype(np.float32)


def load_feature_table(path, gene_ids):
    if not path:
        return None, None
    table = pd.read_csv(path, index_col=0)
    table.index = table.index.astype(str)
    if not table.index.is_unique:
        raise ValueError(f'Duplicate genes in {path}')
    if table.shape[1] == 0:
        raise ValueError(f'No embedding columns in {path}')
    matrix = table.reindex(gene_ids).apply(pd.to_numeric, errors='coerce')
    present = matrix.notna().all(axis=1)
    if not present.any():
        raise ValueError(f'No valid vectors matching expression genes in {path}')
    return torch.from_numpy(matrix.fillna(0).to_numpy(dtype=np.float32)), float(present.mean())


def _annotation_matrix(path, gene_ids, term_name):
    if not path:
        return sp.csr_matrix((len(gene_ids), 0), dtype=np.float32)
    table = pd.read_csv(path, dtype=str)
    if not {'gene', term_name} <= set(table.columns):
        raise ValueError(f'{path} must contain gene and {term_name}')
    table = table[['gene', term_name]].dropna().drop_duplicates()
    mapping = {g: i for i, g in enumerate(gene_ids)}
    table = table.loc[table['gene'].isin(mapping)]
    if table.empty:
        return sp.csr_matrix((len(gene_ids), 0), dtype=np.float32)
    columns, _ = pd.factorize(table[term_name], sort=True)
    rows = table['gene'].map(mapping).to_numpy(dtype=np.int64)
    mat = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, columns)),
                        shape=(len(gene_ids), int(columns.max()) + 1)).tocsr()
    mat.data[:] = 1
    return mat


def _jaccard_from_annotations(matrix):
    n = matrix.shape[0]
    if matrix.shape[1] == 0 or matrix.nnz == 0:
        return sp.csr_matrix((n, n), dtype=np.float32)
    intersections = (matrix @ matrix.T).tocoo()
    counts = np.asarray(matrix.sum(axis=1)).ravel()
    denominator = counts[intersections.row] + counts[intersections.col] - intersections.data
    values = np.divide(intersections.data, denominator, out=np.zeros_like(intersections.data), where=denominator > 0)
    result = sp.coo_matrix((values, (intersections.row, intersections.col)), shape=(n, n)).tocsr()
    result.setdiag(0)
    result.eliminate_zeros()
    return result


def build_functional_edges(gene_ids, go_path, pathway_path, top_k=10, beta_go=0.5, beta_pw=0.5):
    if not go_path and not pathway_path:
        return None
    go = _jaccard_from_annotations(_annotation_matrix(go_path, gene_ids, 'term'))
    pw = _jaccard_from_annotations(_annotation_matrix(pathway_path, gene_ids, 'pathway'))
    similarity = (beta_go * go + beta_pw * pw).tocsr()
    src, dst = [], []
    for gene in range(len(gene_ids)):
        start, end = similarity.indptr[gene], similarity.indptr[gene + 1]
        neighbors = similarity.indices[start:end]
        scores = similarity.data[start:end]
        selected = np.argsort(-scores, kind='stable')[:top_k]
        src.extend(neighbors[selected].tolist())
        dst.extend([gene] * len(selected))
    return undirected_edges(np.stack([src, dst], axis=1) if src else np.empty((0, 2), dtype=np.int64), len(gene_ids))


def undirected_edges(edges, num_nodes):
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    self_edges = np.arange(num_nodes, dtype=np.int64)
    loops = np.stack([self_edges, self_edges], axis=1)
    both = np.concatenate([edges, edges[:, ::-1], loops], axis=0)
    both = np.unique(both, axis=0)
    return torch.from_numpy(both.T.copy()).long()


def map_edges(table, gene_ids, valid_tfs=None):
    mapping = {g: i for i, g in enumerate(gene_ids)}
    if valid_tfs is not None:
        table = table.loc[table['tf'].isin(valid_tfs)]
    table = table.loc[table['tf'].isin(mapping) & table['target'].isin(mapping)]
    pairs = np.stack([table['tf'].map(mapping), table['target'].map(mapping)], axis=1).astype(np.int64)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return np.unique(pairs, axis=0)


def split_pairs(positives, tf_indices, num_genes, seed=0, sparse_cutoff=0.1, negative_ratio=6):
    rng = np.random.default_rng(seed)
    positives = np.asarray(positives, dtype=np.int64).reshape(-1, 2)
    if len(positives) < 5:
        raise ValueError('At least five positive interactions are required for a 60:20:20 split')
    positives = positives[rng.permutation(len(positives))]
    n_train = max(1, int(len(positives) * 0.6))
    n_valid = max(1, int(len(positives) * 0.2))
    pos_parts = [positives[:n_train], positives[n_train:n_train + n_valid], positives[n_train + n_valid:]]
    potential = np.stack(np.meshgrid(tf_indices, np.arange(num_genes), indexing='ij'), axis=-1).reshape(-1, 2)
    potential = potential[potential[:, 0] != potential[:, 1]]
    positive_ids = np.unique(positives[:, 0] * num_genes + positives[:, 1])
    potential_ids = potential[:, 0] * num_genes + potential[:, 1]
    negatives = potential[~np.isin(potential_ids, positive_ids)]
    negatives = negatives[rng.permutation(len(negatives))]
    neg_train_end = int(len(negatives) * 0.6)
    neg_valid_end = int(len(negatives) * 0.8)
    negatives_parts = [negatives[:neg_train_end], negatives[neg_train_end:neg_valid_end], negatives[neg_valid_end:]]
    density = len(positives) / max(1, len(potential))
    if density < sparse_cutoff:
        max_train_neg = min(len(negatives_parts[0]), len(pos_parts[0]) * negative_ratio)
        negatives_parts[0] = negatives_parts[0][:max_train_neg]
    splits = {}
    for name, pos, neg in zip(['train', 'val', 'test'], pos_parts, negatives_parts):
        pairs = np.concatenate([pos, neg], axis=0)
        labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.float32)
        perm = rng.permutation(len(pairs))
        splits[name] = (torch.from_numpy(pairs[perm]).long(), torch.from_numpy(labels[perm]))
    return splits, pos_parts, density


def build_scaffold(train_positives, prior_path, gene_ids, valid_tfs, val_test_pairs):
    n = len(gene_ids)
    blocked = np.unique(np.minimum(val_test_pairs[:, 0], val_test_pairs[:, 1]) * n +
                        np.maximum(val_test_pairs[:, 0], val_test_pairs[:, 1]))
    base = np.asarray(train_positives, dtype=np.int64).reshape(-1, 2)
    if prior_path:
        prior = map_edges(load_reference(prior_path), gene_ids, valid_tfs)
        base = np.concatenate([base, prior], axis=0)
    unordered = np.minimum(base[:, 0], base[:, 1]) * n + np.maximum(base[:, 0], base[:, 1])
    return undirected_edges(base[~np.isin(unordered, blocked)], n)


def save_splits(splits, gene_ids, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, (pairs, labels) in splits.items():
        pd.DataFrame({'tf': gene_ids[pairs[:, 0]], 'target': gene_ids[pairs[:, 1]],
                      'label': labels.numpy().astype(int)}).to_csv(destination / f'{name}_pairs.csv', index=False)
