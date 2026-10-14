import numpy as np
import torch
from biofusegrn.model import BioFuseGRN, SparseGraphAttention
from biofusegrn.data import undirected_edges, split_pairs


def test_attention_shape_and_gradient():
    model = SparseGraphAttention(12, 16, heads=4, dropout=0)
    x = torch.randn(7, 12, requires_grad=True)
    graph = undirected_edges(np.array([[0, 1], [1, 2], [3, 4]]), 7)
    y = model(x, graph)
    assert y.shape == (7, 16)
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_full_model():
    torch.manual_seed(2)
    n = 15
    expression = torch.randn(n, 24)
    graph = undirected_edges(np.array([[0, 1], [1, 2], [2, 3], [3, 4]]), n)
    functional = undirected_edges(np.array([[0, 2], [1, 3]]), n)
    family = torch.randn(n, 13)
    promoter = torch.randn(n, 11)
    pairs = torch.tensor([[0, 6], [2, 9], [1, 5]])
    model = BioFuseGRN(24, n, 13, 11, True, hidden_dim=16, gnn_layers=2, attention_heads=4)
    result = model(expression, graph, functional, family, promoter, pairs)
    assert result['logits'].shape == (3,)
    assert result['embeddings'].shape == (n, 16)
    (result['logits'].square().mean() + result['dgi_loss']).backward()
    assert model.mediator.token.grad is not None
    assert model.dgi_weight.grad is not None
    assert model.functional_embedding.weight.grad is not None


def test_splits_disjoint():
    pairs = np.array([[a, b] for a in range(4) for b in range(4, 20)], dtype=np.int64)
    splits, _, _ = split_pairs(pairs, np.arange(4), 20, seed=4)
    identifier_sets = [set(map(tuple, value[0].numpy())) for value in splits.values()]
    for i in range(len(identifier_sets)):
        for j in range(i + 1, len(identifier_sets)):
            assert identifier_sets[i].isdisjoint(identifier_sets[j])


def test_scaffold_no_reverse_leakage():
    from biofusegrn.data import build_scaffold
    genes = np.array([f'g{i}' for i in range(6)])
    train = np.array([[1, 0], [1, 4]])
    held_out = np.array([[0, 1], [2, 3]])
    edges = build_scaffold(train, None, genes, {'g0', 'g1'}, held_out).T.numpy()
    pairs = set(map(tuple, edges.tolist()))
    assert (1, 0) not in pairs
    assert (0, 1) not in pairs
    assert (1, 4) in pairs
