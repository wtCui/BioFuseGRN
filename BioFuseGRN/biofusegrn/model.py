import torch
from torch import nn
from torch.nn import functional as F


class SparseGraphAttention(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1):
        super().__init__()
        if out_dim % heads:
            raise ValueError('out_dim must be divisible by heads')
        self.heads = heads
        self.channels = out_dim // heads
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, self.channels))
        self.att_dst = nn.Parameter(torch.empty(heads, self.channels))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = dropout
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x, edges):
        n = x.shape[0]
        src, dst = edges[0], edges[1]
        z = self.proj(x).reshape(n, self.heads, self.channels)
        logits = F.leaky_relu((z[src] * self.att_src).sum(-1) + (z[dst] * self.att_dst).sum(-1), 0.2)
        indices = dst[:, None].expand(-1, self.heads)
        maxima = torch.full((n, self.heads), -torch.inf, device=x.device, dtype=logits.dtype)
        maxima.scatter_reduce_(0, indices, logits.detach(), reduce='amax', include_self=True)
        weights = torch.exp(logits - maxima[dst])
        sums = torch.zeros((n, self.heads), device=x.device, dtype=weights.dtype)
        sums.index_add_(0, dst, weights)
        weights = weights / sums[dst].clamp_min(1e-12)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        result = torch.zeros_like(z)
        result.index_add_(0, dst, z[src] * weights.unsqueeze(-1))
        return result.reshape(n, -1) + self.bias


class PriorMediator(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.token = nn.Parameter(torch.randn(dim) * 0.02)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, priors):
        if not priors:
            raise ValueError('At least one biological prior is required')
        query = self.token.view(1, 1, -1).expand(priors[0].shape[0], -1, -1)
        tokens = torch.stack([query[:, 0], *priors], dim=1)
        fused, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        return self.norm(fused[:, 0] + query[:, 0])


class BioFuseGRN(nn.Module):
    def __init__(self, expression_dim, num_genes, family_dim=None, promoter_dim=None, use_functional=True,
                 hidden_dim=128, gnn_layers=2, attention_heads=4, transformer_layers=1,
                 dropout=0.1, use_mediator=True, use_transformer=True, use_graph=True):
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError('hidden_dim must be divisible by attention_heads')
        self.hidden_dim = hidden_dim
        self.use_functional = use_functional
        self.use_mediator = use_mediator
        self.use_transformer = use_transformer
        self.use_graph = use_graph
        self.expression_projection = nn.Sequential(nn.Linear(expression_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        self.local_layers = nn.ModuleList([
            SparseGraphAttention(hidden_dim, hidden_dim, attention_heads, dropout) for _ in range(gnn_layers)
        ])
        self.family_projection = nn.Sequential(nn.Linear(family_dim, hidden_dim), nn.LayerNorm(hidden_dim)) if family_dim else None
        self.promoter_projection = nn.Sequential(nn.Linear(promoter_dim, hidden_dim), nn.LayerNorm(hidden_dim)) if promoter_dim else None
        if use_functional:
            self.functional_embedding = nn.Embedding(num_genes, hidden_dim)
            nn.init.xavier_uniform_(self.functional_embedding.weight)
            self.functional_encoder = SparseGraphAttention(hidden_dim, hidden_dim, attention_heads, dropout)
        self.mediator = PriorMediator(hidden_dim, attention_heads, dropout) if use_mediator else None
        self.dropout = nn.Dropout(dropout)
        self.has_priors = bool(family_dim or promoter_dim or use_functional)
        input_dim = hidden_dim * (2 if self.has_priors else 1)
        if use_transformer:
            layer = nn.TransformerEncoderLayer(input_dim, attention_heads, input_dim * 2, dropout=dropout,
                                               batch_first=True, activation='gelu', norm_first=False)
            self.transformer = nn.TransformerEncoder(layer, num_layers=transformer_layers, enable_nested_tensor=False)
        self.output_projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.link_weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.dgi_weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.link_weight)
        nn.init.xavier_uniform_(self.dgi_weight)

    def compute_priors(self, family, promoter, functional_edges):
        representations = []
        if self.family_projection is not None:
            if family is None:
                raise ValueError('Missing Gene2Vec/family features')
            representations.append(self.family_projection(family))
        if self.promoter_projection is not None:
            if promoter is None:
                raise ValueError('Missing DNABERT/promoter features')
            representations.append(self.promoter_projection(promoter))
        if self.use_functional:
            if functional_edges is None:
                raise ValueError('Missing GO/pathway functional similarity graph')
            gene_vectors = self.functional_embedding.weight
            representations.append(F.elu(self.functional_encoder(gene_vectors, functional_edges)))
        if not representations:
            return None
        if self.use_mediator:
            return self.mediator(representations)
        return torch.stack(representations, dim=0).mean(0)

    def encode(self, expression, graph_edges, priors=None, projected=False):
        h = expression if projected else self.expression_projection(expression)
        if self.use_graph:
            for layer in self.local_layers:
                h = F.elu(layer(self.dropout(h), graph_edges)) + h
        if priors is not None:
            h = torch.cat([h, priors], dim=-1)
        if self.use_transformer:
            h = self.transformer(h.unsqueeze(0)).squeeze(0)
        return self.output_projection(h)

    def link_logits(self, h, pairs):
        return ((h[pairs[:, 0]] @ self.link_weight) * h[pairs[:, 1]]).sum(-1)

    def dgi_loss(self, positive_h, negative_h):
        summary = torch.sigmoid(positive_h.mean(dim=0))
        context = self.dgi_weight @ summary
        pos_logits = positive_h @ context
        neg_logits = negative_h @ context
        return (F.softplus(-pos_logits).mean() + F.softplus(neg_logits).mean()) / 2

    def forward(self, expression, graph_edges, functional_edges, family, promoter,
                pairs=None, contrastive=True):
        priors = self.compute_priors(family, promoter, functional_edges)
        projected = self.expression_projection(expression)
        positive = self.encode(projected, graph_edges, priors, projected=True)
        outputs = {'embeddings': positive}
        if pairs is not None:
            outputs['logits'] = self.link_logits(positive, pairs)
        if contrastive:
            permutation = torch.randperm(expression.shape[0], device=expression.device)
            negative = self.encode(projected[permutation], graph_edges, priors, projected=True)
            outputs['dgi_loss'] = self.dgi_loss(positive, negative)
        return outputs
