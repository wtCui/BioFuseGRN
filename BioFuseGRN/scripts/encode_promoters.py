import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def read_fasta(path):
    records = {}
    name = None
    chunks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if name is not None:
                    records[name] = ''.join(chunks).upper()
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        records[name] = ''.join(chunks).upper()
    return records


def prepare(sequence, k):
    if k <= 0:
        return sequence
    return ' '.join(sequence[i:i + k] for i in range(max(0, len(sequence) - k + 1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fasta', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--kmer', type=int, default=0)
    p.add_argument('--trust-remote-code', action='store_true')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--device', default='auto')
    args = p.parse_args()
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=args.trust_remote_code).to(device).eval()
    sequences = read_fasta(args.fasta)
    genes = list(sequences)
    encoded = []
    with torch.inference_mode():
        for offset in range(0, len(genes), args.batch_size):
            batch = [prepare(sequences[g], args.kmer) for g in genes[offset:offset + args.batch_size]]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=args.max_length,
                               return_tensors='pt').to(device)
            outputs = model(**inputs).last_hidden_state
            mask = inputs['attention_mask'].unsqueeze(-1)
            pooled = (outputs * mask).sum(1) / mask.sum(1).clamp_min(1)
            encoded.append(pooled.cpu().numpy())
    array = np.concatenate(encoded, axis=0)
    frame = pd.DataFrame(array, index=genes, columns=[f'd{i}' for i in range(array.shape[1])])
    frame.index.name = 'gene'
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output)


if __name__ == '__main__':
    main()
