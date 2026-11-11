"""Streaming filtered-ranking audit. Run ``python audit.py --help``."""
import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SEEDS = range(5)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def counts(scores, target, known):
    scores = np.asarray(scores)
    if scores.ndim != 1 or scores.dtype != np.float32 or not np.isfinite(scores).all():
        raise ValueError('Expected a finite one-dimensional float32 score row')
    if not isinstance(target, (int, np.integer)) or not 0 <= target < len(scores):
        raise ValueError('Invalid target ID')
    known = np.asarray(list(known), dtype=np.int64)
    if np.any(known < 0) or np.any(known >= len(scores)):
        raise ValueError('Invalid filter ID')
    keep = np.ones(len(scores), dtype=bool)
    keep[known] = False
    keep[target] = True
    value = scores[target]
    return int(((scores > value) & keep).sum()), int(((scores == value) & keep).sum()), int(keep.sum())


def metrics(b, t):
    b, t = np.asarray(b, dtype=np.int64), np.asarray(t, dtype=np.int64)
    if b.ndim != 1 or b.shape != t.shape or not len(b) or np.any(b < 0) or np.any(t < 1):
        raise ValueError('Invalid B,T arrays')
    h = np.r_[0., np.cumsum(1. / np.arange(1, int((b+t).max())+1, dtype=np.float64))]
    rr = (h[b+t]-h[b])/t
    rr[t == 1] = 1. / (b[t == 1]+1)
    result = {}
    for name, rank in [('optimistic', b+1), ('pessimistic', b+t)]:
        result[name] = {'MRR': float(np.mean(1./rank)), **{f'Hits@{k}': float(np.mean(rank <= k)) for k in (1,3,10)}}
    result['expected_random'] = {'MRR': float(rr.mean()), **{f'Hits@{k}': float(np.mean(np.clip(k-b, 0, t)/t)) for k in (1,3,10)}}
    result['gap'] = result['optimistic']['MRR'] - result['expected_random']['MRR']
    return result


def aggregate(arr, meta):
    n = len(arr['B'])
    b, t, retained = arr['B'], arr['T'], arr['retained']
    if any(len(arr[k]) != n for k in ('row','mode','target','T','retained')) or n != meta['query_count'] or np.any(b < 0) or np.any(t < 1) or np.any(b+t > retained) or np.any(retained > meta['nentity']) or np.any(arr['target'] < 0) or np.any(arr['target'] >= meta['nentity']):
        raise ValueError('Invalid or incomplete query counts')
    expected_rows = meta['rows_per_mode']
    if not np.array_equal(arr['row'], np.tile(np.arange(expected_rows), 2)) or not np.array_equal(arr['mode'], np.repeat([0,1], expected_rows)):
        raise ValueError('Noncanonical or incomplete query order')
    summary = {'overall': metrics(b,t), 'head': metrics(b[:expected_rows],t[:expected_rows]), 'tail': metrics(b[expected_rows:],t[expected_rows:])}
    unique, freq = np.unique(t, return_counts=True)
    summary['ties'] = {'histogram': {str(int(x)): int(y) for x,y in zip(unique,freq)}, 'count': int((t>1).sum()), 'fraction': float((t>1).mean()), 'min': int(t.min()), 'median': float(np.median(t)), 'p90': float(np.percentile(t,90)), 'p99': float(np.percentile(t,99)), 'max': int(t.max())}
    samples = []
    for seed in SEEDS:
        rank = b + np.random.default_rng(seed).integers(1, t+1)
        samples.append({'seed': seed, 'MRR': float(np.mean(1./rank)), **{f'Hits@{k}': float(np.mean(rank<=k)) for k in (1,3,10)}})
    summary['seeds'] = samples
    summary['seed_mean'] = {key: float(np.mean([s[key] for s in samples])) for key in ('MRR','Hits@1','Hits@3','Hits@10')}
    summary['seed_std_ddof1'] = {key: float(np.std([s[key] for s in samples], ddof=1)) for key in summary['seed_mean']}
    summary['seed_mean_minus_expected'] = {key: summary['seed_mean'][key]-summary['overall']['expected_random'][key] for key in summary['seed_mean']}
    return summary


def save_npz(path, **values):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **values)
    os.replace(tmp, path)


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def run(args):
    import torch
    from rotate.common import load_data, truth_indexes
    from rotate.adapter import RotatEAdapter
    from convkb.adapter import ConvKBAdapter
    if args.model == 'legacy_convkb_variant' and not args.allow_legacy:
        raise ValueError('Legacy architecture is diagnostic only: pass --allow-legacy')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    adapter = (RotatEAdapter if args.model == 'rotate' else ConvKBAdapter).load(args.checkpoint, args.data, args.device)
    if args.chunk < 1 or args.batch < 1:
        raise ValueError('Batch and chunk sizes must be positive')
    adapter.chunk_size = args.chunk
    splits, manifest = load_data(args.data)
    if args.limit is not None and (args.limit <= 0 or args.split != 'valid'):
        raise ValueError('Subsets are permitted only on validation')
    triples = splits[args.split][:args.limit] if args.limit else splits[args.split]
    indexes = truth_indexes(splits)
    source = {str(p.relative_to(Path.cwd())): digest(p) for pattern in ('audit.py','rotate/*.py','convkb/*.py','vendor/kg_reeval/*.py') for p in sorted(Path.cwd().glob(pattern))}
    config = {'model': args.model, 'checkpoint_sha256': digest(args.checkpoint), 'data_sha256': {k:v['sha256'] for k,v in manifest['files'].items()}, 'source_sha256': source, 'device': args.device, 'gpu': torch.cuda.get_device_name(0) if args.device == 'cuda' else None, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'cudnn_deterministic': torch.backends.cudnn.deterministic, 'cudnn_benchmark': torch.backends.cudnn.benchmark, 'numpy': np.__version__, 'python': sys.version, 'platform': platform.platform(), 'batch': args.batch, 'chunk': args.chunk, 'split': args.split, 'rows_per_mode': len(triples), 'query_count': 2*len(triples), 'nentity': adapter.nentity, 'filter': 'union(train,valid,test), retain target', 'order': 'head then tail, split row order', 'seeds': list(SEEDS), 'precision': 'float32', 'tf32': False}
    args.out.mkdir(parents=True, exist_ok=True)
    state, meta_path = args.out/'per_query.npz', args.out/'metadata.json'
    if meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding='utf-8'))
        if not args.resume or existing['config'] != config:
            raise ValueError('Existing run or incompatible resume configuration')
        with np.load(state, allow_pickle=False) as old:
            records = {k: old[k].tolist() for k in ('row','mode','target','B','T','retained')}
    else:
        if state.exists():
            raise ValueError('Orphan state file')
        records = {k: [] for k in ('row','mode','target','B','T','retained')}
        save_npz(state, **{k: np.asarray(v,dtype=np.int64) for k,v in records.items()})
        write_json(meta_path, {'config':config, 'status':'running'})
    start = len(records['B'])
    if start > 2*len(triples):
        raise ValueError('Invalid checkpoint length')
    expected_mode = np.repeat([0, 1], len(triples))[:start]
    expected_row = np.tile(np.arange(len(triples)), 2)[:start]
    expected_target = np.r_[triples[:,0], triples[:,2]][:start]
    if any(len(v) != start for v in records.values()) or not np.array_equal(records['mode'], expected_mode) or not np.array_equal(records['row'], expected_row) or not np.array_equal(records['target'], expected_target):
        raise ValueError('Invalid resume query prefix')
    if start and start not in (len(triples), 2*len(triples)) and start % len(triples) % args.batch:
        raise ValueError('Resume must start on a scoring batch boundary')
    for mode_id, mode in enumerate(('head','tail')):
        offset = mode_id*len(triples)
        for i in range(max(0, start-offset), len(triples), args.batch):
            batch = triples[i:i+args.batch]
            scores = adapter.score_candidates(batch, mode)
            if scores.shape != (len(batch), adapter.nentity) or scores.dtype != np.float32 or not np.isfinite(scores).all():
                raise ValueError('Invalid adapter output')
            for j, (h,r,t) in enumerate(batch.tolist()):
                target = h if mode == 'head' else t
                known = indexes[0][(r,t)] if mode == 'head' else indexes[1][(h,r)]
                b, tied, retained = counts(scores[j],target,known)
                for key, value in zip(records, (i+j,mode_id,target,b,tied,retained)):
                    records[key].append(value)
            save_npz(state, **{k: np.asarray(v,dtype=np.int64) for k,v in records.items()})
            print(f'{mode}: {min(i+len(batch),len(triples))}/{len(triples)}', flush=True)
    with np.load(state, allow_pickle=False) as arr:
        summary = aggregate(arr,config)
    write_json(args.out/'metrics.json', summary)
    write_json(meta_path, {'config':config, 'status':'complete', 'completed_utc':datetime.now(timezone.utc).isoformat(), 'per_query_sha256':digest(state)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run')
    p.add_argument('--model', choices=['rotate','legacy_convkb_variant'], required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--split', choices=['valid','test'], default='valid')
    p.add_argument('--limit', type=int)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--chunk', type=int, default=2048)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--allow-legacy', action='store_true')
    a = sub.add_parser('aggregate')
    a.add_argument('directory', type=Path)
    args = parser.parse_args()
    if args.command == 'run':
        run(args)
    else:
        meta = json.loads((args.directory/'metadata.json').read_text(encoding='utf-8'))
        path = args.directory/'per_query.npz'
        if meta['status'] != 'complete' or digest(path) != meta['per_query_sha256']:
            raise ValueError('Incomplete run or per-query hash mismatch')
        with np.load(path, allow_pickle=False) as data:
            result = aggregate(data,meta['config'])
        write_json(args.directory/'metrics.json',result)
        print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
