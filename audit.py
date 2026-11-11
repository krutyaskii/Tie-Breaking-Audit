"""Streaming filtered-ranking audit. Run ``python audit.py --help``."""
import argparse
import hashlib
import json
import os
import platform
import sys
import csv
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager

import numpy as np

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

SEEDS = range(5)
ROOT = Path(__file__).resolve().parent
FIELDS = ('row', 'mode', 'target', 'B', 'T', 'retained')


@contextmanager
def run_lock(directory):
    """OS locks are released on a crash, so no stale lock blocks resume."""
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.run.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            except OSError as error:
                raise ValueError('Another evaluator holds this run directory') from error
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except OSError as error:
                raise ValueError('Another evaluator holds this run directory') from error
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def array_hash(value):
    return hashlib.sha256(np.asarray(value, dtype='<i8').tobytes()).hexdigest()


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
    raw_known = list(known)
    if any(not isinstance(i, (int, np.integer)) for i in raw_known):
        raise ValueError('Filter IDs must be integers')
    known = np.asarray(raw_known, dtype=np.int64)
    if np.any(known < 0) or np.any(known >= len(scores)):
        raise ValueError('Invalid filter ID')
    keep = np.ones(len(scores), dtype=bool)
    keep[known] = False
    keep[target] = True
    value = scores[target]
    return int(((scores > value) & keep).sum()), int(((scores == value) & keep).sum()), int(keep.sum())


def metrics(b, t):
    b, t = np.asarray(b), np.asarray(t)
    if b.dtype.kind not in 'iu' or t.dtype.kind not in 'iu':
        raise ValueError('B,T must contain integer counts')
    b, t = b.astype(np.int64), t.astype(np.int64)
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
    if any(arr[k].ndim != 1 or arr[k].dtype.kind not in 'iu' for k in FIELDS):
        raise ValueError('Per-query arrays must be one-dimensional integer arrays')
    n = len(arr['B'])
    b, t, retained = arr['B'], arr['T'], arr['retained']
    if any(len(arr[k]) != n for k in ('row','mode','target','T','retained')) or n != meta['query_count'] or np.any(b < 0) or np.any(b >= meta['nentity']) or np.any(t < 1) or np.any(t > meta['nentity']) or np.any(b+t > retained) or np.any(retained > meta['nentity']) or np.any(arr['target'] < 0) or np.any(arr['target'] >= meta['nentity']):
        raise ValueError('Invalid or incomplete query counts')
    expected_rows = meta['rows_per_mode']
    if not np.array_equal(arr['row'], np.tile(np.arange(expected_rows), 2)) or not np.array_equal(arr['mode'], np.repeat([0,1], expected_rows)):
        raise ValueError('Noncanonical or incomplete query order')
    if 'target_sha256' in meta and array_hash(arr['target']) != meta['target_sha256']:
        raise ValueError('Targets differ from the recorded split')
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
    for key in ('MRR', 'Hits@1', 'Hits@3', 'Hits@10'):
        low = summary['overall']['pessimistic'][key]
        high = summary['overall']['optimistic'][key]
        values = [summary['overall']['expected_random'][key], *[s[key] for s in samples]]
        if any(not low-1e-12 <= v <= high+1e-12 for v in values):
            raise ValueError('Ranking policy bounds failed')
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


def collect(adapter, triples, indexes, out, config, resume=False, stop_after_batches=None):
    """Persist only complete scoring batches; every state contains its config hash."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    state, meta_path = out/'per_query.npz', out/'metadata.json'
    fingerprint = json_hash(config)
    if meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding='utf-8'))
        if not resume or existing['config'] != config:
            raise ValueError('Existing run or incompatible resume configuration')
        if existing['status'] == 'complete' and digest(state) != existing['per_query_sha256']:
            raise ValueError('Completed state checksum mismatch')
        with np.load(state, allow_pickle=False) as old:
            if old['config_sha256'].item() != fingerprint:
                raise ValueError('State/config hash mismatch')
            if any(old[k].ndim != 1 or old[k].dtype != np.int64 for k in FIELDS):
                raise ValueError('Invalid state array type')
            records = {k: old[k].tolist() for k in FIELDS}
        created = existing['created_utc']
    else:
        if state.exists():
            raise ValueError('Orphan state file')
        records = {k: [] for k in FIELDS}
        created = datetime.now(timezone.utc).isoformat()

    def save(status='running'):
        save_npz(state, config_sha256=np.asarray(fingerprint), **{k: np.asarray(v,dtype=np.int64) for k,v in records.items()})
        payload = {'config':config, 'status':status, 'created_utc':created,
                   'queries_scored':len(records['B']), 'per_query_sha256':digest(state)}
        if status == 'complete':
            payload['completed_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(meta_path, payload)

    start = len(records['B'])
    n = len(triples)
    if not n or start > 2*n or any(len(v) != start for v in records.values()):
        raise ValueError('Invalid checkpoint length')
    if not np.array_equal(records['mode'], np.repeat([0,1], n)[:start]) or not np.array_equal(records['row'], np.tile(np.arange(n),2)[:start]) or not np.array_equal(records['target'], np.r_[triples[:,0],triples[:,2]][:start]):
        raise ValueError('Invalid resume query prefix')
    if any(b < 0 or t < 1 or b+t > c or c > adapter.nentity for b,t,c in zip(records['B'],records['T'],records['retained'])):
        raise ValueError('Invalid resume counts')
    if start not in (0,n,2*n) and start % n % config['batch']:
        raise ValueError('Resume must start on a scoring batch boundary')
    if not meta_path.exists():
        save()
    batches = 0
    for mode_id, mode in enumerate(('head','tail')):
        for i in range(max(0, start-mode_id*n), n, config['batch']):
            batch = triples[i:i+config['batch']]
            scores = adapter.score_candidates(batch, mode)
            if scores.shape != (len(batch), adapter.nentity) or scores.dtype != np.float32 or not np.isfinite(scores).all():
                raise ValueError('Invalid adapter output')
            for j, (h,r,t) in enumerate(batch.tolist()):
                target = h if mode == 'head' else t
                known = indexes[0].get((r,t), set()) if mode == 'head' else indexes[1].get((h,r), set())
                values = (i+j,mode_id,target,*counts(scores[j],target,known))
                for key, value in zip(FIELDS, values):
                    records[key].append(value)
            batches += 1
            if batches % 32 == 0 or i+len(batch) == n or (stop_after_batches and batches >= stop_after_batches):
                save()
                print(f'{mode}: {i+len(batch)}/{n}', flush=True)
            if stop_after_batches and batches >= stop_after_batches:
                return False
    arrays = {k: np.asarray(v,dtype=np.int64) for k,v in records.items()}
    summary = aggregate(arrays, config)
    write_json(out/'metrics.json', summary)
    save('complete')
    return True


def run(args):
    import torch
    from rotate.common import load_data, truth_indexes
    from rotate.adapter import RotatEAdapter
    from convkb.adapter import ConvKBAdapter
    if args.model == 'legacy_convkb_variant' and not args.allow_legacy:
        raise ValueError('Legacy architecture is diagnostic only: pass --allow-legacy')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable')
    if args.model == 'rotate':
        adapter = RotatEAdapter.load(args.checkpoint, args.data, args.device)
        architecture = 'rotate_complex_v1'
    else:
        adapter = ConvKBAdapter.load(args.checkpoint, args.data, args.device, allow_legacy=args.allow_legacy)
        architecture = adapter.architecture
        if (args.model == 'legacy_convkb_variant') != (architecture == 'legacy_convkb_variant'):
            raise ValueError('Requested model identity conflicts with checkpoint architecture')
    if args.chunk < 1 or args.batch < 1 or args.threads < 1:
        raise ValueError('Batch and chunk sizes must be positive')
    adapter.chunk_size = args.chunk
    splits, manifest = load_data(args.data)
    if adapter.metadata['data_manifest'] != manifest:
        raise ValueError('Checkpoint dataset manifest differs from verified benchmark')
    if adapter.metadata.get('score_direction') != 'higher_is_better':
        raise ValueError('Unsupported checkpoint score direction')
    if args.limit is not None and (args.limit <= 0 or args.split != 'valid'):
        raise ValueError('Subsets are permitted only on validation')
    triples = splits[args.split][:args.limit] if args.limit else splits[args.split]
    indexes = truth_indexes(splits)
    source = {p.relative_to(ROOT).as_posix(): digest(p) for pattern in ('audit*.py','rotate/*.py','convkb/*.py','vendor/kg_reeval/*.py') for p in sorted(ROOT.glob(pattern))}
    config = {'model': args.model, 'checkpoint_sha256': digest(args.checkpoint), 'data_sha256': {k:v['sha256'] for k,v in manifest['files'].items()}, 'source_sha256': source, 'device': args.device, 'gpu': torch.cuda.get_device_name(0) if args.device == 'cuda' else None, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'cudnn_deterministic': torch.backends.cudnn.deterministic, 'cudnn_benchmark': torch.backends.cudnn.benchmark, 'numpy': np.__version__, 'python': sys.version, 'platform': platform.platform(), 'batch': args.batch, 'chunk': args.chunk, 'split': args.split, 'rows_per_mode': len(triples), 'query_count': 2*len(triples), 'nentity': adapter.nentity, 'filter': 'union(train,valid,test), retain target', 'order': 'head then tail, split row order', 'seeds': list(SEEDS), 'precision': 'float32', 'tf32': False}
    config.update(architecture=architecture, cpu_threads=args.threads,
                  target_sha256=array_hash(np.r_[triples[:,0],triples[:,2]]),
                  selected_step=adapter.metadata.get('step'), training_seed=adapter.metadata['config'].get('seed'),
                  archive_sha256=digest(args.archive) if args.archive else None,
                  deterministic_algorithms=True, processor=platform.processor(),
                  cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
                  matmul_precision=torch.get_float32_matmul_precision(),
                  interop_threads=torch.get_num_interop_threads(),
                  mkldnn_enabled=torch.backends.mkldnn.enabled,
                  torch_build=torch.__config__.show())
    args.out.mkdir(parents=True,exist_ok=True)
    commit = subprocess.run(['git','-c',f'safe.directory={ROOT.as_posix()}','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
    dirty = subprocess.run(['git','-c',f'safe.directory={ROOT.as_posix()}','status','--porcelain'],cwd=ROOT,capture_output=True,text=True)
    write_json(args.out/'invocation.json', {'argv':sys.argv, 'git_commit':commit.stdout.strip(),
               'git_dirty':dirty.stdout.splitlines(), 'started_utc':datetime.now(timezone.utc).isoformat()})
    started = time.perf_counter()
    complete = collect(adapter, triples, indexes, args.out, config, args.resume, args.stop_after_batches)
    print(f'Elapsed: {time.perf_counter()-started:.2f}s; complete={complete}', flush=True)
    if complete:
        export_report(args.out)


def load_run(directory):
    directory = Path(directory)
    meta = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    path = directory/'per_query.npz'
    if meta['status'] != 'complete' or digest(path) != meta['per_query_sha256']:
        raise ValueError('Incomplete run or per-query hash mismatch')
    with np.load(path, allow_pickle=False) as data:
        if data['config_sha256'].item() != json_hash(meta['config']):
            raise ValueError('Per-query data/config hash mismatch')
        result = aggregate(data, meta['config'])
    return meta, result


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_report(directory):
    directory = Path(directory)
    meta, result = load_run(directory)
    write_json(directory/'metrics.json', result)
    write_csv(directory/'random_seeds.csv', result['seeds'])
    write_json(directory/'tie_histogram.json', result['ties'])
    rows = []
    for scope in ('overall','head','tail'):
        for policy in ('optimistic','expected_random','pessimistic'):
            rows.append({'model':meta['config']['model'], 'split':meta['config']['split'],
                         'scope':scope, 'policy':policy,
                         'queries':meta['config']['query_count'] if scope == 'overall' else meta['config']['rows_per_mode'],
                         'MRR_opt_minus_expected':result[scope]['gap'], **result[scope][policy]})
    write_csv(directory/'main_table.csv', rows)
    table_markdown(directory/'main_table.md', rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    boundaries = np.array([0,1,5,17,65,257,1025,4097,16385])
    hist = result['ties']['histogram']
    totals = [sum(v for t,v in hist.items() if lo <= int(t)-1 < hi) for lo,hi in zip(boundaries[:-1],boundaries[1:])]
    if sum(totals) != meta['config']['query_count']:
        raise ValueError('Binned tie histogram lost queries')
    fig, ax = plt.subplots(figsize=(9,4))
    ax.bar(['0','1–4','5–16','17–64','65–256','257–1024','1025–4096','4097–16384'],totals)
    ax.set(xlabel='Other retained candidates tied with the target (T − 1)', ylabel='Queries', title=f"{meta['config']['model']} / {meta['config']['split']}")
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fig.savefig(directory/'tie_distribution.png',dpi=160)
    plt.close(fig)
    return rows


def table_markdown(path, rows):
    keys = list(rows[0])
    lines = ['| '+' | '.join(keys)+' |', '| '+' | '.join('---' for _ in keys)+' |']
    for row in rows:
        lines.append('| '+' | '.join(f'{row[k]:.12g}' if isinstance(row[k],float) else str(row[k]) for k in keys)+' |')
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run')
    p.add_argument('--model', choices=['rotate','convkb','legacy_convkb_variant'], required=True)
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
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--archive', type=Path)
    p.add_argument('--stop-after-batches', type=int, help='Checkpoint and stop for resume testing')
    a = sub.add_parser('aggregate')
    a.add_argument('directory', type=Path)
    tables = sub.add_parser('table')
    tables.add_argument('directories', nargs='+', type=Path)
    tables.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        with run_lock(args.out):
            run(args)
    elif args.command == 'aggregate':
        export_report(args.directory)
        print(json.dumps(load_run(args.directory)[1],indent=2))
    else:
        rows = []
        signatures = []
        for directory in args.directories:
            meta, _ = load_run(directory)
            signatures.append((meta['config']['split'],meta['config']['rows_per_mode'],meta['config']['data_sha256']))
            rows.extend(export_report(directory))
        if any(s != signatures[0] for s in signatures):
            raise ValueError('Cannot combine runs with different splits or datasets')
        args.out.mkdir(parents=True,exist_ok=True)
        write_csv(args.out/'main_table.csv',rows)
        table_markdown(args.out/'main_table.md',rows)


if __name__ == '__main__':
    main()
