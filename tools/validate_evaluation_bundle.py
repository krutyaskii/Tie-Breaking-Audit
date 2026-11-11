"""Independently validate a returned audit ZIP and import compact test evidence.

Uses only NumPy and the standard library; does not execute code from the ZIP.
Example: python tools/validate_evaluation_bundle.py bundle.zip --out reports/final_audit
"""
import argparse
import csv
import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    'rotate': '5ec0bbc39e9568f6362ac38010cc752531dfbdba7aaea029661838707ce8b622',
    'convkb': 'dfa921d625ae229320106a6321844991a385ca4f9a3cb87117eb29e212cbfe46',
}
ARCHIVES = {
    'rotate': '2c66eecac4fe499f8c3d695cb107d1c6025cda6758c875ea0461ba59421a0d1b',
    'convkb': '9008666f453d8306531b6cbf884cbd79b22d380198ba5124a205e47745065a5b',
}
KEYS = ('MRR', 'Hits@1', 'Hits@3', 'Hits@10')
FIELDS = ('row', 'mode', 'target', 'B', 'T', 'retained')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(content):
    return hashlib.sha256(content).hexdigest()


def array_sha(values):
    return sha(np.asarray(values, dtype='<i8').tobytes())


def close(actual, expected, label):
    require(math.isfinite(float(actual)) and math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-13), f'{label}: {actual} != {expected}')


def rank_metrics(ranks):
    return {'MRR': math.fsum(1 / int(r) for r in ranks) / len(ranks),
            **{f'Hits@{k}': sum(int(r) <= k for r in ranks) / len(ranks) for k in (1,3,10)}}


def direct_metrics(b, t):
    # Independent direct summation; no harmonic-number subtraction.
    expected_rr = [math.fsum(1/r for r in range(int(x)+1, int(x+y)+1))/int(y) for x,y in zip(b,t)]
    return {'optimistic': rank_metrics(b+1), 'pessimistic': rank_metrics(b+t),
            'expected_random': {'MRR':math.fsum(expected_rr)/len(b),
                                **{f'Hits@{k}': math.fsum(min(max(k-int(x),0),int(y))/int(y) for x,y in zip(b,t))/len(b) for k in (1,3,10)}}}


def load_benchmark(repo, data):
    manifest = json.loads((repo/'reports/data_manifest.json').read_text(encoding='utf-8'))
    for name, reference in manifest['files'].items():
        require(sha((data/name).read_bytes()) == reference['sha256'], f'Dataset hash: {name}')
    entities = {label:int(number) for number,label in (line.split('\t') for line in (data/'entities.dict').read_text(encoding='utf-8').splitlines())}
    relations = {label:int(number) for number,label in (line.split('\t') for line in (data/'relations.dict').read_text(encoding='utf-8').splitlines())}
    heads,tails = defaultdict(set),defaultdict(set)
    test = []
    for split in ('train','valid','test'):
        for line in (data/f'{split}.txt').read_text(encoding='utf-8').splitlines():
            h,r,t = line.split('\t')
            h,r,t = entities[h],relations[r],entities[t]
            heads[(r,t)].add(h)
            tails[(h,r)].add(t)
            if split == 'test':
                test.append((h,r,t))
    test = np.asarray(test,dtype=np.int64)
    require(test.shape == (20466,3), 'Unexpected test size')
    targets = np.r_[test[:,0],test[:,2]]
    retained = np.asarray([14541-len(heads[(r,t)])+1 for h,r,t in test] +
                          [14541-len(tails[(h,r)])+1 for h,r,t in test],dtype=np.int64)
    return manifest, targets, retained


def validate(bundle, repo, data):
    blob = bundle.read_bytes()
    sidecar = bundle.with_suffix('.zip.sha256')
    if sidecar.exists():
        require(sidecar.read_text(encoding='utf-8').split()[0].lower() == sha(blob), 'ZIP sidecar mismatch')
    manifest, targets, retained = load_benchmark(repo,data)
    report = {'passed':False, 'source_bundle_sha256':sha(blob), 'source_bundle_bytes':len(blob),
              'sidecar_verified':sidecar.exists(), 'verification':'Independent direct sums; no model scoring or ZIP source execution', 'models':{}}
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = z.namelist()
        require(len(names) == len(set(names)), 'Duplicate ZIP members')
        for name in names:
            require(not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts and '\\' not in name and ':' not in name, 'Unsafe ZIP path')
        sums = json.loads(z.read('SHA256SUMS.json'))
        require(set(names)-{'SHA256SUMS.json','README.md'} == set(sums), 'Incomplete ZIP hash coverage')
        for name, expected in sums.items():
            require(sha(z.read(name)) == expected, f'ZIP member checksum: {name}')
        report['verified_internal_files'] = len(sums)
        source_matches = []
        for name in names:
            if name.startswith('source/'):
                local = repo/name.removeprefix('source/')
                require(local.is_file(), f'Local source missing: {local}')
                require(z.read(name).replace(b'\r\n',b'\n') == local.read_bytes().replace(b'\r\n',b'\n'), f'Run source differs from local source: {name}')
                source_matches.append(name)
        report['source_files_matching_repository_after_newline_normalization'] = len(source_matches)
        expected_tables = {}
        for model in ('rotate','convkb'):
            prefix = f'results/{model}/'
            meta = json.loads(z.read(prefix+'metadata.json'))
            cfg = meta['config']
            require(meta['status'] == 'complete' and meta['queries_scored'] == 40932, f'{model}: incomplete')
            require(cfg['model'] == model and cfg['split'] == 'test' and cfg['query_count'] == 40932 and cfg['rows_per_mode'] == 20466 and cfg['nentity'] == 14541, f'{model}: benchmark contract')
            require(cfg['architecture'] == ('convkb_hrt_v1' if model == 'convkb' else 'rotate_complex_v1'), 'Wrong architecture')
            require(cfg['checkpoint_sha256'] == CHECKPOINTS[model] and cfg['archive_sha256'] == ARCHIVES[model], 'Wrong frozen weights')
            require(cfg['selected_step'] == (6000 if model == 'convkb' else 30000) and cfg['training_seed'] == 42, 'Wrong selection metadata')
            require(cfg['data_sha256'] == {k:v['sha256'] for k,v in manifest['files'].items()}, 'Wrong dataset/dictionaries')
            require(cfg['seeds'] == list(range(5)) and cfg['precision'] == 'float32' and not cfg['tf32'], 'Wrong evaluation protocol')
            require(cfg['deterministic_algorithms'] and cfg['device'] == 'cuda', 'Unexpected numerical execution settings')
            for name, expected in cfg['source_sha256'].items():
                require(sha(z.read('source/'+name)) == expected, f'Run/source link: {name}')
            state_blob = z.read(prefix+'per_query.npz')
            require(sha(state_blob) == meta['per_query_sha256'], 'Per-query checksum')
            with np.load(io.BytesIO(state_blob),allow_pickle=False) as state:
                require(state['config_sha256'].item() == sha(json.dumps(cfg,sort_keys=True).encode()), 'State/config link')
                arrays = {k:state[k].copy() for k in FIELDS}
            for name, values in arrays.items():
                require(values.shape == (40932,) and values.dtype == np.int64, f'{model}/{name}: shape/dtype')
            require(np.array_equal(arrays['row'],np.tile(np.arange(20466),2)), 'Rows/order/duplicates')
            require(np.array_equal(arrays['mode'],np.repeat([0,1],20466)), 'Modes/order')
            require(np.array_equal(arrays['target'],targets) and cfg['target_sha256'] == array_sha(targets), 'Targets do not match test')
            require(np.array_equal(arrays['retained'],retained), 'Retained counts differ from benchmark filtering')
            b,t = arrays['B'],arrays['T']
            require(np.all((b >= 0) & (b < 14541) & (t >= 1) & (t <= 14541) & (b+t <= retained)), 'Count bounds')
            saved = json.loads(z.read(prefix+'metrics.json'))
            for scope, indexes in [('overall',slice(None)),('head',slice(0,20466)),('tail',slice(20466,None))]:
                calculated = direct_metrics(b[indexes],t[indexes])
                gap = calculated['optimistic']['MRR']-calculated['expected_random']['MRR']
                close(saved[scope]['gap'],gap,f'{model}/{scope}/gap')
                for policy, values in calculated.items():
                    for key, value in values.items():
                        close(saved[scope][policy][key],value,f'{model}/{scope}/{policy}/{key}')
                    expected_tables[(model,scope,policy)] = (40932 if scope == 'overall' else 20466, gap, values)
                for key in KEYS:
                    require(calculated['pessimistic'][key]-1e-13 <= calculated['expected_random'][key] <= calculated['optimistic'][key]+1e-13, 'Policy bounds')
            hist = {str(k):v for k,v in sorted(Counter(map(int,t)).items())}
            require(saved['ties']['histogram'] == hist, 'Histogram mismatch')
            require(saved['ties']['count'] == int((t>1).sum()), 'Tie count')
            close(saved['ties']['fraction'],float((t>1).mean()), 'Tie fraction')
            for key, value in {'min':int(t.min()),'max':int(t.max()),'median':float(np.median(t)),'p90':float(np.percentile(t,90)),'p99':float(np.percentile(t,99))}.items():
                close(saved['ties'][key],value,'Tie quantile '+key)
            require(json.loads(z.read(prefix+'tie_histogram.json')) == saved['ties'], 'Separate histogram mismatch')
            seeded = []
            for seed in range(5):
                ranks = b+np.random.default_rng(seed).integers(1,t+1)
                values = rank_metrics(ranks)
                seeded.append(values)
                require(saved['seeds'][seed]['seed'] == seed, 'Seed order')
                for key, value in values.items():
                    close(saved['seeds'][seed][key],value,f'Seed {seed}/{key}')
                    require(saved['overall']['pessimistic'][key]-1e-13 <= value <= saved['overall']['optimistic'][key]+1e-13, 'Seed metric bounds')
            for key in KEYS:
                mean = math.fsum(s[key] for s in seeded)/5
                std = math.sqrt(math.fsum((s[key]-mean)**2 for s in seeded)/4)
                close(saved['seed_mean'][key],mean,'Seed mean')
                close(saved['seed_std_ddof1'][key],std,'Seed sample std')
                close(saved['seed_mean_minus_expected'][key],mean-saved['overall']['expected_random'][key],'Seed expectation deviation')
            seed_rows = list(csv.DictReader(io.StringIO(z.read(prefix+'random_seeds.csv').decode())))
            require(len(seed_rows) == 5, 'Seed CSV length')
            for seed, row in enumerate(seed_rows):
                require(int(row['seed']) == seed, 'CSV seed order')
                for key in KEYS:
                    close(row[key],seeded[seed][key],'Seed CSV')
            checks = json.loads(z.read(f'results/{model}_adapter_checks.json'))
            require(checks['passed'] and checks['checkpoint_sha256'] == CHECKPOINTS[model], 'Adapter check identity')
            require(all(checks['checks'][mode]['compatible'] and checks['checks'][mode]['exact_repeatability'] for mode in ('head','tail')), 'Adapter compatibility/repeatability')
            elapsed = (datetime.fromisoformat(meta['completed_utc'])-datetime.fromisoformat(meta['created_utc'])).total_seconds()
            report['models'][model] = {'passed':True,'queries':40932,'head_queries':20466,'tail_queries':20466,
                'target_tie_count':int((t>1).sum()),'target_tie_fraction':float((t>1).mean()),'max_T':int(t.max()),
                'overall':saved['overall'],'elapsed_seconds':elapsed,'device':cfg['device'],'gpu':cfg['gpu'],
                'checkpoint_sha256':CHECKPOINTS[model], 'selected_step':cfg['selected_step'],
                'checks':['canonical query IDs','targets against test','retained counts against truth union',
                          'direct reciprocal sums','head/tail/overall policies','five seeds and ddof=1',
                          'exact histogram and quantiles','checkpoint/data/source/state hashes']}
        for path in ('results/main_table.csv','results/rotate/main_table.csv','results/convkb/main_table.csv'):
            rows = list(csv.DictReader(io.StringIO(z.read(path).decode())))
            expected_keys = set(expected_tables) if path == 'results/main_table.csv' else {k for k in expected_tables if k[0] == path.split('/')[1]}
            require(len(rows) == len(expected_keys), 'Table row count')
            seen = set()
            for row in rows:
                key = (row['model'],row['scope'],row['policy'])
                require(key in expected_keys and key not in seen and row['split'] == 'test', 'Table identity')
                seen.add(key)
                count,gap,values = expected_tables[key]
                require(int(row['queries']) == count, 'Table query count')
                close(row['MRR_opt_minus_expected'],gap,'Table gap')
                for name,value in values.items():
                    close(row[name],value,'Table metric')
        test_log = z.read('results/tests.log').decode()
        test_count = re.search(r'^Ran (\d+) tests?\b', test_log, re.MULTILINE)
        require(test_count is not None and int(test_count.group(1)) >= 23 and re.search(r'^OK\s*$', test_log, re.MULTILINE), 'Colab test log')
        report['passed'] = True
        files = {}
        for name in names:
            if name.startswith(('results/rotate/','results/convkb/')) or name in {
                'results/main_table.csv','results/main_table.md','results/evaluation_environment.json',
                'results/input_checks.json','results/requirements-evaluation-actual.txt',
                'results/rotate_adapter_checks.json','results/convkb_adapter_checks.json'}:
                files[name.removeprefix('results/')] = z.read(name)
        for name in ('tests','rotate_test','convkb_test'):
            files[name+'_output.txt'] = z.read('results/'+name+'.log')
        files['SOURCE_BUNDLE.sha256'] = (sha(blob)+'  '+bundle.name+'\n').encode()
    return report,files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle',type=Path)
    parser.add_argument('--repo',type=Path,default=ROOT)
    parser.add_argument('--data',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    report,files = validate(args.bundle,args.repo,args.data or args.repo/'data/FB15k-237')
    args.out.mkdir(parents=True,exist_ok=True)
    files['validation_report.json'] = (json.dumps(report,indent=2)+'\n').encode()
    for name,content in files.items():
        path = args.out/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(content)
    (args.out/'ARTIFACT_SHA256SUMS.json').write_text(json.dumps({k:sha(v) for k,v in sorted(files.items())},indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
