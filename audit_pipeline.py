"""Verify frozen packages, pilot on validation, score test, and export evidence."""
import argparse
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

from audit import ROOT, digest, write_json

ARCHIVES = {
    'rotate': '2c66eecac4fe499f8c3d695cb107d1c6025cda6758c875ea0461ba59421a0d1b',
    'convkb': '9008666f453d8306531b6cbf884cbd79b22d380198ba5124a205e47745065a5b',
}
CHECKPOINTS = {
    'rotate': '5ec0bbc39e9568f6362ac38010cc752531dfbdba7aaea029661838707ce8b622',
    'convkb': 'dfa921d625ae229320106a6321844991a385ca4f9a3cb87117eb29e212cbfe46',
}


def unpack_verified(archive, destination, model):
    import hashlib
    if digest(archive) != ARCHIVES[model]:
        raise ValueError(f'{model}: archive hash mismatch')
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive members')
        for name in names:
            path = (destination/name).resolve()
            if not path.is_relative_to(destination) or '\\' in name or ':' in name:
                raise ValueError('Unsafe archive path')
        sums = json.loads(z.read('SHA256SUMS.json'))
        if set(names) - {'SHA256SUMS.json'} != set(sums):
            raise ValueError('Manifest does not cover every archive file')
        for name, value in sums.items():
            if hashlib.sha256(z.read(name)).hexdigest() != value:
                raise ValueError(f'Internal hash mismatch: {name}')
        if hashlib.sha256(z.read('model.pt')).hexdigest() != CHECKPOINTS[model]:
            raise ValueError('Unexpected frozen checkpoint')
        for name in names:
            path = destination/name
            data = z.read(name)
            if path.exists() and path.read_bytes() != data:
                raise ValueError(f'Existing package changed: {path}')
            path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():
                path.write_bytes(data)
    summary = json.loads((destination/'summary.json').read_text())
    handoff = json.loads((destination/'handoff.json').read_text())
    history = [json.loads(line) for line in (destination/'validation.jsonl').read_text().splitlines() if line.strip()]
    best = max(history,key=lambda row:row['MRR_expected_random'])
    if best['step'] != handoff['selected_step'] or best['step'] != summary['best_validation_step']:
        raise ValueError('Validation selection metadata disagree')
    if handoff['test_queries_scored'] != 0:
        raise ValueError('Unexpected prior test use in package')
    return {'archive_sha256':ARCHIVES[model], 'checkpoint_sha256':CHECKPOINTS[model],
            'verified_internal_files':len(sums), 'selected_step':best['step'],
            'validation_MRR':best['MRR_expected_random'], 'validation_records':len(history)}


def check_examples(model, bundle, device, out):
    import torch
    from rotate.adapter import RotatEAdapter
    from convkb.adapter import ConvKBAdapter
    from rotate.common import load_data
    adapter = (RotatEAdapter if model == 'rotate' else ConvKBAdapter).load(bundle/'model.pt',ROOT/'data/FB15k-237',device)
    examples = np.load(bundle/'validation_score_examples.npz',allow_pickle=False)
    splits, manifest = load_data(ROOT/'data/FB15k-237')
    if adapter.metadata['data_manifest'] != manifest:
        raise ValueError('Checkpoint dataset manifest differs from benchmark')
    handoff = json.loads((bundle/'handoff.json').read_text())
    if adapter.metadata['step'] != handoff['selected_step']:
        raise ValueError('Checkpoint step differs from validation selection')
    if not np.array_equal(examples['triples'],splits['valid'][:8]):
        raise ValueError('Archive examples are not the first eight validation triples')
    reports = {}
    for mode in ('head','tail'):
        key = 'scores_'+mode if model == 'rotate' else mode+'_scores'
        actual = adapter.score_candidates(examples['triples'],mode)
        expected = examples[key]
        if expected.shape != actual.shape or expected.dtype != np.float32 or not np.isfinite(expected).all():
            raise ValueError('Invalid archived score array')
        compatible = bool(np.allclose(actual,expected,atol=1e-4,rtol=1e-5))
        repeatable = bool(np.array_equal(actual,adapter.score_candidates(examples['triples'],mode)))
        reports[mode] = {'max_abs_error':float(np.abs(actual-expected).max()),
                         'atol':1e-4,'rtol':1e-5,'compatible':compatible,'exact_repeatability':repeatable}
    passed = all(r['compatible'] and r['exact_repeatability'] for r in reports.values())
    write_json(out,{'passed':passed,'device':device,'checkpoint_sha256':CHECKPOINTS[model],
                    'architecture':getattr(adapter,'architecture','rotate_complex_v1'),
                    'checks':reports,'note':'Tolerance compares devices; ties still use exact float32 equality.'})
    if not passed:
        raise ValueError('Archived example compatibility or repeatability failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rotate-archive',type=Path,required=True)
    parser.add_argument('--convkb-archive',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=ROOT/'runs/final_audit')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke-only',action='store_true')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True,exist_ok=True)
    test = subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-v'],cwd=ROOT,capture_output=True,text=True)
    (args.out/'tests.log').write_text(test.stdout+test.stderr,encoding='utf-8')
    if test.returncode:
        raise ValueError('Tests failed; see tests.log')
    if not (ROOT/'data/FB15k-237/train.txt').exists():
        subprocess.run([sys.executable,'prepare_data.py'],cwd=ROOT,check=True)
    checks = {}
    commands = {}
    for model in ('rotate','convkb'):
        archive = getattr(args,model+'_archive').resolve()
        bundle = args.out/'packages'/model
        checks[model] = unpack_verified(archive,bundle,model)
        check_examples(model,bundle,args.device,args.out/f'{model}_adapter_checks.json')
        # Fix these settings on validation before touching test.
        batch, chunk = (8,2048) if model == 'rotate' else (2,512)
        command = [sys.executable,'-B',str(ROOT/'audit.py'),'run','--model',model,
                   '--checkpoint',str(bundle/'model.pt'),'--archive',str(archive),
                   '--data',str(ROOT/'data/FB15k-237'),'--batch',str(batch),'--chunk',str(chunk),
                   '--threads',str(args.threads),'--device',args.device]
        pilot = [*command,'--split','valid','--limit','8','--out',str(args.out/(model+'_pilot'))]
        if args.resume:
            pilot.append('--resume')
        subprocess.run(pilot,cwd=ROOT,check=True)
        commands[model] = command
    write_json(args.out/'input_checks.json',checks)
    if not args.smoke_only:
        for model, command in commands.items():
            full = [*command,'--split','test','--out',str(args.out/model)]
            if args.resume:
                full.append('--resume')
            with (args.out/f'{model}_test.log').open('a',encoding='utf-8') as log:
                with subprocess.Popen(full,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as child:
                    for line in child.stdout:
                        log.write(line)
                        log.flush()
                        print(f'{model}: {line}',end='',flush=True)
                    if child.wait():
                        raise subprocess.CalledProcessError(child.returncode,full)
    if not args.smoke_only:
        subprocess.run([sys.executable,'-B',str(ROOT/'audit.py'),'table',str(args.out/'rotate'),str(args.out/'convkb'),'--out',str(args.out)],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-B',str(ROOT/'audit_bundle.py'),str(args.out)],cwd=ROOT,check=True)


if __name__ == '__main__':
    main()
