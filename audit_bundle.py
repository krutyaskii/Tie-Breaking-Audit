"""Export audit evidence and source, excluding checkpoints and datasets."""
import argparse
import importlib.metadata
import json
import platform
import sys
import zipfile
from pathlib import Path

from audit import ROOT, digest, write_json


def build(directory):
    directory = Path(directory).resolve()
    lock = {name:importlib.metadata.version(name) for name in ('numpy','torch','matplotlib','psutil')}
    write_json(directory/'evaluation_environment.json',{'python':sys.version,'platform':platform.platform(),'packages':lock})
    index = '--extra-index-url https://download.pytorch.org/whl/cpu\n' if lock['torch'].endswith('+cpu') else ''
    (directory/'requirements-evaluation-actual.txt').write_text(index+''.join(f'{k}=={v}\n' for k,v in lock.items()),encoding='utf-8')
    files = {}
    for path in sorted(directory.rglob('*')):
        relative = path.relative_to(directory)
        if path.is_file() and not any(p in ('packages','__pycache__') for p in relative.parts) and path.suffix in ('.json','.csv','.md','.npz','.png','.log','.txt'):
            files['results/'+relative.as_posix()] = path
    for pattern in ('audit*.py','prepare_data.py','requirements-evaluation.txt','rotate/*.py','convkb/*.py','vendor/**/*.py','vendor/**/SOURCE.json','vendor/**/LICENSE','tests/*.py','docs/*AUDIT*.md','reports/EVALUATION_REPORT.md'):
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file():
                files['source/'+path.relative_to(ROOT).as_posix()] = path
    manifest = {name:digest(path) for name,path in files.items()}
    archive = directory.parent/'evaluation_validation_bundle.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,path in files.items():
            z.write(path,name)
        z.writestr('SHA256SUMS.json',json.dumps(manifest,indent=2)+'\n')
        z.writestr('README.md','Audit evidence. Verify SHA256SUMS.json before use.\n'
                   'Run source/audit.py aggregate on a complete result directory to recompute metrics.\n'
                   'Check metadata status, split and query_count; validation pilots are preliminary.\n'
                   'Weights and datasets are deliberately distributed separately.\n')
    archive.with_suffix('.zip.sha256').write_text(digest(archive)+'  '+archive.name+'\n',encoding='utf-8')
    print(archive)
    return archive


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    build(parser.parse_args().directory)
