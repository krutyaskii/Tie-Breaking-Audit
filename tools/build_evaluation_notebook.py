"""Create a self-contained GPU evaluation notebook from the current source."""
import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
paths = [*ROOT.glob('audit*.py'), ROOT/'prepare_data.py', ROOT/'requirements-evaluation.txt', ROOT/'reports/data_manifest.json']
for directory in ('rotate','convkb','vendor','tests'):
    paths.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
payload = io.BytesIO()
with zipfile.ZipFile(payload,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for path in sorted(paths):
        info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(),date_time=(1980,1,1,0,0,0))
        info.external_attr = 0o644 << 16
        z.writestr(info,path.read_bytes(),compress_type=zipfile.ZIP_DEFLATED)
encoded = base64.b64encode(payload.getvalue()).decode('ascii')
cells = []


def cell(kind, source):
    if kind == 'code':
        compile(source,'<evaluation-notebook>','exec')
    entry = {'cell_type':kind,'metadata':{'id':f'audit_{len(cells):02d}'},'source':source.splitlines(keepends=True)}
    if kind == 'code':
        entry.update(execution_count=None,outputs=[])
    cells.append(entry)


cell('markdown', '# Frozen-checkpoint tie-breaking audit\n\nSelect a T4 GPU, then run cells in order. '
     'The notebook embeds the evaluator source. Upload the original RotatE ZIP and '
     'the corrected ConvKB ZIP (20261111); hashes are checked before loading. '
     'The GPU workflow is prepared but has not been executed in the local CPU environment.\n')
cell('code', "import base64, io, os, zipfile\nfrom pathlib import Path\nPROJECT = Path('/content/tie_breaking_audit')\nPROJECT.mkdir(exist_ok=True)\n"
     f"with zipfile.ZipFile(io.BytesIO(base64.b64decode({encoded!r}))) as z:\n    z.extractall(PROJECT)\nos.chdir(PROJECT)\n")
cell('code', "import subprocess, sys\nsubprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-evaluation.txt'], check=True)\n"
     "import torch\nassert torch.cuda.is_available(), 'Select a GPU runtime and reconnect.'\nprint(torch.__version__, torch.cuda.get_device_name(0))\n")
cell('markdown', 'Mount Drive to preserve resumable counts across Colab disconnects. Keep the same '
     'packages, GPU, source, batch and chunk sizes when resuming. Changed environments are rejected.\n')
cell('code', "from google.colab import drive, files\ndrive.mount('/content/drive')\n"
     "OUTPUT = Path('/content/drive/MyDrive/TieBreakingAudit/final_audit')\n"
     "uploaded = files.upload()\nimport hashlib\n"
     "hash_to_path = {hashlib.sha256(data).hexdigest(): str(Path(name).resolve()) for name, data in uploaded.items()}\n"
     "from audit_pipeline import ARCHIVES\nrotate_zip = hash_to_path[ARCHIVES['rotate']]\nconvkb_zip = hash_to_path[ARCHIVES['convkb']]\n")
cell('code', "command = [sys.executable, '-u', 'audit_pipeline.py', '--rotate-archive', rotate_zip,\n"
     "           '--convkb-archive', convkb_zip, '--out', str(OUTPUT), '--device', 'cuda', '--threads', '4']\n"
     "if (OUTPUT / 'rotate_pilot' / 'metadata.json').exists():\n    command.append('--resume')\n"
     "subprocess.run(command, check=True)\n")
cell('markdown', 'The runner verifies both packages, runs all tests, compares validation examples, '
     'pilots both models, then scores 40,932 test queries per model. Model progress is saved in '
     '`rotate_test.log` and `convkb_test.log`. Re-run the preceding cell to resume with matching settings.\n')
cell('code', "bundle = OUTPUT.parent / 'evaluation_validation_bundle.zip'\nfiles.download(str(bundle))\nfiles.download(str(bundle.with_suffix('.zip.sha256')))\n"
     "print((OUTPUT / 'main_table.md').read_text())\n")
notebook = {'nbformat':4,'nbformat_minor':5,'metadata':{'accelerator':'GPU','colab':{'provenance':[]},'kernelspec':{'name':'python3','language':'python','display_name':'Python 3'}},'cells':cells}
target = ROOT/'notebooks/Evaluation_Colab.ipynb'
target.write_text(json.dumps(notebook,indent=2)+'\n',encoding='utf-8')
print(target)
