"""Check unchanged evidence and repeatability of generated report artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generated',type=Path,default=ROOT/'runs/gosha_materials')
    args = parser.parse_args()
    evidence = ROOT/'reports/final_audit'
    reference = json.loads((evidence/'ARTIFACT_SHA256SUMS.json').read_text())
    if not all(digest(evidence/name)==value for name,value in reference.items()):
        raise ValueError('Frozen result evidence changed')
    files = [*sorted((HERE/'tables').glob('*.csv')),*sorted((HERE/'tables').glob('*.tex')),HERE/'numbers.tex']
    for source in files:
        relative = source.relative_to(HERE)
        if source.read_bytes() != (args.generated/relative).read_bytes():
            raise ValueError(f'Generated numeric/text artifact differs: {relative}')
    figures = {p.relative_to(HERE).as_posix():p.read_bytes()==(args.generated/p.relative_to(HERE)).read_bytes()
               for p in sorted((HERE/'figures').glob('*')) if p.suffix in ('.pdf','.png')}
    summary = {'passed':True,'unchanged_evidence_files':len(reference),
               'identical_generated_numeric_and_tex_files':len(files),
               'same_environment_figure_byte_equality':figures}
    print(json.dumps(summary,indent=2))
