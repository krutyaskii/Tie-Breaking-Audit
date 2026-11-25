"""Render and inspect compiled previews. Requires requirements-preview.txt."""
import argparse
import json
from pathlib import Path
import re

import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def inspect(pdf, out):
    document = fitz.open(pdf)
    require_two = pdf.name=='preview.pdf'
    if require_two and len(document)!=2:
        raise ValueError(f'Two-column preview has {len(document)} pages, expected two')
    log = pdf.with_suffix('.log')
    overfull = re.findall(r'Overfull \\[hv]box[^\n]*',log.read_text(encoding='utf-8',errors='replace')) if log.exists() else []
    text = '\n'.join(page.get_text() for page in document)
    if '??' in text:
        raise ValueError('Unresolved citation/reference marker in PDF')
    outside = []
    for number,page in enumerate(document,1):
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line['spans']:
                    x0,y0,x1,y1 = span['bbox']
                    if span['text'].strip() and (x0<106 or x1>506 or y0<68 or y1>765):
                        outside.append({'page':number,'text':span['text'],'bbox':span['bbox']})
        page.get_pixmap(matrix=fitz.Matrix(1.7,1.7),alpha=False).save(out/f'{pdf.stem}_page_{number}.png')
    result = {'file':pdf.name,'pages':len(document),'unresolved_markers':False,
              'overfull_boxes':overfull,'text_outside_margin_tolerance':outside,
              'hyperlinks':sum(len(p.get_links()) for p in document),
              'passed':not overfull and not outside}
    if not result['passed']:
        raise ValueError(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/gosha_pdf_qa')
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    results = [inspect(HERE/name,args.out) for name in ('preview.pdf','preview_neurips.pdf')]
    (HERE/'pdf_validation.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(results,indent=2))
