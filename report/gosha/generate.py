"""Verify saved counts independently, then regenerate Gosha's tables and figures.

No checkpoint, dataset download, PyTorch import, or GPU is required. Input
evidence is read-only. Actual target/filter checks against the benchmark are
performed separately by tools/validate_evaluation_bundle.py.
"""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KEYS = ('MRR', 'Hits@1', 'Hits@3', 'Hits@10')
POLICIES = ('optimistic', 'expected_random', 'pessimistic')
LABELS = {'rotate':'RotatE', 'convkb':'ConvKB'}
N = 40932


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(content):
    return hashlib.sha256(content).hexdigest()


def close(actual, expected, label, tolerance=1e-13):
    error = abs(float(actual)-float(expected))
    require(math.isfinite(float(actual)) and error <= tolerance,
            f'{label}: saved={actual}, recalculated={expected}')
    return error


def rank_metrics(ranks):
    return {'MRR':math.fsum(1/int(r) for r in ranks)/len(ranks),
            **{f'Hits@{k}':sum(int(r)<=k for r in ranks)/len(ranks) for k in (1,3,10)}}


def direct_metrics(b, t):
    rr = [math.fsum(1/r for r in range(int(x)+1,int(x+y)+1))/int(y) for x,y in zip(b,t)]
    return {'optimistic':rank_metrics(b+1), 'pessimistic':rank_metrics(b+t),
            'expected_random':{'MRR':math.fsum(rr)/len(rr),
                **{f'Hits@{k}':math.fsum(min(max(k-int(x),0),int(y))/int(y) for x,y in zip(b,t))/len(b) for k in (1,3,10)}}}


def verify_tables(folder, rows):
    csv_rows = list(csv.DictReader((folder/'main_table.csv').open(encoding='utf-8',newline='')))
    require(len(csv_rows) == len(rows), 'CSV row count')
    for saved, expected in zip(csv_rows,rows):
        require(set(saved) == set(expected), 'CSV columns')
        for key,value in expected.items():
            if isinstance(value,float):
                close(saved[key],value,'CSV '+key)
            else:
                require(saved[key] == str(value),'CSV identity/order '+key)
    lines = (folder/'main_table.md').read_text(encoding='utf-8').strip().splitlines()
    keys = [x.strip() for x in lines[0].strip('|').split('|')]
    require(len(lines)-2 == len(rows), 'Markdown row count')
    for line, expected in zip(lines[2:],rows):
        values = [x.strip() for x in line.strip('|').split('|')]
        require(len(values) == len(keys), 'Markdown columns')
        for key,value in zip(keys,values):
            reference = f'{expected[key]:.12g}' if isinstance(expected[key],float) else str(expected[key])
            require(value == reference,'Markdown rounded value '+key)


def read_verified_results():
    source = ROOT/'reports/final_audit'
    sums = json.loads((source/'ARTIFACT_SHA256SUMS.json').read_text(encoding='utf-8'))
    for name, expected in sums.items():
        require(sha((source/name).read_bytes()) == expected,'Artifact checksum '+name)
    models,all_rows,seed_rows,tie_rows = {},[],[],[]
    max_error = 0.0
    for model in ('rotate','convkb'):
        folder = source/model
        meta = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        cfg = meta['config']
        saved = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        require(meta['status']=='complete' and cfg['split']=='test' and cfg['query_count']==N,
                'Incomplete/non-test input')
        raw = (folder/'per_query.npz').read_bytes()
        require(sha(raw)==meta['per_query_sha256'],'Per-query checksum')
        with np.load(folder/'per_query.npz',allow_pickle=False) as archive:
            require(archive['config_sha256'].item()==sha(json.dumps(cfg,sort_keys=True).encode()),'Config link')
            arrays = {k:archive[k].copy() for k in ('row','mode','target','B','T','retained')}
        require(all(v.shape==(N,) and v.dtype==np.int64 for v in arrays.values()),'Array shape/dtype')
        require(np.array_equal(arrays['row'],np.tile(np.arange(N//2),2)),'Canonical rows')
        require(np.array_equal(arrays['mode'],np.repeat([0,1],N//2)),'Canonical modes')
        require(len(set(zip(arrays['row'],arrays['mode'])))==N,'Duplicate query IDs')
        require(sha(np.asarray(arrays['target'],dtype='<i8').tobytes())==cfg['target_sha256'],'Target hash')
        require(np.all((arrays['target']>=0)&(arrays['target']<cfg['nentity'])),'Target bounds')
        b,t,c = arrays['B'],arrays['T'],arrays['retained']
        require(np.all((b>=0)&(b<cfg['nentity'])&(t>=1)&(t<=cfg['nentity'])&(b+t<=c)&(c<=cfg['nentity'])),'B/T/retained bounds')
        rows = []
        for scope, selected in [('overall',slice(None)),('head',slice(0,N//2)),('tail',slice(N//2,None))]:
            direct = direct_metrics(b[selected],t[selected])
            gap = direct['optimistic']['MRR']-direct['expected_random']['MRR']
            max_error = max(max_error,close(saved[scope]['gap'],gap,'Direct MRR gap'))
            for policy in POLICIES:
                for key in KEYS:
                    max_error = max(max_error,close(saved[scope][policy][key],direct[policy][key],f'{model}/{scope}/{policy}/{key}'))
                    require(direct['pessimistic'][key]-1e-13<=direct[policy][key]<=direct['optimistic'][key]+1e-13,'Policy bounds')
                # Preserve source values after independent validation; no hand-entered measurements.
                rows.append({'model':model,'split':'test','scope':scope,'policy':policy,
                             'queries':N if scope=='overall' else N//2,
                             'MRR_opt_minus_expected':saved[scope]['gap'],**saved[scope][policy]})
        verify_tables(folder,rows)
        all_rows.extend(rows)
        hist = {str(k):v for k,v in sorted(Counter(map(int,t)).items())}
        require(saved['ties']['histogram']==hist and sum(hist.values())==N,'Exact histogram')
        require(saved['ties']['count']==int((t>1).sum()),'Target tie count')
        close(saved['ties']['fraction'],float((t>1).mean()),'Target tie fraction')
        for key,value in {'min':t.min(),'max':t.max(),'median':np.median(t),'p90':np.percentile(t,90),'p99':np.percentile(t,99)}.items():
            close(saved['ties'][key],value,'Tie quantile '+key)
        require(json.loads((folder/'tie_histogram.json').read_text())==saved['ties'],'Separate histogram')
        samples = []
        for seed in range(5):
            ranks = b+np.random.default_rng(seed).integers(1,t+1)
            metrics = rank_metrics(ranks)
            require(saved['seeds'][seed]['seed']==seed,'Seed order')
            for key,value in metrics.items():
                max_error = max(max_error,close(saved['seeds'][seed][key],value,'Seed metric'))
                require(saved['overall']['pessimistic'][key]-1e-13<=value<=saved['overall']['optimistic'][key]+1e-13,'Seed bounds')
            samples.append(metrics)
            seed_rows.append({'model':model,**saved['seeds'][seed]})
        original_csv = list(csv.DictReader((folder/'random_seeds.csv').open(encoding='utf-8',newline='')))
        require(len(original_csv)==5,'Seed CSV rows')
        for seed,row in enumerate(original_csv):
            require(int(row['seed'])==seed,'Seed CSV order')
            for key in KEYS:
                close(row[key],samples[seed][key],'Seed CSV metric')
        for key in KEYS:
            mean = math.fsum(v[key] for v in samples)/5
            std = math.sqrt(math.fsum((v[key]-mean)**2 for v in samples)/4)
            close(saved['seed_mean'][key],mean,'Seed mean')
            close(saved['seed_std_ddof1'][key],std,'Seed ddof=1')
            close(saved['seed_mean_minus_expected'][key],mean-saved['overall']['expected_random'][key],'Seed deviation')
        tie_rows.append({'model':model,'queries':N,'target_tie_count':saved['ties']['count'],
                         'target_tie_fraction':saved['ties']['fraction'],'target_tie_percent':100*saved['ties']['fraction'],
                         'seeded_MRR_mean':saved['seed_mean']['MRR'],
                         'seeded_MRR_sample_std':saved['seed_std_ddof1']['MRR'],
                         'seed_mean_minus_expected_MRR':saved['seed_mean_minus_expected']['MRR']})
        models[model] = {'metadata':meta,'metrics':saved}
    verify_tables(source,all_rows)
    verification = {'passed':True,'queries_per_model':N,'verified_artifact_files':len(sums),
                    'independent_method':'Direct reciprocal summation; fresh default_rng draws; sample std ddof=1',
                    'max_abs_direct_metric_difference':max_error,
                    'torch_installed':importlib.util.find_spec('torch') is not None,
                    'numpy':np.__version__,'python':sys.version,'platform':platform.platform(),
                    'scope':'No model scoring. Actual target/filter-vs-dataset validation is a separate recorded check.',
                    'source_manifest_sha256':sha((source/'ARTIFACT_SHA256SUMS.json').read_bytes())}
    return models,all_rows,seed_rows,tie_rows,verification


def write_csv(path, rows):
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def scientific(value, digits=3):
    mantissa, exponent = f'{value:.{digits-1}e}'.split('e')
    return rf'{mantissa}\times10^{{{int(exponent)}}}'


def tex_table(path, caption, label, header, rows, columns):
    lines = [r'\begin{table*}[t]',r'\centering',r'\small',r'\setlength{\tabcolsep}{3pt}',
             r'\caption{'+caption+'}',r'\label{'+label+'}',
             r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}'+columns+'@{}}',r'\toprule',
             ' & '.join(header)+r' \\',r'\midrule']
    lines += [' & '.join(row)+r' \\' for row in rows]
    lines += [r'\bottomrule',r'\end{tabular*}',r'\end{table*}']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')


def generate(out):
    models,rows,seeds,ties,verification = read_verified_results()
    out.mkdir(parents=True,exist_ok=True)
    tables,figures = out/'tables',out/'figures'
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    main = []
    for model, values in models.items():
        m = values['metrics']['overall']
        main.append({'model':model,'optimistic_MRR':m['optimistic']['MRR'],
                     'expected_random_MRR':m['expected_random']['MRR'],'pessimistic_MRR':m['pessimistic']['MRR'],
                     'delta_MRR':m['gap'],**{k:m['expected_random'][k] for k in KEYS[1:]}})
    write_csv(tables/'main_results.csv',main)
    write_csv(tables/'all_policies.csv',rows)
    write_csv(tables/'head_tail_results.csv',[r for r in rows if r['scope']!='overall'])
    write_csv(tables/'evaluation_seeds.csv',seeds)
    write_csv(tables/'tie_seed_summary.csv',ties)
    table_rows = [[LABELS[r['model']],*[f'{r[k]:.9f}' for k in ('optimistic_MRR','expected_random_MRR','pessimistic_MRR')],
                   '$'+scientific(r['delta_MRR'])+'$',*[f'{r[k]:.4f}' for k in KEYS[1:]]] for r in main]
    tex_table(tables/'main_results.tex',
              'Full filtered test audit (40,932 queries per model). O/E/P denote optimistic, analytic expected random, and pessimistic ranking. '
              'MRR is rounded to nine decimals; $\\Delta$ is computed before rounding and remains nonzero. Hits are expected random.',
              'tab:gosha-main',['Model','MRR O','MRR E','MRR P',r'$\Delta$','H@1','H@3','H@10'],table_rows,'lrrrrrrr')
    tie_table = [[LABELS[r['model']],f"{r['target_tie_count']} / 40,932",f"{r['target_tie_percent']:.6f}\\%",
                  f"${r['seeded_MRR_mean']:.9f}\\pm {scientific(r['seeded_MRR_sample_std'])}$"] for r in ties]
    tex_table(tables/'tie_seed_summary.tex','Exact target ties and five evaluation draws (seeds 0--4). '
              'Mean $\\pm$ sample standard deviation uses $\\mathrm{ddof}=1$ and describes tie-position randomness, not training uncertainty.',
              'tab:gosha-ties',['Model','Tied queries','Fraction',r'Seeded MRR mean $\pm$ SD'],tie_table,'lrrr')
    detail_rows = [[LABELS[r['model']],r['scope'].title(),{'optimistic':'O','expected_random':'E','pessimistic':'P'}[r['policy']],
                    f"{r['MRR']:.9f}",*[f'{r[k]:.6f}' for k in KEYS[1:]]] for r in rows if r['scope']!='overall']
    tex_table(tables/'head_tail_results.tex','Head/tail breakdown, 20,466 queries per mode. O/E/P as in the main table.',
              'tab:gosha-head-tail',['Model','Mode','Policy','MRR','H@1','H@3','H@10'],detail_rows,'lllrrrr')
    paper = json.loads((HERE/'paper_reference.json').read_text(encoding='utf-8'))
    write_csv(tables/'paper_reference.csv',[{'model':m,**r} for m,r in paper['models'].items()])
    macros = [r'% Generated from independently checked saved artifacts; do not edit numbers by hand.']
    for model,values in models.items():
        prefix = 'Rotate' if model=='rotate' else 'ConvKB'
        m = values['metrics']
        macros += [rf'\providecommand{{\Gosha{prefix}MRR}}{{{m["overall"]["expected_random"]["MRR"]:.9f}}}',
                   rf'\providecommand{{\Gosha{prefix}Gap}}{{\ensuremath{{{scientific(m["overall"]["gap"])}}}}}',
                   rf'\providecommand{{\Gosha{prefix}Ties}}{{{m["ties"]["count"]}}}']
    (out/'numbers.tex').write_text('\n'.join(macros)+'\n',encoding='utf-8',newline='\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'xtick.labelsize':9,'ytick.labelsize':8,
                         'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':220})
    fig,ax = plt.subplots(figsize=(3.25,2.35))
    percentages = [r['target_tie_percent'] for r in ties]
    bars = ax.bar([LABELS[r['model']] for r in ties],percentages,color=['#2166ac','#b65c24'],width=.5)
    ax.set_ylim(0,max(percentages)*1.55)
    ax.set_ylabel('Queries with target ties (%)')
    ax.set_xlabel('40,932 filtered test queries per model')
    for bar,r in zip(bars,ties):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(percentages)*.045,
                f"{r['target_tie_count']} / 40,932\n{r['target_tie_percent']:.6f}%",ha='center',va='bottom',fontsize=8)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(pad=.6)
    for ext in ('pdf','png'):
        metadata = {'Creator':'Tie-Breaking Audit report generator'}
        if ext == 'pdf':
            metadata.update(CreationDate=None,ModDate=None)
        fig.savefig(figures/f'target_ties.{ext}',metadata=metadata)
    plt.close(fig)
    fig,ax = plt.subplots(figsize=(3.6,2.35))
    x = [models['rotate']['metrics']['overall']['gap'],models['convkb']['metrics']['overall']['gap'],paper['models']['convkb']['gap']]
    y = [2,1,0]
    ax.scatter(x,y,color=['#2166ac','#2166ac','#b65c24'],s=35,zorder=3)
    ax.set_xscale('log')
    ax.set_xlim(1e-11,1.5)
    ax.set_ylim(-.5,2.55)
    ax.set_yticks(y,['RotatE: this audit','ConvKB: this audit','ConvKB: paper'])
    ax.set_xlabel(r'MRR gap $\Delta$ (logarithmic scale)')
    for v,height in zip(x,y):
        ax.annotate(f'{v:.3g}',(v,height),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
    ax.grid(axis='x',alpha=.25)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(pad=.6)
    for ext in ('pdf','png'):
        metadata = {'Creator':'Tie-Breaking Audit report generator'}
        if ext == 'pdf':
            metadata.update(CreationDate=None,ModDate=None)
        fig.savefig(figures/f'gap_comparison.{ext}',metadata=metadata)
    plt.close(fig)
    (out/'source_validation.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8',newline='\n')
    inputs = {str(p.relative_to(ROOT)).replace('\\','/'):sha(p.read_bytes()) for p in [
        HERE/'generate.py',HERE/'paper_reference.json',
        ROOT/'reports/final_audit/ARTIFACT_SHA256SUMS.json',
        *[ROOT/f'reports/final_audit/{m}/{f}' for m in models for f in ('per_query.npz','metadata.json','metrics.json')]]}
    (out/'generation_inputs.json').write_text(json.dumps(inputs,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=HERE)
    generate(parser.parse_args().out)
