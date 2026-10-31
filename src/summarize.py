"""Build tables, plots and evidence-based discussion without invented scores."""
import csv
import json
from pathlib import Path
from .common import read,write_csv


def summarize(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    root=Path(root)
    out=root/'results'
    metrics=read(out/'metrics.json')
    rows=list(metrics.values())
    write_csv(out/'metrics.csv',rows)
    df=pd.DataFrame(rows)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key,title in zip(axes,['bleu','chrf'],['BLEU (13a)','chrF (beta=2)']):
        ax.bar(df.model,df[key],color=['#2463a6','#15988d','#d39435','#8556b3'][:len(df)])
        ax.set_title(title)
        ax.set_ylabel('Score (0-100)')
        ax.set_ylim(0,max(5,float(df[key].max())*1.2))
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('English to Hindi | same held-out test sentences')
    fig.tight_layout()
    fig.savefig(out/'quality_comparison.png',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(df.model,df.parameters/1e6,color='#2463a6')
    axes[0].set_ylabel('Total parameters (millions)')
    axes[1].bar(df.model,df.seconds_per_sentence,color='#15988d')
    axes[1].set_ylabel('Inference seconds per sentence')
    for ax in axes: ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out/'compute_comparison.png',dpi=180)
    plt.close(fig)
    temp=pd.concat([pd.read_csv(out/f'{name}_temperature_metrics.csv') for name in metrics],ignore_index=True)
    temp.to_csv(out/'temperature_metrics.csv',index=False)
    grouped=temp.groupby(['model','temperature']).agg(bleu_mean=('bleu','mean'),bleu_std=('bleu','std'),
        chrf_mean=('chrf','mean'),chrf_std=('chrf','std')).reset_index()
    grouped.to_csv(out/'temperature_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4))
    for name,g in grouped.groupby('model'):
        ax.errorbar(g.temperature,g.chrf_mean,yerr=g.chrf_std.fillna(0),marker='o',capsize=3,label=name)
    ax.set(xlabel='Temperature (0 = greedy)',ylabel='chrF on fixed temperature subset')
    ax.legend()
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out/'temperature_analysis.png',dpi=180)
    plt.close(fig)
    table='| Model | BLEU | chrF | Training s | Inference s | Parameters |\n|---|---:|---:|---:|---:|---:|\n'
    for r in rows:
        train='N/A (prompting)' if r['training_seconds'] is None else f"{r['training_seconds']:.2f}"
        table+=f"| {r['model']} | {r['bleu']:.2f} | {r['chrf']:.2f} | {train} | {r['inference_seconds']:.2f} | {r['parameters']:,} |\n"
    complete=set(metrics)=={'basic','mt5','llama','gemma'}
    best=max(rows,key=lambda r:(r['chrf'],r['bleu']))
    conclusion=(f"The best model in this run by the predeclared primary metric, chrF, is **{best['model']}** "
        f"(chrF {best['chrf']:.2f}; BLEU {best['bleu']:.2f}). "
        'This ranking is specific to the frozen test subset and configuration; it does not establish that one architecture is intrinsically superior.'
        if complete else 'The four-model experiment is incomplete. No overall winner can yet be named.')
    observations=[]
    for name in metrics:
        with (out/f'{name}_human_review.csv').open(encoding='utf-8-sig') as f:
            review=list(csv.DictReader(f))
        flagged=[r for r in review if r['automatic_flags']]
        observations.append(f"{name}: {len(flagged)} test translations triggered at least one heuristic review flag. "
            'Flags are screening signals, not confirmed ethical violations; digit-script differences can be benign.')
    temperature=[]
    for name,g in grouped.groupby('model'):
        bestt=g.loc[g.chrf_mean.idxmax()]
        greedy=g[g.temperature==0].iloc[0]
        temperature.append(f"{name}: greedy chrF {greedy.chrf_mean:.2f}; highest observed mean chrF "
            f"{bestt.chrf_mean:.2f} at temperature {bestt.temperature:g}. Nonzero settings use three seeds. "
            'These test-subset observations are descriptive and must not be used to retune the main benchmark.')
    text='# Measured experiment results\n\n'+table+'\n## Conclusion\n\n'+conclusion+\
        '\n\n## Sampling-temperature analysis\n\n'+'\n\n'.join(temperature)+\
        '\n\n## Output screening\n\n'+'\n\n'.join(observations)+\
        '\n\nHuman inspection must confirm adequacy, bias, invented facts, cultural meaning and sensitive-content handling. '+\
        'Use the saved translations and ethical probes; do not call heuristic flags proven bias or hallucination.\n'
    (out/'RESULTS.md').write_text(text,encoding='utf-8')
    readme=root/'README.md'
    if readme.exists():
        s=readme.read_text(encoding='utf-8')
        a,b='<!-- RESULTS_START -->','<!-- RESULTS_END -->'
        if a in s and b in s:
            readme.write_text(s.split(a)[0]+a+'\n\n'+text+'\n'+b+s.split(b,1)[1],encoding='utf-8')
    print(text)
    return text
