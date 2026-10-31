"""Run with: python run.py --models all. Read START_HERE.md first."""
import argparse
import gc
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path
import torch
from src.common import read,dump,seed_everything
from src.prepare import prepare
from src.train import train_basic,train_mt5
from src.evaluate import evaluate_model,hf_predictor


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='config.json')
    parser.add_argument('--output',default='.')
    parser.add_argument('--models',nargs='+',default=['all'],choices=['all','basic','mt5','llama','gemma'])
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--cpu-threads',type=int,default=4)
    args=parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    cfg=read(args.config)
    root=Path(args.output).resolve()
    (root/'results').mkdir(parents=True,exist_ok=True)
    seed_everything(cfg['seed'])
    chosen=['basic','mt5','llama','gemma'] if 'all' in args.models else args.models
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    previous=root/'results/run_config.json'
    if previous.exists() and read(previous)!=cfg:
        raise ValueError('Config changed. Use a new --output directory to avoid mixing experiments.')
    dump(previous,cfg)
    env={'python':platform.python_version(),'platform':platform.platform(),'device':str(device),
         'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
         'torch':torch.__version__,'cpu_threads':args.cpu_threads,
         'versions':{k:importlib.metadata.version(k) for k in ['transformers','datasets','sacrebleu','sentencepiece']}}
    old_metrics = root/'results/metrics.json'
    if old_metrics.exists():
        for previous_result in read(old_metrics).values():
            if previous_result.get('device', str(device)) != str(device) or previous_result.get('hardware', env['gpu'] or platform.machine()) != (env['gpu'] or platform.machine()):
                raise ValueError('Hardware differs from saved measurements. Use a new --output directory and rerun all models.')
    dump(root/'results/environment.json',env)
    if (root/'results/data_manifest.json').exists() and all((root/'data'/f'{s}.json').exists() for s in ['train','validation','test']):
        data={s:read(root/'data'/f'{s}.json') for s in ['train','validation','test']}
    else:
        data=prepare(cfg,root)
    if args.prepare_only:
        print('Data prepared and split manifest saved.')
        return
    examples=[r for r in data['train'] if len(r['en'].split())<=12 and len(r['hi'].split())<=15][:2]
    if len(examples)!=2:
        raise ValueError('Need two short training examples for shared few-shot prompts')
    dump(root/'results/prompt_examples.json',examples)
    results_file=root/'results/metrics.json'
    results=read(results_file) if results_file.exists() else {}
    revisions_file=root/'results/model_revisions.json'
    revisions=read(revisions_file) if revisions_file.exists() else {}
    for name in chosen:
        seed_everything(cfg['seed'])
        if name!='basic':
            from huggingface_hub import HfApi,hf_hub_download
            model_id=cfg['models'][name]
            revision=revisions.get(name) or HfApi().model_info(model_id).sha
            # Fail clearly before weight download when gated access has not been granted.
            hf_hub_download(model_id,'config.json',revision=revision)
            revisions[name]=revision
            dump(revisions_file,revisions)
        start=time.perf_counter()
        if name=='basic':
            model,tok,train_seconds,history=train_basic(cfg,data,root,device)
            predict=lambda text,temperature=None:model.translate(text,tok,device,cfg['max_new_tokens'],temperature)
            adaptation='trained from scratch; best validation NLL checkpoint'
        elif name=='mt5':
            model,tok,train_seconds,history=train_mt5(cfg,data,root,device,revisions[name])
            predict=hf_predictor(model,tok,cfg,device,name,examples)
            adaptation='full supervised fine-tuning; best validation NLL checkpoint'
        else:
            from transformers import AutoTokenizer,AutoModelForCausalLM
            tok=AutoTokenizer.from_pretrained(cfg['models'][name],revision=revisions[name])
            if tok.pad_token_id is None: tok.pad_token=tok.eos_token
            # Float32 on all models: fits a 16 GB GPU at batch 1 for these small variants.
            model=AutoModelForCausalLM.from_pretrained(cfg['models'][name],revision=revisions[name],torch_dtype=torch.float32).to(device)
            predict=hf_predictor(model,tok,cfg,device,name,examples)
            train_seconds=None
            adaptation='two-shot prompting; no fine-tuning'
        setup=max(0.,time.perf_counter()-start-(train_seconds or 0.))
        results[name]=evaluate_model(name,model,predict,cfg,data,root,train_seconds,adaptation,setup)
        results[name]['device'] = str(device)
        results[name]['hardware'] = env['gpu'] or platform.machine()
        dump(results_file,results)
        print(json.dumps(results[name],indent=2),flush=True)
        del predict,model,tok
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    from src.summarize import summarize
    summarize(root)

if __name__=='__main__':
    main()
