import functools
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .basic import BasicTransformer, PAD, collate_basic
from .common import dump, timed


def create_basic(cfg, vocab_size):
    return BasicTransformer(vocab_size, **{k:cfg[k] for k in ['d_model','nhead','layers','ff_dim','dropout']})


def train_basic(cfg, data, root, device):
    import sentencepiece as spm
    root = Path(root)
    ckpt = root/'checkpoints/basic'
    ckpt.mkdir(parents=True, exist_ok=True)
    corpus = ckpt/'tokenizer_train.txt'
    # Tokenizer learns ONLY from the training split.
    corpus.write_text('\n'.join(r[k] for r in data['train'] for k in ['en','hi']), encoding='utf-8')
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(ckpt/'tokenizer'),
        vocab_size=cfg['vocab_size'], model_type='bpe', character_coverage=1.0,
        pad_id=PAD, unk_id=1, bos_id=2, eos_id=3, hard_vocab_limit=False,
        shuffle_input_sentence=False, num_threads=1)
    sp = spm.SentencePieceProcessor(model_file=str(ckpt/'tokenizer.model'))
    model = create_basic(cfg, sp.vocab_size()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['basic_lr'], weight_decay=0.01)
    collate = functools.partial(collate_basic, sp=sp)
    train = DataLoader(data['train'], batch_size=cfg['basic_batch'], shuffle=True, collate_fn=collate)
    val = DataLoader(data['validation'], batch_size=cfg['basic_batch'], collate_fn=collate)
    history = []

    def fit():
        best = float('inf')
        for epoch in range(1, cfg['basic_epochs']+1):
            model.train()
            total, count = 0., 0
            for src, tgt in train:
                src, tgt = src.to(device), tgt.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(src, tgt[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, sp.vocab_size()), tgt[:,1:].reshape(-1),
                                       ignore_index=PAD, label_smoothing=0.1)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite basic Transformer loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()
                n = tgt[:,1:].ne(PAD).sum().item()
                total += loss.item()*n
                count += n
            model.eval()
            vtotal, vcount = 0., 0
            with torch.no_grad():
                for src,tgt in val:
                    src,tgt = src.to(device),tgt.to(device)
                    logits = model(src,tgt[:,:-1])
                    loss = F.cross_entropy(logits.reshape(-1,sp.vocab_size()),tgt[:,1:].reshape(-1),ignore_index=PAD,reduction='sum')
                    vtotal += loss.item()
                    vcount += tgt[:,1:].ne(PAD).sum().item()
            vl = vtotal/vcount
            history.append({'epoch':epoch,'train_smoothed_loss':total/count,'validation_nll':vl})
            print('basic',history[-1],flush=True)
            if vl < best:
                best = vl
                torch.save(model.state_dict(),ckpt/'best.pt')
            dump(root/'results/basic_history.json',history)
    _, elapsed = timed(fit)
    model.load_state_dict(torch.load(ckpt/'best.pt',map_location=device,weights_only=True))
    dump(ckpt/'config.json',{'config':cfg,'vocab_size':sp.vocab_size()})
    return model,sp,elapsed,history


def train_mt5(cfg,data,root,device,revision):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Adafactor, DataCollatorForSeq2Seq
    root = Path(root)
    ckpt = root/'checkpoints/mt5'
    ckpt.mkdir(parents=True,exist_ok=True)
    name = cfg['models']['mt5']
    tok = AutoTokenizer.from_pretrained(name,revision=revision,use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(name,revision=revision).to(device)
    # mT5 fp16 can overflow: use float32, memory-efficient Adafactor and batch size 1.
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    def encode(rows):
        result=[]
        for row in rows:
            x=tok('translate English to Hindi: '+row['en'])
            y=tok(text_target=row['hi'])
            if max(len(x['input_ids']),len(y['input_ids']))>cfg['max_input_tokens']:
                raise ValueError('mT5 sequence too long; increase max_input_tokens. No silent truncation.')
            x['labels']=y['input_ids']
            result.append(x)
        return result
    collator=DataCollatorForSeq2Seq(tok,model=model,padding=True,label_pad_token_id=-100)
    train=DataLoader(encode(data['train']),batch_size=cfg['mt5_batch'],shuffle=True,collate_fn=collator)
    val=DataLoader(encode(data['validation']),batch_size=cfg['mt5_batch'],collate_fn=collator)
    opt=Adafactor(model.parameters(),lr=cfg['mt5_lr'],scale_parameter=False,relative_step=False,warmup_init=False)
    history=[]
    def fit():
        best=float('inf')
        acc=cfg['mt5_accumulation']
        for epoch in range(1,cfg['mt5_epochs']+1):
            model.train()
            total,count=0.,0
            opt.zero_grad(set_to_none=True)
            for i,batch in enumerate(train):
                batch={k:v.to(device) for k,v in batch.items()}
                loss=model(**batch).loss
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite mT5 loss')
                # Last accumulation group may be smaller; correct its denominator.
                group_start=(i//acc)*acc
                group_size=min(acc,len(train)-group_start)
                (loss/group_size).backward()
                if (i+1)%acc==0 or i+1==len(train):
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                n=batch['labels'].ne(-100).sum().item()
                total+=loss.item()*n
                count+=n
                if (i+1)%500==0:
                    print(f'mT5 epoch {epoch}: {i+1}/{len(train)} batches',flush=True)
            model.eval()
            vtotal,vcount=0.,0
            with torch.no_grad():
                for batch in val:
                    batch={k:v.to(device) for k,v in batch.items()}
                    n=batch['labels'].ne(-100).sum().item()
                    vtotal+=model(**batch).loss.item()*n
                    vcount+=n
            vl=vtotal/vcount
            history.append({'epoch':epoch,'train_nll':total/count,'validation_nll':vl})
            print('mt5',history[-1],flush=True)
            if vl<best:
                best=vl
                model.save_pretrained(ckpt)
                tok.save_pretrained(ckpt)
            dump(root/'results/mt5_history.json',history)
    _,elapsed=timed(fit)
    del opt,model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model=AutoModelForSeq2SeqLM.from_pretrained(ckpt).to(device)
    model.config.use_cache=True
    model.eval()
    return model,tok,elapsed,history
