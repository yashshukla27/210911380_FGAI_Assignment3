"""Shared reproducibility, data and measurement helpers."""
import csv
import hashlib
import json
import random
import re
import time
import unicodedata
from pathlib import Path


def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text)).strip()


def clean_pair(item, max_chars=240, max_words=40):
    tr = item.get('translation', item)
    en, hi = normalize(tr['en']), normalize(tr['hi'])
    if not en or not hi or en.casefold() == hi.casefold():
        return None
    if max(len(en), len(hi)) > max_chars or max(len(en.split()), len(hi.split())) > max_words:
        return None
    if not re.search('[A-Za-z]', en) or not re.search('[\u0900-\u097f]', hi):
        return None
    return {'en': en, 'hi': hi, 'id': hashlib.sha256((en+'\0'+hi).encode()).hexdigest()[:20]}


def unique_pairs(items, seen_en=None, seen_hi=None):
    """Prevent either a source or reference appearing across partitions."""
    seen_en = set() if seen_en is None else seen_en
    seen_hi = set() if seen_hi is None else seen_hi
    result = []
    for item in items:
        if item is None:
            continue
        en, hi = item['en'].casefold(), item['hi']
        if en in seen_en or hi in seen_hi:
            continue
        seen_en.add(en)
        seen_hi.add(hi)
        result.append(item)
    return result


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_csv(path, rows):
    if not rows:
        raise ValueError('Cannot write an empty result table')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def synchronize():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    synchronize()
    start = time.perf_counter()
    result = fn()
    synchronize()
    return result, time.perf_counter() - start


def scores(predictions, references):
    from sacrebleu.metrics import BLEU, CHRF
    if len(predictions) != len(references) or not predictions:
        raise ValueError('Predictions and references must have the same nonzero length')
    bleu, chrf = BLEU(tokenize='13a'), CHRF(char_order=6, word_order=0, beta=2)
    b = bleu.corpus_score(predictions, [references])
    c = chrf.corpus_score(predictions, [references])
    return {'bleu': b.score, 'chrf': c.score,
            'bleu_signature': str(bleu.get_signature()), 'chrf_signature': str(chrf.get_signature())}
