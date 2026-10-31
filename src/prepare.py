"""Freeze official dataset splits, filtering duplicates before sampling."""
import hashlib
import itertools
import random
from pathlib import Path
from .common import clean_pair, unique_pairs, dump, read


def prepare(cfg, root):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    root = Path(root)
    manifest_path = root/'results/data_manifest.json'
    revision = read(manifest_path)['revision'] if manifest_path.exists() else HfApi().dataset_info(cfg['dataset']).sha
    seen_en, seen_hi, data, stats = set(), set(), {}, {}
    # Test has priority; all eligible held-out text is excluded from training.
    for split in ['test', 'validation', 'train']:
        stream = load_dataset(cfg['dataset'], split=split, streaming=True, revision=revision)
        if split == 'train':
            # Bounded shuffled stream: a resource-limited sample, not a uniform corpus sample.
            stream = stream.shuffle(seed=cfg['seed'], buffer_size=10000)
            stream = itertools.islice(stream, cfg['train_pool'])
        raw = list(stream)
        cleaned = [clean_pair(r, cfg['max_chars'], cfg['max_words']) for r in raw]
        eligible = unique_pairs(cleaned, seen_en, seen_hi)
        random.Random(cfg['seed']).shuffle(eligible)
        n = cfg[f'{split}_size']
        if len(eligible) < n:
            raise ValueError(f'{split}: only {len(eligible)} eligible pairs; need {n}. Lower split size or relax filters.')
        data[split] = eligible[:n]
        dump(root/'data'/f'{split}.json', data[split])
        digest = hashlib.sha256('\n'.join(r['id'] for r in data[split]).encode()).hexdigest()
        stats[split] = {'examined': len(raw), 'after_content_filter': sum(x is not None for x in cleaned),
                        'after_deduplication': len(eligible), 'selected': n, 'ordered_id_sha256': digest}
    sets = [{r['en'].casefold() for r in data[s]} for s in ['train','validation','test']]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i+1,3))
    manifest = {'dataset':cfg['dataset'], 'revision':revision, 'seed':cfg['seed'], 'splits':stats,
        'sampling_note':'Official partitions retained; bounded buffered shuffle of the training stream; short-sentence subset.',
        'normalization':'Unicode NFC, collapse whitespace, preserve case/punctuation/Devanagari.',
        'deduplication':'casefolded English OR exact normalized Hindi; held-out partitions take priority.'}
    dump(root/'results/data_manifest.json', manifest)
    return data
