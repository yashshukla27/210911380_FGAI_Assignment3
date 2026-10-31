# English to Hindi Machine Translation with Four Transformer Models

**Student:** Yash Shukla · **Registration:** 210911380  
**Course:** Foundations of Generative AI (ICT4403), Assignment 3

## Problem statement and objective

Build and compare a basic encoder-decoder Transformer, mT5, LLaMA and Gemma for English-to-Hindi translation. Train the basic model, fine-tune mT5 and apply the assignment-permitted prompting method to the decoder-only models. Evaluate quality, time, size and responsible-AI risks on identical held-out examples.

[Open the notebook in Google Colab](https://colab.research.google.com/github/yashshukla27/210911380_FGAI_Assignment3/blob/main/FGAI_Assignment3_Colab.ipynb) · [Submission PDF](submission/Yash_Shukla_210911380_FGAI_Assignment3.pdf)

## Execution status

This repository contains the implementation, the six component/API checks that passed in the original execution environment, a measured CPU baseline and submission documents. The latest environment lacks PyTorch, so those checks could not be rerun there. The basic baseline returned empty greedy translations; its zero scores and unusually short inference time reflect a failed translation run, not useful efficiency. It is not a claim that all four models have been run. Only numbers written by `run.py` are experimental results. A complete submission requires successful runs of all four models, inspection of the actual outputs, a verified GitHub URL and the final PDF. Consult `results/STATUS.json` for the preparation status. A single-model run cannot establish the best of four.

## Dataset and source

- Dataset: [IIT Bombay English-Hindi Parallel Corpus](https://huggingface.co/datasets/cfilt/iitb-english-hindi).
- Original source: https://www.cfilt.iitb.ac.in/~moses/iitb_en_hi_parallel/
- Corpus paper: Kunchukuttan, Mehta and Bhattacharyya (2018), https://aclanthology.org/L18-1548/.
- Published split metadata: 1,659,083 train, 520 validation, 2,507 test sentence pairs. These are corpus counts, not selected experimental counts.
- Default selected subset: 2,000 training, 100 validation, 100 test pairs, subject to cleaning. Exact selected counts and dataset revision are saved in `results/data_manifest.json`.
- Use official partitions. Test has first priority, then validation, then training when removing overlapping English or Hindi sentences. Normalize Unicode to NFC; collapse whitespace; preserve case, punctuation and Hindi script; discard empty/identical pairs and pairs exceeding 240 characters or 40 whitespace-separated words; require Latin letters in English and Devanagari in Hindi.
- Randomize a bounded training stream with a buffer of 10,000 and examine 50,000 rows. This economical short-sentence sample is **not a uniform sample of the full corpus** and limits generalization.
- Train the basic model's joint BPE tokenizer only on training text. Fit no tokenizer or parameter using the test set. Official held-out text may nevertheless have appeared in pretrained models' unknown training corpora.
- Download data from its source. Review the source's current terms; this project does not assign a new license to third-party text or model weights. Keep downloaded data and weights out of GitHub.

## Models and adaptation

| Model | Architecture | Method | Default settings |
|---|---|---|---|
| Basic Transformer | Encoder-decoder | Train from scratch | Joint BPE up to 8k; d=128; 2 layers on each side; 4 heads; FFN=512; dropout 0.1; AdamW 3e-4; 6 epochs |
| `google/mt5-small` | Encoder-decoder | Full supervised fine-tuning | Adafactor 5e-4; 2 epochs; batch 1; accumulation 16; float32; gradient checkpointing |
| `meta-llama/Llama-3.2-1B-Instruct` | Decoder-only | Two-shot prompting | The same two training examples used for both causal models; no weight updates |
| `google/gemma-3-1b-it` | Decoder-only | Two-shot prompting | Native chat template; same translation instruction and demonstrations |

Basic and mT5 checkpoints are selected by lowest validation negative log-likelihood. Basic training uses label smoothing 0.1; its training loss is not directly comparable with unsmoothed validation NLL. Model parameter counts are measured from the actual loaded models, not inferred from names. All models use float32 in this implementation to avoid mT5 float16 overflow and simplify precision comparisons. Downloads, preprocessing and model setup are separate from training time.

## Execution and dependencies

**Easiest:** Upload `FGAI_Assignment3_Colab.ipynb` to Google Colab and follow the numbered cells. That notebook reconstructs all source files automatically. Select a GPU runtime; a 16 GB GPU is a practical starting point. Runtime availability, RAM and execution time vary; no fixed completion time is promised.

Before running LLaMA or Gemma, sign in to Hugging Face, obtain access on the official model pages and create a read token. Model license/access decisions must be completed by you. Use the notebook's hidden login input or Colab secret `HF_TOKEN`; never put the token in code, outputs or GitHub.

For a local Python 3.11/3.12 environment:

```bash
python -m pip install -r requirements.txt
huggingface-cli login
python -m unittest discover -s tests -v
python run.py --prepare-only
python run.py --models basic
python run.py --models mt5
python run.py --models llama
python run.py --models gemma
python finalize.py --repo-url https://github.com/yashshukla27/210911380_FGAI_Assignment3
```

The four separate commands retain the same frozen data and configuration. `python run.py --models all` runs them sequentially. Start a new `--output` directory if changing hyperparameters. To resume after Colab disconnects, use a persistent Drive project directory. Completed models remain saved, but an interrupted model's training restarts at epoch one when rerun. Do not mix metrics from a CPU run with GPU timings in the final comparison; run all four in the same environment.

## Evaluation protocol

- Primary ranking: corpus **chrF**, character order 6, word order 0, beta 2; BLEU is secondary and breaks an exact chrF tie. BLEU uses SacreBLEU's `13a` tokenizer. Both use one Hindi reference per source and range from 0 to 100. Hindi morphology and valid paraphrases can reduce lexical overlap; inspect translations as well.
- Main benchmark: same ordered 100 test IDs, greedy decoding, one beam, batch size 1, up to 128 new tokens for every model. Generated output token budgets are equal but represent different text lengths because tokenizers differ. Log length-cap hits. Never crop output using the reference or remove bad examples from evaluation.
- Decoder-only output removes prompt tokens before decoding. Chat-template overhead remains part of its inference cost. Prompt instructions explicitly ask for translation only and preservation of names, numbers and negation.
- Timings use a monotonic clock and CUDA synchronization; one validation sentence warms up the model first. Inference includes tokenization, generation and decoding. Download and initialization time are excluded. Training/fine-tuning time includes the epoch loop, validation and checkpoint saves; no pretraining cost is counted. Prompted models have N/A adaptation time, not a falsely measured zero.
- Record exact parameter counts, updated parameter counts, weight bytes, CUDA peak allocated memory during benchmark inference (if applicable), hardware, software versions, model revisions, split hashes and random seeds.
- Do not call a result an architecture-only comparison. Size, pretrained data, instruction tuning, training budget, task adaptation and tokenization differ substantially. Prompting avoids task training but does not remove large-model memory or generation costs.

<!-- RESULTS_START -->

# Measured experiment results

| Model | BLEU | chrF | Training s | Inference s | Parameters |
|---|---:|---:|---:|---:|---:|
| basic | 0.00 | 0.00 | 47.30 | 0.28 | 2,982,208 |

## Conclusion

The four-model experiment is incomplete. No overall winner can yet be named.

## Sampling-temperature analysis

basic: greedy chrF 0.00; highest observed mean chrF 4.89 at temperature 1. Nonzero settings use three seeds. These test-subset observations are descriptive and must not be used to retune the main benchmark.

## Output screening

basic: 100 test translations triggered at least one heuristic review flag. Flags are screening signals, not confirmed ethical violations; digit-script differences can be benign.

Human inspection must confirm adequacy, bias, invented facts, cultural meaning and sensitive-content handling. Use the saved translations and ethical probes; do not call heuristic flags proven bias or hallucination.

<!-- RESULTS_END -->

## Sampling temperature analysis

Main scores use greedy decoding. Independently evaluate temperatures 0.3, 0.7 and 1.0 on the same first 25 frozen test examples with seeds 42, 43 and 44. Use full-distribution sampling (top_p=1, top_k=0) so temperature is the only sampling filter. Save per-seed predictions, mean BLEU/chrF and sample standard deviations. Greedy has one deterministic result. This subset analysis is descriptive, not a license to choose test-set hyperparameters. Higher temperature can increase diversity but can also damage faithfulness; actual direction must be read from measurements.

## Responsible and ethical AI

Read `ETHICS.md` and fill in `results/*_human_review.csv` and `results/*_ethical_probes.csv` after the run. These contain actual generated translations, not fabricated examples. A generated report lists heuristic flags only; a Hindi-capable reviewer must decide whether each flagged translation is actually wrong. `finalize.py` checks that a minimum review sample has been rated before marking the package ready.

## Files produced

- `results/metrics.json`, `metrics.csv`: measured scores, times, sizes, metric signatures.
- `results/*_translations.csv`: source, reference, actual output and token-cap flag for every test ID.
- `results/*_temperature*.csv`: seeded sampling outputs and summaries.
- `results/*_human_review.csv`, `*_ethical_probes.csv`: actual-output inspection worksheets.
- `results/*_history.json`: epoch losses; checkpoints stored locally.
- `results/quality_comparison.png`, `compute_comparison.png`, `temperature_analysis.png`: real plots generated from metrics.
- `results/RESULTS.md`: measured comparison and a conditional winner.
- `submission/`: personalized Word/PDF cover and technical report generated by `finalize.py`.

## Conclusion

The completed experiment will identify the highest-chrF model on this specific subset. A smaller trained encoder-decoder can offer lower inference cost, while pretrained models may benefit from extensive language knowledge; neither outcome is assumed in advance. An incomplete run has no defensible overall winner.

## References

1. Vaswani et al. (2017), Attention Is All You Need: https://arxiv.org/abs/1706.03762
2. Kunchukuttan et al. (2018), IIT Bombay corpus: https://aclanthology.org/L18-1548/
3. Xue et al. (2021), mT5: https://arxiv.org/abs/2010.11934 ; model: https://huggingface.co/google/mt5-small
4. Meta, LLaMA 3.2 model card: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
5. Google, Gemma 3 model card: https://huggingface.co/google/gemma-3-1b-it
6. Post (2018), A Call for Clarity in Reporting BLEU Scores: https://aclanthology.org/W18-6319/ ; implementation: https://github.com/mjpost/sacrebleu
7. Popovic (2015), chrF: https://aclanthology.org/W15-3049/
