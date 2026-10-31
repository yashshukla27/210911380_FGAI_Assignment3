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
