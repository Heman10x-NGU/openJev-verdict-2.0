# RL Agent evaluation

Metrics after calibration. Zero-shot = task families held out of training entirely.

## In-task

| task family | questions | accuracy | ECE | NLL |
|---|---|---|---|---|
| conversation outcomes | 3600 | 0.482 | 0.019 | 0.693 |
| email triage | 2691 | 0.732 | 0.017 | 0.595 |
| emotion and tone | 1825 | 0.906 | 0.018 | 0.238 |
| inference and fact checking | 3022 | 0.883 | 0.054 | 0.340 |
| instruction-following tasks | 600 | 0.878 | 0.046 | 0.302 |
| intent and routing | 1475 | 0.991 | 0.009 | 0.181 |
| moderation and safety | 2708 | 0.967 | 0.061 | 0.153 |
| reading comprehension | 770 | 0.847 | 0.083 | 0.409 |
| response quality scoring | 3146 | 0.581 | 0.023 | 1.009 |
| robustness checks | 744 | 0.851 | 0.108 | 1.058 |
| search relevance | 733 | 0.628 | 0.066 | 0.728 |
| sentiment and rating | 961 | 0.442 | 0.438 | 3.545 |
| topic classification | 749 | 0.939 | 0.029 | 0.196 |

Overall: accuracy 0.753, ECE 0.030, Brier 0.308, accuracy at 50% coverage 0.947

## Zero-shot

| task family | questions | accuracy | ECE | NLL |
|---|---|---|---|---|
| emotion and tone | 600 | 0.583 | 0.318 | 1.976 |
| instruction-following tasks | 600 | 0.863 | 0.045 | 0.319 |
| moderation and safety | 600 | 0.797 | 0.171 | 1.415 |
| sentiment and rating | 600 | 0.362 | 0.291 | 1.798 |

Overall: accuracy 0.651, ECE 0.204, Brier 0.532, accuracy at 50% coverage 0.818

## Latency

```
{
  "1_questions": {
    "p50_ms": 38.4,
    "p95_ms": 42.1
  },
  "10_questions": {
    "p50_ms": 156.0,
    "p95_ms": 158.4
  },
  "50_questions": {
    "p50_ms": 721.4,
    "p95_ms": 733.0
  }
}
```
