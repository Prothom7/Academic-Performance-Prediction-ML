# WEKA Model Experiment Ledger

All values below are from the recorded WEKA runs. Blank cells mean the metric was undefined or was not recorded.

| Model | Accuracy | Kappa | W. Precision | W. Recall | W. F1 | ROC-AUC | PRC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| ZeroR | 28.0148% | 0.0000 | — | 0.280 | — | 0.498 | 0.213 |
| OneR | 32.7929% | 0.0942 | — | 0.328 | — | 0.546 | 0.237 |
| NaiveBayes | 34.6985% | 0.1559 | 0.341 | 0.347 | 0.334 | 0.648 | 0.322 |
| BayesNet | 34.5848% | 0.1548 | 0.341 | 0.346 | 0.334 | 0.647 | 0.322 |
| Logistic | 34.0728% | 0.1467 | 0.336 | 0.341 | 0.332 | 0.647 | 0.318 |
| SimpleLogistic | 35.2958% | 0.1554 | 0.351 | 0.353 | 0.337 | 0.652 | 0.324 |
| SMO | 33.9022% | 0.1439 | 0.342 | 0.339 | 0.332 | 0.641 | 0.293 |
| MultilayerPerceptron | 56.0011% | 0.4402 | 0.560 | 0.560 | 0.560 | 0.757 | 0.533 |
| J48 | 52.6451% | 0.3969 | 0.530 | 0.526 | 0.525 | 0.757 | 0.483 |
| RandomForest baseline | 60.3242% | 0.4855 | 0.634 | 0.603 | 0.604 | 0.863 | 0.724 |
| RandomTree | 55.5461% | 0.4356 | 0.555 | 0.555 | 0.555 | 0.734 | 0.449 |
| REPTree | 38.4812% | 0.2092 | 0.385 | 0.385 | 0.381 | 0.678 | 0.362 |
| Bagging + REPTree | 50.9386% | 0.3691 | 0.515 | 0.509 | 0.508 | 0.769 | 0.552 |
| AdaBoostM1 + DecisionStump | 30.5176% | 0.0502 | — | 0.305 | — | 0.542 | 0.234 |
| LogitBoost | 35.5518% | 0.1545 | 0.355 | 0.356 | 0.333 | 0.653 | 0.328 |
| RandomCommittee | 58.2480% | 0.4649 | 0.588 | 0.582 | 0.583 | 0.848 | 0.704 |
| RandomSubSpace | 53.2139% | 0.3925 | 0.556 | 0.532 | 0.529 | 0.783 | 0.587 |
| IBk (K=1) | 57.2810% | 0.4551 | 0.577 | 0.573 | 0.572 | 0.740 | 0.479 |
| KStar | 58.0489% | 0.4639 | 0.581 | 0.580 | 0.579 | 0.851 | 0.702 |
| LWL + DecisionStump | 34.1013% | 0.1158 | — | 0.341 | — | 0.648 | 0.325 |
| PART | 50.9386% | 0.3752 | 0.509 | 0.509 | 0.509 | 0.747 | 0.469 |
| JRip | 36.4050% | 0.1449 | 0.420 | 0.364 | 0.307 | 0.610 | 0.309 |
| DecisionTable | 38.3959% | 0.1863 | 0.459 | 0.384 | 0.343 | 0.644 | 0.335 |
| HoeffdingTree | 34.6985% | 0.1560 | 0.341 | 0.347 | 0.334 | 0.648 | 0.322 |
| LMT | 58.1342% | 0.4677 | 0.581 | 0.581 | 0.581 | 0.781 | 0.544 |
| DecisionStump | 30.5176% | 0.0502 | — | 0.305 | — | 0.542 | 0.234 |
| ClassificationViaRegression + M5P | 46.1604% | 0.3105 | 0.458 | 0.462 | 0.456 | 0.719 | 0.436 |

## Models unavailable / skipped in this WEKA installation

- Ridor
- SimpleCart
- ConjunctiveRule
- RBFNetwork
- SGD

These are recorded as **skipped**, not as failed model results.
