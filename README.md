# model2vec-ticket-triage

## Results and Experiments

Below you will find a table with the evaluation accuracy of the trained models on the test set.

| Model | Learning Rate | Boosted classes | Boost factor | Epochs | Eval Accuracy (test set) |
| --- | --- | --- | --- | --- | --- |
| (baseline) ticket_triage_potion-base-32M_1e-3 | 1e-3 | - | - | 15 | 0.60 |
| ticket_triage_potion-base-32M_1e-3-oversample | 1e-3 | - | - | 15 | 0.61 | 
| ticket_triage_potion-base-32M_1e-3-oversample-boosting | 1e-3 | [2, 3, 7] | 3 | 15 | 0.63 |
| ticket_triage_potion-base-32M_1e-3-oversample-boosting-wd | 1e-3 | [2, 3, 7] | 3 | 15 | 0.64 |
| ticket_triage_potion-base-32M_5e-3-oversample-boosting-wd | 5e-3 | [2, 3, 7] | 3 | 10 | 0.70 |
| ticket_triage_potion-base-32M_8e-3-oversample-boosting-wd | 8e-3 | [2, 3, 7] | 3 | 15 |  |

