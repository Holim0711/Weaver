# Augmentation Experiments
## Configurations
- model: Wide ResNet 28-10
- optimizer: SGD with neterov (lr=0.1, m=0.9, wd=5e-4)
- scheduler: Linear Warmup Cosine Annealing
- max epochs: 200
- batch size:128

## Accuracy Results
| Attempt | #1  | #2  | #3  | mean |
| :---:   | :-: | :-: | :-: | :-:  |
| AutoAugment | 97.63 | 97.42 | 97.37 | 97.47 |
| RandAugment | 97.25 | 97.39 | 97.36 | 97.33 |
