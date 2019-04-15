#       Vodafone | Predict Store Footfall

### Scope:
- The overall goal of the challenge is to predict the store footfall’s label and value.

### Data Source:
- The dataset describes the profiles of VF customers nearby the Vodafone stores and the stores’ characteristics.

### Evaluation:
- The evaluation metrics are the Residual Sum of Squares (regression task) and Accuracy (classification task):
```latex
RRS = \frac{1}{N} \sum_{i=1}^{N}(y^{_{i}}-f(x^{_{i}}))^{^{2}}
```
```latex
Acc = \frac{TP+TN}{TP+TN+FP+FN}
```
