# Train metrics for one layer model

## Classifier

The following hidden layers added on the top of BERTimbau.
```python
classifier = nn.Sequential(
    nn.Linear(bertimbau.config.hidden_size, num_labels),
)
```

#### Without freezing BERTimbau layers - Elapsed time: 00:16:19
| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0                | 0.97      | 0.71   | 0.82     | 51      |
| 1                | 0.92      | 0.99   | 0.95     | 269     |
| 2                | 0.95      | 0.93   | 0.94     | 187     |
| **Accuracy**     | -         | -      | 0.94     | 507     |
| **Macro Avg**    | 0.95      | 0.87   | 0.90     | 507     |
| **Weighted Avg** | 0.94      | 0.94   | 0.93     | 507     |



#### Freezing BERTimbau layers - Elapsed time: 00:16:18
| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0                | 0.84      | 0.80   | 0.82     | 51      |
| 1                | 0.94      | 0.96   | 0.95     | 269     |
| 2                | 0.96      | 0.94   | 0.95     | 187     |
| **Accuracy**     | -         | -      | 0.94     | 507     |
| **Macro Avg**    | 0.91      | 0.90   | 0.91     | 507     |
| **Weighted Avg** | 0.94      | 0.94   | 0.94     | 507     |



#### Oversampled without freezing BERTimbau layers (**Best Model**) - Elapsed time: 00:20:36
| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0                | 0.95      | 0.98   | 0.96     | 197     |
| 1                | 0.95      | 0.98   | 0.96     | 256     |
| 2                | 0.98      | 0.92   | 0.95     | 199     |
| **Accuracy**     | -         | -      | 0.96     | 652     |
| **Macro Avg**    | 0.96      | 0.96   | 0.96     | 652     |
| **Weighted Avg** | 0.96      | 0.96   | 0.96     | 652     |



#### Oversampled freezing - Elapsed time: 00:20:35
| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0                | 0.90      | 0.98   | 0.94     | 197     |
| 1                | 0.96      | 0.94   | 0.95     | 256     |
| 2                | 0.96      | 0.90   | 0.93     | 199     |
| **Accuracy**     | -         | -      | 0.94     | 652     |
| **Macro Avg**    | 0.94      | 0.94   | 0.94     | 652     |
| **Weighted Avg** | 0.94      | 0.94   | 0.94     | 652     |
