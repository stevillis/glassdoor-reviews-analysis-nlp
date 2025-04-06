# Train metrics for small layer model

## Classifier

The following hidden layers added on the top of BERTimbau.
```python
classifier = nn.Sequential(
    nn.Linear(BERTIMBAU_HIDDEN_SIZE, 50),
    nn.ReLU(),
    nn.Linear(50, num_labels),
)
```

#### Without freezing BERTimbau layers - Elapsed time: 00:16:47
| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0                | 0.82      | 0.82   | 0.82     | 51      |
| 1                | 0.98      | 0.91   | 0.95     | 269     |
| 2                | 0.89      | 0.98   | 0.93     | 187     |
| **Accuracy**     | -         | -      | 0.93     | 507     |
| **Macro Avg**    | 0.90      | 0.91   | 0.90     | 507     |
| **Weighted Avg** | 0.93      | 0.93   | 0.93     | 507     |



#### Freezing BERTimbau layers - Elapsed time: 00:16:52
| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0                | 0.91      | 0.76   | 0.83     | 51      |
| 1                | 0.93      | 0.97   | 0.95     | 269     |
| 2                | 0.96      | 0.95   | 0.95     | 187     |
| **Accuracy**     | -         | -      | 0.94     | 507     |
| **Macro Avg**    | 0.93      | 0.89   | 0.91     | 507     |
| **Weighted Avg** | 0.94      | 0.94   | 0.94     | 507     |



#### Oversampled without freezing BERTimbau layers (**Best Model**) - Elapsed time: 00:21:18
| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0                | 0.96      | 0.98   | 0.97     | 197     |
| 1                | 0.96      | 0.95   | 0.95     | 256     |
| 2                | 0.95      | 0.96   | 0.95     | 199     |
| **Accuracy**     | -         | -      | 0.96     | 652     |
| **Macro Avg**    | 0.96      | 0.96   | 0.96     | 652     |
| **Weighted Avg** | 0.96      | 0.96   | 0.96     | 652     |



#### Oversampled freezing - Elapsed time: 00:21:18
| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0                | 0.96      | 0.98   | 0.97     | 197     |
| 1                | 0.92      | 0.98   | 0.95     | 256     |
| 2                | 0.99      | 0.87   | 0.93     | 199     |
| **Accuracy**     | -         | -      | 0.95     | 652     |
| **Macro Avg**    | 0.96      | 0.95   | 0.95     | 652     |
| **Weighted Avg** | 0.95      | 0.95   | 0.95     | 652     |
