# Train metrics for one layer model

## Classifier

The following hidden layers added on the top of BERTimbau.
```python
classifier = nn.Sequential(
    nn.Linear(BERTIMBAU_HIDDEN_SIZE, 300),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(300, 100),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(50, num_labels),
)
```

#### Without freezing BERTimbau layers - Elapsed time: 00:16:27
| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0                | 0.72      | 0.76   | 0.74     | 51      |
| 1                | 0.91      | 0.98   | 0.94     | 269     |
| 2                | 0.98      | 0.85   | 0.91     | 187     |
| **Accuracy**     | -         | -      | 0.91     | 507     |
| **Macro Avg**    | 0.87      | 0.87   | 0.87     | 507     |
| **Weighted Avg** | 0.92      | 0.91   | 0.91     | 507     |




#### Freezing BERTimbau layers - Elapsed time: 00:16:27
| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0                | 0.79      | 0.80   | 0.80     | 51      |
| 1                | 0.94      | 0.96   | 0.95     | 269     |
| 2                | 0.97      | 0.93   | 0.95     | 187     |
| **Accuracy**     | -         | -      | 0.93     | 507     |
| **Macro Avg**    | 0.90      | 0.90   | 0.90     | 507     |
| **Weighted Avg** | 0.93      | 0.93   | 0.93     | 507     |




#### Oversampled without freezing BERTimbau layers - Elapsed time: 00:20:49
| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0                | 0.94      | 0.98   | 0.96     | 197     |
| 1                | 0.95      | 0.96   | 0.95     | 256     |
| 2                | 0.97      | 0.92   | 0.95     | 199     |
| **Accuracy**     | -         | -      | 0.95     | 652     |
| **Macro Avg**    | 0.95      | 0.95   | 0.95     | 652     |
| **Weighted Avg** | 0.95      | 0.95   | 0.95     | 652     |




#### Oversampled freezing (**Best Model**) - Elapsed time: 00:20:48
| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0                | 0.95      | 0.98   | 0.96     | 197     |
| 1                | 0.95      | 0.95   | 0.95     | 256     |
| 2                | 0.97      | 0.93   | 0.95     | 199     |
| **Accuracy**     | -         | -      | 0.96     | 652     |
| **Macro Avg**    | 0.96      | 0.96   | 0.96     | 652     |
| **Weighted Avg** | 0.96      | 0.96   | 0.96     | 652     |
