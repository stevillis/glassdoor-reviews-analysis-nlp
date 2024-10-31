# Train metrics for best architecture model

## Classifier 1

The following hidden layers added on the top of BERTimbau.
```python
classifier = nn.Sequential(
    nn.Linear(bertimbau.config.hidden_size, 300),
    nn.ReLU(),
    nn.Linear(300, 100),
    nn.ReLU(),
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, num_labels),
)
```

#### Without freezing BERTimbau layers

|              | precision | recall | f1-score | support |
| ------------ | --------- | ------ | -------- | ------- |
| 0            | 0.75      | 0.76   | 0.76     | 51      |
| 1            | 0.95      | 0.97   | 0.96     | 269     |
| 2            | 0.96      | 0.91   | 0.94     | 187     |
| accuracy     |           |        | 0.93     | 507     |
| macro avg    | 0.89      | 0.88   | 0.88     | 507     |
| weighted avg | 0.93      | 0.93   | 0.93     | 507     |


#### Freezing BERTimbau layers
|              | precision | recall | f1-score | support |
| ------------ | --------- | ------ | -------- | ------- |
| 0            | 0.93      | 0.76   | 0.84     | 51      |
| 1            | 0.94      | 0.97   | 0.95     | 269     |
| 2            | 0.94      | 0.95   | 0.94     | 187     |
| accuracy     |           |        | 0.94     | 507     |
| macro avg    | 0.94      | 0.89   | 0.91     | 507     |
| weighted avg | 0.94      | 0.94   | 0.94     | 507     |


#### Oversampled without freezing BERTimbau layers
|              | precision | recall | f1-score | support |
| ------------ | --------- | ------ | -------- | ------- |
| 0            | 0.94      | 0.98   | 0.96     | 197     |
| 1            | 0.93      | 0.96   | 0.95     | 256     |
| 2            | 0.98      | 0.89   | 0.93     | 199     |
| accuracy     |           |        | 0.95     | 652     |
| macro avg    | 0.95      | 0.95   | 0.95     | 652     |
| weighted avg | 0.95      | 0.95   | 0.95     | 652     |


#### Oversampled freezing (**BEST MODEL**)
|              | precision | recall | f1-score | support |
| ------------ | --------- | ------ | -------- | ------- |
| 0            | 0.97      | 0.98   | 0.98     | 197     |
| 1            | 0.95      | 0.98   | 0.96     | 256     |
| 2            | 0.98      | 0.93   | 0.96     | 199     |
| accuracy     |           |        | 0.97     | 652     |
| macro avg    | 0.97      | 0.97   | 0.97     | 652     |
| weighted avg | 0.97      | 0.97   | 0.97     | 652     |
