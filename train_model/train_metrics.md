# Train metrics for different classifiers

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
| 0            | 0.84      | 0.73   | 0.78     | 51      |
| 1            | 0.95      | 0.97   | 0.96     | 269     |
| 2            | 0.94      | 0.95   | 0.94     | 187     |
| accuracy     |           |        | 0.94     | 507     |
| macro avg    | 0.91      | 0.88   | 0.89     | 507     |
| weighted avg | 0.94      | 0.94   | 0.94     | 507     |


#### Freezing BERTimbau layers
|              | precision | recall | f1-score | support |
| ------------ | --------- | ------ | -------- | ------- |
| 0            | 0.73      | 0.88   | 0.80     | 51      |
| 1            | 0.96      | 0.95   | 0.96     | 269     |
| 2            | 0.96      | 0.93   | 0.94     | 187     |
| accuracy     |           |        | 0.93     | 507     |
| macro avg    | 0.88      | 0.92   | 0.90     | 507     |
| weighted avg | 0.94      | 0.93   | 0.93     | 507     |


#### Oversampled without freezing BERTimbau layers
|              | precision | recall | f1-score | support |
| ------------ | --------- | ------ | -------- | ------- |
| 0            | 0.90      | 0.98   | 0.94     | 197     |
| 1            | 0.93      | 0.96   | 0.94     | 256     |
| 2            | 0.98      | 0.86   | 0.92     | 199     |
| accuracy     |           |        | 0.94     | 652     |
| macro avg    | 0.94      | 0.93   | 0.93     | 652     |
| weighted avg | 0.94      | 0.94   | 0.94     | 652     |


#### Oversampled freezing (**BEST MODEL**)
|              | precision | recall | f1-score | support |
| ------------ | --------- | ------ | -------- | ------- |
| 0            | 0.96      | 0.98   | 0.97     | 197     |
| 1            | 0.92      | 0.98   | 0.95     | 256     |
| 2            | 0.98      | 0.88   | 0.93     | 199     |
| accuracy     |           |        | 0.95     | 652     |
| macro avg    | 0.96      | 0.95   | 0.95     | 652     |
| weighted avg | 0.95      | 0.95   | 0.95     | 652     |


## Classifier 2
A layer with 3 outputs added on the top of freezed BERTimbau.
```python
classifier = nn.Sequential(
   nn.Linear(bertimbau.config.hidden_size, num_labels),
)
```

#### Without oversampling
|              | precision | recall | f1-score | support |
| ------------ | --------- | ------ | -------- | ------- |
| 0            | 0.00      | 0.00   | 0.00     | 51      |
| 1            | 0.64      | 0.98   | 0.77     | 269     |
| 2            | 0.81      | 0.40   | 0.54     | 187     |
| accuracy     |           |        | 0.67     | 507     |
| macro avg    | 0.48      | 0.46   | 0.44     | 507     |
| weighted avg | 0.64      | 0.67   | 0.61     | 507     |


#### With oversampling
Here is the transformed markdown table with the new values:

|              | precision | recall | f1-score | support |
| ------------ | --------- | ------ | -------- | ------- |
| 0            | 0.87      | 0.59   | 0.71     | 197     |
| 1            | 0.64      | 0.96   | 0.77     | 256     |
| 2            | 0.88      | 0.58   | 0.70     | 199     |
| accuracy     |           |        | 0.73     | 652     |
| macro avg    | 0.80      | 0.71   | 0.72     | 652     |
| weighted avg | 0.78      | 0.73   | 0.73     | 652     |


### Classifier 3
The following hidden layers added on the top of BERTimbau.
```python
classifier = nn.Sequential(
    nn.Linear(bertimbau.config.hidden_size, 300),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(300, 100),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(100, num_labels),
)
```

#### Without freezing BERTimbau layers
|              | precision | recall | f1-score | support |
| ------------ | --------- | ------ | -------- | ------- |
| 0            | 0.83      | 0.75   | 0.78     | 51      |
| 1            | 0.92      | 0.97   | 0.94     | 269     |
| 2            | 0.96      | 0.90   | 0.93     | 187     |
| accuracy     |           |        | 0.92     | 507     |
| macro avg    | 0.90      | 0.87   | 0.89     | 507     |
| weighted avg | 0.92      | 0.92   | 0.92     | 507     |

#### Freezing BERTimbau layers
|              | precision | recall | f1-score | support |
| ------------ | --------- | ------ | -------- | ------- |
| 0            | 0.91      | 0.80   | 0.85     | 51      |
| 1            | 0.95      | 0.97   | 0.96     | 269     |
| 2            | 0.94      | 0.95   | 0.95     | 187     |
| accuracy     |           |        | 0.94     | 507     |
| macro avg    | 0.94      | 0.91   | 0.92     | 507     |
| weighted avg | 0.94      | 0.94   | 0.94     | 507     |

#### With oversampling (Kaggle free tier ended up)
