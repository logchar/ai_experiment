import pandas as pd
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss

from setfit import SetFitModel, SetFitTrainer

test_data = pd.read_csv(
    'test.csv',
    names=['label', 'title', 'text'],
    header=0,
    usecols=['label', 'text'],
)

test_dataset = Dataset.from_pandas(test_data)
classify1 = test_dataset.filter(lambda x: x["label"] == 1).to_pandas()
classify2 = test_dataset.filter(lambda x: x["label"] == 2).to_pandas()
classify3 = test_dataset.filter(lambda x: x["label"] == 3).to_pandas()
classify4 = test_dataset.filter(lambda x: x["label"] == 4).to_pandas()

length = 20
classify1 = classify1.sample(frac=1).reset_index(drop=True)[:length]
classify2 = classify2.sample(frac=1).reset_index(drop=True)[:length]
classify3 = classify3.sample(frac=1).reset_index(drop=True)[:length]
classify4 = classify4.sample(frac=1).reset_index(drop=True)[:length]

train_dataframe = pd.concat([classify1, classify2, classify3, classify4]).sample(frac=1).reset_index(drop=True)
train_dataset = Dataset.from_pandas(train_dataframe)

print(train_dataframe)
print(train_dataset)
print(test_dataset)

model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

print(1)

trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss_class=CosineSimilarityLoss,
    batch_size=length,
    num_iterations=20,
    num_epochs=4
)

# Train and evaluate!
trainer.train()
metrics = trainer.evaluate()

print(metrics)
