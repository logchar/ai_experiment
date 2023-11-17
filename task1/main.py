import numpy as np
import pandas as pd
import torch
from torch import nn


class Net(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, dim_out):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(dim_in, dim_hidden1), nn.BatchNorm1d(dim_hidden1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(dim_hidden1, dim_hidden2), nn.BatchNorm1d(dim_hidden2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(dim_hidden2, dim_out))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


data = pd.read_csv(
    "./data/event.csv",
    usecols=["事件等级", "事件类型", "事件名称", "目的端口"],
    encoding="gb18030",
)
data.columns = ["level", "type", "name", "port"]
str_data = data
reflection = list(set(data["type"]))
# print(data)
# print(reflection)

data["level"] = pd.factorize(data["level"])[0]
data["type"] = pd.factorize(data["type"])[0]
data["name"] = pd.factorize(data["name"])[0]
data["port"] = pd.factorize(data["port"])[0]
data = data.sample(frac=1).reset_index(drop=True)

print(data)

total_labels = np.array(data["type"])
total_labels = pd.get_dummies(total_labels, dtype=np.float32)

data = data.drop("type", axis=1)
total_labels = np.array(total_labels)
total_features = np.array(data)
train_labels = torch.as_tensor(total_labels[:66000])
test_labels = torch.as_tensor(total_labels[66000:])
train_features = torch.as_tensor(total_features[:66000], dtype=torch.float32)
test_features = torch.as_tensor(total_features[66000:], dtype=torch.float32)

# print(train_features)
# print(train_labels)
# print(test_features)
# print(test_labels)

net = Net(3, 400, 128, 18)

optimizer = torch.optim.SGD(net.parameters(), lr=0.015)

loss_function = torch.nn.CrossEntropyLoss()

for i in range(50):
    for j in range(220):
        batch_features = train_features[j*300:(j+1)*300]
        batch_labels = train_labels[j*300:(j+1)*300]
        out = net(batch_features)
        loss = loss_function(out, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i*220+j) % 50 == 0:
            print(i*220+j, loss)

tot_count = 0
for i in range(10):
    batch_features = test_features[i*300:(i+1)*300]
    batch_labels = test_labels[i*300:(i+1)*300]
    out = net(batch_features)
    predict = nn.functional.softmax(out, dim=-1).detach().numpy()
    predict = np.array(predict)
    predict = np.argmax(predict, axis=1)
    expection = np.argmax(np.array(batch_labels), axis=1)
    print(predict)
    print(expection)
    count = 0
    for j in range(300):
        if predict[j] == expection[j]:
            count = count + 1
    tot_count = tot_count + count
    print(count)

print(tot_count/3000)