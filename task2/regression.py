import numpy as np
import pandas as pd
import torch
from torch import nn


class Net(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.BatchNorm1d(dim_hidden), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(dim_hidden, dim_out))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.squeeze(-1)
        return x


# 数据处理
data = pd.read_csv(
    "data.csv",
    nrows=10000
)
data = data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'job_name', 'task_name', 'inst_name', 'worker_name', 'gpu_type'])
data.fillna(0, inplace=True)
label = data['end_time'].to_numpy() - data['start_time'].to_numpy()
feature = data.drop(columns=['end_time', 'start_time']).to_numpy()

# print(feature.shape)
print(label)
# print(data.info)

train_features = torch.as_tensor(feature[:7000], dtype=torch.float32)
train_labels = torch.as_tensor(label[:7000], dtype=torch.float32)
test_features = torch.as_tensor(feature[7000:], dtype=torch.float32)
test_labels = torch.as_tensor(label[7000:], dtype=torch.float32)

print(train_features)
print(train_labels)
print(test_features)
print(test_labels)

net = Net(18, 256, 1)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

loss_function = torch.nn.MSELoss()

# 训练
batch_size = 10
for i in range(30):
    feature_batch = train_features[i*batch_size: (i+1)*batch_size]
    label_batch = train_labels[i*batch_size: (i+1)*batch_size]

    out = net(feature_batch)
    loss = loss_function(out, label_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)
