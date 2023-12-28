import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
import numpy as np


# 加载数据
data = pd.read_csv('names.csv')
names = data['Name']
features = data['Features']

# 使用TF-IDF向量化器提取特征
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(features)

num_clusters = 30

# 运行K-means聚类
model = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

labels = model.labels_
true_labels = data['TrueLabel']
ari = adjusted_rand_score(true_labels, labels)
print(f"Adjusted Rand Index: {ari}")

result = pd.DataFrame({'Name': names, 'Cluster': labels})  
print(result)
