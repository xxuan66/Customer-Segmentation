import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. 读取数据集
file_path = 'Online Retail.xlsx'
data = pd.read_excel(file_path)

# 2. 数据清理
# 去除缺失的 CustomerID 行
data_cleaned = data.dropna(subset=['CustomerID'])

# 将 InvoiceNo 转为字符串类型
data_cleaned['InvoiceNo'] = data_cleaned['InvoiceNo'].astype(str)

# 删除退货订单及交易金额小于 0 的记录
data_cleaned = data_cleaned[~data_cleaned['InvoiceNo'].str.startswith('C')]
data_cleaned = data_cleaned[data_cleaned['Quantity'] > 0]
data_cleaned = data_cleaned[data_cleaned['UnitPrice'] > 0]

# 计算总消费金额
data_cleaned['TotalPrice'] = data_cleaned['Quantity'] * data_cleaned['UnitPrice']

# 3. 特征工程
customer_data = data_cleaned.groupby('CustomerID').agg(
    TotalQuantity=('Quantity', 'sum'),
    TotalPrice=('TotalPrice', 'sum'),
    MaxQuantity=('Quantity', 'max'),
    MaxPrice=('UnitPrice', 'max'),
    AvgQuantity=('Quantity', 'mean'),
    AvgPrice=('UnitPrice', 'mean')
).reset_index()

# 4. 数据标准化
scaler = StandardScaler()
scaled_customer_data = scaler.fit_transform(customer_data[['TotalQuantity', 'TotalPrice', 'MaxQuantity', 'MaxPrice', 'AvgQuantity', 'AvgPrice']])

# 5. 使用肘部法确定最佳 K 值
inertia = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_customer_data)
    inertia.append(kmeans.inertia_)

# 绘制肘部法图形
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.show()

# 6. K-Means 聚类分析（选择 K=3）
kmeans = KMeans(n_clusters=3, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(scaled_customer_data)

# 7. 查看每个群组的平均特征
cluster_summary = customer_data.groupby('Cluster').mean()
print(cluster_summary)

# 8. 可视化聚类结果
plt.figure(figsize=(8, 5))
plt.scatter(scaled_customer_data[:, 0], scaled_customer_data[:, 1], c=customer_data['Cluster'], cmap='viridis')
plt.title('Customer Segmentation with K-Means Clustering')
plt.xlabel('TotalQuantity (scaled)')
plt.ylabel('TotalPrice (scaled)')
plt.show()
