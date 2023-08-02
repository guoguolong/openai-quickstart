import pandas as pd
import secret;
import ast
import openai
openai.api_key = secret.OPENAI_API_KEY

# 导入 NumPy 包，NumPy 是 Python 的一个开源数值计算扩展。这种工具可用来存储和处理大型矩阵，
# 比 Python 自身的嵌套列表（nested list structure)结构要高效的多。
import numpy as np
# 从 matplotlib 包中导入 pyplot 子库，并将其别名设置为 plt。
# matplotlib 是一个 Python 的 2D 绘图库，pyplot 是其子库，提供了一种类似 MATLAB 的绘图框架。
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
# 从 sklearn.manifold 模块中导入 TSNE 类。
# TSNE (t-Distributed Stochastic Neighbor Embedding) 是一种用于数据可视化的降维方法，尤其擅长处理高维数据的可视化。
# 它可以将高维度的数据映射到 2D 或 3D 的空间中，以便我们可以直观地观察和理解数据的结构。
from sklearn.manifold import TSNE


EMBEDDING_DATAPATH = "../openai_api/data/fine_food_reviews_with_embeddings_1k.csv"
# EMBEDDING_DATAPATH = "./data/target.csv"
df_embedded = pd.read_csv(EMBEDDING_DATAPATH, index_col=0)
df_embedded["embedding_vec"] = df_embedded["embedding"].apply(ast.literal_eval)

assert df_embedded['embedding_vec'].apply(len).nunique() == 1

# # 将嵌入向量列表转换为二维 numpy 数组
matrix = np.vstack(df_embedded['embedding_vec'].values)
# # 创建一个 t-SNE 模型，t-SNE 是一种非线性降维方法，常用于高维数据的可视化。
# # n_components 表示降维后的维度（在这里是2D）
# # perplexity 可以被理解为近邻的数量
# # random_state 是随机数生成器的种子
# # init 设置初始化方式
# # learning_rate 是学习率。
# tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
# vis_dims = tsne.fit_transform(matrix) # 使用 t-SNE 对数据进行降维，得到每个数据点在新的2D空间中的坐标
# colors = ["red", "darkorange", "gold", "turquoise", "darkgreen"] # 定义了五种不同的颜色，用于在可视化中表示不同的等级
# # 从降维后的坐标中分别获取所有数据点的横坐标和纵坐标
# x = [x for x,y in vis_dims]
# y = [y for x,y in vis_dims]

# # 根据数据点的评分（减1是因为评分是从1开始的，而颜色索引是从0开始的）获取对应的颜色索引
# color_indices = df_embedded.Score.values - 1

# # 确保你的数据点和颜色索引的数量匹配
# assert len(vis_dims) == len(df_embedded.Score.values)

# # 创建一个基于预定义颜色的颜色映射对象
# colormap = matplotlib.colors.ListedColormap(colors)
# # 使用 matplotlib 创建散点图，其中颜色由颜色映射对象和颜色索引共同决定，alpha 是点的透明度
# plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)

# # 为图形添加标题
# plt.title("Amazon ratings visualized in language using t-SNE")
# plt.show()


n_clusters = 4
kmeans = KMeans(n_clusters = n_clusters, init='k-means++', random_state=42)
kmeans.fit(matrix)
df_embedded['Cluster'] = kmeans.labels_
# 首先为每个聚类定义一个颜色。
colors = ["red", "green", "blue", "purple"]

# 然后，你可以使用 t-SNE 来降维数据。这里，我们只考虑 'embedding_vec' 列。
tsne_model = TSNE(n_components=2, random_state=42)
vis_data = tsne_model.fit_transform(matrix)

# 现在，你可以从降维后的数据中获取 x 和 y 坐标。
x = vis_data[:, 0]
y = vis_data[:, 1]

# 'Cluster' 列中的值将被用作颜色索引。
color_indices = df_embedded['Cluster'].values

# 创建一个基于预定义颜色的颜色映射对象
colormap = matplotlib.colors.ListedColormap(colors)

# 使用 matplotlib 创建散点图，其中颜色由颜色映射对象和颜色索引共同决定
plt.scatter(x, y, c=color_indices, cmap=colormap)

# 为图形添加标题
plt.title("Clustering visualized in 2D using t-SNE")

# 显示图形
plt.show()