# deepwalk-quickstart
DeepWalk算法是一种用于学习图（如社交网络、推荐系统网络等）中节点的向量表示的算法。这种向量表示能够捕捉和表达节点间的关系和结构特征。下面我将一步一步地解释DeepWalk算法的基本概念和工作原理。

### 1. 图的概念

在开始之前，我们需要理解图的基本概念。图由节点（也称为顶点）和边组成，节点代表实体，边代表实体间的关系。例如，在社交网络中，节点可以是人，边可以表示人与人之间的友谊关系。

### 2. 随机游走（Random Walk）

DeepWalk算法的第一步是在图中进行随机游走。随机游走是指从图中的一个节点开始，随机选择一个邻接的节点移动到那里，然后重复这个过程多次，形成一个节点序列。这个过程类似于在图中随机“漫步”，每一步都是随机选择的[1][2][3][4].

### 3. 使用Word2Vec模型

获得足够数量的随机游走序列后，DeepWalk使用Word2Vec模型来学习节点的向量表示。Word2Vec是一种常用于自然语言处理的模型，它可以将文本中的单词转换为向量形式。在DeepWalk中，我们可以将每个节点看作是一个“单词”，将整个游走序列看作是一个“句子”[1][2][3].

### 4. Skip-Gram模型

DeepWalk具体使用的是Word2Vec的一种变体，称为Skip-Gram模型。Skip-Gram模型的目标是通过一个节点预测其在游走序列中的上下文节点。例如，如果我们的游走序列是[A, B, C, D, E]，并且我们正在考虑节点C，Skip-Gram模型将尝试使用节点C的向量来预测节点B和D（假设上下文窗口大小为1）[1][2][3].

### 5. 训练过程

在训练过程中，模型通过调整向量来最大化预测上下文节点的概率。这意味着模型会学习到哪些节点经常一起出现在游走序列中，这些节点在向量空间中应该彼此接近。通过这种方式，模型能够捕捉节点间的关系和网络的结构特性[1][2][3].

### 6. 向量表示的应用

一旦训练完成，每个节点都会有一个向量表示。这些向量可以用于多种应用，如节点分类、链接预测或者节点聚类。例如，如果两个节点在向量空间中彼此非常接近，我们可以推断它们在图中可能属于同一个社区[1][2][3].

### 总结

DeepWalk算法通过结合图的随机游走和自然语言处理中的Word2Vec技术，有效地学习图中节点的向量表示。这些表示捕捉了节点间的关系和网络的整体结构，对于理解和分析复杂网络非常有用。

Citations:
[1] https://yunlongs.cn/2019/04/26/NE-Deepwalk/
[2] https://juejin.cn/post/7197404867802972218
[3] https://www.cnblogs.com/Lee-yl/p/12670515.html
[4] https://zjt-blog.readthedocs.io/zh/latest/embeddings/DeepWalk.html
[5] https://mumaxu.github.io/2019/04/26/DeepWalk/
[6] https://blog.csdn.net/google19890102/article/details/79756289
[7] https://blog.csdn.net/gsq0854/article/details/117587606
[8] https://developer.aliyun.com/article/1368092


```
import networkx as nx
import numpy as np
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

# 创建一个图
G = nx.karate_club_graph()

# 模拟随机游走
def random_walk(G, node, walk_length):
    walk = [node]  # 保持节点的原始类型
    for _ in range(walk_length - 1):
        neighbors = list(G.neighbors(walk[-1]))
        if len(neighbors) > 0:
            walk.append(np.random.choice(neighbors))  # 直接添加邻居节点
    return walk

# 生成多个随机游走
walks = []
num_walks = 10
walk_length = 10
for node in G.nodes():
    for _ in range(num_walks):
        walks.append(random_walk(G, node, walk_length))

# 将节点转换为字符串，因为Word2Vec期望字符串输入
walks_str = [[str(node) for node in walk] for walk in walks]

# 使用Word2Vec学习嵌入
model = Word2Vec(walks_str, vector_size=2, window=5, min_count=0, sg=1, workers=1, epochs=10)

# 可视化嵌入
node_embeddings = np.array([model.wv[str(node)] for node in G.nodes()])  # 使用字符串形式的节点标识符
plt.figure(figsize=(8, 8))
pos = {node: node_embeddings[index] for index, node in enumerate(G.nodes())}
nx.draw(G, pos=pos, with_labels=True, node_color='lightblue')
plt.title('Node Embeddings Visualized')
plt.show()

```
