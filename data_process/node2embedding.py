from node2vec import Node2Vec
import networkx as nx
import pandas as pd
from data_process.util import dataset_name,dataset_path

def node2emb(graph_pd,embedding_size,embedding_save_file):
    # graph_pd=pd.read_csv('../data/hf/graph/graph.csv')
    # 定义图的节点和边
    nodes=set(graph_pd['code1'].unique())|set(graph_pd['code2'].unique())
    edges=[tuple(x) for x in graph_pd.values]

    G_Di = nx.DiGraph()
    G_Di.add_nodes_from(nodes)
    G_Di.add_weighted_edges_from(edges)

    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    node2vec = Node2Vec(G_Di, dimensions=embedding_size, walk_length=30, num_walks=200, workers=4,p=1.0,q=1.0)  # Use temp_folder for big graphs

    print("start")
    # Embed nodes
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # Save embeddings for later use
    model.wv.save_word2vec_format(embedding_save_file)
if __name__ == '__main__':
    graph_pd = pd.read_csv(dataset_path+'{}/graph/graph.csv'.format(dataset_name))
    embedding_size=96
    print(embedding_size)
    embedding_save_flie=dataset_path+'{}/input/node_embedding_'.format(dataset_name)+str(embedding_size)
    node2emb(graph_pd, embedding_size, embedding_save_flie)