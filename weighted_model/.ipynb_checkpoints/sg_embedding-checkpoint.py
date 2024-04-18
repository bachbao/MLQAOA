import numpy as np
import networkx as nx
from numpy import errstate, sqrt, isinf
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

class SGEmbedding:

    def __init__(self, embedding_dimension=2, eigenvalue_normalization=True):
        self.embedding_dimension = embedding_dimension
        self.eigenvalue_normalization = eigenvalue_normalization
    
    def fit(self, graph):

        adjacency_matrix = nx.to_scipy_sparse_array(graph)

        if type(adjacency_matrix) == sparse.csr_array:
            adj_matrix = adjacency_matrix
        elif sparse.isspmatrix(adjacency_matrix) or type(adjacency_matrix) == np.ndarray:
            adj_matrix = sparse.csr_matrix(adjacency_matrix)
        else:
            raise TypeError(
                'The argument must be a Numpy array or a Scipy Sparse matrix.'
            )
        n_nodes, m_nodes = adj_matrix.shape
        if n_nodes != m_nodes:
            raise ValueError('Adjacency matrix must be a square matrix.')
        if csgraph.connected_components(adj_matrix, directed=False)[0] > 1:
            raise ValueError('The graph must be connected.')
        
        degrees = adj_matrix.dot(np.ones(n_nodes))
        degree_matrix = sparse.diags(degrees, format='csr')
        laplacian = degree_matrix - adj_matrix

        with errstate(divide='ignore'):
            degrees_inv_sqrt = 1.0 / sqrt(degrees)
        degrees_inv_sqrt[isinf(degrees_inv_sqrt)] = 0
        weight_matrix = sparse.diags(degrees_inv_sqrt, format='csr')

        laplacian = weight_matrix.dot(laplacian.dot(weight_matrix))

        eigenvalues, eigenvectors = eigsh(laplacian, min(self.embedding_dimension + 1, n_nodes - 1), which='SM')
        self.eigenvalues_ = eigenvalues[1:]

        graph_vector = []

        for i in range(self.embedding_dimension):
            for j in range(n_nodes):
                graph_vector.append(eigenvectors[j][i])
        
        self.embedding_ = graph_vector

        return self

