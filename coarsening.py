import math
import random
import numpy as np
from sklearn.neighbors import KDTree
from networkit.graph import Graph
from networkit import graphtools, nxadapter, readGraph, Format, Graph


class EmbeddingCoarsening:
    # G: graph to sparsify
    # d: dimension of the embedding
    # ratio: ratio of edges to sparsifyß
    def __init__(self, G, d, ratio):
        self.G = G
        self.sG = graphtools.toWeighted(G)
        self.d = d
        self.n = G.numberOfNodes()
        self.space = np.random.rand(self.n, d)
        self.M = set()
        self.R = -1
        self.ratio = ratio

    # sparsify graph
    def sparsify(self):
        # return if no edges should be sparsified
        if self.ratio == 0:
            return
        # compute number of edges to remove
        removeCount = int(self.ratio * self.sG.numberOfEdges())

        # compute length of each edge in embedding
        edgeDist = []
        edgeMap = {}
        for u, v in self.sG.iterEdges():
            w = self.sG.weight(u, v)
            d = 0
            for i in range(self.d):
                d += (self.space[u][i] - self.space[v][i]) ** 2
            d = w * np.sqrt(d)
            edgeDist.append((d, u, v))
            edgeMap[(u, v)] = d
            edgeMap[(v, u)] = d

        # sort edges by their distance
        edgeDist.sort()

        # remove shortest edges
        for i in range(removeCount):
            # get endpoints of edge to remove
            u = edgeDist[i][1]
            v = edgeDist[i][2]

            # find minimum weight adjacent edge
            minE_u = None
            minE_v = None
            for x in self.sG.iterNeighbors(u):
                if v != x:
                    if minE_u == None or edgeMap[(u, x)] < edgeMap[minE_u]:
                        minE_u = (u, x)
            for x in self.sG.iterNeighbors(v):
                if u != x:
                    if minE_v == None or edgeMap[(v, x)] < edgeMap[minE_v]:
                        minE_v = (v, x)

            # reweight edges to preserve weights
            w = self.sG.weight(u, v)
            if minE_u != None and (minE_v == None or edgeMap[minE_u] < edgeMap[minE_v]):
                u1 = minE_u[0]
                u2 = minE_u[1]
                if self.sG.weight(u1, u2) != 0:
                    self.sG.increaseWeight(u1, u2, w)
            elif minE_v != None and (minE_u == None or edgeMap[minE_v] < edgeMap[minE_u]):
                v1 = minE_v[0]
                v2 = minE_v[1]
                if self.sG.weight(v1, v2) != 0:
                    self.sG.increaseWeight(v1, v2, w)
            self.sG.removeEdge(u, v)

    # compute distance objective for a node in the embedding
    def nodeObj(self, p, c):
        obj = 0
        for x in c:
            for i in range(self.d):
                obj += x[self.d] * (p[i] - x[i]) ** 2
        return obj

    # get a random point within the embedding
    def randPoint(self):
        theta = random.uniform(0, 2 * math.pi)
        phi = random.uniform(0, math.pi)
        x = math.cos(theta) * math.sin(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(phi)
        return [x, y, z]

    # compute the optimal position of a node in the embedding
    def optimal(self, u):
        k = 2 * self.sG.weightedDegree(u)
        a = 1
        b = -2 * k
        c = k ** 2
        X = self.space[u]
        temp = [0 for _ in range(self.d)]
        for v in self.sG.iterNeighbors(u):
            w = self.sG.weight(u, v)
            for i in range(self.d):
                temp[i] += 2 * w * self.space[v][i]
        for i in range(self.d):
            c -= temp[i] ** 2
        lambda1 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / 2 * a
        lambda2 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / 2 * a
        p = k - lambda1
        q = k - lambda2
        if p == 0 and q == 0:
            return X, 0
        if p != 0:
            p1 = [temp[i] / p for i in range(self.d)]
        if q != 0:
            p2 = [temp[i] / q for i in range(self.d)]
        if p == 0:
            return p2
        if q == 0:
            return p1
        p1d = 0
        p2d = 0
        for v in self.sG.iterNeighbors(u):
            t1 = 0
            t2 = 0
            x = self.space[v]
            for i in range(self.d):
                t1 += (p1[i] - x[i]) ** 2
                t2 += (p2[i] - x[i]) ** 2
            p1d += np.sqrt(t1)
            p2d += np.sqrt(t2)
        if p1d > p2d:
            d = 0
            for i in range(self.d):
                d += (p1[i] - X[i]) ** 2
            return p1, np.sqrt(d)
        if p2d > p1d:
            d = 0
            for i in range(self.d):
                d += (p2[i] - X[i]) ** 2
            return p2, np.sqrt(d)

    # evaluate the objective of optimizing the embedding
    def coarseObj(self):
        o = 0
        for u, v, w in self.sG.iterEdgesWeights():
            for i in range(self.d):
                o += w * (self.space[u][i] - self.space[v][i]) ** 2
        print('Current Obj (to be minimized):', o)

    # iterate through nodes and optimize the location in embedding to maximize distance between neighbors
    def embed(self, nodes):
        n = self.sG.numberOfNodes()
        change = 0
        for i in nodes:
            res, c = self.optimal(i)
            self.space[i] = res
            change += c
        return change / n

    # match each vertex with the nearest neighbor in the embedding greedily
    def match(self):
        n = self.sG.numberOfNodes()
        tree = KDTree(self.space)
        ind = tree.query_radius(self.space, 0)
        used = set()
        clusters = []
        singletons = []
        t = 0
        for x in ind:
            if x[0] in used:
                continue
            elif len(x) == 1:
                singletons.append(x[0])
            else:
                clusters.append(x)
                for y in x:
                    used.add(y)
                t += len(x)
        used = set()
        for c in clusters:
            k = len(c)
            if k % 2 == 1:
                singletons.append(c[k - 1])
            for i in range(int(k / 2)):
                self.M.add((c[2 * i], c[2 * i + 1]))
                used.add(c[2 * i])
                used.add(c[2 * i + 1])
        indices = []
        newspace = []
        k = len(singletons)
        if k % 2 == 1:
            k = k - 1
            self.R = singletons[k]
            used.add(self.R)
        if k == 0:
            return
        for i in range(k):
            x = singletons[i]
            indices.append(x)
            newspace.append(self.space[x])
        newspace = np.array(newspace)
        tree = KDTree(newspace)
        ind = tree.query(newspace, k=min(40, k), return_distance=False)

        unused = []
        for i in range(len(ind)):
            idx = indices[i]
            ct = 0
            if idx not in used:
                for j in ind[i]:
                    jdx = indices[j]
                    if jdx not in used and idx != jdx and (ct >= 10 or not self.sG.hasEdge(idx, jdx)):
                        self.M.add((idx, jdx))
                        used.add(idx)
                        used.add(jdx)
                        break
                    ct += 1
        for i in range(n):
            if i not in used:
                unused.append(i)
        m = len(unused)
        for i in range(int(m / 2)):
            self.M.add((unused[2 * i], unused[2 * i + 1]))

    # construct coarse graph from a matching of nodes
    def coarsen(self):
        n = self.sG.numberOfNodes()
        i = 0
        j = int(n / 2)
        self.mapCoarseToFine = {}
        self.mapFineToCoarse = {}
        idx = 0
        count = 1
        nodes = [i for i in range(n)]
        random.shuffle(nodes)
        change = self.embed(nodes)
        while change > 0.01 and count < 31:
            change = self.embed(nodes)
            count += 1
        print(count, 'iterations until embedding convergence')
        self.sparsify()
        self.match()
        for u, v in self.M:
            self.mapCoarseToFine[idx] = [u, v]
            self.mapFineToCoarse[u] = idx
            self.mapFineToCoarse[v] = idx
            idx += 1
        if n % 2 == 1:
            self.mapCoarseToFine[idx] = [self.R]
            self.mapFineToCoarse[self.R] = idx
            idx += 1
        self.cG = Graph(n=idx, weighted=True, directed=False)
        for u, v in self.sG.iterEdges():
            cu = self.mapFineToCoarse[u]
            cv = self.mapFineToCoarse[v]
            self.cG.increaseWeight(cu, cv, self.G.weight(u, v))
        self.cG.removeSelfLoops()
        self.cG.indexEdges()
