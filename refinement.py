import time
import random
import numpy as np
import MQLib as mq
import networkx as nx
import pickle
from sortedcontainers import SortedKeyList
from networkit.graph import Graph
from networkit import graphtools, nxadapter, readGraph, Format
from QIRO_MC import QIRO_MC
import Generating_Problems as Generator
from weighted_model.sg_embedding import *
from classical_solver import find_maxcut
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from scipy.optimize import minimize
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
service = QiskitRuntimeService()

def matrices_to_graphs(matrix_list):
    g_list = []
    for matrix in matrix_list:
        array = np.array(matrix)
        g = from_numpy_array(array)
        g_list.append(g)
    return g_list

with open(r"weighted_model/Complete_graphs_ver3/g_list_updated.pkl", "rb") as input_file:
    g_list = pickle.load(input_file)
weighted_spectral = SGEmbedding(embedding_dimension=1)
embedding_model = []
for i in range(len(g_list)):
    embedding_model.append((weighted_spectral.fit(g_list[i])).embedding_)
print('Completed')

def find_min_index(vector):
    length = len(vector)
    sorted_vector = sorted(vector)
    min_value = sorted_vector[0]
    for i in range(len(vector)):
        if vector[i] == min_value:
            min_index = vector.index(vector[i])
    return min_index

def weighted_distance(v_1, v_2):
    '''
    This function calculates the Euclidean distance between two (graph) embedded vectors.
    '''
    
    diffs_1 = []
    diffs_2 = []  
    for i in range(len(v_1)):
        
        diffs_1.append(v_1[i] - v_2[i])
        diffs_2.append(v_1[i] + v_2[i])  
    return min(np.linalg.norm(diffs_1), np.linalg.norm(diffs_2))

print('Loading params...')
gamma_params_file = open('weighted_model/Complete_graphs_ver3/optimal_gammas.txt')
gamma_params = np.loadtxt(gamma_params_file).reshape(5087, 3)#(4047, 3) #(3162, 3)

beta_params_file = open('weighted_model/Complete_graphs_ver3/optimal_betas.txt')
beta_params = np.loadtxt(beta_params_file).reshape(5087, 3)#(4047, 3) #(3162, 3)
print('Completed')

class Refinement:
    def __init__(self, G, spsize, solver, solution):
        self.G = G
        self.n = G.numberOfNodes()
        self.gainmap = [0 for _ in range(self.n)]
        self.passes = 0
        self.spsize = spsize
        self.solver = solver
        self.solution = solution
        self.buildGain()
        # print("Solution: ", solution)
        self.obj = self.calc_obj(G, solution)
        self.last_subprob = None
        self.unused = SortedKeyList([i for i in range(self.n)])
        self.locked_nodes = set()
        self.alpha = 0.2
        self.randomness = 1.5
        self.bound = 3
        self.increase = -1
        self.done = False

    # refines the coarsest level with MQLib
    def refine_coarse(self):
        self.solution, obj = self.mqlibSolve(5, G=self.G)
        self.obj = self.calc_obj(self.G, self.solution)
        return self.obj

    # compute the maxcut objective at this level
    def calc_obj(self, G, solution):
        obj = 0
        n = G.numberOfNodes()
        for u, v in G.iterEdges():
            obj += G.weight(u, v) * (
                    2 * float(solution[u]) * float(solution[v]) - float(solution[u]) - float(solution[v]))
        return -1 * obj

    # solve a maxcut instance using MQLib for a set running time
    def mqlibSolve(self, t=0.1, G=None):
        start = time.perf_counter()
        if G == None:
            G = self.G
            n = self.G.numberOfNodes()
        else:
            n = G.numberOfNodes()
        print(G)
        Q = np.zeros((n, n))
        for u, v, w in G.iterEdgesWeights():
            Q[u][u] -= w
            Q[v][v] -= w
            if u < v:
                Q[u][v] = 2 * w
            else:
                Q[v][u] = 2 * w
        Q = Q / 2
        i = mq.Instance('M', Q)

        def f(s):
            return 1

        res = mq.runHeuristic("BURER2002", i, t, f, 100)

        terminate = time.perf_counter()
        print(terminate - start, 'seconds solving using mqlib')
        objective = res['objval']
        print(f'With objective {objective}')
        return (res['solution'] + 1) / 2, res['objval']

    # solve a maxcut instance using qaoa
    def qaoa(self, p=3, G=None, solver = "QIRO"):
        global service
        s = time.perf_counter()
        n = G.numberOfNodes()
        G = nxadapter.nk2nx(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        if solver == "QIRO": 
            problem = Generator.MaxCut(G)
            qiro = QIRO_MC(problem, nc=10, strategy = "Max", no_correlation = 1,temperature = 8)
            qiro.execute()
            i = 0
            res = {}
            for x in qiro.solution:
                if x < 0: 
                    res[i] = 1
                if x > 0: 
                    res[i] = 0
                i += 1
            t = time.perf_counter()
            print(t - s, 'seconds solving using qaoa using RQAOA-inspired QIRO')
            return res
        
        elif solver == "graph-learning":
            test_graph_vector = weighted_spectral.fit(G).embedding_
            dists = []
            for i in range(len(embedding_model)):
                dist = weighted_distance(embedding_model[i], test_graph_vector)
                dists.append(dist)
            index = find_min_index(dists)
            gamma = gamma_params[index]
            beta = beta_params[index]
            print("Gamma: ", gamma)
            print("Beta: ", beta)
            num_layer = 3
            qc = QuantumCircuit(n)
            # initial_state
            qc.h(range(n))
            for layer_index in range(num_layer):
                # problem unitary
                for pair in list(G.edges()):
                    qc.cx(pair[0], pair[1])
                    qc.rz(gamma[layer_index] * G[pair[0]][pair[1]]['weight'], pair[1])
                    qc.cx(pair[0], pair[1])
                # mixer unitary
                for qubit in range(n):
                    qc.rx(2 * beta[layer_index], qubit)
            qc.measure_all()
            print("Actual depth of circuit: ", qc.depth())
            # Perform simulation and choose the best solution
            backend = service.backend('ibmq_qasm_simulator')
        # ### Noisy simulation
        # backend = FakeHanoiV2()
        # transpiled_circuit = transpile(qc, backend)
        # job = backend.run(transpiled_circuit)
        # counts = job.result().get_counts()
        ###
            pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
            isa_circuit = pm.run(qc)

            sampler = Sampler(backend=backend)
            job = sampler.run([(isa_circuit,)], shots=1024 * 10)
            job_result = job.result()
            pub_result = job.result()[0]
            counts = pub_result.data.meas.get_counts()
            ## Best objective
            sub_problem_obj = 0
            current_solution = ""
            G = nxadapter.nx2nk(G)
            for key in counts.keys():
                i= 0
                res = {}
                for x in key:
                    res[i] = x
                    i += 1
                if self.calc_obj(G, res) > sub_problem_obj:
                    current_solution = key
                    sub_problem_obj = self.calc_obj(G, res)
            L = current_solution
            ## 
            i = 0
            res = {}
            for x in L:
                res[i] = x
                i += 1
            t = time.perf_counter()
            print(t - s, 'seconds solving using graph-learning qaoa')
            return res

    # compute the gain for each node
    def buildGain(self):
        for u, v, w in self.G.iterEdgesWeights():
            if float(self.solution[u]) == float(self.solution[v]):
                self.gainmap[u] += w
                self.gainmap[v] += w
            else:
                self.gainmap[u] -= w
                self.gainmap[v] -= w
        self.gainlist = SortedKeyList([i for i in range(self.n)], key=lambda x: self.gainmap[x] + 0.01 * x)

    # update the gain after changing the solution
    def updateGain(self, S, changed):
        for u in changed:
            for v in self.G.iterNeighbors(u):
                if v not in changed:
                    w = 2 * self.G.weight(u, v) * (1 + self.alpha)
                    if S[u] == S[v]:
                        self.gainmap[v] += w
                    else:
                        self.gainmap[v] -= w
                        
    # construct a subproblem using a random subset and choosing by highest gain
    def randGainSubProb(self, count):
        if self.n >= 2 * self.spsize:
            sample_size = max(int(0.3 * self.n), 2 * self.spsize)
        else:
            sample_size = self.n
        if count != 0:
            sample = random.sample(range(self.n), sample_size)
            nodes = [i for i in sample]
            nodes.sort(reverse=True, key=lambda x: self.gainmap[x])
        else:
            nodes = list(range(self.n))
            nodes.sort(reverse=True, key=lambda x: self.gainmap[x])
        spnodes = nodes[:self.spsize]

        subprob = Graph(n=len(spnodes) + 2, weighted=True, directed=False)
        mapProbToSubProb = {}
        i = 0
        idx = 0
        change = set()
        while i < len(spnodes):
            u = spnodes[i]
            change.add(u)
            mapProbToSubProb[u] = idx
            idx += 1
            i += 1
        self.last_subprob = spnodes

        keys = mapProbToSubProb.keys()
        j = 0
        while j < len(spnodes):
            u = spnodes[j]
            spu = mapProbToSubProb[u]
            for v in self.G.iterNeighbors(u):
                w = self.G.weight(u, v)
                if v not in keys:
                    if float(self.solution[v]) == 0:
                        spv = idx
                    else:
                        spv = idx + 1
                    subprob.increaseWeight(spu, spv, w)
                else:
                    spv = mapProbToSubProb[v]
                    if u < v:
                        subprob.increaseWeight(spu, spv, w)
            j += 1
        total = subprob.totalEdgeWeight()
        subprob.increaseWeight(idx, idx + 1, self.G.totalEdgeWeight() - total)
        # print("Map problem to subproblem: ", mapProbToSubProb)
        return (subprob, mapProbToSubProb, idx)

    # refine the current level
    def refine(self):
        count = 0
        total_count = 0
        print(f"Start refinement")
        while count < 3 and total_count < 10:
            subprob = self.randGainSubProb(count)
            mapProbToSubProb = subprob[1]
            if self.solver == 'qaoa':
                print("Start running QAOA")
                n_nodes = subprob[0].numberOfNodes()
                nx_G = nxadapter.nk2nx(subprob[0])
                S = self.qaoa(p=3, G=subprob[0], solver="graph-learning")
            else:
                S, new_obj = self.mqlibSolve(G=subprob[0])
                #print(f"MQLIB solution: {S, new_obj}")
            new_sol = self.solution.copy()
            total_count += 1
            keys = mapProbToSubProb.keys()
            for i in keys:
                new_sol[i] = S[mapProbToSubProb[i]]
            changed = set()
            for u in self.last_subprob:
                if float(self.solution[u]) != float(new_sol[u]):
                    changed.add(u)
            # print("Solution is changed ! ---")
            new_obj = self.obj
            for u in changed:
                for v in self.G.iterNeighbors(u):
                    if v not in changed:
                        w = self.G.weight(u, v)
                        if float(new_sol[u]) == float(new_sol[v]):
                            new_obj -= w
                        else:
                            new_obj += w
            # print("New solution: ", new_obj)
            count += 1
            if new_obj >= self.obj:
                self.updateGain(new_sol, changed)
                self.solution = new_sol.copy()
                if new_obj > self.obj:
                    count = 0
                    self.obj = new_obj

    def refineLevel(self):
        self.refine()