import random
import argparse
import time
import numpy as np
import scipy
import faulthandler
import multiprocessing
import warnings
from networkit import graphtools, readGraph, Format
from coarsening import EmbeddingCoarsening
from refinement import Refinement

T = 0
warnings.filterwarnings("ignore")
random.seed(int(time.perf_counter()))
np.random.seed(int(time.perf_counter()))
faulthandler.enable()

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-g", type=str, default="None", help="graph file")
parser.add_argument("-sp", type=int, default=18, help="size of subproblems")
parser.add_argument("-S", type=str, default="mqlib", help="subproblem solver")
parser.add_argument("-f", type=str, default="elist", help="graph format")
parser.add_argument("-e", type=str, default='cube', help='shape of embedding')
parser.add_argument("-c", type=int, default=0, help='coarse only')
parser.add_argument("-sparse", type=float, default=0, help='ratio to sparsify')
args = parser.parse_args()
sptime = 0
flag = True

print('Completed')


# run refinement in parallel
def parallel(ref):
    s = int(time.perf_counter() * ref[2])
    random.seed(s)
    np.random.seed(s)
    R = Refinement(ref[0], args.sp, args.S, ref[1])
    R
    return R.solution, R.obj


class MaxcutSolver:
    def __init__(self, fname, sp, solver, ratio):
        self.problem_graph = readGraph("./graphs/" + fname, Format.EdgeListSpaceOne)
        self.hierarchy = []
        self.hierarchy_map = []
        self.spsize = sp
        self.solver = solver
        self.solution = None
        self.obj = 0
        self.start = time.perf_counter()
        self.ratio = ratio
        random.seed(int(time.perf_counter()))
        np.random.seed(int(time.perf_counter()))

    # randomly perturb a solution
    def noisySolution(self, ratio):
        S = self.solution.copy()
        for i in range(int(len(S) * ratio)):
            k = random.randint(0, len(S) - 1)
            S[k] = 1 - S[k]
        return S

    # solves the maxcut problem using multilevel methods
    def solve(self):
        global sptime
        G = graphtools.toWeighted(self.problem_graph)
        print(G)
        s = time.perf_counter()
        # coarsen the graph down to subproblem size
        # counter = 0
        s_0 = time.perf_counter()
        while G.numberOfNodes() > self.spsize:
            E = EmbeddingCoarsening(G, 3, self.ratio)
            E.coarsen()
            print(E.cG)
            self.hierarchy.append(E)
            G = E.cG
        t = time.perf_counter()
        print(t - s, 'sec coarsening')
        self.hierarchy.reverse()

        # refine the coarsest level
        R = Refinement(G, self.spsize, 'mqlib', [random.randint(0, 1) for _ in range(G.numberOfNodes())])
        self.coarse_obj = R.refine_coarse()
        self.obj = R.obj
        self.solution = R.solution

        # iterate through and refine every other level
        # modify to use qaoa solver at the last refinement only
        print("Number of hierarchy level: ", len(self.hierarchy))
        self.solver = 'qaoa'  # QAOA
        for i in range(len(self.hierarchy)):
            E = self.hierarchy[i]
            if i != len(self.hierarchy) - 1:
                G = E.G
                self.solver = 'qaoa'
            else:
                G = self.problem_graph
                #self.solver = 'mqlib'
            fineToCoarse = E.mapFineToCoarse
            print('Level', i + 1, 'Nodes:', G.numberOfNodes(), 'Edges:', G.numberOfEdges())
            S = [0.0 for _ in range(G.numberOfNodes())]
            for j in range(len(S)):
                S[j] = float(self.solution[fineToCoarse[j]])
            self.solution = S
            sptime -= time.perf_counter()
            R = Refinement(G, self.spsize, self.solver, self.solution)
            R.refineLevel()
            sptime += time.perf_counter()
            self.solution = R.solution
            self.obj = R.obj
            print('Objective:', self.obj)
        t_0 = time.perf_counter()
        dur = t_0 - s_0
        objective = R.calc_obj(self.problem_graph, self.solution)
        print("Final objective: ", objective)
        print("Final self objective:", self.obj)
        # print("mqlibsolve is solving ...")
        # s = time.perf_counter()
        # mqsol, _ = R.mqlibSolve(t=sptime,G=self.problem_graph)
        # t = time.perf_counter()
        # print(f"mqlibsolve in {t-s} s")
        # mqobj = R.calc_obj(self.problem_graph, mqsol)
        # print('mqlib ratio:',self.obj / mqobj)
        return (objective, dur)
