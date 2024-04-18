import numpy as np
import itertools as it
import copy


class MAX2SAT:
    """Max 2-SAT problem generator."""
    def __init__(self, num_var, num_clauses, seed=42):
        self.num_var = num_var
        self.num_clauses = num_clauses
        self.num_per_clause = 2
        self.var_list = []
        self.cnf = None
        self.cnf_start = None
        self.matrix = None
        self.position_translater = None
        self.random_seed = seed
        self.generate_formula()
        self.SAT_to_Hamiltonian()
        self.matrix_start = copy.deepcopy(self.matrix)

    def generate_formula(self):
        """This generates a random MAX-2-SAT formula."""
        variables = [i for i in range(1, self.num_var + 1)]
        self.cnf = []

        rg = np.random.default_rng(self.random_seed)

        for j in range(int(self.num_clauses)):
            var_in_clause = rg.choice(variables, self.num_per_clause, replace=True)
            neg_pos_in_clause = rg.choice([1, -1], self.num_per_clause, replace=True)
            for var in var_in_clause:
                if var not in self.var_list:
                    self.var_list.append(var)

            # check if the two variables in the clause are the same
            if var_in_clause[0] == var_in_clause[1]:
                # if variables have opposite polarity and the clause is therefore automatically fulfilled, it is not
                # added to the formula
                if neg_pos_in_clause[0] != neg_pos_in_clause[1]:
                    pass
                # the two variables have also the same polarity, therefore it is sufficient to add the variable only
                # once to the clause
                else:
                    clause = [var_in_clause[0] * neg_pos_in_clause[0]]
                    self.cnf.append(clause)
            # adding clause with two different variables ("standard case")
            else:
                clause = [var_in_clause[0] * neg_pos_in_clause[0], var_in_clause[1] * neg_pos_in_clause[1]]
                self.cnf.append(clause)

        # copy of cnf formula will be needed in the recursive algorithms
        self.cnf_start = copy.deepcopy(self.cnf)
        self.var_list.sort()
        self.remain_var_list = copy.deepcopy(self.var_list)

    def add_off_element(self, i, j, const):
        """Adds an off-diagonal element to the Hamiltonian matrix."""
        if np.abs(i) >= np.abs(j):
            self.matrix[np.abs(i), np.abs(j)] += np.sign(i) * np.sign(j) * const
        else:
            self.matrix[np.abs(j), np.abs(i)] += np.sign(i) * np.sign(j) * const

    def add_diag_element(self, i, const):
        """Adds a diagonal element to the Hamiltonian matrix."""
        self.matrix[np.abs(i), np.abs(i)] += -np.sign(i) * const

    def SAT_to_Hamiltonian(self):
        """Converts the MAX-2-SAT formula to a Hamiltonian matrix."""
        self.single_dict = {} # do we ever use this?
        self.coupling_dict = {} # or this?

        self.position_translater = {0}
        for clause in self.cnf:
            positive_clause = []
            for element in range(len(clause)):
                positive_clause.append(abs(clause[element]))
            self.position_translater = self.position_translater.union(set(positive_clause))
        self.position_translater = list(self.position_translater)
        self.position_translater.sort()

        self.matrix = np.zeros((len(self.position_translater), len(self.position_translater)))

        for clause in self.cnf:
            if len(clause) > 1:
                ind_0 = self.position_translater.index(np.abs(clause[0])) * np.sign(clause[0])
                ind_1 = self.position_translater.index(np.abs(clause[1])) * np.sign(clause[1])
                self.add_off_element(i=int(ind_0), j=int(ind_1), const=1 / 4)
                self.add_diag_element(i=int(ind_0), const=1 / 4)
                self.add_diag_element(i=int(ind_1), const=1 / 4)
            elif len(clause) == 1:
                ind_0 = self.position_translater.index(np.abs(clause[0])) * np.sign(clause[0])
                self.add_diag_element(i=int(ind_0), const=1 / 2)

    def calc_energy(self, assignment, single_energy_vector, correl_energy_matrix):
        E_calc = np.sign(assignment).T @ correl_energy_matrix @ np.sign(assignment) + single_energy_vector.T @ np.sign(assignment)
        return E_calc

    def calc_violated_clauses(self, solution):
        E = 0
        E_diff = 1
        for clause in self.cnf_start:
            if len(clause) != 1:
                for variable in solution:
                    if np.abs(clause[0]) == np.abs(variable):
                        if np.sign(clause[0]) == np.sign(variable):
                            E_diff = 0
                for variable in solution:
                    if np.abs(clause[1]) == np.abs(variable):
                        if np.sign(clause[1]) == np.sign(variable):
                            E_diff = 0
            else:
                for variable in solution:
                    if np.abs(clause[0]) == np.abs(variable):
                        if np.sign(clause[0]) == np.sign(variable):
                            E_diff = 0
            E = E + E_diff
            E_diff = 1
        return E


class MIS:
    """Maximum Independent Set problem generator. Have implement Penny Lane simulator here """
    def __init__(self, graph, alpha=1.1):
        """Init takes in a networkx graph object, and a penalty factor alpha."""
        self.graph = copy.deepcopy(graph)
        self.alpha = alpha
        self.matrix = None
        self.position_translater = None # index to graph's label
        self.var_list = None

        # compute the matrix (i.e., Hamiltonian) from the graph. Also sets the varlist!
        self.graph_to_matrix()
        self.remain_var_list = copy.deepcopy(self.var_list)

        self.num_qubits = None
        self.H_driver = None
        self.H_mixer = None
        self.pauli_words = []
        self.get_qml_hamiltonian()


    def get_qml_hamiltonian(self):
        self.num_qubits = self.graph.number_of_nodes()
        wire_map = dict(zip(self.graph.nodes, range(self.num_qubits)))
        inv_wire_map = dict(zip(range(self.num_qubits), self.graph.nodes))
        print("num qubits", self.num_qubits)
        print("wire map", wire_map)
        self.H_driver = 0
        self.H_mixer = 0
        for q in self.graph.nodes:
            term_str = 'I'*wire_map[q] + 'Z' + 'I'*(self.num_qubits-wire_map[q]-1)
            mixer_str = 'I'*wire_map[q] + 'X' + 'I'*(self.num_qubits-wire_map[q]-1)
            print("term, mixer: ", term_str, mixer_str)
            print((qml.pauli.string_to_pauli_word(term_str, wire_map)))
            self.pauli_words.append(qml.pauli.string_to_pauli_word(term_str, wire_map))
            self.H_driver += (-1./2+ (self.alpha/4.) * self.graph.degree[q]) * qml.pauli.string_to_pauli_word(term_str, wire_map)
            self.H_mixer += 1. * qml.pauli.string_to_pauli_word(mixer_str, wire_map)
        for (q1, q2) in self.graph.edges:
            term_str = list('I' * self.num_qubits)
            term_str[wire_map[q1]] = 'Z'
            term_str[wire_map[q2]] = 'Z'
            term_str = ''.join(term_str)
            print('edge: ', (q1,q2), term_str)
            print((qml.pauli.string_to_pauli_word(term_str, wire_map)))
            self.pauli_words.append(qml.pauli.string_to_pauli_word(term_str, wire_map))
            self.H_driver += (self.alpha/4.) *  qml.pauli.string_to_pauli_word(term_str, wire_map)

    def add_off_element(self, i, j, coeff):
        if np.abs(i) >= np.abs(j):
            self.matrix[np.abs(i), np.abs(j)] += coeff
        else:
            self.matrix[np.abs(j), np.abs(i)] += coeff

    def add_diag_element(self, i, coeff):
        self.matrix[np.abs(i), np.abs(i)] += coeff

    
    def graph_to_matrix(self):
        
        # matrix is one dimension larger due to the 0-th row and column, which are set to 0 by convention.
        self.matrix = np.zeros((self.graph.number_of_nodes() + 1, self.graph.number_of_nodes() + 1))


        # Transform graph nodes in ordered list of variables. These run from 1 -> n (instead of 0 -> n-1)
        variable_set = set()
        for node_shifted in self.graph.nodes:
            node = node_shifted + 1
            variable_set.add(node)
        variables = list(variable_set)
        variables.sort()
        self.var_list = copy.deepcopy(variables)

        # Filling the matrix (here the type of optimization problem is encoded, MIS in this case)
        # we skip the zeroth index, which is set to 0 by convention
        for variable in variables:
            idx = variables.index(variable) + 1
            self.add_diag_element(idx, -1/2)

        for correlation in self.graph.edges:
            # the first correlation + 1 comes from the fact that graph nodes run from 0...n-1
            # the fact that we add another +1 to the index is because the variables list runs 1....n 
            # and the indices in the matrix run 0...n
            idx1, idx2 = variables.index(correlation[0] + 1) + 1, variables.index(correlation[1] + 1) + 1
            self.add_off_element(idx1, idx2, self.alpha/4)
            self.add_diag_element(idx1, self.alpha/4)
            self.add_diag_element(idx2, self.alpha/4)

        # we define the appropriate position translater
        self.position_translater = [0] + variables

    def evaluate_solution(self, solution):
        """Returns the number of violations and the size of the set found."""
        number_of_violations = 0
        size_of_set = len(*np.where(np.array(solution) > 0.))
        for n1, n2 in self.graph.edges:
            if solution[n1] > 0. and solution[n2] > 0.:
                number_of_violations += 1
        
        return number_of_violations, size_of_set

    def calc_energy(self, assignment, single_energy_vector, correl_energy_matrix):
        E_calc = np.sign(assignment).T @ correl_energy_matrix @ np.sign(assignment) + single_energy_vector.T @ np.sign(assignment)
        return E_calc
    
        
class MaxCut:
    """Weighted MaxCut problem, with weight on the node. Even if the original problem has no weight on the node,
    due to update rule, we will eventually have weight on node."""
    def __init__(self, graph):
        """."""
        self.graph = copy.deepcopy(graph)
        self.matrix = None
        self.position_translater = None
        self.var_list = None

        # compute the matrix (i.e., Hamiltonian) from the graph. Also sets the varlist!
        self.graph_to_matrix()
        self.remain_var_list = copy.deepcopy(self.var_list)

    # This is the same as MIS, no modification needed 
    def add_off_element(self, i, j, coeff):
        if np.abs(i) >= np.abs(j):
            self.matrix[np.abs(i), np.abs(j)] += coeff
        else:
            self.matrix[np.abs(j), np.abs(i)] += coeff
    # This is the same as MIS, no modification needed
    def add_diag_element(self, i, coeff):
        self.matrix[np.abs(i), np.abs(i)] += coeff

    
    def graph_to_matrix(self):
        # matrix is one dimension larger due to the 0-th row and column, which are set to 0 by convention.
        self.matrix = np.zeros((self.graph.number_of_nodes() + 1, self.graph.number_of_nodes() + 1))

        # Transform graph nodes in ordered list of variables. These run from 1 -> n (instead of 0 -> n-1)
        variable_set = set()
        for node_shifted in self.graph.nodes:
            node = node_shifted + 1
            variable_set.add(node)
        variables = list(variable_set)
        variables.sort()
        self.var_list = copy.deepcopy(variables)
          # The following is to create self-loop (disable if you do not consider that was the case)
#         for variable in variables:
#             idx = variables.index(variable) + 1 
#             node = variable-1
            # The -1 is due to the variable is from 1 -> n where as node run from 0 -> n-1 
#             self.add_diag_element(idx, self.graph[node][node]['weight']) 

        for edge in self.graph.edges:
            # the first correlation + 1 comes from the fact that graph nodes run from 0...n-1
            # the fact that we add another +1 to the index is because the variables list runs 1....n 
            # and the indices in the matrix run 0...n
            idx1, idx2 = variables.index(edge[0] + 1) + 1, variables.index(edge[1] + 1) + 1
            # This is due to to we could have self-loop 
            if idx1 != idx2: 
                weight_ij = self.graph[edge[0]][edge[1]]['weight']
                self.add_off_element(idx1, idx2, weight_ij)

        # we define the appropriate position translater
        self.position_translater = [0] + variables

    def evaluate_solution(self, solution):
        """Returns the weight of the partition"""
        total_weight = 0
        for n1, n2 in self.graph.edges:
            if n1 != n2:
                if np.sign(solution[n1])*np.sign(solution[n2]) < 0:
                    total_weight += self.graph[n1][n2]['weight']
            else: 
                total_weight += self.graph[n1][n2]['weight']
        
        return total_weight 
    
    
    def calc_energy(self, assignment, single_energy_vector, correl_energy_matrix):
        E_calc = np.sign(assignment).T @ correl_energy_matrix @ np.sign(assignment) + single_energy_vector.T @ np.sign(assignment)
        return E_calc
    
        