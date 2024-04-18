import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import copy
import Calculating_Expectation_Values as Expectation_Values
from Generating_Problems import MaxCut
import networkx as nx
from classical_solver import find_maxcut
from copy import deepcopy 


class QIRO_MC(Expectation_Values.ExpectationValues):
    """
    :param problem_input: The problem object that shall be solved
    :param nc: size of remaining subproblems that are solved by brute force
    This class is responsible for the whole QIRO (Max Cut) procedure; the output represents the optimized bitstring solution in the form
    of a dictionary as well as a list of optimal parameters from each elimination step
    """

    def __init__(self, problem_input, nc, strategy,no_correlation,temperature,radius = 2):
        super().__init__(problem=problem_input)
        # let us use the problem graph as the reference, and this current graph as the dynamic
        # object from which we will eliminate nodes:
        self.graph = copy.deepcopy(self.problem.graph)
        self.nc = nc
        self.assignment = []
        self.solution = []
        self.strategy = strategy
        self.no_correlation = no_correlation
        self.temperature = temperature
        # These dictionaries are used to store the known two-points correlations  
        self.same_parity_dict = {node: [node] for node in self.graph.nodes()}
        self.opposite_parity_dict = {node: [] for node in self.graph.nodes()} 
        self.radius = radius 
     
    
    def corr_dict_update(self,variables,exp_value_sign,fixing_list,assignments):
        """ This function is used to deal with updating all the variables in which we have fixed correlations in previous round """ 

        if len(variables) == 1: 
            # In case we choose one-point correlation, we should update all the fixed correlation accordingly. For instance,
            # if we know that node 1 and node 3 have opposite parity and node 1 is assign 1, node 3 should be assign -1. 
            node = variables[0] - 1 
            
            # Assign all the correlated_node (same parity node) the same value 
            for corr_node in self.same_parity_dict[node]:
                fixing_list.append([corr_node+1]) 
                assignments.append(exp_value_sign)
                
            # Pop (node,correlated_node) out of same_parity_dict 
            self.same_parity_dict.pop(node) 
            
            # Assign all the correlated_node (opposite parity) the opposite parity 
            for corr_node in self.opposite_parity_dict[node]: 
                fixing_list.append([corr_node+1]) 
                assignments.append(-exp_value_sign)
            # Pop the node out the the opposite_parity_dict
            self.opposite_parity_dict.pop(node)
            
        else: 
            node_large = variables[0] - 1 
            node_small = variables[1] - 1
            
            # Update the correlation dictionary and pop out the remove node 
            if exp_value_sign == 1: 
                self.same_parity_dict[node_large] = self.same_parity_dict[node_large] + self.same_parity_dict[node_small]
                self.opposite_parity_dict[node_large] = self.opposite_parity_dict[node_large] + self.opposite_parity_dict[node_small]
            else: 
                self.same_parity_dict[node_large] = self.same_parity_dict[node_large] + self.opposite_parity_dict[node_small]
                self.opposite_parity_dict[node_large] = self.opposite_parity_dict[node_large] + self.same_parity_dict[node_small]   
             
            self.same_parity_dict.pop(node_small) 
            self.opposite_parity_dict.pop(node_small)      
        return fixing_list, assignments 
            
                
       
    def update_single(self, variable_index, exp_value_sign):
        """Updates Hamiltonian according to fixed single point correlation"""
        node = variable_index - 1
        fixing_list = []
        assignments = []
        ns = set(self.graph.neighbors(node))
        ns.remove(node) 
        
        # nn -> neighbor node 
        for nn in ns: 
            # We add weight to the neighbor of the chosen node. The reason is w_ij Z_i Z_j -> (+/-) w_ij Z_j 
            # Due to it is better to work with edge in networkx, weight of a node is done by self-loop 
            nn_weight = self.graph[node][nn]['weight']
            self.graph[nn][nn]['weight'] = self.graph[nn][nn]['weight'] + nn_weight*exp_value_sign
        
        # in any case we remove the node which was selected by correlations:
        self.graph.remove_node(node)
        fixing_list, assignments = self.corr_dict_update([variable_index],exp_value_sign,fixing_list,assignments)
        
        # reinitailize the problem object with the new, updated, graph:
        self.problem = MaxCut(self.graph)
        return fixing_list, assignments
    
    
    def update_correlation(self, variables, exp_value_sign):
        """Updates Hamiltonian according to fixed two point correlation"""
        fixing_list = []
        assignments = []
        
        # The node we keep and the node we remove  
        fix_node = variables[0]-1  
        remove_node = variables[1]-1
        
        # Obtain neighbor of the remove_node and filter out the remove_node as well as fix_node 
        ns = set(self.graph.neighbors(remove_node)) - {fix_node,remove_node}
             
        # Contracted the remove_node 
        old_graph = copy.deepcopy(self.problem.graph)
        self.graph = nx.contracted_nodes(self.graph, fix_node, remove_node,self_loops = False)

        # Update the weight of edges in the new graph: 
        for n in ns: 
            if old_graph.has_edge(fix_node,n):
                self.graph[fix_node][n]['weight'] = old_graph[remove_node][n]['weight']*exp_value_sign  + old_graph[fix_node][n]['weight']
            else: 
                self.graph[fix_node][n]['weight'] = old_graph[remove_node][n]['weight']*exp_value_sign
            # This is a rude way to catch a subtle bug (dont remove it) 
            if(self.graph[fix_node][n]['weight'] == 0): 
                self.graph.remove_edge(fix_node,n)
        # reinitailize the problem object with the new, updated, graph:
        fixing_list, assignments = self.corr_dict_update(variables,exp_value_sign,fixing_list,assignments)
        self.problem = MaxCut(self.graph)
        return fixing_list, assignments
    
    
    def prune_graph(self):
        """Prunes the graph by removing all connected components that have less than nc nodes. The assignments are determined
        to be the maximum independent sets of the connected components. The self.graph is updated correspondingly."""
        fixing_list = [] 
        assignments = [] 
        nodes_to_remove = []
        
        # get connected components
        connected_components = copy.deepcopy(list(nx.connected_components(self.graph)))
        
        for component in connected_components:
            if len(component) < self.nc:
                subgraph = self.graph.subgraph(component)
                _, maxcut = find_maxcut(subgraph)
                maxcut = maxcut[0] 
                for node in maxcut.keys(): 
                    variable = node + 1 
                    match maxcut[node]:
                        case 1: sign = 1 
                        case 0: sign = -1 
                    fixing_list, assignments = self.corr_dict_update([variable],sign,fixing_list,assignments)
                nodes_to_remove.extend(component)
        # Remove node
        for node in nodes_to_remove:
            self.graph.remove_node(node)

        # Re-initialize the problem 
        self.problem = MaxCut(self.graph)

        return fixing_list, assignments
    def statistic_approach(self,exp_value_coeffs,exp_value_signs,exp_values):
        """ This approach is based on Goemans-Williamson algorithm. From the one-point and two-point functions, we can calculate the mean and covariance matrix. We assume that the underlying assumption is of the multivariables gaussian distribution N(0,M) since MaxCut problem has Z_2 symmetry  """
        no_nodes = self.graph.number_of_nodes()
        node_index_dict = dict( zip(self.graph.nodes(),range(no_nodes)) ) 
        M = np.identity(no_nodes) 
        # Copy of exp_arrays 
        copy_exp_value_coeffs = [] 
        copy_exp_value_signs = [] 
        copy_exp_values = [] 
        
        # Filling off diagonal terms in the covariance matrix 
        for index,coeff in enumerate(exp_value_coeffs):
            node_1 = coeff[0] - 1
            node_2 = coeff[1] - 1
            index_i = node_index_dict[node_1] 
            index_j = node_index_dict[node_2] 
            value = exp_values[index] 
            M[index_i,index_j] = value 
            M[index_j,index_i] = value 
            
#         try:
#             inverse_M = np.linalg.inv(M)
#             print("Matrix is invertible.")
#             print(M)
#             print(inverse_M)
#         except np.linalg.LinAlgError:
#             print("Matrix is not invertible.")    
        # Sampling 
        sample = np.random.multivariate_normal(np.zeros(no_nodes),M) 
        
        # Reconstruct exp_array based on the sample 
        
        for index,coeff in enumerate(exp_value_coeffs): 
            node_1 = coeff[0] - 1 
            node_2 = coeff[1] - 1 
            index_i = node_index_dict[node_1] 
            index_j = node_index_dict[node_2]       
            copy_exp_value_coeffs.append(coeff) 
            value = sample[index_i]*sample[index_j] 
            copy_exp_values.append(value) 
            copy_exp_value_signs.append(np.sign(value).astype(int)) 
                                          
        return copy_exp_value_coeffs,copy_exp_value_signs,copy_exp_values 
    
    def picking_correlation_functions(self,exp_value_coeffs,exp_value_signs,exp_values): 
        """ This approach is based on Goemans-Williamson algorithm. From the one-point and two-point functions, we can calculate the mean and covariance matrix. We assume that the underlying assumption is of the multivariables gaussian distribution N(mu,M) """
        copy_graph = deepcopy(self.graph)
        copy_exp_value_coeffs = [] 
        copy_exp_value_signs = [] 
        copy_exp_values = [] 
        counter = self.no_correlation 
        # We pick out one correlation function for each loop based on the chosenstrategy 
        while counter > 0 and not nx.is_empty(copy_graph):           
            match self.strategy: 
                case "Max":
                    chosen_index = np.argmax(np.abs(exp_values))                
                case "Soft_Max":    
                    weight_vector = [np.exp(abs(value)*self.temperature) for value in exp_values]
                    weight_vector = weight_vector/np.sum(weight_vector) 
                    chosen_index = np.random.choice(range(len(exp_values)), 1, False, weight_vector)[0] 
        
            chosen_coeff = exp_value_coeffs[chosen_index] 
            chosen_sign = exp_value_signs[chosen_index] 
            chosen_value = exp_values[chosen_index] 
            
            # Appending the chosen element 
            copy_exp_value_coeffs.append(chosen_coeff) 
            copy_exp_value_signs.append(chosen_sign) 
            copy_exp_values.append(chosen_value) 
            

            # The following mumbo jumbo is to deal with
            nodes_in_radius = set()  
            for label in chosen_coeff: 
                node = label - 1
                nodes_in_radius = nodes_in_radius.union(set(nx.ego_graph(copy_graph,node,distance = self.radius)))
            copy_graph.remove_nodes_from(nodes_in_radius)
            
            # Remove all the one-point and two-points correlations related to the nodes_in_radius set 
            remove_index_list = [index for index, coeff in enumerate(exp_value_coeffs) if any( (label-1) in nodes_in_radius for label in coeff)]              
            exp_value_coeffs = [coeff for index,coeff in enumerate(exp_value_coeffs) if index not in remove_index_list]
            exp_value_signs = [sign for index,sign in enumerate(exp_value_signs) if index not in remove_index_list]            
            exp_values = [value for index,value in enumerate(exp_values) if index not in remove_index_list]
            counter -= 1             
            
        return copy_exp_value_coeffs, copy_exp_value_signs, copy_exp_values 

    def graph_renormalization(self): 
        """ This function is used at each iteration to make sure that the graph edge weight is normalized to be in [-1,1] """
        max_weight = 0
        for _, _, data in self.graph.edges(data=True):
            max_weight = max(max_weight, abs(data['weight']))
        for edge in self.graph.edges():
            i = edge[0]
            j = edge[1]
            self.graph[i][j]['weight'] = self.graph[i][j]['weight']/max_weight
        self.problem = MaxCut(self.graph)
        
#     def edge_cutting(self,exp_value_coeffs,exp_value_signs,exp_values): 
#         """ This function is used to cut the edge if the correlation is smaller than some threshold  and update the graph according to RQAOA """
#         copy_exp_value_coeffs = [] 
#         copy_exp_value_signs = [] 
#         copy_exp_values = [] 
#         thresh_hold = 0.01
#         for index,value in enumerate(exp_values): 
#             if abs(value) > thresh_hold: 
#                 copy_exp_value_coeffs.append(exp_value_coeffs[index])
#                 copy_exp_value_signs.append(exp_value_signs[index])
#                 copy_exp_values.append(exp_values[index])
#             else: 
#                 coeff = exp_value_coeffs[index]
#                 node_1 = coeff[0]-1
#                 node_2 = coeff[1]-1
#                 value = exp_values[index]
#                 print(f'Remove edge {[node_1,node_2]} with value {value}')
#                 self.graph.remove_edge(node_1,node_2)
#         return copy_exp_value_coeffs, copy_exp_value_signs, copy_exp_values 

    def execute(self, energy='best'):
        """Main QIRO function which produces the solution by applying the QIRO procedure."""
        self.opt_gamma = []
        self.opt_beta = []
        self.fixed_correlations = []
        step_nr = 0
        self.coeff_value_dict = {}


        while self.graph.number_of_nodes() > 0:
            step_nr += 1
            # print(f"Step: {step_nr}. Number of nodes: {self.graph.number_of_nodes()}.")
            # Drawing the graph
#             pos = nx.circular_layout(self.graph)
#             nx.draw(self.graph, pos, with_labels=True)
#             edge_weights = nx.get_edge_attributes(self.graph, 'weight')
#             nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_weights)
#             plt.show()

            fixed_variables = []            
            
            # Obtain ALL the one-point and two-point correlation functions 
            exp_value_coeffs, exp_value_signs, exp_values = self.optimize()
            
            # The following is to weed out one-point correlation functions (disable this if you have self-loop)
            number_of_node = self.graph.number_of_nodes()      
            exp_value_coeffs = exp_value_coeffs[number_of_node:] 
            exp_value_signs = exp_value_signs[number_of_node:] 
            exp_values = exp_values[number_of_node:]   
            self.coeff_value_dict = {tuple(sorted(coeff)): value for coeff,value in zip(exp_value_coeffs,exp_values)}
            
            # statistical approach 
#             exp_value_coeffs, exp_value_signs, exp_values = self.statistic_approach(exp_value_coeffs,exp_value_signs,exp_values)   
            
            # Picking out correlations function based on self.strategy 
            exp_value_coeffs, exp_value_signs, exp_values = self.picking_correlation_functions(exp_value_coeffs,exp_value_signs,exp_values)            
            # print(f'Chosen coeffs {exp_value_coeffs}, and values {exp_values}')
            for index in range(len(exp_value_coeffs)): 
                exp_value_coeff = exp_value_coeffs[index]
                exp_value_sign = exp_value_signs[index]
                exp_value = exp_values[index]
                if exp_value_sign == 0:
                    exp_value_sign = 1
                exp_value = exp_values[index]
                if len(exp_value_coeff) == 1: 
                    holder_fixed_variables, assignments = self.update_single(*exp_value_coeff,exp_value_sign)
                    fixed_variables += holder_fixed_variables 
                    for var, assignment in zip(holder_fixed_variables,assignments): 
                        self.fixed_correlations.append([var,int(assignment),exp_value])
                else:
                    holder_fixed_variables, assignments = self.update_correlation(exp_value_coeff,exp_value_sign)
                    fixed_variables += holder_fixed_variables 
                    for var, assignment in zip(holder_fixed_variables,assignments):
                        self.fixed_correlations.append([var,int(assignment),exp_value])
            
            # perform pruning.
            pruned_variables, pruned_assignments = self.prune_graph()
#             print(f"Pruned {len(pruned_variables)} variables.")
            for var, assignment in zip(pruned_variables, pruned_assignments):
                if var is None:
                    raise Exception("Variable to be eliminated is None. WTF?")
                self.fixed_correlations.append([var, assignment, None])
            fixed_variables += pruned_variables
            
            # Renormalizing the graph 
            self.graph_renormalization()
        
#         print(f'Fixed correaltion list: {self.fixed_correlations}')
        solution = [var[0] * assig for var, assig, _ in self.fixed_correlations]
        sorted_solution = sorted(solution, key=lambda x: abs(x))
        # print(f"Solution: {sorted_solution}")
        self.solution = np.array(sorted_solution).astype(int)

