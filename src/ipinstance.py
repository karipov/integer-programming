from dataclasses import dataclass
import numpy  as np
from docplex.mp.model import Model
import heapq as pq

@dataclass(frozen=True)
class IPConfig:
    numTests: int # number of tests
    numDiseases: int # number of diseases
    costOfTest: np.ndarray #[numTests] the cost of each test
    A: np.ndarray #[numTests][numDiseases] 0/1 matrix if test is positive for disease


#  * File Format
#  * #Tests (i.e., n)
#  * #Diseases (i.e., m)
#  * Cost_1 Cost_2 . . . Cost_n
#  * A(1,1) A(1,2) . . . A(1, m)
#  * A(2,1) A(2,2) . . . A(2, m)
#  * . . . . . . . . . . . . . .
#  * A(n,1) A(n,2) . . . A(n, m)
def data_parse(filename : str) :
    try:
        with open(filename,"r") as fl:
            numTests = int(fl.readline().strip()) #n 
            numDiseases = int(fl.readline().strip()) #m

            costOfTest = np.array([float(i) for i in fl.readline().strip().split()])

            A = np.zeros((numTests,numDiseases))
            for i in range(0,numTests):
                A[i,:] = np.array([int(i) for i in fl.readline().strip().split() ])

        return numTests, numDiseases, costOfTest, A

    except Exception as e:
        print(f"Error reading instance file. File format may be incorrect.{e}")
        exit(1)

class Node:
    def __init__(self, lp_objective_value: float, decisions: dict = {}):
        # floating point value of the LP relaxation
        self.lp_objective_value = lp_objective_value
        # decisions made: i -> 1 if test i is used, 0 otherwise
        self.decisions = decisions
    

class IPInstance:
    def __init__(self,filename : str) -> None:
        numT,numD,cst,A = data_parse(filename)
        self.numTests = numT
        self.numDiseases = numD
        self.costOfTest = cst
        self.A = A
        self.model = Model() #CPLEX solver
        
        # add problem constraints: use_vars[i] = 1 if test i is used, 0 otherwise
        self.use_vars = [self.model.continuous_var(name='use_{0}'.format(i), lb=0, ub=1)
                         for i in range(self.numTests)]

        # for every pair of diseases, there must be at least one test that can distinguish them
        for i in range(self.numDiseases):
            for j in range(self.numDiseases):
                if (i!= j): 
                    diff = [np.abs(self.A[k][j] - self.A[k][i]) for k in range(self.numTests)]
                    diff_x_use = [diff[k] * self.use_vars[k] for k in range(self.numTests)]
                    self.model.add_constraint(self.model.sum(diff_x_use) >= 1)
        
        # minimize the costs of all the used tests
        costs = [self.use_vars[i] * self.costOfTest[i] for i in range(self.numTests)]
        self.model.minimize(self.model.sum(costs))
        
        # store the incumbent
        self.incumbent: Node = Node(float('inf'))
        
        # initialize the priority queue to the root node
        self.model.solve_lp()
        self.priority_queue = pq.heapify([Node(self.model.objective_value)])
  
    def toString(self):
        out = ""
        out = f"Number of test: {self.numTests}\n"
        out += f"Number of diseases: {self.numDiseases}\n"
        cst_str = " ".join([str(i) for i in self.costOfTest])
        out += f"Cost of tests: {cst_str}\n"
        A_str = "\n".join([" ".join([str(j) for j in self.A[i]]) for i in range(0,self.A.shape[0])])
        out += f"A:\n{A_str}"
        
        return out

    def branch(parent: Node):
        """ Gives us two children nodes by branching on the next variable """
        
        # choose variable to branch on
        
        # make the children nodes (solve the LP relaxation)
        
        # assess the nodes (incumbent, prune, add to queue)
    
    def pick_variable(self):
        """ Heuristics for picking the next variable """
        pass
  
    def solve_lp(self):
        """ Solve the LP relaxation """
        sol = self.model.solve()

        obj_value = self.model.objective_value
        if sol:
            self.model.print_information()
       
            print(f"Objective Value: {obj_value}")
        else:
            print("No solution found!")

    def solve_ip(self):
        """ Branch and bound implementation to solve IP using LP"""
        
        while not self.pq:
            # pick the next node to explore
            parent_node = pq.heappop(self.pq)

            # check if the node is dominated by the incumbent, if so prune it
            if parent_node.lp_objective_value >= self.incumbent.lp_objective_value:
                continue
            
            # branch on the node
            self.branch()
            
            
            
            
        


    


    

  