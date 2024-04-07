from dataclasses import dataclass
import numpy  as np
from docplex.mp.model import Model
import heapq as pq
from util import check_integer_solution
from functools import total_ordering

@dataclass(frozen=True)
class IPConfig:
    numTests: int # number of tests
    numDiseases: int # number of diseases
    costOfTest: np.ndarray #[numTests] the cost of each test
    A: np.ndarray #[numTests][numDiseases] 0/1 matrix if test is positive for disease
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

# TODO: make sure that the Node is ordered correctly in the priority queue
# potentially use dataclass ?
@total_ordering
class Node:
    def __init__(self, lp_objective_value: float, decisions: dict = {}, next_variable = None):
        # floating point value of the LP relaxation
        self.lp_objective_value = lp_objective_value
        # decisions made: i -> 1 if test i is used, 0 otherwise
        self.decisions = decisions
        # next variable that will be picked
        self.next_variable = next_variable
    
    def __eq__(self, other):
        if not isinstance(other, __class__):
            return NotImplemented
        return self.lp_objective_value == other.lp_objective_value

    def __lt__(self, other):
        if not isinstance(other, __class__):
            return NotImplemented
        return self.lp_objective_value < other.lp_objective_value
    

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
        
        print("[INFO] created initial vars")

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
        objective_value, solution = self.solve_lp()

        self.priority_queue = []
        pq.heappush(self.priority_queue,
                    Node(objective_value, {}, self.pick_variable({}, solution)))
  
    def toString(self):
        out = ""
        out = f"Number of test: {self.numTests}\n"
        out += f"Number of diseases: {self.numDiseases}\n"
        cst_str = " ".join([str(i) for i in self.costOfTest])
        out += f"Cost of tests: {cst_str}\n"
        A_str = "\n".join([" ".join([str(j) for j in self.A[i]]) for i in range(0,self.A.shape[0])])
        out += f"A:\n{A_str}"
        return out

    def branch(self, parent: Node):
        """ Gives us two children nodes by branching on the next variable """
        
        # make the children nodes (solve the LP relaxation for each child node)
        for decision in [0, 1]:
            # create a new node with the parent's decisions + the new decision
            new_decisions = parent.decisions.copy()
            new_decisions[parent.next_variable] = decision
            
            # need to update the model with the new decisions and solve the LP relaxation
            decision_constraints = self.add_decisions(new_decisions)
            new_objective, new_solution = self.solve_lp()
            # make sure to remove the decision constraints for the next iteration
            self.remove_decisions(decision_constraints) 
            
            # check if the solution is feasible
            if new_objective is None:
                continue
            
            # check if dominated by the incumbent
            if new_objective >= self.incumbent.lp_objective_value:
                continue
            
            # check if we have a new incumbent / solution is IP
            if check_integer_solution(new_solution, self.numTests):
                # check if the new integer solution is better than the incumbent
                if new_objective < self.incumbent.lp_objective_value:
                    # if so, update the incumbent
                    self.incumbent = Node(new_objective, new_decisions)
                continue
                
            # pre-choose some variable to branch on based on current decisions + some heuristic
            next_variable = self.pick_variable(new_decisions, new_solution)
            
            # add the new node to the priority queue
            new_node = Node(new_objective, new_decisions, next_variable)
            pq.heappush(self.priority_queue, new_node)
    
    def pick_variable(self, current_decisions: dict, solution) -> str:
        """ Heuristics for picking the next variable """
        use_vars = [f"use_{i}" for i in range(self.numTests)]
        use_vars_available = [i for i in use_vars if i not in current_decisions]
        
        assert len(use_vars_available) > 0, "No variables left to branch on"
        
        # check how close each variable is to being an integer, select the one that is the farthest
        # TODO: apparently, solution[i] simply returns 0 and doesn't fail. do we do smth about it?
        thresh = [abs(solution[i] - round(solution[i])) for i in use_vars_available]
        
        # TODO: branch only on fractional variables (not integer ones!)
        # TODO: incorporate randomness because we are using a very greedy approach
        # top-k selection (select randomly from top 3 variables farthest from being integer)
        
        
        assert max(thresh) != 0, "No fractional variables found"
        
        # return the variable name
        return use_vars_available[thresh.index(max(thresh))]

    def add_decisions(self, new_decisions: dict):
        """ Update the model with the new decisions """
        # add new decisions to the model
        # print("[INFO]", new_decisions)
        decision_constraints = [self.use_vars[int(i.split('_')[1])] == new_decisions[i] for i in new_decisions]
        new_constraints = self.model.add_constraints(decision_constraints)
        
        # return the new constraints (they will need to be cleared in the future)
        return new_constraints

    def remove_decisions(self, old_constraints):
        """ Remove the decisions from the model """
        self.model.remove_constraints(old_constraints)

    def solve_lp(self):
        """ Solve the LP relaxation """
        solution = self.model.solve()
        
        if not solution:
            # print("No solution found!")
            return None, None
        else:
            # self.model.print_information()
            # print(f"Objective Value: {self.model.objective_value}")
            return self.model.objective_value, solution

    def solve_ip(self) -> Node:
        """ Branch and bound implementation to solve IP using LP"""
        while self.priority_queue:
            # pick the next node to explore
            parent_node = pq.heappop(self.priority_queue)

            # branch on the node
            self.branch(parent_node)
        
        # when done, return the incumbent
        print("[INFO] done with branch and bound")
        return self.incumbent
            
            
            
            
        


    


    

  