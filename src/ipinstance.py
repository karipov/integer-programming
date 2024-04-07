from dataclasses import dataclass
import numpy  as np
from docplex.mp.model import Model
import heapq as pq
from util import check_integer_solution

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

# TODO: make sure that the Node is ordered correctly in the priority queue
# potentially use dataclass ?
class Node:
    def __init__(self, lp_objective_value: float, decisions: dict = {}):
        # floating point value of the LP relaxation
        self.lp_objective_value = lp_objective_value
        # decisions made: i -> 1 if test i is used, 0 otherwise
        self.decisions = decisions
        print("I made a node")
        #TODO ADD something that stores the values of the decision variables per node, then make sure that
        #these values can be stored
    

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
        
        print("I made vars")

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
        print("created initial incumbent")
        # initialize the priority queue to the root node
        self.solve_lp()
        print(self.model.objective_value)
        self.priority_queue = []
        pq.heappush(self.priority_queue, Node(self.model.objective_value)) #??
        print(self.priority_queue)
        self.solve_ip()
        print("solve IP")
        
        
  
    def toString(self):
        out = ""
        out = f"Number of test: {self.numTests}\n"
        out += f"Number of diseases: {self.numDiseases}\n"
        cst_str = " ".join([str(i) for i in self.costOfTest])
        out += f"Cost of tests: {cst_str}\n"
        A_str = "\n".join([" ".join([str(j) for j in self.A[i]]) for i in range(0,self.A.shape[0])])
        out += f"A:\n{A_str}"
        return out

    def branch(self, parent: Node, solution):
        """ Gives us two children nodes by branching on the next variable """
        
        # choose variable to branch on based on current decisions + some heuristic
        print("calling branch")
        picked_variable = self.pick_variable(parent.decisions, solution)
        print("picked var =" + str(picked_variable))
        # make the children nodes (solve the LP relaxation for each child node)
        for decision in [0, 1]:
            # create a new node with the parent's decisions + the new decision
            new_decisions = parent.decisions.copy()
            new_decisions[picked_variable] = decision
            
            # need to update the model with the new decisions and solve the LP relaxation
            self.update_model(new_decisions)
            new_objective, new_solution = self.solve_lp()
            
            # check if the solution is feasible
            if new_objective is None:
                print("new objective is infeasible")
                break #i think this is correct
            
            # check if dominated by the incumbent
            if new_objective >= self.incumbent.lp_objective_value:
                print("new objective is dominated")
                break
            
            # check if we have a new incumbent / solution is IP
            if check_integer_solution(new_solution):
                print("new objective is new solution")
                # TODO: update the incumbent, potentially prune the queue #what does potentially prune queue mean 
                if new_objective < self.incumbent.lp_objective_value:
                    self.incumbent: Node = Node(new_objective, new_decisions)
                    # pq_copy = pq[:]
                    # pq_copy.heapify(pq_copy)
                    # for item in pq_copy:
                    #     print(item)
                break
            
            print("new objective is none of the above")
            # add the new node to the priority queue
            new_node = Node(new_objective, new_decisions)
            pq.heappush(self.pq, new_node)
            print(self.pq)
    
    def pick_variable(self, current_decisions: dict, solution) -> str:
        """ Heuristics for picking the next variable """
        # use_vars = ["use_" + i for i in range(self.numTests)]
        use_vars_available = [i for i in self.use_vars if i not in current_decisions]
        
        assert len(use_vars_available) > 0, "No variables left to branch on"
        
        # check how close each variable is to being an integer, select the one that is the farthest
        # TODO: apparently, solution[i] simply returns 0 and doesn't fail. do we do smth about it?

        thresh = [abs(solution.get_var_values[var] - round(solution.get_var_values[var])) for var in use_vars_available]
        
        assert max(thresh) != 0, "No fractional variables found"
        
        # return the variable name
        return use_vars_available[thresh.index(max(thresh))]

    def update_model(self, new_decisions: dict):
        """ Update the model with the new decisions """
        # remove all the old decisions
        print("In update model")
        decision_constraints = ["decision_" + i for i in range(self.numTests)]
        for decision in decision_constraints:
            self.model.remove_constraint(decision)
        
        # add all the decisions back + the new ones
        for decision in new_decisions:
            # TODO: need to add names for these decisions ???
            self.model.add_constraint(self.use_vars[decision] == new_decisions[decision])

    def solve_lp(self):
        """ Solve the LP relaxation """
        print("called solve() in LP")
        solution = self.model.solve()
        
        if not solution:
            print("No solution found!")
            return None, None
        else:
            
            self.model.print_information()
            print(f"Objective Value: {self.model.objective_value}")
            for i, var in enumerate(self.use_vars):
                print(f"Value of {var}: {solution.get_var_value(var)}")
                self.use_vars[i] = solution.get_var_value(var)  # Update the use_vars variables

            # self.model.objective_value,
            return solution

    def solve_ip(self) -> Node:
        """ Branch and bound implementation to solve IP using LP"""
        print("here's pq:")
        print(self.priority_queue)
        
        # pq.heapify(pq)
        # print(list(pq))
        count = 0
        while self.priority_queue:
            print("iteration:" + str(count))
            print("in while loop")
            # pick the next node to explore
            parent_node = pq.heappop(self.priority_queue)
            sol = self.solve_lp
            print("sol time")
            print(self.model.objective_value)
            # branch on the node
            print("calling branch")
            self.branch(parent_node, sol)
            count +=1
        print("out of while loop, printing incumbent")
        print(self.incumbent.lp_objective_value)
        # when done, return the incumbent
        return self.incumbent
            
            
            
            
        


    


    

  