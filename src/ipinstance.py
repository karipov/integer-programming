from dataclasses import dataclass
import numpy as np
from docplex.mp.model import Model
import heapq as pq
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


def check_integer_solution(solution, num_tests) -> bool:
    """
    Checks if our LP relaxation solution is a solution to the IP problem
    """
    threshold = 1e-3
    thresh_check = lambda x: abs(x - round(x)) < threshold
    all_values = [solution.get_value(f"use_{i}") for i in range(num_tests)]
    return all(map(thresh_check, all_values))

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
    def __init__(self, filename: str) -> None:
        numT,numD,cst,A = data_parse(filename)
        self.numTests = numT
        self.numDiseases = numD
        self.costOfTest = cst
        self.A = A
        self.model = Model(checker="off") #CPLEX solver
        np.random.seed(44)
        
        # add problem constraints: use_vars[i] = 1 if test i is used, 0 otherwise
        self.use_vars = [self.model.continuous_var(name='use_{0}'.format(i), lb=0, ub=1)
                         for i in range(self.numTests)]
        
        print("[INFO] created decision variables")
        
        # ---------------------- DECISION VARIABLES ----------------------

        # for every pair of diseases, there must be at least one test that can distinguish them
        constraint_count = 0
        batch_constraints = []
        for i in range(self.numDiseases):
            for j in range(i + 1, self.numDiseases):
                if i != j:
                    diff = [np.abs(self.A[k][j] - self.A[k][i]) for k in range(self.numTests)]
                    diff_x_use = [diff[k] * self.use_vars[k] for k in range(self.numTests)]
                    constraint_count += 1
                    batch_constraints.append(self.model.sum(diff_x_use) >= 1)
                    # self.model.add_constraint(self.model.sum(diff_x_use) >= 1)
        self.model.add_constraints(batch_constraints)
        print("[INFO] added constraints:", constraint_count)
        
        # minimize the costs of all the used tests
        costs = [self.use_vars[i] * self.costOfTest[i] for i in range(self.numTests)]
        self.model.minimize(self.model.sum(costs))
        
        # ------------------ INITIALIZE BRANCH AND BOUND -----------------
        
        # intialize heuristic variables related to heuristics and search strategies
        # COST_MULTIPLIER = 1
        # self.cost_effective_use = {}
        # for i in range(self.numTests):
        #     discriminative = 0
        #     for j in range(self.numDiseases):
        #         for k in range(i + 1, self.numDiseases):
        #             if (j != k):
        #                 discriminative += np.abs(self.A[i][j] - self.A[i][k])
        #     self.cost_effective_use[f"use_{i}"] = discriminative \
        #         / (self.costOfTest[i] * COST_MULTIPLIER)
        self.mixed_search_switched = False
        
        # initialize incumbent variables
        self.incumbent: Node = Node(float('inf'))
        self.incumbent_changes = 0
        self.dominated_counter = 0

        # initialize the priority queue to the root node
        objective_value, solution = self.solve_lp()

        self.queue = []
        # the "pq" module works over python lists, so no need to initialize a stack structure
        pq.heappush(self.queue,
                    Node(objective_value, {}, self.pick_variable({}, solution, "fractional")))
  
    def toString(self):
        out = ""
        out = f"Number of test: {self.numTests}\n"
        out += f"Number of diseases: {self.numDiseases}\n"
        cst_str = " ".join([str(i) for i in self.costOfTest])
        out += f"Cost of tests: {cst_str}\n"
        A_str = "\n".join([" ".join([str(j) for j in self.A[i]]) for i in range(0,self.A.shape[0])])
        out += f"A:\n{A_str}"
        return out

    def branch(self, parent: Node, heuristic: str, search_strategy: str):
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
                self.dominated_counter += 1
                continue
            
            # check if we have a new incumbent / solution is IP
            if check_integer_solution(new_solution, self.numTests):
                # check if the new integer solution is better than the incumbent
                if new_objective < self.incumbent.lp_objective_value:
                    # if so, update the incumbent
                    self.incumbent = Node(new_objective, new_decisions)
                    self.incumbent_changes += 1
                    print(f"[INFO] incumbent: value {self.incumbent.lp_objective_value} counter {self.incumbent_changes}")
                continue
                
            # pre-choose some variable to branch on based on current decisions + some heuristic
            next_variable = self.pick_variable(new_decisions, new_solution, heuristic)
            
            # add the new node to the priority queue
            new_node = Node(new_objective, new_decisions, next_variable)
            
            if search_strategy == "best_first":
                pq.heappush(self.queue, new_node)
            elif search_strategy == "depth_first":
                self.queue.append(new_node)
            elif search_strategy == "mixed":
                if self.mixed_search_switched:
                    pq.heappush(self.queue, new_node)
                else:
                    self.queue.append(new_node)
            elif search_strategy == "mixed_random":
                self.queue.append(new_node)
            else:
                raise ValueError("Invalid search strategy")
    
    def pick_variable(self, current_decisions: dict, solution, heuristic: str) -> str:
        """ Heuristics for picking the next variable """
        use_vars = [f"use_{i}" for i in range(self.numTests)]
        use_vars_available = [i for i in use_vars if i not in current_decisions]
        assert len(use_vars_available) > 0, "No variables left to branch on"
        
        # filter only the variables that are not integers
        is_integer = lambda x: abs(x - round(x)) < 1e-5
        use_vars_noninteger = [i for i in use_vars_available if not is_integer(solution[i])]
        # print("[INFO] fractional variables:", {i: round(solution[i], 4) for i in use_vars_noninteger})
        # print("[INFO] integer variables:", {i: round(solution[i], 4) for i in use_vars_available if i not in use_vars_noninteger})
        # print("[INFO] fractional variables cost-effective:", {i: round(self.cost_effective_use[i], 4) for i in use_vars_noninteger})
        
        # early exit here if using random heuristic
        if heuristic == "random":
            return str(np.random.choice(use_vars_noninteger))
        
        # check how close each variable is to being an integer
        # TODO: apparently, solution[i] simply returns 0 and doesn't fail. do we do smth about it?
        thresh = {i: abs(solution[i] - round(solution[i])) for i in use_vars_noninteger}
        assert max(thresh) != 0, "No fractional variables found"
        
        # choose more sophisticated heuristic
        chosen_var = None
        if heuristic == "cost_effective":
            # get the heuristic of each test and sort by highest value first
            use_vars_cost = [(i, round(self.cost_effective_use[i] * thresh[i], 2))
                             for i in use_vars_noninteger]
            chosen_var = max(use_vars_cost, key=lambda x: x[1])[0]
        elif heuristic == "fractional":
            # choose the most fractional variable
            chosen_var = max(thresh, key=thresh.get)
        elif heuristic == "random_top_fractional":
            # choose a random variable from the top 10%
            n = int(len(use_vars_noninteger) * 0.1) if len(use_vars_noninteger) > 10 else 1
            top_fractional = pq.nlargest(n, thresh, key=thresh.get)
            chosen_var = str(np.random.choice(top_fractional))
        else:
            raise ValueError("Invalid heuristic")

        # print("[INFO] chosen variable:", chosen_var, "with value:", solution[chosen_var])
        # print()
        return chosen_var

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
        print("[INFO] starting branch and bound")
        
        heuristic = "fractional"
        search_strategy = "depth_first" # best_first / depth_first / mixed / mixed_random
        
        print("[INFO] using heuristic:", heuristic)
        print("[INFO] using search strategy:", search_strategy)
        
        counter = 0
        while self.queue:
            counter += 1
            if counter % 50 == 0:
                print(f"[INFO] nodes visited: {counter} (dominated: {self.dominated_counter})")

            # pick the next node to explore
            if search_strategy == "best_first":
                parent_node = pq.heappop(self.queue)
            elif search_strategy == "depth_first":
                parent_node = self.queue.pop()
            elif search_strategy == "mixed":
                # switch to best-first search after a certain number of incumbent changes
                if self.incumbent_changes >= 1:
                    if not self.mixed_search_switched:
                        print("[INFO] switching to best-first at node", counter)
                        pq.heapify(self.queue)
                        self.mixed_search_switched = True

                    parent_node = pq.heappop(self.queue)
                else:
                    parent_node = self.queue.pop()
            elif search_strategy == "mixed_random":
                if self.incumbent_changes >= 1 and (not self.mixed_search_switched):
                    print("[INFO] switching to random heuristic at node", counter)
                    self.mixed_search_switched = True
                    heuristic = "random" # after switching, use random heuristic

                parent_node = self.queue.pop()
            else:
                raise ValueError("Invalid search strategy")

            # branch on the node
            self.branch(parent_node, heuristic, search_strategy)
        
        # when done, return the incumbent
        print("[INFO] final iteration", counter)
        print("[INFO] done with branch and bound")
        return self.incumbent
            
            
            
            
        


    


    

  