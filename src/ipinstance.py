from dataclasses import dataclass
import numpy as np
from docplex.mp.model import Model
import queue
import threading
from functools import total_ordering

@dataclass(frozen=True)
class IPConfig:
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
        # print(f"Error reading instance file. File format may be incorrect.{e}")
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
        self.model_lock = threading.Lock()
        np.random.seed(44)

        # ---------------------- DECISION VARIABLES ----------------------

        self.use_vars = self.model.continuous_var_list(self.numTests, lb=0, ub=1, name=lambda x: f'use_{x}')
        print("[INFO] created decision variables")
        
        # ------------------------- CONSTRAINTS  -------------------------

        # each disease must be detected by at least one test
        constraint_count = 0
        batch_constraints = []
        for i in range(self.numDiseases):
            for j in range(i + 1, self.numDiseases):
                if i != j:
                    diff = [np.abs(self.A[k][j] - self.A[k][i]) for k in range(self.numTests)]
                    constraint_count += 1
                    diff_x_use = self.model.scal_prod(self.use_vars, diff)
                    batch_constraints.append(self.model.sum(diff_x_use) >= 1)
        self.model.add_constraints(batch_constraints)
        print("[INFO] added constraints:", constraint_count)

        # minimize the costs of all the used tests
        costs = self.model.scal_prod(self.use_vars, self.costOfTest)
        self.model.minimize(self.model.sum(costs))

        # ------------------ INITIALIZE BRANCH AND BOUND -----------------

        # initialize the queue to the root node
        objective_value, solution = self.solve_lp(self.model)

        self.queue: queue.Queue = queue.LifoQueue()
        self.queue.put(Node(objective_value, {}, self.pick_variable({}, solution)))

        # initialize incumbent variables
        self.incumbent: Node = Node(float('inf'))
        self.incumbent_lock = threading.Lock()
        
        self.done = threading.Event()
  
    def toString(self):
        out = ""
        out = f"Number of test: {self.numTests}\n"
        out += f"Number of diseases: {self.numDiseases}\n"
        cst_str = " ".join([str(i) for i in self.costOfTest])
        out += f"Cost of tests: {cst_str}\n"
        A_str = "\n".join([" ".join([str(j) for j in self.A[i]]) for i in range(0,self.A.shape[0])])
        out += f"A:\n{A_str}"
        return out

    def branch(self, model, parent: Node):
        """ Gives us two children nodes by branching on the next variable """
        
        # make the children nodes (solve the LP relaxation for each child node)
        for decision in [0, 1]:
            # create a new node with the parent's decisions + the new decision
            new_decisions = parent.decisions.copy()
            new_decisions[parent.next_variable] = decision

            # need to update the model with the new decisions and solve the LP relaxation
            constraints = self.add_decisions(model, new_decisions)
            new_objective, new_solution = self.solve_lp(model)
            self.remove_decisions(model, constraints)
            
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
                    with self.incumbent_lock:
                        # print("[INFO] new incumbent found:", new_objective)
                        self.incumbent = Node(new_objective, new_decisions)
                continue

            # pre-choose some variable to branch on based on current decisions
            next_variable = self.pick_variable(new_decisions, new_solution)

            # add the new node to the priority queue
            new_node = Node(new_objective, new_decisions, next_variable)
            
            # add the new node to the queue
            self.queue.put(new_node)
    
    def pick_variable(self, current_decisions: dict, solution) -> str:
        """ Heuristics for picking the next variable """
        use_vars = [f"use_{i}" for i in range(self.numTests)]
        use_vars_available = [i for i in use_vars if i not in current_decisions]
        
        # check how close each variable is to being an integer
        thresh = {i: abs(solution[i] - round(solution[i])) for i in use_vars_available}

        # choose the most fractional variable
        return max(thresh, key=thresh.get)

    def add_decisions(self, model, new_decisions: dict):
        """ Update the model with the new decisions """
        # add new decisions to the model
        use_names = [f"use_{i}" for i in range(self.numTests)]
        use_vars = [model.get_var_by_name(i) for i in use_names]
        
        decision_constraints = [use_vars[int(i.split('_')[1])] == new_decisions[i] for i in new_decisions]
        new_constraints = model.add_constraints(decision_constraints)
        
        # return the new constraints (they will need to be cleared in the future)
        return new_constraints

    def remove_decisions(self, model, old_constraints):
        """ Remove the decisions from the model """
        model.remove_constraints(old_constraints)

    def solve_lp(self, model):
        """ Solve the LP relaxation """
        solution = model.solve()

        if not solution:
            return None, None
        else:
            return model.objective_value, solution
    
    def solve_ip_worker(self):
        with self.model_lock:
            model_clone = self.model.clone()

        idx = threading.get_ident() % 1000
        # print("[INFO] starting thread", idx)

        counter = 0
        while self.done.is_set() == False:
            # if counter % 100 == 0:
            #     print(f"[INFO] thread {idx} has solved {counter} nodes")

            # pick the next node to explore
            parent_node = self.queue.get(block=True)
            
            # branch on the node
            self.branch(model_clone, parent_node)
            
            # mark the task as done
            self.queue.task_done()
            
            counter += 1
        
        # print("[INFO] thread", idx, "is done at node", counter)

    def solve_ip(self) -> Node:
        """ Branch and bound implementation to solve IP using LP"""
        print("[INFO] starting branch and bound")
        
        # start parallel processing
        threads = []
        for _ in range(5):
            t = threading.Thread(target=self.solve_ip_worker)
            t.daemon = True
            t.start()
            threads.append(t)

        # wait for all threads to finish
        self.queue.join()
        self.done.set()
        
        # print("[INFO] waiting for threads to finish")
        # for t in threads:
        #     t.join()

        # when done, return the incumbent
        print("[INFO] done with branch and bound")
        return self.incumbent
