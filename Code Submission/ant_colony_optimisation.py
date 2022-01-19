# Imports
from operator import truediv
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import time
import math as m
import random


# Function and CLass definitions
class Print_order():
    def __init__(self, poster_order, fitness = 0):
        self.poster_order = poster_order
        self.fitness = fitness
        
class Poster(): # stores details of a poster
    def __init__(self, id, ink_total, inks):
        # Poster ID number
        self.id = id
        #ink_total referring to total number of inks used for this poster
        self.ink_total = ink_total
        # inks referring a list of inks used for this poster [ints]
        self.inks = inks
        
  
class Print_instance(): # i.e. unit of print time (can consist of multiple posters)
    def __init__(self, inks, poster_count = 1):
        # final_inks referring to the inks used in the print instance
        self.final_inks = inks
        # posterCount referring to the number of posters to be printed in this print instance.
        self.poster_count = poster_count

def get_orders_details(filename):
    """
    Getting and returning order_list from input
    :returns: orders_list - [] of Poster objects
    :returns: N = Int - Number of inks
    :returns: T = Int - Time constraint
    """
    
    f = open(filename)
    
    # Header Handling
    header_list = f.readline().split(" ")
    N = header_list[0]
    T = header_list[2]
    
    # Orders Handling 
    orders_list = []
    id_count = 0
    
    for line in f:
        if line[-1] == '\n':
            line = line[:-1]
                    
        order_line = line.split(" ")
        ink_total = order_line.pop(0)
        inks = order_line
        poster = Poster(id_count, ink_total, inks)
        orders_list.append(poster)
        # Incrementing id_count
        id_count += 1
    f.close()
    
    # Removing of any erroneous (frome Python's file import) empty line Poster objects
    for i in range(len(orders_list)):
        if orders_list[i] == "\n":
            del orders_list[i]
            
    return orders_list, int(N), int(T)


def tetrisify(path):
    """
    :param path:
    :returns: List of Print_instance objects
    """
    
    tetrified_path = []
    # TODO: candidate begins as Print_instance list
    # Converting Poster list to Print_instance list
    for poster in path:
        tetrified_path.append(Print_instance(poster.inks))
    
    # Checking for duplication of inks 
    # len(candidate)- 1 so that last poster does not check
    for i in range(len(tetrified_path) - 1):
        if i+1 < len(tetrified_path):
            can_compress = True
            for ink in tetrified_path[i].final_inks:
                # For ink in poster...
                if ink in tetrified_path[i + 1].final_inks:
                    # If ink in next poster's list of inks...
                    can_compress = False
                    # break

            # Creating combined or individual Print_instance objects to append
            # to tetrisified_candidate list.
            if can_compress:
                # If no duplicated inks...
                combined_inks = tetrified_path[i].final_inks + tetrified_path[i + 1].final_inks
                tetrified_path[i].final_inks = combined_inks
                tetrified_path[i].poster_count += 1
                
                # Deleting 2nd print_instance, as it is combined
                del tetrified_path[i + 1]
                i -= 1
    
    return tetrified_path

def is_tetrifiable(i, j):
    """
    Function used to establish whether i Poster can compress with j Poster
    :returns boolean:
    """
    can_compress = True
    
    # Checking for any duplicate inks
    for ink in i.inks:
        if ink in j.inks:
            can_compress = False
    
    return can_compress


def convert_path(path, orders_list):
    """Given a path and the orders_list, return the path in Poster objects form"""
    returned_path = []
    
    for index in path:
        returned_path.append(orders_list[index])
    
    return returned_path

def fitness(path, orders_list, T, N, success_income = 10, fail_cost = 5):
    """
    Calculating a fitness score of a path, given the T constraint and N
    :returns (income - loss) score:
    """
    # [:len(path)-1] removes the return to first poster in the path
    posters_path = convert_path(path, orders_list)[:len(path)-1]
    tetrisified_path = tetrisify(posters_path)
    
    income = 0
    loss = 0
    
    # For each Print_instance unit of time, until time constraint T... 
    for i in range(T):
        try:
            poster_count = tetrisified_path[i].poster_count
            
            if len(tetrisified_path[i].final_inks) < N:
                income += poster_count * success_income
            else:
                loss += poster_count * fail_cost
        except:
            print("Error here")
            
    return (income - loss)

def edge_value(i,j):
    """Function used to evaluate the potential value of the edge (i -> j)"""
    value = 1
    if is_tetrifiable(i, j):
        value = 2
        
    return value
    

class AntColonyOptimizer:
    def __init__(self, ants, rho, intensification, alpha=1.0, beta=0.0, beta_decay=0,
                 exploitation=.1):
        """
        Ant colony optimizer.  Traverses a graph and finds either the max or min distance between nodes.
        :param ants: number of ants to traverse the graph
        :param ro: rate at which pheromone evaporates
        :param intensification: constant added to the best path
        :param alpha: weighting of pheromone
        :param beta: weighting of heuristic (1/distance)
        :param beta_decay: rate at which beta decays (optional)
        :param exploitation: probability to choose the best route
        """
        # Parameters
        self.ants = ants
        self.rho = rho
        self.reinforce_pheromone = intensification
        self.alpha = alpha
        self.beta = beta
        self.beta_decay = beta_decay
        self.exploitation = exploitation

        # Internal representations
        self.pheromone_matrix = None
        self.heuristic_matrix = None
        self.probability_matrix = None

        self.map = None
        self.allowed_nodes = None

        # Internal stats
        self.best_series = []
        self.best = None
        self.fitted = False
        self.best_path = None
        self.fit_time = None

        # Plotting values
        self.stopped_early = False

    def __str__(self):
        string = "Ant Colony Optimizer"
        string += "\n--------------------"
        string += "\nDesigned to optimize either the minimum or maximum distance between nodes in a square matrix that behaves like a distance matrix."
        string += "\n--------------------"
        string += "\n Number of ants:\t\t\t\t{}".format(self.ants)
        string += "\n Rho:\t\t\t{}".format(self.rho)
        string += "\n Intensification factor:\t\t{}".format(self.reinforce_pheromone)
        string += "\n Alpha:\t\t\t{}".format(self.alpha)
        string += "\n Beta:\t\t\t\t{}".format(self.beta)
        string += "\n Beta decay rate:\t\t{}".format(self.beta_decay)
        string += "\n Choose Best Percentage:\t\t{}".format(self.exploitation)
        string += "\n--------------------"
        string += "\n USAGE:"
        string += "\n Number of ants influences how many paths are explored each iteration."
        string += "\n Alpha and beta affect how much influence the pheromones or the distance heuristic weigh an ants' decisions."
        string += "\n Beta decay reduces the influence of the visibility over time."
        string += "\n Choose best is a percentage of how often an ant will choose the best route over probabilistically choosing a route based on pheromones."
        string += "\n--------------------"
        if self.fitted:
            string += "\n\nThis optimizer has been fitted."
        else:
            string += "\n\nThis optimizer has NOT been fitted."
        return string

    def _initialize(self):
        """
        Initializes the model by creating the various matrices and generating the list of available nodes
        """
        assert self.map.shape[0] == self.map.shape[1]
        num_nodes = self.map.shape[0]
        self.pheromone_matrix = np.ones((num_nodes, num_nodes))
        # Remove the diagonal since there is no pheromone from node i to itself
        self.pheromone_matrix[np.eye(num_nodes) == 1] = 0
        self.heuristic_matrix = 1 / self.map
        self.probability_matrix = (self.pheromone_matrix ** self.alpha) * (
                self.heuristic_matrix ** self.beta)  # element by element multiplcation
        self.allowed_nodes = list(range(num_nodes))

    def _reinstate_nodes(self):
        """
        Resets available nodes to all nodes for the next iteration
        """
        self.allowed_nodes = list(range(self.map.shape[0]))

    def _update_probabilities(self):
        """
        After evaporation and intensification, the probability matrix needs to be updated.  This function
        does that.
        """
        self.probability_matrix = (self.pheromone_matrix ** self.alpha) * (
                self.heuristic_matrix ** self.beta)

    def _choose_next_node(self, from_node):
        """
        Chooses the next node based on probabilities.  If p < p_exploitation, then the best path is chosen, otherwise
        it is selected from a probability distribution weighted by the pheromone.
        :param from_node: the node the ant is coming from
        :return: index of the node the ant is going to
        """
        numerator = self.probability_matrix[from_node, self.allowed_nodes]
        
        if np.random.random() < self.exploitation:
            next_node = np.argmax(numerator)
        else:
            denominator = np.sum(numerator)
            probabilities = numerator / denominator
            next_node = np.random.choice(range(len(probabilities)), p=probabilities)
        return next_node

    def _remove_node(self, node):
        self.allowed_nodes.remove(node)

    def _evaluate(self, orders_list, paths, mode):
        """
        Evaluates the solutions of the ants by testing it against the fitness function.
        :param paths: solutions from the ants
        :param mode: max or min
        :return: The best path, and the best score
        """
        scores = []
        # Determine fitness score for each path
        for i in range(len(paths)):
            scores.append(fitness(paths[i], orders_list, T, N))
        
        # Handling mode (should be max for best fitness score)
        if mode == 'min':
            best = np.argmin(scores)
        elif mode == 'max':
            best = np.argmax(scores)
            
        return paths[best], scores[best]

    def _evaporation(self):
        """
        Evaporate some pheromone as the inverse of the evaporation rate.  Also evaporates beta if desired.
        """
        self.pheromone_matrix *= (1 - self.rho)
        self.beta *= (1 - self.beta_decay)

    def _reinforce(self, best_path):
        """
        Increases the pheromone by some scalar for the best route.
        :param best_path: best path
        """
        i = best_path[0]
        j = best_path[1]
        self.pheromone_matrix[i, j] += self.reinforce_pheromone

    def fit(self, orders_list, map_matrix, iterations=100, mode='max', early_stopping_count=20, verbose=False):
        """
        Fits the ACO to a specific map.  This was designed with the Traveling Salesman problem in mind.
        :param map_matrix: Distance matrix or some other matrix with similar properties
        :param iterations: number of iterations
        :param mode: whether to get the minimum path or maximum path
        :param early_stopping_count: how many iterations of the same score to make the algorithm stop early
        :return: the best score
        """
        if verbose: print("Beginning ACO Optimization with {} iterations...".format(iterations))
        self.map = map_matrix
        start = time.time()
        self._initialize()
        num_equal = 0

        for i in range(iterations):
            start_iter = time.time()
            paths = []
            path = []

            for ant in range(self.ants):
                current_node = self.allowed_nodes[np.random.randint(0, len(self.allowed_nodes))]
                start_node = current_node
                while True:
                    path.append(current_node)
                    self._remove_node(current_node)
                    if len(self.allowed_nodes) != 0:
                        current_node_index = self._choose_next_node(current_node)
                        current_node = self.allowed_nodes[current_node_index]
                    else:
                        break

                path.append(start_node)  # go back to start
                self._reinstate_nodes()
                paths.append(path)
                path = []

            best_path, best_score = self._evaluate(orders_list, paths, mode)

            if i == 0:
                best_score_so_far = best_score
            else:
                if mode == 'min':
                    if best_score < best_score_so_far:
                        best_score_so_far = best_score
                        self.best_path = best_path
                elif mode == 'max':
                    if best_score > best_score_so_far:
                        best_score_so_far = best_score
                        self.best_path = best_path

            if best_score == best_score_so_far:
                num_equal += 1
            else:
                num_equal = 0

            self.best_series.append(best_score)
            self._evaporation()
            self._reinforce(best_path)
            self._update_probabilities()

            if verbose: print("Best score at iteration {}: {}; overall: {} ({}s)"
                              "".format(i, round(best_score, 2), round(best_score_so_far, 2),
                                        round(time.time() - start_iter)))

            if best_score == best_score_so_far and num_equal == early_stopping_count:
                self.stopped_early = True
                print("Stopping early due to {} iterations of the same score.".format(early_stopping_count))
                break

        self.fit_time = round(time.time() - start)
        self.fitted = True

        if mode == 'min':
            self.best = self.best_series[np.argmin(self.best_series)]
            if verbose: print(
                "ACO fitted.  Runtime: {} minutes.  Best score: {}".format(self.fit_time // 60, self.best))
            return self.best
        elif mode == 'max':
            self.best = self.best_series[np.argmax(self.best_series)]
            if verbose: print(
                "ACO fitted.  Runtime: {} minutes.  Best score: {}".format(self.fit_time // 60, self.best))
            return self.best
        else:
            raise ValueError("Invalid mode!  Choose 'min' or 'max'.")

    def plot(self):
        """
        Plots the score over time after the model has been fitted.
        :return: None if the model isn't fitted yet
        """
        if not self.fitted:
            print("Ant Colony Optimizer not fitted!  There exists nothing to plot.")
            return None
        else:
            fig, ax = plt.subplots(figsize=(20, 15))
            ax.plot(self.best_series, label="Best Run")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Performance")
            ax.text(.8, .6,
                    'Ants: {}\nEvap Rate: {}\nIntensify: {}\nAlpha: {}\nBeta: {}\nBeta Evap: {}\nChoose Best: {}\n\nFit Time: {}m{}'.format(
                        self.ants, self.rho, self.reinforce_pheromone, self.alpha,
                        self.beta, self.beta_decay, self.exploitation, self.fit_time // 60,
                        ["\nStopped Early!" if self.stopped_early else ""][0]),
                    bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10}, transform=ax.transAxes)
            ax.legend()
            plt.title("Ant Colony Optimization Results (best: {})".format(np.round(self.best, 2)))
            plt.show()
        
    def print_pheromone(self):
        data = self.pheromone_matrix
        fig, ax = plt.subplots()
        im = ax.imshow(data, cmap=plt.get_cmap('hot'), interpolation='nearest',
                        vmin=np.min(data), vmax=np.max(data))
        fig.colorbar(im)
        plt.show()
        return 0


orders_list, N, T = get_orders_details("orders.txt")
_NODES = len(orders_list)

fitness_matrix = np.zeros((_NODES, _NODES))
for i in range(_NODES):
    for j in range(_NODES):
        fitness_matrix[i][j] = edge_value(orders_list[i], orders_list[j])

optimiser = AntColonyOptimizer(ants=15, rho=.2, intensification=2, alpha=0.6, beta=1,
                               beta_decay=0, exploitation=.1)

best = optimiser.fit(orders_list, fitness_matrix, 100)

print(optimiser)

print(optimiser.best_path)
print(best)
optimiser.plot()