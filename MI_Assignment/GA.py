# Imports
import numpy as np
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


def roulette_wheel_selection(sorted_population, max_pop_size, survivor_count, T, N):
    pop_fitness = 0
    candidate_possibilities = []
    
    # Total population fitness
    for candidate in sorted_population:
        pop_fitness += fitness(candidate, T, N)
    
    # Candidate probability
    for candidate in sorted_population:
        candidate_possibilities.append(fitness(candidate, T, N) / pop_fitness)
        
    returned_population = []
    for i in range(max_pop_size - survivor_count):
        # Selecting random candidate based on probabilities
        index = np.random.choice(len(candidate_possibilities), p=candidate_possibilities)
        returned_population.append(sorted_population[index])
    
    return returned_population

def elitism_selection(sorted_population, max_pop_size, survivor_count):
    """
    "Keep best half" selection approach
    """
    
    elitism_population = []   

    for i in range(max_pop_size - survivor_count):
        elitism_population.append(sorted_population[i])
        
    return elitism_population


# TODO: consider different crossover techniques(?), e.g. N-point crossover
def single_point_crossover(parent1, parent2):
    """
    Is not a naive crossover approach - does not allow for duplication of orders.
    :returns: child1, child2
    """
    crossover_index = random.randint(1, len(parent1))
    
    # 1st parent
    p1_left = parent1[0:crossover_index]
    p1_right = parent1[crossover_index:(len(parent1))]
    # 2nd parent
    p2_left = parent2[0:crossover_index]
    p2_right = parent2[crossover_index:(len(parent1))]
    
    # Root
    child1 = np.array(p1_left)
    child2 = np.array(p2_left)
    
    # Child 1
    for val in p2_right:
        if val not in child1:
            child1 = np.append(child1, val)
        else: 
            for val in p2_left:
                if val not in child1:
                    child1 = np.append(child1, val)
                    break
    
    # Child 2
    for val in p1_right:
        if val not in child2:
            child2 = np.append(child2, val)
        else:
            for val in p1_left:
                if val not in child2:
                    child2 = np.append(child2, val)
                    break
    
    return child1, child2

  
def mutate(child, mutation_rate):
    rand = random.random()  # random float between 0 and 1.
    
    if rand < mutation_rate:
        # If mutation occurs...
        # Getting of random 2 random indexes to swap
        swap_index1 = random.randint(0, len(child)-1)
        swap_index2 = random.randint(0, len(child)-1)

        # Ensuring swap index are not the same
        while swap_index1 == swap_index2:
            swap_index2 = random.randint(0, len(child)-1)

        # Swapping
        temp = child[swap_index1]
        child[swap_index1] = child[swap_index2]
        child[swap_index2] = temp

    return child
    

def tetrisify(candidate):
    """
    :param candidate:
    :returns: List of Print_instance objects
    """
    
    tetrified_candidate = []
    # Converting Poster list to Print_instance list
    for poster in candidate:
        tetrified_candidate.append(Print_instance(poster.inks))
    
    # Checking for duplication of inks 
    for i in range(len(tetrified_candidate)):
        can_compress = False
        
        j = i+1
        level_count = 1
        for x in range(len(tetrified_candidate[:i+1])):
            # ^ i.e. progressively checking lower posters if no blocking inks
            if j < len(tetrified_candidate):
            
                all_inks_compress = True
                
                for ink in tetrified_candidate[j].final_inks:
                    if ink in tetrified_candidate[j-level_count].final_inks:
                        # If ink is found then the two inks cannot be merged 
                        all_inks_compress = False
                if all_inks_compress:
                    can_compress = True
                    level_count += 1
        if can_compress:
            # If can_compress, combine the inks of once compressed
            # This approach does allow for posters to 'slip past' posters
            #
            # e.g. poster 2 will merge with poster 0
            # poster 0 = 1, 5
            # poster 1 = 0, 1, 3, 5
            # poster 2 = 4
            #
            # In this case, 4 will pass through poster 1 as there are no blocking inks
            # and instead merge with poster 0
            combined_inks = tetrified_candidate[j].final_inks + tetrified_candidate[j - (level_count - 1)].final_inks
            tetrified_candidate[j - (level_count - 1)].final_inks = combined_inks
            tetrified_candidate[j - (level_count - 1)].poster_count += 1
            
            del tetrified_candidate[j]
        
    return tetrified_candidate


def fitness(candidate, T, N, success_income = 10, fail_cost = 5):
    """
    Calculating a fitness score of a candidate solution, given the T constraint and N
    :returns (income - loss) score:
    """
    tetrisified_candidate = tetrisify(candidate)
    
    income = 0
    loss = 0
    
    # For each Print_instance unit of time, until time constraint T... 
    for i in range(T):
        try:
            poster_count = tetrisified_candidate[i].poster_count
            
            if len(tetrisified_candidate[i].final_inks) < N:
                income += poster_count * success_income
            else:
                loss += poster_count * fail_cost
        except:
            print("Error here")
            
    return (income - loss)

    
def sort_by_fitness(population, T, N):
    # TODO: Sort function using fitness function
    
    print_order_population = []
    for ind in population:
        print_order = Print_order(ind, fitness(ind, T, N,))
        print_order_population.append(print_order)
   
    # Sorting population using list of Print_Order objects (to more easily store fitness)
    print_order_population.sort(key=lambda x: x.fitness, reverse=True)
   
    sorted_population = []
    for print_order in print_order_population:
        sorted_population.append(print_order.poster_order)
    
    return sorted_population


def get_next_generation(sorted_population, max_pop_size, survivor_count, mutation_rate, T, N):    
    # Using elitism_selection (best half) approach:
    survivor_population = elitism_selection(sorted_population, max_pop_size, survivor_count)
    
    ##########################################################################################
    
    # Using Roulette Wheel Selection:
    # survivor_population = roulette_wheel_selection(sorted_population, max_pop_size, survivor_count, T, N)
    
    # If survivor_population is odd, then duplicate the best fitting candidate and add to front of survivor population
    if len(survivor_population) % 2 == 1:
        survivor_population.insert(0, survivor_population[0])
    
    children = []
    
    # Crossover
    i = 0
    step = 2
    while i < len(survivor_population):
        # Getting parents
        p1 = survivor_population[i]
        p2 = survivor_population[i+1]
        
        # Crossing over children using parents
        child1, child2 = single_point_crossover(p1, p2)
        
        # Mutating children
        mutated1 = mutate(child1, mutation_rate)
        mutated2 = mutate(child2, mutation_rate)
        # Appending to children list for next population
        children.append(mutated1)
        children.append(mutated2)

        # Consider pairs of individuals in population for crossover
        i += step       
    
    # Trimming returned population such that it fits max_pop_size
    return_population = survivor_population + children
    return_population = return_population[:max_pop_size]
    return return_population


def genetic_algorithm(orders, T, N, max_pop_size = 20, max_generations = 100, survivor_count = 10, mutation_rate = 0.1):
    """
    :param orders: 
    :param T: Time constraint
    :param N: Total Ink count (2 <= N <= 32) -- Consider odd/even calculations
    :param max_pop_size:
    :param max_generations:
    :param survivor_count = 10:
    :param mutation_rate = 0.1:
    :returns optimal_solution_found: Fittest candidate solution
    :returns optimal_fitness_score: Fitness score of fittest candidate solution
    :returns generation_bests: [] of population's fittest candidate solution
    :returns generation_bests_score: [] of population's fittest candidate solution's fitness score
    :returns generation_score_averages: [] of average fitness score per generation
    """
    
    # Defining initial population with random permutations
    population = [] 
    for i in range (max_pop_size):
        # TODO: check this permutation implementation
        printOrders = np.random.permutation(orders)
        population.append(printOrders)
    
    optimal_solution_found = None
    # TODO: convert solution to solution print order(below)
    optimal_fitness_score = -np.inf
    generation_bests = []
    generation_bests_score = []
    generation_score_averages = []
    # Generation loop
    for i in range(max_generations):
        total_generation_score = 0
        # Sorting population by fitness
        sorted_population = sort_by_fitness(population, T, N,)
        
        # TODO: consider population average for analysis (graph representation)
        
        # Comparing population optimal 
        optimal_candidate = sorted_population[0]
        population_best = fitness(optimal_candidate, T, N)
        if population_best > optimal_fitness_score: # Maximisation 
            optimal_fitness_score = population_best
            optimal_solution_found = optimal_candidate
        generation_bests.append(optimal_candidate)
        generation_bests_score.append(fitness(optimal_candidate, T, N))
        
        # Generation bests score average
        for ind in sorted_population:
            total_generation_score += fitness(ind, T, N)
        generation_score_averages.append(total_generation_score / max_pop_size)
        
        population = get_next_generation(sorted_population, max_pop_size, survivor_count, mutation_rate, T, N)        
        
    
    return optimal_solution_found, optimal_fitness_score, generation_bests, generation_bests_score, generation_score_averages

# -+- Main Function Calling-+-
orders_list, N, T  = get_orders_details("orders.txt")

optimal_solution_found, optimal_fitness_score, generation_bests, \
    generation_bests_score, generation_score_average = genetic_algorithm(orders_list, T, N, 6, 20, 2, 0.1)
solution_string = ""
poster_count = 0

# Output (# of Posters to print, and then the ordered order ID of each)
print(str(len(optimal_solution_found)) + " posters to print")
for poster in optimal_solution_found:
    print(poster.id)

#"""
# Testing / Inner workings prints statements below

print(f"Best Solution:\n{solution_string} Fitness Score: {optimal_fitness_score}")
print(f"Generation Bests:\n")

for i in range(len(generation_bests_score)):
    # Printing Generation bests, and Generation average fitness scores
    print(generation_bests_score[i])
    print(generation_score_average[i])
    
    for poster in generation_bests[i]:
        print(poster.inks)
#"""