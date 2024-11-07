import argparse
import random
import time
from tqdm import tqdm
from utils import save_files, egraph_preprocess
from random_extractor import random_generate_dags, one_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',
                        type=str,
                        default='examples/math_syn/math_synthetic_d9r4i0.dot')
    parser.add_argument('--time_limit', type=int, default=60)
    parser.add_argument('--load_cost', action='store_true', default=False)
    parser.add_argument('--num_of_generations', type=int, default=300)
    parser.add_argument('--num_of_paths', type=int, default=30)
    parser.add_argument('--num_of_tour_particips', type=int, default=10)
    parser.add_argument('--tournament_prob', type=float, default=0.7)
    parser.add_argument('--crossover_prob', type=float, default=0.8)
    parser.add_argument('--mutation_prob', type=float, default=0.8)
    parser.add_argument('--choose_prob', type=float, default=0.4)
    parser.add_argument('--quad_cost_file',
                        type=str,
                        default=None,
                        help='path to the quadratic cost file')
    parser.add_argument('--mlp_cost_file',
                        type=str,
                        default=None,
                        help='path to the mlp cost file')

    return parser.parse_args()


class Population:

    def __init__(self):
        self.population = []  #element is a class one_path

    def __len__(self):
        return len(self.population)

    def extend(self, new_paths):
        self.population.extend(new_paths)

    def append(self, new_path):
        self.population.append(new_path)


class NSGA2forEgraph:

    def __init__(self,
                 num_of_generations=50,
                 num_of_paths=100,
                 num_of_tour_particips=50,
                 tournament_prob=0.6,
                 crossover_prob=0.8,
                 mutation_prob=0.8,
                 choose_prob=0.5,
                 time_limit=60,
                 egraph=None):

        self.num_of_generations = num_of_generations
        self.num_of_paths = num_of_paths
        self.num_of_tour_particips = num_of_tour_particips
        self.tournament_prob = tournament_prob
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.egraph = egraph
        self.choose_prob = choose_prob
        self.time_limit = time_limit

    def population_initialization(self):
        record_data, generated_paths, reach_time_limit \
            = random_generate_dags(self.egraph, self.choose_prob, self.num_of_paths, self.time_limit)
        population = Population()
        population.extend(generated_paths)  #convert to population
        return population, record_data, reach_time_limit

    def __tournament(self, population):
        participants = random.sample(population.population,
                                     self.num_of_tour_particips)
        best = participants[0]
        for participant in participants:
            if participant.superior(best) and self.__select_with_prob(
                    self.tournament_prob):
                best = participant
        return best

    def __select_with_prob(self, prob):
        if random.random() <= prob:
            return True
        return False

    def __mutate(self, child_path):
        if (random.random() <= self.mutation_prob):
            can_choose_enodes = child_path.enodes.copy()
            i = 0
            while True:
                i += 1
                # print(f"Trying mutating for {i} times outer! : different enodes in path")
                mutation_enode = random.sample(can_choose_enodes, 1)[0]
                can_choose_enodes.remove(mutation_enode)  # remove this
                eclass = self.egraph.enodes[mutation_enode].belong_eclass_id
                j = 0
                if len(self.egraph.eclasses[eclass].enode_id) > 1:
                    enodes_all = self.egraph.eclasses[eclass].enode_id.copy()
                    while len(enodes_all
                              ) > 1:  # try all the enodes in this class
                        j += 1
                        enodes_all.remove(mutation_enode)
                        mutation_enode = random.sample(enodes_all, 1)[0]
                        backup_choice_enode = child_path.choices[eclass]
                        child_path.choices[eclass] = mutation_enode
                        cycles = child_path.find_cycles(
                            self.egraph.enodes, self.egraph.root_classes)
                        # print(f"Trying mutating for {i} times inner! : different enodes in one eclass")
                        if cycles == []:  # can mutate, then over
                            #update, will change the path from the mutate enode to the leaf
                            child_path.enodes = child_path.get_path(
                                self.egraph.enodes, self.egraph.root_classes)
                            child_path.enodes_to_tensor(
                                self.egraph.enodes_tensor)
                            child_path.cost = child_path.dag_cost(self.egraph)
                            return True
                        else:
                            child_path.choices[eclass] = backup_choice_enode
                            if can_choose_enodes == []:
                                return False
                else:
                    if can_choose_enodes == []:
                        return False

    def __crossover_one_path(self, start_crossover_enode, this_path,
                             other_path, enodes):
        #find the path from the crossover_enodes to the leaf
        backup_choices = this_path.choices.copy()
        start_crossover_ecls = self.egraph.enodes[
            start_crossover_enode].belong_eclass_id
        other_path_enodes = other_path.get_path(enodes, [start_crossover_ecls])
        other_path_ecls = [
            self.egraph.enodes[node].belong_eclass_id
            for node in other_path_enodes
        ]
        for ecls in other_path_ecls:
            this_path.choices[ecls] = other_path.choices[ecls]
        cycles = this_path.find_cycles(enodes, this_path.root_classes)
        if cycles != []:
            # print(f"Can not crossover because of cycles, try another one")
            this_path.choices = backup_choices
            return False
        else:  #can mutate, continue update this path
            # print(f"crossover success!")
            return True

    def __crossover(self, parent_path1, parent_path2):
        #can not change parent_path1 and parent_path2!
        #can only copy choices here!
        child1 = one_path(self.egraph.root_classes)
        child1.choices = parent_path1.choices.copy()
        child2 = one_path(self.egraph.root_classes)
        child2.choices = parent_path2.choices.copy()

        if (random.random() <= self.crossover_prob):
            assert len(parent_path1.enodes) == len(set(parent_path1.enodes))
            assert len(parent_path2.enodes) == len(set(parent_path2.enodes))
            common_enodes = list(
                set(parent_path1.enodes) & set(parent_path2.enodes))
            if not common_enodes:
                # print("Can not crossover because of no common enodes, try another one")
                return False
            else:
                while common_enodes:
                    crossover_enode = random.sample(common_enodes, 1)[0]
                    common_enodes.remove(crossover_enode)
                    #crossover to get child1, happen inplace
                    if not self.__crossover_one_path(
                            crossover_enode, child1, parent_path2,
                            self.egraph.enodes):  #cycle
                        continue
                    #crossover to get child2, happen inplace
                    if not self.__crossover_one_path(
                            crossover_enode, child2, parent_path1,
                            self.egraph.enodes):  #cycle
                        continue
                    else:  #crossover success, update path informations
                        child1.enodes = child1.get_path(
                            self.egraph.enodes, self.egraph.root_classes)
                        child1.enodes_to_tensor(self.egraph.enodes_tensor)
                        child1.cost = child1.dag_cost(self.egraph)
                        child2.enodes = child2.get_path(
                            self.egraph.enodes, self.egraph.root_classes)
                        child2.enodes_to_tensor(self.egraph.enodes_tensor)
                        child2.cost = child2.dag_cost(self.egraph)
                        return child1, child2

            # print("Can not crossover because of cycle, try another one")
            return False
        else:  # just copy the parent_path1 and parent_path2
            child1.enodes = parent_path1.enodes.copy()
            child1.cost = parent_path1.cost
            child2.enodes = parent_path2.enodes.copy()
            child2.cost = parent_path2.cost
            return child1, child2

    # @profile
    def __create_children(self, population):
        children = []
        while len(children) < len(population):
            parent1 = self.__tournament(population)
            parent2 = self.__tournament(population)
            i = 0
            while parent1.enodes == parent2.enodes:  # choose different parents
                parent2 = self.__tournament(population)
                i += 1
                if i > len(population) * 5:
                    # print("Can not find different parents, they are all same?")
                    # breakpoint()
                    break
            #try crossover
            crs_result = self.__crossover(parent1, parent2)
            if crs_result == None:
                child1, child2 = crs_result
                # print("Don't crossover this time!")
            elif crs_result != False:  #crossover success
                # print(f"Crossover success at 1 times!")
                child1, child2 = crs_result
            # i=0
            # j=0
            while crs_result == False:  # must crossover this time, because we already judge don't crossover at above
                # i+=1
                # print(f"Crossover failed for {i} times!")
                parent1 = self.__tournament(population)
                parent2 = self.__tournament(population)
                while parent1.enodes == parent2.enodes:  # choose different parents
                    parent2 = self.__tournament(population)
                #try crossover
                crs_result = self.__crossover(parent1, parent2)
                if crs_result == None:
                    crs_result = False
                    continue
                if crs_result != False and crs_result != None:  #crossover success
                    # j+=1
                    # print(f"Crossover success at {j} times!")
                    child1, child2 = crs_result
                    break
            self.__mutate(child1)
            # print("child1 mutate success!")
            self.__mutate(child2)
            # print("child2 mutate success!")
            children.append(child1)
            children.append(child2)

        return children

    def __generation_and_test_strategy(self, children, parents_population):
        #adopt "generation and test" strategy for generating new population among (original population + children be generated)
        assert len(children) == len(parents_population.population)
        population_all_list = children + parents_population.population
        new_population_list = sorted(
            population_all_list,
            key=lambda path: path.cost)[:self.num_of_paths]
        new_population = Population()
        new_population.population = new_population_list
        self.population = new_population
        return new_population

    def __evolve_for_one_generation(self, population):
        children = self.__create_children(population)
        new_population = self.__generation_and_test_strategy(
            children, population)
        return new_population

    def evolutions(self, initilization_time, initilized_population):
        time_start = time.time()
        cost_time_dic = {"cost": [], "time": []}
        population = initilized_population
        # for i in tqdm(range(self.num_of_generations),
        #               desc=f'Evolving {self.num_of_generations} Generations'):
        for i in range(self.num_of_generations):
            population = self.__evolve_for_one_generation(population)
            used_time = time.time() - time_start + initilization_time
            if used_time > self.time_limit:
                return cost_time_dic
            sorted_population = sorted(population.population,
                                       key=lambda path: path.cost)
            cost_time_dic["cost"].append(sorted_population[0].cost)
            cost_time_dic["time"].append(used_time)
        return cost_time_dic


def main(exp_id):
    args = get_args()
    egraph = egraph_preprocess(args)
    GA_solver = NSGA2forEgraph(args.num_of_generations, args.num_of_paths,
                               args.num_of_tour_particips,
                               args.tournament_prob, args.crossover_prob,
                               args.mutation_prob, args.choose_prob,
                               args.time_limit, egraph)
    initialized_population, initilization_record, reach_time_limit = GA_solver.population_initialization(
    )
    initilization_best = min(initilization_record["cost"])
    used_time = initilization_record["time"][-1]
    if not reach_time_limit:
        evolution_record = GA_solver.evolutions(used_time,
                                                initialized_population)
        cost_all = initilization_record["cost"] + evolution_record["cost"]
        time_all = initilization_record["time"] + evolution_record["time"]
    else:
        cost_all = initilization_record["cost"]
        time_all = initilization_record["time"]
    best_cost = min(cost_all)
    best_time = time_all[cost_all.index(best_cost)]
    record_data = {"cost": cost_all, "time": time_all}
    save_files(best_cost, best_time, record_data, "Genetic",
               args.quad_cost_file, args.mlp_cost_file, args.input_file,
               exp_id)


if __name__ == "__main__":
    for i in range(3):
        main(3)
