import argparse
from tqdm import tqdm
import time
import numpy as np
from collections import defaultdict
from utils import egraph_preprocess, save_files
from dag_greedy import UniqueQueue, ExtractionResult, FasterGreedyDagExtractor
import random


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',
                        type=str,
                        default='examples/math_syn/math_synthetic_d9r4i0.dot')
    parser.add_argument('--time_limit', type=int, default=60)
    parser.add_argument('--load_cost', action='store_true', default=False)
    parser.add_argument('--num_of_paths', type=int, default=100)
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


class Cost_history:

    def __init__(self, costs=None, choice=None):
        self.costs = costs or set()
        self.choice = choice


class one_path(ExtractionResult):

    def __init__(self, root_classes):
        self.enodes = []
        self.cost = 0
        self.choices = {}
        self.root_classes = root_classes  # don't change

    def superior(self, other_path):
        better = (self.cost > other_path.cost)
        return better

    def get_path(self, egraph_enodes, roots):
        choose_enodes = []
        visited_eclass = set()
        todo = list(roots)
        while todo:
            cid = todo.pop()
            node = self.choices[cid]
            if cid in visited_eclass:
                continue
            visited_eclass.add(cid)
            choose_enodes.append(node)
            for child in egraph_enodes[node].eclass_id:
                todo.append(child)
        return choose_enodes

    def enodes_to_tensor(self,
                         enodes_tensor):  # convert enodes to int index tensor
        # int_index_list =[]
        # for node in self.enodes:
        #     int_index_list.append(self.enode_map[node])
        enodes_tensor[0, self.enodes] = 1

    def dag_cost(self, egraph):
        if egraph.quad_cost_mat is not None:
            quad_loss = egraph.quad_cost(egraph.enodes_tensor)
            loss = quad_loss.item()
        elif egraph.mlp is not None:
            mlp_loss = egraph.mlp_cost(egraph.enodes_tensor)
            loss = mlp_loss.item()
        else:
            linear_loss = egraph.linear_cost(egraph.enodes_tensor)
            loss = linear_loss.item()
        egraph.enodes_tensor.zero_()  #free the enodes tensor space
        return loss


class RandomDagExtractor():

    # @profile
    def update_cost_his(self, node, costs, egraph_enodes):
        if not egraph_enodes[node].eclass_id:
            return Cost_history({egraph_enodes[node].belong_eclass_id}, node)

        children_classes = list(
            set(child for child in egraph_enodes[node].eclass_id))

        cid = egraph_enodes[node].belong_eclass_id
        if cid in children_classes:
            return False  #cycle

        result = costs[children_classes[0]].costs.copy()
        for child_cid in children_classes[1:]:
            result.update(costs[child_cid].costs)

        if cid in result:
            return False  #cycle
        else:
            result.add(cid)
            return Cost_history(result, node)

    # @profile
    def extract(self, egraph, choose_prob):
        analysis_pending = UniqueQueue()
        analysis_pending.extend(egraph.leaf_nodes)
        parents = egraph.parents
        class_reject_times = defaultdict(int)
        costs = {}
        while analysis_pending:
            node = analysis_pending.pop()
            class_id = egraph.enodes[node].belong_eclass_id
            if class_id in costs:
                continue
            elif all(child_class in costs
                     for child_class in egraph.enodes[node].eclass_id):
                random_prob = random.random()
                if egraph.node_probability[
                        node] == 1:  #only one e-node in this e-class
                    pass
                elif random_prob > choose_prob:  # weighted uniform random_choice
                    class_reject_times[class_id] += 1
                    if class_reject_times[
                            class_id] < 1 / egraph.node_probability[node]:
                        analysis_pending.insert(node)
                        continue
                #class's last e-node or probability allow. choose it
                cost_his = self.update_cost_his(node, costs, egraph.enodes)
                if cost_his == False:  # cycle
                    # print(f"class{class_id} has cycle!")
                    if class_reject_times[
                            class_id] >= 1 / egraph.node_probability[
                                node]:  #already reject all the e-nodes
                        choices = egraph.eclasses[class_id].enode_id.copy()
                        choices.remove(node)
                        # analysis_pending.insert(random.random.choice(Eclasses[class_id]))
                        analysis_pending.insert(random.random.choice(choices))
                    continue
                else:
                    costs[class_id] = cost_his
                    analysis_pending.extend(parents[class_id])
            else:
                analysis_pending.insert(node)
        path = one_path(egraph.root_classes)
        for cid, cost_set in costs.items():
            path.choose(cid, cost_set.choice)
        if path.find_cycles(egraph.enodes, egraph.root_classes) != []:
            for i, cycle in enumerate(
                    path.find_cycles(egraph.enodes, egraph.root_classes)):
                print(f"Cycle {i}: {cycle}")
            return False
        path.enodes = path.get_path(egraph.enodes, egraph.root_classes)
        path.enodes_to_tensor(egraph.enodes_tensor)
        path.cost = path.dag_cost(egraph)
        return path


def random_generate_one_dag(egraph, extractor, choose_prob=0.4):
    # sample path and DFS check acylic
    path = extractor.extract(egraph, choose_prob)
    return path


def random_generate_dags(egraph, choose_prob, num_of_paths=30, time_limit=60):
    extractor = RandomDagExtractor()
    paths_num = 0
    generated_paths = []
    cost_time_dic = {"cost": [], "time": []}
    reach_time_limit = False
    start_time = time.time()
    # for i in tqdm(range(num_of_paths),
    #               desc=f"Randomly Extract {num_of_paths} DAGs"):
    for i in range(num_of_paths):
        path = random_generate_one_dag(egraph, extractor, choose_prob)
        if path == False:
            # print(
            #     "This path cycle, next!"
            # )  #Theoretically we may meet this in complex graph, but not yet
            pass
        else:
            if time.time() - start_time > time_limit:
                reach_time_limit = True
                # print(f"Time limit reached, only sampled {paths_num} DAGs!")
                return cost_time_dic, generated_paths, reach_time_limit
            paths_num += 1
            cost_time_dic["cost"].append(path.cost)
            cost_time_dic["time"].append(round(time.time() - start_time, 4))
            generated_paths.append(path)
    # print(f"Successfully randomly sampled {paths_num} DAG!")
    return cost_time_dic, generated_paths, reach_time_limit


def print_and_save_data(cost_time_dic, args):
    tile_cost_list = []
    for i, cost in enumerate(cost_time_dic["cost"]):
        tile_cost_list.append(cost)
        if (i + 1) % 50 == 0:
            min_tile_cost = min(tile_cost_list)
            min_tile_cost_index = tile_cost_list.index(min_tile_cost)
            min_cost_time = cost_time_dic["time"][i - 49 + min_tile_cost_index]
            tile_cost_list = []
            # print(
            #     f"Best cost for path {i-49} ~ {i+1} is {min_tile_cost}, time is {min_cost_time}s"
            # )
    best_cost = min(cost_time_dic["cost"])
    best_time = cost_time_dic["time"][cost_time_dic["cost"].index(best_cost)]
    # print(f"\nBest cost is {best_cost}, time is {best_time}s")
    save_files(best_cost, best_time, cost_time_dic, "Random_Extractor",
               args.quad_cost_file, args.mlp_cost_file, args.input_file)


def main():
    args = get_args()
    egraph = egraph_preprocess(args)
    cost_time_dic, _, _ = random_generate_dags(egraph, args.choose_prob,
                                               args.num_of_paths,
                                               args.time_limit)
    print_and_save_data(cost_time_dic, args)


if __name__ == "__main__":
    main()
