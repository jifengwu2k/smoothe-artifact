import argparse
import numpy as np
from collections import defaultdict
from collections import deque
import time
import json
import math
import itertools
from egraph_data import EGraphData
from utils import find_root_classes
import os


class CostSet:

    def __init__(self, costs=None, total=0, choice=None):
        self.costs = costs or {}
        self.total = total
        self.choice = choice


class FasterGreedyDagExtractor:

    def calculate_cost_set(self, node, costs, cost_of_node, egraph_enodes,
                           best_cost):
        if not egraph_enodes[node].eclass_id:
            return CostSet(
                {egraph_enodes[node].belong_eclass_id: cost_of_node[node]},
                cost_of_node[node], node)

        children_classes = list(
            set(child for child in egraph_enodes[node].eclass_id))

        cid = egraph_enodes[node].belong_eclass_id
        if cid in children_classes:
            return CostSet({}, float("inf"), node)

        first_cost = costs[children_classes[0]]
        if (
                # len(children_classes) == 1
                cost_of_node[node] + first_cost.total > best_cost):
            return CostSet({}, float("inf"), node)

        result = costs[children_classes[0]].costs.copy()
        for child_cid in children_classes[1:]:
            result.update(costs[child_cid].costs)

        contain = cid in result
        result[cid] = cost_of_node[node]
        result_cost = float("inf") if contain else sum(result.values())

        return CostSet(result, result_cost, node)

    # @profile
    def extract(self, cost_of_node, egraph_enodes, egraph_eclasses=None):
        "int index for elcass id and enode id"
        parents = defaultdict(list)
        analysis_pending = UniqueQueue()

        for node in egraph_enodes:
            if egraph_enodes[node].eclass_id == [] or egraph_enodes[
                    node].eclass_id == set():
                analysis_pending.insert(node)  #leaf node
            else:
                for child_class in egraph_enodes[node].eclass_id:
                    parents[child_class].append(node)

        costs = {}
        while analysis_pending:
            node = analysis_pending.pop()
            class_id = egraph_enodes[node].belong_eclass_id

            if all(child_class in costs
                   for child_class in egraph_enodes[node].eclass_id):
                if class_id in costs:
                    prev_cost = costs.get(class_id).total
                    # prev_choice = costs.get(class_id).choice
                else:
                    prev_cost = float("inf")

                cost_set = self.calculate_cost_set(node, costs, cost_of_node,
                                                   egraph_enodes, prev_cost)
                if cost_set.total < prev_cost:
                    costs[class_id] = cost_set
                    analysis_pending.extend(parents[class_id])
                # elif cost_set.total == prev_cost and prev_cost != float("inf"):
                #     if cost_set.choice < prev_choice:  # we remove the randomness
                #         costs[class_id] = cost_set
                #         analysis_pending.extend(parents[class_id])

        result = ExtractionResult()
        for cid, cost_set in costs.items():
            result.choose(cid, cost_set.choice)

        return result, costs


class BaslineGreedyDagExtractor:

    def extract(self, cost_of_node, egraph_enodes, egraph_eclasses):
        "int index for elcass id and enode id"
        costs = {}
        do_something = True
        while do_something:
            do_something = False
            for eclass in egraph_eclasses:
                if eclass not in costs:
                    eclass_cost = np.inf
                else:
                    eclass_cost = costs[eclass].total
                    prev_choice = costs[eclass].choice
                child_enode_cost = np.inf
                best_enode = None
                for enode in egraph_eclasses[eclass].enode_id:
                    if eclass in egraph_enodes[enode].eclass_id:
                        pass  #cycle
                    if all(child_class in costs
                           for child_class in egraph_enodes[enode].eclass_id):
                        this_enode_cost = cost_of_node[enode] + sum(
                            costs[child_class].total
                            for child_class in egraph_enodes[enode].eclass_id)
                        if this_enode_cost < child_enode_cost:
                            child_enode_cost = this_enode_cost
                            best_enode = enode
                if child_enode_cost == np.inf:
                    pass
                else:
                    if child_enode_cost < eclass_cost:
                        costs[eclass] = CostSet({}, child_enode_cost,
                                                best_enode)
                        do_something = True
                    elif child_enode_cost == eclass_cost and eclass_cost != np.inf:
                        if best_enode < prev_choice:  #remove the randomness
                            costs[eclass] = CostSet({}, child_enode_cost,
                                                    best_enode)
                            do_something = True

        result = ExtractionResult()
        for cid, cost_set in costs.items():
            result.choose(cid, cost_set.choice)

        return result, costs


class UniqueQueue:

    def __init__(self):
        self.set = set()
        self.queue = deque()

    def insert(self, item):
        if item not in self.set:
            self.set.add(item)
            self.queue.append(item)

    def extend(self, items):
        for item in items:
            self.insert(item)

    def pop(self):
        if not self.queue:
            return None
        item = self.queue.popleft()
        self.set.remove(item)
        return item

    def __bool__(self):
        return bool(self.queue)


class ExtractionResult:

    def __init__(self):
        self.choices = {}
        self.final_dag = []

    def choose(self, class_id, node_id):
        self.choices[class_id] = node_id

    def find_cycles(self, egraph_enodes, roots):
        status = defaultdict(lambda: "Todo")
        cycles = []
        for root in roots:
            self._cycle_dfs(egraph_enodes, root, status, cycles)
        return cycles

    def _cycle_dfs(self, egraph_enodes, class_id, status, cycles):
        if status[class_id] == "Done":
            return
        elif status[class_id] == "Doing":
            cycle_start = False
            cycle_path = []
            for class_idx, class_status in status.items():
                if class_idx == class_id:  #the start of the cycle
                    cycle_start = True
                    assert (class_status == 'Doing')
                if not cycle_start:
                    continue
                else:
                    if class_status == 'Doing':  #Doing means in cycle
                        cycle_path.append([class_idx])
                cycles.append(cycle_path)
            return

        status[class_id] = "Doing"
        node = self.choices[class_id]
        for child in egraph_enodes[node].eclass_id:
            self._cycle_dfs(egraph_enodes, child, status, cycles)
        status[class_id] = "Done"

    def dag_cost(self, egraph_enodes, roots, cost, quad_cost=None):
        choose_enodes = []
        costs = {}
        todo = list(roots)
        while todo:
            cid = todo.pop()
            node = self.choices[cid]
            if cid in costs:
                continue
            costs[cid] = cost[node]
            choose_enodes.append(node)
            for child in egraph_enodes[node].eclass_id:
                todo.append(child)
        linear_cost = sum(costs.values())
        if quad_cost == None:
            return linear_cost, choose_enodes
        else:
            extra_quad_cost = 0
            # print(f"len(choose_enodes) = {len(choose_enodes)}")
            for a, b in itertools.combinations(choose_enodes, 2):
                extra_quad_cost += quad_cost[(a, b)] if (a,
                                                         b) in quad_cost else 0
                extra_quad_cost += quad_cost[(b, a)] if (b,
                                                         a) in quad_cost else 0
            # breakpoint()
            return extra_quad_cost, choose_enodes,


def choose_extractor(method):
    assert method in ["faster", "baseline"
                      ], "we only support 'faster' and 'baseline' method"
    if method == "faster":
        return FasterGreedyDagExtractor()
    elif method == "baseline":
        return BaslineGreedyDagExtractor()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",
                        type=str,
                        default="examples/cunxi_test_egraph2.dot")
    parser.add_argument("--method",
                        choices=["faster", "baseline"],
                        default="faster")
    parser.add_argument("--load_cost", action="store_true", default=False)
    parser.add_argument("--ini_greedy", action="store_true", default=False)

    return parser.parse_args()


def greedy(egraph, method,
           ini_greedy):  #egraph's type is EGraphData or BaseEGraph
    "use int index, needs some preprocess for EGraphData/BaseEGraph"
    if ini_greedy:  # use cost after preprocess
        cost = egraph.processed_cost_per_node.cpu().numpy().tolist(
        )  #calculate on cpu
    else:
        cost = egraph.cost_per_node.cpu().numpy().tolist()  #calculate on cpu
    extractor = choose_extractor(method)
    start_time = time.time()
    result, cost_history = extractor.extract(cost, egraph.enodes,
                                             egraph.eclasses)
    end_time = time.time()
    root_classes = find_root_classes(egraph)
    assert result.find_cycles(egraph.enodes, root_classes) == []
    dag_cost, choose_enodes = result.dag_cost(egraph.enodes, root_classes,
                                              cost)
    time_consume = end_time - start_time
    # print(f"Time consume: {time_consume} cost: {dag_cost}")
    if not ini_greedy:
        return dag_cost, time_consume
    else:
        return choose_enodes  #already in int index


def main():
    args = get_args()
    # we do greedy on cpu
    egraph = EGraphData(args.input_file,
                        load_cost=args.load_cost,
                        drop_self_loops=False,
                        device="cpu")
    dag_cost, time_consume = greedy(egraph, args.method, args.ini_greedy)
    save_data = {
        "name": args.input_file,
        "dag": dag_cost,
        "micros": math.floor(time_consume * 10**6),
    }
    saved_file_path = args.input_file.replace(".json",
                                              f"_{args.method}_greedy.json")
    saved_file_path = saved_file_path.replace(".dot",
                                              f"_{args.method}_greedy.json")
    saved_file_path = saved_file_path.replace("dataset", "logs/heuristic")
    os.makedirs(os.path.dirname(saved_file_path), exist_ok=True)
    with open(saved_file_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
