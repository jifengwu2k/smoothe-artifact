import re
import os
import json
import logging
import argparse
import pulp
import time
from pulp import PULP_CBC_CMD
from collections import defaultdict
from egraph_data import EGraphData
from ortools.linear_solver import pywraplp


def get_eclass(enode):
    if '.' in enode:
        return enode.split('.')[0]
    elif '__' in enode:
        return enode.split('__')[0]


# Function to build and solve the ILP
def ortools_solve_ilp(EClasses, EEdges, root_classes, cost, acyclic, solver,
                      time_limit, verbose, target):
    # Instantiate solver
    if solver == 'cbc':
        solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            print("Could not create CBC solver.")
            return
    elif solver == 'scip':
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            print("Could not create CBC solver.")
            return
    else:
        print("Unknown solver option.")
        return

    if verbose:
        solver.EnableOutput()

    # Variables
    M = len(EClasses)
    x = {i: solver.IntVar(0, 1, f"x_{i}") for i in EEdges.keys()}
    t = {m: solver.NumVar(0, 1, f"t_{m}") for m in EClasses.keys()}

    # Objective function
    solver.Minimize(sum(cost[i] * x[i] for i in EEdges.keys()))

    # Constraints
    for eclass in root_classes:
        # one e-node should be chosen from the root e-class
        solver.Add(sum(x[i] for i in EClasses[eclass]) == 1)

    for i, edges in EEdges.items():
        for edge in edges:
            solver.Add(x[i] <= sum(x[j] for j in EClasses[edge]))

    if acyclic:
        eps = 1 / (len(EClasses) * 2)
        A = 100
        for i, edges in EEdges.items():
            for edge in edges:
                solver.Add(t[get_eclass(i)] - t[edge] - eps + A *
                           (1 - x[i]) >= 0)

    if target is not None:
        solver.parameters.optimal_objective_limit = target
    # the time limit is in milliseconds
    solver.SetTimeLimit(time_limit * 1000)
    status = solver.Solve()

    # printing solution status
    if status == pywraplp.Solver.OPTIMAL:
        status = 'OPTIMAL'
    elif status == pywraplp.Solver.FEASIBLE:
        status = 'FEASIBLE'
    else:
        status = 'INFEASIBLE'
    print(f'Status: {status}')

    return solver.Objective().Value(), status


# cplex solver is not supported by ortools by default.
def pulp_solve_ilp(EClasses, EEdges, root_classes, cost, acyclic, solver,
                   time_limit, verbose, target):
    prob = pulp.LpProblem("Egraph_Extraction", pulp.LpMinimize)

    # Variables
    x = pulp.LpVariable.dicts("x", EEdges.keys(), 0, 1,
                              cat='Integer')  # x[i] = 1 if e-node i is chosen
    # t = pulp.LpVariable.dicts("t", EClasses.keys(), 0,
    #                           cat='Integer')  # t[m] = 1 if e-class m is chosen
    t = pulp.LpVariable.dicts("t", EClasses.keys(), 0, 1)
    # Objective function
    if target is not None:
        prob += pulp.lpSum([cost[i] * x[i] for i in EEdges.keys()
                            ]) <= target, "TargetConstraint"
    else:
        prob += pulp.lpSum([cost[i] * x[i] for i in EEdges.keys()])

    # Constraints
    for eclass in root_classes:
        # one e-node should be chosen from the root e-class
        prob += pulp.lpSum([x[i] for i in EClasses[eclass]]) == 1

    for i, edges in EEdges.items():
        for edge in edges:
            prob += x[i] <= pulp.lpSum([x[j] for j in EClasses[edge]])

    if acyclic:
        eps = 1 / (len(EClasses) * 2)
        A = 100
        for i, edges in EEdges.items():
            for edge in edges:
                prob += t[get_eclass(i)] - t[edge] - eps + A * (1 - x[i]) >= 0

    # Solve the ILP
    if solver == 'cplex':
        solver = pulp.CPLEX_CMD(msg=verbose,
                                options=["set timelimit " + str(time_limit)])
        # solver.logPath = './exp_logs/cplex.log'
        status = prob.solve(solver=solver)
    elif solver == 'cbc':
        status = prob.solve(PULP_CBC_CMD(msg=verbose, timeLimit=time_limit))

    print(f'Status: {pulp.LpStatus[status]}')

    if status == 1:  # Check if the problem is solved optimally
        # Extract and print the solved variable values
        solved_x = [v for v in EEdges.keys() if pulp.value(x[v]) == 1]

        print("Solved x values:")
        print(solved_x)
    return pulp.value(prob.objective), pulp.LpStatus[status]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',
                        type=str,
                        default='examples/cunxi_test_egraph2.dot')
    parser.add_argument('--acyclic', action='store_true', default=False)
    parser.add_argument('--time_limit', type=int, default=60)
    parser.add_argument('--target', type=float, default=None)
    parser.add_argument('--solver',
                        type=str,
                        default='cplex',
                        choices=['cplex', 'cbc', 'scip'])
    parser.add_argument('--verbose', action='store_true', default=False)
    return parser.parse_args()


def main():
    args = get_args()
    # if args.verbose:
    #     logging.basicConfig(level=print)
    get_eclass_dict = None
    load_cost = args.input_file.endswith('.dot')
    egraph = EGraphData(args.input_file,
                        load_cost=load_cost,
                        drop_self_loops=False)
    EClasses, EEdges = egraph.input_dict['classes'], egraph.input_dict['nodes']
    if hasattr(egraph, 'root') and isinstance(egraph.root, list) and len(
            egraph.root) > 0:
        root_classes = [str(i) for i in egraph.root]
    elif hasattr(egraph, 'root') and isinstance(egraph.root, int):
        root_classes = [str(egraph.root)]
    elif 'root_' in args.input_file:
        pattern = r'root_(\d+)\.dot'
        match = re.search(pattern, args.input_file)
        root_classes = [match.group(1)]
    else:
        visited_classes = set()
        for e, c in EEdges.items():
            for i in c:
                visited_classes.add(i)
        root_classes = set(EClasses.keys()) - visited_classes
    print(f'Root classes: {root_classes}')

    if load_cost:
        # map enode to its label
        label_map = egraph.input_dict['labels']
        # map label to its cost
        cost_map = egraph.label_cost
        cost = {}
        for node in EEdges.keys():
            if label_map[node] in cost_map:
                cost[node] = cost_map[label_map[node]]
            else:
                cost[node] = cost_map['default']
    else:
        if hasattr(egraph, 'enode_cost'):
            cost = {}
            for node in EEdges.keys():
                cost[node] = egraph.enode_cost[egraph.enode_map[node]]
        else:
            cost = {node: 1 for node in EEdges.keys()}

    start_time = time.time()
    file_name = os.path.splitext(os.path.basename(args.input_file))[0] + '_ilp'
    file_name = f'logs/ilp_log/{file_name}_solver_{args.solver}_time_{args.time_limit}.json'
    log_name = file_name.replace('.json', '.log')
    kwargs = {
        'EClasses': EClasses,
        'EEdges': EEdges,
        'root_classes': root_classes,
        'cost': cost,
        'acyclic': args.acyclic,
        'solver': args.solver,
        'time_limit': args.time_limit,
        'verbose': args.verbose,
        'target': args.target
    }
    if args.solver in ['cplex', 'cbc']:
        cost, status = pulp_solve_ilp(**kwargs)
    else:
        cost, status = ortools_solve_ilp(**kwargs)
    log = {
        'cost': cost,
        'time_limit': args.time_limit,
        'runtime': time.time() - start_time,
        'solver': args.solver,
        'acyclic': args.acyclic,
        'input_file': args.input_file,
        'status': status
    }

    with open(file_name, 'w') as f:
        f.write(json.dumps(log) + '\n')


if __name__ == "__main__":
    main()
