import os
import json
import torch
from egraph_data import EGraphData
import re
from collections import defaultdict


def find_root_classes(egraph):  #int index
    # find the int index root classes
    if (hasattr(egraph, "root") and isinstance(egraph.root, list)
            and len(egraph.root) > 0):
        root_classes = [egraph.class_mapping[i] for i in egraph.root]
    elif hasattr(egraph, "root") and isinstance(egraph.root, int):
        root_classes = [egraph.class_mapping[egraph.root]]
    else:
        root_classes = []
        for eclass_id, eclass_info in egraph.eclasses.items():
            if len(eclass_info.in_nodes) == 0:
                root_classes.append(eclass_id)
            else:
                in_enodes = set(eclass_info.in_nodes.keys())
                contain_enodes = set(eclass_info.enode_id)
                " This eclass doesn't pointed by enodes from other eclass but by its own enodes."
                " And not all its enodes points to it"
                " Occurred in diospyros"
                if len(in_enodes) < len(contain_enodes) and in_enodes.issubset(
                        contain_enodes):
                    root_classes.append(eclass_id)
    return root_classes


# def get_eclass(enode):
#     if "." in enode:
#         return enode.split(".")[0]
#     elif "__" in enode:
#         return enode.split("__")[0]
#     else:
#         raise ValueError("Not correct enode format")

##string index preprocess, now we adopt int index, so we abandon this function
# def data_preprocess(input_file,
#                     load_cost: bool,
#                     quad_cost_file=None,
#                     mlp_cost_file=None,
#                     device='cuda'):
#     egraph = EGraphData(input_file, load_cost=load_cost, device=device)
#     EClasses, EEdges = egraph.input_dict["classes"], egraph.input_dict["nodes"]
#     if (hasattr(egraph, "root") and isinstance(egraph.root, list)
#             and len(egraph.root) > 0):
#         root_classes = [str(i) for i in egraph.root]
#     elif hasattr(egraph, "root") and isinstance(egraph.root, int):
#         root_classes = [str(egraph.root)]
#     elif 'root_' in input_file:
#         pattern = r'root_(\d+)\.dot'
#         match = re.search(pattern, input_file)
#         root_classes = [match.group(1)]
#     else:
#         visited_classes = set()
#         for e, c in EEdges.items():
#             for i in c:
#                 visited_classes.add(i)
#         root_classes = set(EClasses.keys()) - visited_classes

#     if load_cost:
#         # map enode to its label
#         label_map = egraph.input_dict["labels"]
#         # map label to its cost
#         cost_map = egraph.label_cost
#         cost = {}
#         for node in EEdges.keys():
#             if label_map[node] in cost_map:
#                 cost[node] = cost_map[label_map[node]]
#             else:
#                 cost[node] = cost_map["default"]
#     else:
#         if hasattr(egraph, "enode_cost"):
#             cost = {}
#             for node in EEdges.keys():
#                 cost[node] = egraph.enode_cost[egraph.enode_map[node]]
#         else:
#             cost = {node: 1 for node in EEdges.keys()}

#     egraph.node_to_class = {}
#     for eclass, enodes in EClasses.items():
#         egraph.node_to_class.update({enode: eclass for enode in enodes})
#     #define for easy to call
#     egraph.EClasses = EClasses
#     egraph.EEdges = EEdges
#     egraph.root_classes = root_classes
#     egraph.cost = cost
#     egraph.mlp = None
#     egraph.quad_cost_mat = None
#     if quad_cost_file:
#         egraph.init_quad_cost(quad_cost_file)
#     elif mlp_cost_file:
#         egraph.init_mlp_cost(mlp_cost_file)
#     return egraph


def egraph_preprocess(args):
    device = 'cuda'
    egraph = EGraphData(args.input_file,
                        load_cost=args.load_cost,
                        drop_self_loops=False,
                        device=device)

    root_classes = find_root_classes(egraph)
    setattr(egraph, "root_classes", root_classes)
    setattr(egraph, "mlp", None)
    setattr(egraph, "quad_cost_mat", None)
    if args.quad_cost_file:
        egraph.init_quad_cost(args.quad_cost_file)
    elif args.mlp_cost_file:
        egraph.init_mlp_cost(args.mlp_cost_file)
    enodes_tensor = torch.zeros(1, len(egraph.enode_map), dtype=torch.float32)
    enodes_tensor = enodes_tensor.to(device)
    setattr(egraph, "enodes_tensor", enodes_tensor)
    #calculate the uniform probability for each e-node
    node_probability = {}
    for eclass, enodes in egraph.eclasses.items():
        node_probability.update(
            {enode: 1 / len(enodes.enode_id)
             for enode in enodes.enode_id})
    setattr(egraph, "node_probability", node_probability)
    parents = defaultdict(list)
    for node in egraph.enodes:
        for child_class in egraph.enodes[node].eclass_id:
            parents[child_class].append(node)
    setattr(egraph, "parents", parents)
    leaf_nodes = []
    for node in egraph.enodes:
        if egraph.enodes[node].eclass_id == [] or egraph.enodes[
                node].eclass_id == set():
            leaf_nodes.append(node)
    setattr(egraph, "leaf_nodes", leaf_nodes)
    return egraph


def save_files(best_cost, best_time, cost_time_dic, method, quad_cost,
               mlp_cost, input_file, exp_id):
    total_file_name = input_file
    parts = total_file_name.rsplit("/", 3)
    # dir_name = parts[-3] + "/" + parts[-2] if len(parts) > 1 else None
    dir_name = 'logs/genetic/' + parts[-2] if len(parts) > 1 else None
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    base_name = os.path.splitext(os.path.join(dir_name, parts[-1]))[0]
    if quad_cost:
        file_suffix = "_quad_cost"
    elif mlp_cost:
        file_suffix = "_mlp_cost"
    else:
        file_suffix = "_linear_cost"
    file_suffix += f"_{exp_id}"
    exp_file_name = method + file_suffix + ".json"
    saved_file_path = base_name + file_suffix + ".json"

    save_data = {
        "name": input_file,
        "Best Cost": best_cost,
        "Best Cost Time": f"{best_time}s",
    }
    with open(saved_file_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)
