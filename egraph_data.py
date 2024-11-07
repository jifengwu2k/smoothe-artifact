import re
import os
import json
import pickle
import logging
from collections import defaultdict
from graphviz import Source

import torch
import torch.nn as nn
import torch.distributions as dist


def bool_to_index(bool_tensor):
    range_tensor = torch.arange(bool_tensor.shape[-1], device='cuda')
    range_tensor = range_tensor.expand(bool_tensor.shape)
    return range_tensor[bool_tensor].tolist()


class ENode:

    def __init__(self, eclass_id, belong_eclass_id, label):
        self.eclass_id = eclass_id
        self.belong_eclass_id = belong_eclass_id
        self.normalized_prob = 0
        self.depth = 0
        self.label = label

    def __repr__(self) -> str:
        return f'ENode : {self.eclass_id}, belong_eclass_id: {self.belong_eclass_id}'


class EClass:

    def __init__(self, enode_id, hidden_dim):
        self.enode_id = enode_id
        self.in_nodes = defaultdict(int)
        self.visited_in_nodes = defaultdict(int)
        self.hidden_dim = hidden_dim
        # included = True if at least one in_node is sampled
        # or self is a source eclass
        self.included = False
        self.predecessor_embeddings = []
        self.predecessor_enodes = []
        self.normalized_prob = 1

    def add_in_node(self, in_node):
        self.in_nodes[in_node] += 1

    def add_visited_in_node(self, in_node):
        self.visited_in_nodes[in_node] += 1

    def __repr__(self) -> str:
        return f'EClass: {self.enode_id} in_nodes: {self.in_nodes} visited_in_nodes: {self.visited_in_nodes}'


class MLP(nn.Module):

    def __init__(self, input_width):
        super(MLP, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_width, 64, bias=False),
                                 nn.ReLU(), nn.Linear(64, 64, bias=False),
                                 nn.ReLU(), nn.Linear(64, 8, bias=False),
                                 nn.ReLU(), nn.Linear(8, 1, bias=False))

    def forward(self, x):
        return self.net(x)


class EGraphData:

    def __init__(self,
                 input_file,
                 hidden_dim=32,
                 load_cost=False,
                 compress=False,
                 drop_self_loops=False,
                 device='cuda'):
        self.eclasses = {}
        self.enodes = {}
        self.hidden_dim = hidden_dim
        self.load_cost = load_cost
        self.label_cost = False
        self.drop_self_loops = drop_self_loops
        self.compress = compress
        self.device = device

        if load_cost:
            head, _ = os.path.split(input_file)
            self.load_cost_from_file(head)

        if input_file.endswith('.dot'):
            self.from_dot_file(input_file)
        elif input_file.endswith('.json'):
            self.from_json_file(input_file)
        elif input_file.endswith('.pickle'):
            self.from_pickle_file(input_file)
        else:
            raise NotImplementedError

        # if quad_cost is not None:
        #     self.quad_cost = pickle.load(open(quad_cost, 'rb'))
        # if drop_self_loops:
        #     self.drop_self_loops()
        # fix the mapping and cost_per_node
        # self.export_egraph(input_file)  # for debug preprocess
        self.set_cost_per_node()

    def __repr__(self) -> str:
        return f'EGraph: EClass {self.eclasses} ENode {self.enodes}'

    # def drop_self_loops(self):
    #     enode_tobe_removed = set()
    #     for enode in self.enodes:
    #         for eclass in self.enodes[enode].eclass_id:
    #             if enode in self.eclasses[eclass].enode_id:
    #                 enode_tobe_removed.add(enode)
    #     for enode in enode_tobe_removed:
    #         del self.enodes[enode]
    #         for eclass in self.eclasses:
    #             if enode in self.eclasses[eclass].in_nodes:
    #                 del self.eclasses[eclass].in_nodes[enode]

    def export_egraph(self, input_file_name):
        #indeed int index eclass and enode,
        #convert to str type due to the compulsory conversion of dumping to json file
        debug_dict = {}
        enodes_debug = {}
        eclasses_debug = {}
        for enode_id, enode in self.enodes.items():
            enodes_debug[str(enode_id)] = {
                'children_eclass_id':
                [str(eclass_id) for eclass_id in list(enode.eclass_id)],
                'belong to eclass':
                str(enode.belong_eclass_id),
                'cost':
                self.enode_cost[enode_id]
            }
        for eclass_id, eclass in self.eclasses.items():
            eclasses_debug[str(eclass_id)] = {
                'contain enode_id':
                [str(enode_id) for enode_id in list(eclass.enode_id)],
                'pointed by enode_id':
                [str(in_nodes) for in_nodes in list(eclass.in_nodes)]
            }
        debug_dict["self.enodes"] = enodes_debug
        debug_dict["self.eclasses"] = eclasses_debug
        try:
            debug_dict["root_eclasses"] = [
                str(self.class_mapping[r]) for r in self.root
            ]
        except AttributeError:
            pass
        file_path = '/'.join(input_file_name.split("/")[1:])
        dir_name = 'export_egraph/' + '/'.join(file_path.split("/")[:-1])
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_name = file_path.split('.')[-2].split('/')[-1]
        export_file_name = dir_name + "/" + file_name + ".json"
        with open(export_file_name, 'w') as f:
            json.dump(debug_dict, f, indent=4)

    def from_dict(self, input_dict):
        self.input_dict = input_dict
        enode_map = {}
        node_to_class_id = {}  #tmp reverse dict
        self.class_mapping = {}
        enode_count = 0
        eclass_count = 0

        enode_cost = {}
        for enode_id, eclass_id in input_dict['nodes'].items():
            input_dict['nodes'][enode_id] = set(eclass_id)
            label = input_dict['labels'][enode_id]
            if label in self.label_cost:
                enode_cost[enode_id] = self.label_cost[label]
            else:
                enode_cost[enode_id] = self.label_cost['default']

        def drop_self_loops():
            enode_tobe_removed = set()
            for enode_id, eclass_id in input_dict['nodes'].items():
                for eclass in eclass_id:
                    if enode_id in input_dict['classes'][eclass]:
                        enode_tobe_removed.add((enode_id, eclass))
                        break

            for enode, eclass in enode_tobe_removed:
                del input_dict['nodes'][enode]
                del input_dict['labels'][enode]
                assert enode in input_dict['classes'][eclass]
                input_dict['classes'][eclass].remove(enode)
                self.raw_nodes_mapping[enode] = []
            return len(enode_tobe_removed)

        def compress():
            inv_class2node = defaultdict(list)
            for enode_id, eclass_id in input_dict['nodes'].items():
                for eclass in eclass_id:
                    inv_class2node[eclass].append(enode_id)

            merged_eclass_id = []
            merged_enode_id = []
            for eclass_id, enode_id in input_dict['classes'].items():
                assert len(enode_id) > 0
                if len(enode_id) == 1:
                    parent_enodes = inv_class2node[eclass_id]
                    if len(parent_enodes) != 1:
                        # skip potential root eclass
                        # skip potential cse
                        continue
                    if hasattr(self, 'root') and eclass_id in self.root:
                        continue
                    child_eclasses = input_dict['nodes'][enode_id[0]]

                    for parent_enode in set(parent_enodes):
                        input_dict['nodes'][parent_enode] = input_dict[
                            'nodes'][parent_enode].union(child_eclasses)
                        input_dict['nodes'][parent_enode].remove(eclass_id)

                        self.raw_nodes_mapping[
                            parent_enode] = self.raw_nodes_mapping[
                                parent_enode] + self.raw_nodes_mapping[
                                    enode_id[0]]
                        self.raw_nodes_mapping[enode_id[0]] = []

                    for child_class in child_eclasses:
                        inv_class2node[child_class] = inv_class2node[
                            child_class] + parent_enodes
                        inv_class2node[child_class].remove(enode_id[0])

                    merged_eclass_id.append(eclass_id)
                    merged_enode_id.append(enode_id[0])

            for eclass_id in merged_eclass_id:
                del input_dict['classes'][eclass_id]
            for enode_id in merged_enode_id:
                del input_dict['nodes'][enode_id]
                del input_dict['labels'][enode_id]
            return len(merged_eclass_id)

        # preprocessing
        self.raw_num_enodes = len(input_dict['nodes'])
        self.raw_num_eclasses = len(input_dict['classes'])
        self.raw_nodes_mapping = {k: [k] for k in input_dict['nodes']}
        if self.drop_self_loops and len(input_dict['classes']) > 10:
            assert self.compress
            total_self_loops = 0
            total_merged = 0
            while True:
                l1 = drop_self_loops()
                l2 = compress()
                total_self_loops += l1
                total_merged += l2
                if l1 + l2 == 0:
                    break
            logging.info(f'Deleted {total_self_loops} self-loops nodes')
            logging.info(f'Merged {total_merged} singleton classes')

        self.enode_cost = [1] * self.raw_num_enodes

        for enode, v in self.raw_nodes_mapping.items():
            if v:
                enode_map[enode] = enode_count
                enode_count += 1
        preprocessed_num_nodes = enode_count
        for enode, v in self.raw_nodes_mapping.items():
            if not v:
                enode_map[enode] = enode_count
                enode_count += 1
        # self.raw_nodes_mapping = {
        #     enode_map[k]: [enode_map[v] for v in vs]
        #     for k, vs in self.raw_nodes_mapping.items()
        # }
        nodes2raw_key = []
        nodes2raw_value = []
        for k, vs in self.raw_nodes_mapping.items():
            for v in vs:
                nodes2raw_key.append(enode_map[k])
                nodes2raw_value.append(enode_map[v])
        nodes2raw_key = torch.tensor(nodes2raw_key,
                                     dtype=torch.long,
                                     device=self.device)
        nodes2raw_value = torch.tensor(nodes2raw_value,
                                       dtype=torch.long,
                                       device=self.device)
        self.nodes2raw = torch.sparse_coo_tensor(
            indices=torch.stack([nodes2raw_key, nodes2raw_value]),
            values=torch.ones(len(nodes2raw_key), device=self.device),
            size=(preprocessed_num_nodes, self.raw_num_enodes))

        for eclass_id, enode_id in input_dict['classes'].items():
            # map enode_id (str) to enode_num_id (int)
            enode_num_id = []
            for node in enode_id:
                enode_num_id.append(enode_map[node])
                node_to_class_id[enode_map[node]] = eclass_count
            self.eclasses[eclass_count] = EClass(enode_num_id, self.hidden_dim)
            self.class_mapping[
                eclass_id] = eclass_count  # map eclass_id(str) to eclass_id(int)
            eclass_count += 1

        for (enode_id,
             eclass_id), (_, label) in zip(input_dict['nodes'].items(),
                                           input_dict['labels'].items()):
            self.enodes[enode_map[enode_id]] = ENode(
                eclass_id={self.class_mapping[i]
                           for i in eclass_id},
                belong_eclass_id=node_to_class_id[enode_map[enode_id]],
                label=label)
            # if self.load_cost or self.label_cost:
            #     self.enode_cost[enode_map[enode_id]] = enode_cost[enode_id]
            for eclass in eclass_id:
                self.eclasses[self.class_mapping[eclass]].add_in_node(
                    enode_map[enode_id])

        for enode_id in self.raw_nodes_mapping.keys():
            self.enode_cost[enode_map[enode_id]] = enode_cost[enode_id]

        self.enode_map = enode_map
        self.processed_cost_per_node = self.nodes2raw @ torch.tensor(
            self.enode_cost, dtype=torch.float, device=self.device)

        return self

    def from_json_file(self, json_file):
        with open(json_file, 'r') as f:
            input_dict = json.load(f)
        # the format from the extraction gym repo
        # (https://github.com/egraphs-good/extraction-gym)

        if 'classes' in input_dict:
            # format 1)
            if isinstance(input_dict['classes'], list):
                input_dict['classes'] = {
                    str(k): [str(vi) for vi in v]
                    for k, v in enumerate(input_dict['classes'])
                }
            if isinstance(input_dict['nodes'], list):
                input_dict['nodes'] = {
                    str(k): [str(vi) for vi in v]
                    for k, v in enumerate(input_dict['nodes'])
                }
            self.from_dict(input_dict)
        else:
            # enode_map = {}
            # assert 'root_eclasses' in input_dict
            # self.enode_cost = [0] * len(input_dict['nodes'])
            class_out_list = defaultdict(list)
            label_cost = defaultdict(int)
            # class_in_list = defaultdict(list)

            new_dict = {'nodes': {}, 'classes': {}, 'labels': {}}
            for i, node in enumerate(input_dict['nodes']):
                cur_enode = input_dict['nodes'][node]
                pattern1 = r'(\d+)__\d+'
                pattern2 = r'(\d+).\d+'
                eclass_list = []
                for child in cur_enode['children']:
                    p1_result = re.findall(pattern1, child)
                    p2_result = re.findall(pattern2, child)
                    if len(p1_result) > 0:
                        eclass_list.append(p1_result[0])
                    else:
                        eclass_list.append(p2_result[0])
                new_dict['nodes'][node] = eclass_list
                # new_dict['labels'][node] = cur_enode['op']
                new_dict['labels'][node] = node

                class_out_list[cur_enode['eclass']].append(node)
                label_cost[node] = cur_enode['cost']
                # self.enode_cost[i] = cur_enode['cost']
            new_dict['classes'] = class_out_list
            self.label_cost = label_cost

            # enode_map[node] = i
            # self.enodes[i] = ENode(eclass_list)

            # for i, _class in enumerate(class_out_list):
            #     self.eclasses[_class] = EClass(class_out_list[_class],
            #                                    self.hidden_dim)
            #     if _class in class_in_list:
            #         self.eclasses[_class].in_nodes = class_in_list[_class]

            # self.enode_map = enode_map
            # self.class_mapping = dict(
            #     zip(self.eclasses.keys(), range(len(self.eclasses))))
            self.root = input_dict['root_eclasses']
            self.from_dict(new_dict)
            # self.root = [
            #     self.class_mapping[r] for r in input_dict['root_eclasses']
            # ]

        return self

    def from_pickle_file(self, pickle_file):
        with open(pickle_file, 'rb') as f:
            input_dict = pickle.load(f)
        if isinstance(input_dict['classes'], list):
            input_dict['classes'] = {
                str(k): [str(vi) for vi in v]
                for k, v in enumerate(input_dict['classes'])
            }
        if isinstance(input_dict['nodes'], list):
            input_dict['nodes'] = {
                str(k): [str(vi) for vi in v]
                for k, v in enumerate(input_dict['nodes'])
            }
        self.from_dict(input_dict)
        return self

    def from_dot_file(self, dot_file):
        src = Source.from_file(dot_file)

        # Parse the graph into a more usable structure
        graph = src.source.split('\n')[2:-1]

        # Initialize our node and class lists
        classes = defaultdict(list)
        nodes = defaultdict(list)
        labels = {}

        # Now parse the lines
        pattern = r'[^\s]+=[^\s]+'
        for line in graph:
            line = line.strip()
            if line.startswith('subgraph'):
                # This is a new class
                class_id = int(re.split('_| ', line)[-2])
                # logging.info(f'Class {class_id}')
            elif line.startswith('}'):
                # This is the end of a class
                continue
            elif re.match(pattern, line):
                # This is graph level label
                # logging.info(f'Graph label {line}')
                continue
            elif '->' in line:
                # This is an edge
                src_node = re.split(' -> |:', line)[0]
                dst_class = line.split(' -> ')[1].split('.')[0]
                nodes[src_node].append(dst_class)
                # logging.info(f'Edge from enode {src_node} to eclass {dst_class}')
            else:
                # This is a node
                node_id = line.split('[')[0]
                class_id = node_id.split('.')[0]
                label = line.split('label = ')[1].replace('"', '').replace(
                    ']', '').strip()
                labels[node_id] = label
                classes[class_id].append(node_id)
                nodes[node_id] = []
                # logging.info(f'Node {node_id} with label {label}')

        input_dict = {
            "classes": classes,
            "nodes": nodes,
            "labels": labels,
        }
        self.from_dict(input_dict)
        return self

    def load_cost_from_file(self, path):
        # data = open(cost_file, "r").read()
        # lines = data.split("\n")
        # result = {}
        # for line in lines:
        #     # If the line contains "=>"
        #     if "=>" in line:
        #         # Split the line at "=>" and take the first part as the key and the second part as the value
        #         key = line.split("=>")[0].strip().replace("Math::", "").replace(
        #             "(..)", "")
        #         value = int(line.split("=>")[1].split(",")[0].strip())

        #         # Add the key-value pair to the dictionary
        #         result[key] = value
        cost = open(os.path.join(path, 'cost.txt'), "r").read()
        language = open(os.path.join(path, 'language.txt'), "r").read()

        matches1 = re.findall(r'\w+::(\w+)\(.+\) => (-?\d+(\.\d+)?),', cost)
        dict1 = {name: float(value) for name, value, _ in matches1}

        # Now, let's verify which keys from string2 exist in dict1
        matches2 = re.findall(r'"(.+)" = (\w+)\(', language)
        dict2 = {
            name: dict1[value]
            for name, value in matches2 if value in dict1
        }
        if 'default' not in dict2:
            dict2['default'] = 0
        self.label_cost = dict2

    def class_to_id(self, classes):
        class_ids = []
        for eclass in bool_to_index(classes):
            for k, v in self.class_mapping.items():
                if v == eclass:
                    class_ids.append(k)
                    break
        return class_ids

    def node_to_id(self, nodes):
        node_ids = []
        for enode in bool_to_index(nodes):
            for k, v in self.enode_map.items():
                if v == enode:
                    node_ids.append(k)
                    break
        return node_ids

    def set_cost_per_node(self):
        cost_per_node = []
        if hasattr(self, 'enode_cost'):
            cost_per_node = torch.tensor(self.enode_cost).float().to(
                self.device)
        else:
            cost_per_node = torch.empty(len(self.enodes)).to(self.device)
            cost_per_node.zero_()
            cost_per_node += 1
        self.cost_per_node = nn.Parameter(cost_per_node, requires_grad=False)

    @torch.no_grad()
    def init_mlp_cost(self, mlp_weight_file):
        self.mlp = MLP(self.raw_num_enodes)

        for param in self.mlp.parameters():
            param.requires_grad = False
        # self.mlp.net[-1].weight.abs_()

        if os.path.exists(mlp_weight_file):
            self.mlp.load_state_dict(torch.load(mlp_weight_file))
        else:
            logging.warning(f'{mlp_weight_file}_mlp.pth not found')
            logging.warning(f'Initializing mlp with random weights')
            torch.save(self.mlp.state_dict(), mlp_weight_file)
        self.mlp.to(self.device)

    def init_quad_cost(self, quad_cost_file):
        quad_cost = pickle.load(open(quad_cost_file, 'rb'))
        row = []
        col = []
        val = []
        for (r, c), cost in quad_cost.items():
            row.append(self.enode_map[r])
            col.append(self.enode_map[c])
            val.append(cost)

        row = torch.tensor(row, dtype=torch.long, device=self.device)
        col = torch.tensor(col, dtype=torch.long, device=self.device)
        val = torch.tensor(val, dtype=torch.float, device=self.device)
        self.quad_cost_mat = torch.sparse_coo_tensor(
            indices=torch.stack([row, col]),
            values=val,
            size=(self.raw_num_enodes, self.raw_num_enodes))

    def mlp_cost(self, enodes):
        mlp_loss = self.mlp(enodes).squeeze(-1)
        return mlp_loss + self.linear_cost(enodes)

    def quad_cost(self, enodes):
        quad_loss = enodes @ self.quad_cost_mat
        quad_loss *= enodes
        return quad_loss.sum(dim=1) + self.linear_cost(enodes)

    def linear_cost(self, enodes):
        linear_loss = (self.cost_per_node * enodes).sum(dim=1)
        return linear_loss
