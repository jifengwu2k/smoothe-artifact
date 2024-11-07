import numpy as np
import os
import json
from collections import defaultdict

# load egraph names
dataset_names = os.listdir('dataset')

data = defaultdict(list)
for dataset in dataset_names:
    for egraph in os.listdir(os.path.join('dataset', dataset)):
        if egraph.endswith('dot') or egraph.endswith('json'):
            data[dataset].append(egraph)

# load & process smoothe results
raw_smoothe_result = [{}, {}, {}]
smoothe_result = defaultdict(dict)
smoothe_diff = defaultdict(dict)
smoothe_time = defaultdict(dict)


def get_results(data):
    if 'inference_loss' in data:
        loss = np.array(data['inference_loss'])
    elif 'cost' in data:
        loss = np.array(data['cost'])
    time = np.array(data['time'])
    if len(time) == len(loss) + 1:
        time = time[1:]
    return np.min(loss[time < 60]), min(time.max(), 60)


for dataset in dataset_names:
    for i in range(3):
        raw_smoothe_result[i][dataset] = json.load(
            open(os.path.join('logs', dataset + f'_linear_{i}.json'), 'r'))
    for egraph_name in data[dataset]:
        temp_cost = []
        temp_time = []
        for i in range(3):
            egraph = raw_smoothe_result[i][dataset][egraph_name]
            cost, time = get_results(egraph)
            temp_cost.append(cost)
            temp_time.append(time)
        smoothe_result[dataset][egraph_name] = temp_cost
        smoothe_time[dataset][egraph_name] = temp_time
        # smoothe_diff[dataset][egraph_name] = np.max(temp)  - np.min(temp)

# load ILP results
TIME = 900
cbc_result = defaultdict(dict)
scip_result = defaultdict(dict)
cplex_result = defaultdict(dict)
cplex_optimal = defaultdict(dict)

cbc_time = defaultdict(dict)
scip_time = defaultdict(dict)
cplex_time = defaultdict(dict)
cplex_optimal_time = defaultdict(dict)
for dataset in data:
    for egraph in data[dataset]:
        if egraph.endswith('.json'):
            egraph_name = egraph[:-5]
        elif egraph.endswith('.dot'):
            egraph_name = egraph[:-4]
        else:
            raise ValueError
        cbc = json.load(
            open(
                os.path.join(
                    'logs/ilp_log/',
                    f'{egraph_name}_ilp_solver_cbc_time_{TIME}.json')))
        scip = json.load(
            open(
                os.path.join(
                    'logs/ilp_log/',
                    f'{egraph_name}_ilp_solver_scip_time_{TIME}.json')))
        cplex = json.load(
            open(
                os.path.join(
                    'logs/ilp_log',
                    f'{egraph_name}_ilp_solver_cplex_time_{TIME}.json')))
        cplex_10h = json.load(
            open(
                os.path.join(
                    'logs/ilp_log',
                    f'{egraph_name}_ilp_solver_cplex_time_{36000}.json')))

        if cbc['status'] == 'Not Solved':
            cbc_result[dataset][egraph] = float('nan')
        else:
            cbc_result[dataset][egraph] = cbc['cost']
        cbc_time[dataset][egraph] = cbc['runtime']

        if scip['status'] == 'INFEASIBLE':
            scip_result[dataset][egraph] = float('nan')
        else:
            scip_result[dataset][egraph] = scip['cost']
        scip_time[dataset][egraph] = scip['runtime']

        if cplex['status'] == 'Infeasible':
            cplex_result[dataset][egraph] = float('nan')
        else:
            cplex_result[dataset][egraph] = cplex['cost']
        cplex_time[dataset][egraph] = cplex['runtime']

        if cplex_10h['status'] == 'Infeasible':
            cplex_optimal[dataset][egraph] = float('nan')
        else:
            cplex_optimal[dataset][egraph] = cplex_10h['cost']
        cplex_optimal_time[dataset][egraph] = cplex_10h['runtime']

        smoothe_baseline = np.mean(smoothe_result[dataset][egraph])

# load greedy results
base_greedy_result = defaultdict(dict)
base_greedy_time = defaultdict(dict)

greedy_result = defaultdict(dict)
greedy_time = defaultdict(dict)
for dataset in dataset_names:
    path = os.path.join('logs/heuristic', dataset)
    for egraph in data[dataset]:
        if egraph.endswith('.json'):
            egraph_name = egraph[:-5]
        elif egraph.endswith('.dot'):
            egraph_name = egraph[:-4]
        else:
            raise ValueError
        r = json.load(
            open(os.path.join(path, f'{egraph_name}_faster_greedy.json')))
        greedy_result[dataset][egraph] = r['dag']
        greedy_time[dataset][egraph] = r['micros'] * 1e-6

        r = json.load(
            open(os.path.join(path, f'{egraph_name}_baseline_greedy.json')))
        base_greedy_result[dataset][egraph] = r['dag']
        base_greedy_time[dataset][egraph] = r['micros'] * 1e-6


# process data
def new_mean(l, method='geo'):
    l = np.array(l)
    l = l[np.isnan(l) == False]
    if method == 'geo':
        if l.max() < 0:
            l = -l
            geo_mean = l.prod()**(1.0 / len(l))
            geo_mean = -geo_mean
        else:
            geo_mean = l.prod()**(1.0 / len(l))
        return geo_mean
    elif method == 'arith':
        arith_mean = l.mean()
        return arith_mean


norm_results = defaultdict(list)
norm_worst = defaultdict(list)
fails = defaultdict(list)
for dataset in data:
    cplex = []
    cbc = []
    scip = []
    greedy = []
    base_greedy = []
    smoothe1 = []
    smoothe2 = []
    smoothe3 = []
    for egraph in smoothe_result[dataset]:
        optimal = cplex_optimal[dataset][egraph]
        abs_optimal = abs(optimal)
        smoothe1.append((smoothe_result[dataset][egraph][0]) / abs_optimal)
        smoothe2.append((smoothe_result[dataset][egraph][1]) / abs_optimal)
        smoothe3.append((smoothe_result[dataset][egraph][2]) / abs_optimal)
        cplex.append((cplex_result[dataset][egraph]) / abs_optimal)
        cbc.append((cbc_result[dataset][egraph]) / abs_optimal)
        scip.append((scip_result[dataset][egraph]) / abs_optimal)
        greedy.append((greedy_result[dataset][egraph]) / abs_optimal)
        base_greedy.append((base_greedy_result[dataset][egraph]) / abs_optimal)
        # genetic.append((genetic_result[dataset][egraph]-optimal) / abs_optimal)
        # random.append((random_result[dataset][egraph]-optimal) / abs_optimal)

    norm_results['SmoothE1'].append(new_mean(smoothe1))
    norm_results['SmoothE2'].append(new_mean(smoothe2))
    norm_results['SmoothE3'].append(new_mean(smoothe3))
    norm_results['CPLEX'].append(new_mean(cplex))
    norm_results['CBC'].append(new_mean(cbc))
    norm_results['SCIP'].append(new_mean(scip))
    norm_results['Greedy'].append(new_mean(greedy))
    norm_results['BaseGreedy'].append(new_mean(base_greedy))

    fails['CBC'].append(np.isnan(cbc).sum())
    fails['SCIP'].append(np.isnan(scip).sum())

    norm_worst['SmoothE1'].append(np.max(smoothe1))
    norm_worst['SmoothE2'].append(np.max(smoothe2))
    norm_worst['SmoothE3'].append(np.max(smoothe3))
    norm_worst['CPLEX'].append(np.max(cplex))
    norm_worst['CBC'].append(np.max(cbc))
    norm_worst['SCIP'].append(np.max(scip))
    norm_worst['Greedy'].append(np.max(greedy))
    norm_worst['BaseGreedy'].append(np.max(base_greedy))

for i in range(len(data)):
    norm = [norm_results[f'SmoothE{k}'][i] for k in range(1, 4)]
    norm_w = [norm_worst[f'SmoothE{k}'][i] for k in range(1, 4)]

    norm_results['avg_SmoothE'].append(new_mean(norm))
    norm_results['avg_SmoothE_diff'].append(np.max(norm) - np.min(norm))

    norm_results['worst_SmoothE'].append(new_mean(norm_w))
    norm_results['worst_SmoothE_diff'].append(np.max(norm_w) - np.min(norm_w))


# output results
def str2(num):
    if num < 0:
        return f'${(num+1) / abs(num) * 100:.1f}\%$'
    elif num < 2:
        return f'${abs(num-1)*100:.1f}\%$'
    elif num >= 2:
        return f'${(num):.1f}\\times$'
    else:
        return 'Inf'


def str_dec(num):
    return f'{num*100:.1f}\%'


def str_sec(dic):
    values = [v for v in dic.values()]
    if isinstance(values[0], list):
        num0 = np.mean([v[0] for v in values])
        num1 = np.mean([v[1] for v in values])
        num2 = np.mean([v[2] for v in values])
        l = [num0, num1, num2]
        return f'${np.mean(l):.1f} {{\scriptscriptstyle \pm {(np.max(l)- np.min(l))/2:.1f}}}$'
    else:
        num = np.clip(values, 0, 900).mean()
        return f'{num:.1f}'


def print_dataset(dataset, i):
    # CPLEX
    cplex_time_str = str_sec(cplex_time[dataset])
    cplex_norm = f"{str2(norm_worst['CPLEX'][i])} / {str2(norm_results['CPLEX'][i])}"
    cplex_cell = f"{cplex_time_str}<br>{cplex_norm}"

    # SCIP
    scip_time_str = str_sec(scip_time[dataset])
    if fails['SCIP'][i] > 0:
        scip_time_str += f' ({fails["SCIP"][i]})'
    scip_norm = f"{str2(norm_worst['SCIP'][i])} / {str2(norm_results['SCIP'][i])}"
    scip_cell = f"{scip_time_str}<br>{scip_norm}"

    # CBC
    cbc_time_str = str_sec(cbc_time[dataset])
    if fails['CBC'][i] > 0:
        cbc_time_str += f' ({fails["CBC"][i]})'
    cbc_norm = f"{str2(norm_worst['CBC'][i])} / {str2(norm_results['CBC'][i])}"
    cbc_cell = f"{cbc_time_str}<br>{cbc_norm}"

    # Base Greedy
    base_greedy_time_str = str_sec(base_greedy_time[dataset])
    base_greedy_norm = f"{str2(norm_worst['BaseGreedy'][i])} / {str2(norm_results['BaseGreedy'][i])}"
    base_greedy_cell = f"{base_greedy_time_str}<br>{base_greedy_norm}"

    # Greedy
    greedy_time_str = str_sec(greedy_time[dataset])
    greedy_norm = f"{str2(norm_worst['Greedy'][i])} / {str2(norm_results['Greedy'][i])}"
    greedy_cell = f"{greedy_time_str}<br>{greedy_norm}"

    # SmoothE
    smoothe_time_str = str_sec(smoothe_time[dataset])
    smoothe_norm = (
        f"{str2(norm_results['worst_SmoothE'][i])} ± {str_dec(norm_results['worst_SmoothE_diff'][i]/2)} / "
        f"{str2(norm_results['avg_SmoothE'][i])} ± {str_dec(norm_results['avg_SmoothE_diff'][i]/2)}"
    )
    smoothe_cell = f"{smoothe_time_str}<br>{smoothe_norm}"

    return f'| {dataset} | {cplex_cell} | {scip_cell} | {cbc_cell} | {greedy_cell} | {base_greedy_cell} | {smoothe_cell} |\n'


index_map = {dataset: i for i, dataset in enumerate(dataset_names)}

res = '## Table 2 \n'
res += '| Dataset | CPLEX | SCIP | CBC | Heuristic | Heuristic+ | SmoothE |\n'
res += '|---|---|---|---|---|---|---|\n'
for dataset in ['diospyros', 'flexc', 'impress', 'rover', 'tensat']:
    res += print_dataset(dataset, index_map[dataset])


def str_num(num, digit=2):
    if digit == 2:
        return f'{num:.3f}'
    elif digit == 1:
        if num > 900:
            num = 900
        return f'{num:.1f}'


# Iterate over each egraph and append rows to the Markdown table
def print_breakdown(dataset):
    ret = ''
    mapping = {
        'tensat': [
            'nasneta.json', 'nasrnn.json', 'bert.json', 'vgg.json',
            'resnet50.json'
        ],
        'rover': [
            'fir_8_tap_5iteration_egraph.json',
            'fir_8_tap_6iteration_egraph.json',
            'fir_8_tap_7iteration_egraph.json',
            'fir_8_tap_8iteration_egraph.json',
            'box_filter_3iteration_egraph.json',
            'box_filter_4iteration_egraph.json',
            'box_filter_5iteration_egraph.json',
            'mcm_3_7_21_original_8iteration_egraph.json',
            'mcm_3_7_21_original_9iteration_egraph.json',
        ]
    }
    for egraph in mapping[dataset]:
        egraph_name = egraph.split('.')[0]
        if 'fir' in egraph_name:
            egraph_name = 'fir\_' + egraph_name[10]
        elif 'box' in egraph_name:
            egraph_name = 'box\_' + egraph_name[11]
        elif 'mcm' in egraph_name:
            egraph_name = 'mcm\_' + egraph_name[20]

        # CPLEX
        cplex_val = str_num(cplex_result[dataset][egraph])
        cplex_time_val = str_num(cplex_time[dataset][egraph], 1)
        cplex_cell = f'{cplex_val} / {cplex_time_val}'

        # SCIP
        scip_val = str_num(scip_result[dataset][egraph])
        scip_time_val = str_num(scip_time[dataset][egraph], 1)
        scip_cell = f'{scip_val} / {scip_time_val}'

        # CBC
        cbc_val = str_num(cbc_result[dataset][egraph])
        cbc_time_val = str_num(cbc_time[dataset][egraph], 1)
        cbc_cell = f'{cbc_val} / {cbc_time_val}'

        # Base Greedy
        base_greedy_val = str_num(base_greedy_result[dataset][egraph])
        base_greedy_time_val = str_num(base_greedy_time[dataset][egraph], 1)
        base_greedy_cell = f'{base_greedy_val} / {base_greedy_time_val}'

        # Greedy
        greedy_val = str_num(greedy_result[dataset][egraph])
        greedy_time_val = str_num(greedy_time[dataset][egraph], 1)
        greedy_cell = f'{greedy_val} / {greedy_time_val}'

        # SmoothE
        smoothe_mean = str_num(np.mean(smoothe_result[dataset][egraph]))
        smoothe_diff = str_num((np.max(smoothe_result[dataset][egraph]) -
                                np.min(smoothe_result[dataset][egraph])) / 2)
        smoothe_time_mean = str_num(np.mean(smoothe_time[dataset][egraph]), 1)
        smoothe_time_diff = str_num(
            (np.max(smoothe_time[dataset][egraph]) -
             np.min(smoothe_time[dataset][egraph])) / 2, 1)
        smoothe_cell = f'{smoothe_mean} ± {smoothe_diff} / {smoothe_time_mean} ± {smoothe_time_diff}'

        # Combine all cells into a single Markdown row
        ret += f'| {egraph_name} | {cplex_cell} | {scip_cell} | {cbc_cell} | {base_greedy_cell} | {greedy_cell} | {smoothe_cell} |\n'
    return ret


# output Table 3
res += '\n'
res += '| Egraph | CPLEX | SCIP | CBC | Heuristic | Heuristic+ | SmoothE |\n'
res += '|---|---|---|---|---|---|---|\n'

res += print_breakdown('tensat')
res += '|---|---|---|---|---|---|---|\n'
res += print_breakdown('rover')
res = res.replace('nan', 'Fails')

# output Table 4
res += '\n'
res += '## Table 4 \n'
res += '| Dataset | CPLEX | SCIP | CBC | Heuristic | Heuristic+ | SmoothE |\n'
res += '|---|---|---|---|---|---|---|\n'
for dataset in ['set', 'maxsat']:
    res += print_dataset(dataset, index_map[dataset])

res = res.replace('.000', '')
res = res.replace('Inf / Inf', 'Failed')
res = res.replace('Inf', 'Failed')
res = res.replace('100.0\%', '2.0\\times')

# output results to table.md
with open('table.md', 'w') as f:
    f.write(res)
