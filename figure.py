import re
import argparse
import os
import json
from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np


def new_mean(l, method='geo'):
    l = np.array(l)
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


def get_results(data):
    if 'inference_loss' in data:
        loss = np.array(data['inference_loss'])
    elif 'cost' in data:
        loss = np.array(data['cost'])
    time = np.array(data['time'])
    if len(time) == len(loss) + 1:
        time = time[1:]
    return np.min(loss[time < 60]), min(time.max(), 60)


# Figure 6
def parse_cplex_log(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    results = []
    elapsed_time_lines = []
    solution_count = []
    start_reading = False
    current_value = None

    # Regex patterns
    elapsed_time_pattern = re.compile(
        r"Elapsed time = ([\d\.]+) sec\..*solutions = (\d+)")
    header_pattern = re.compile(
        r"\s*Node\s+Left\s+Objective\s+IInf\s+Best Integer\s+Best Bound\s+ItCnt\s+Gap"
    )

    # First pattern for lines with all numerical values
    full_info_pattern_1 = re.compile(
        r"^\s*(\d+)\s+(\d+)\s+([\d\.\-e\+]+)\s+(\d+)\s+([\d\.\-e\+]+)\s+([\d\.\-e\+]+)\s+(\d+)\s+([\d\.]+)%"
    )

    # Second pattern for lines with a label (e.g., "ZeroHalf:") in the sixth column
    full_info_pattern_2 = re.compile(
        r"^\s*(\d+)\s+(\d+)\s+([\d\.\-e\+]+)\s+(\d+)\s+([\d\.\-e\+]+)\s+(\w+:\s*\d+)\s+(\d+)\s+([\d\.]+)%"
    )

    # Function to process a range of lines for changes in value
    def process_lines(start_line, end_line, last_elapsed_time,
                      current_elapsed_time):
        nonlocal current_value
        for i in range(start_line, end_line):
            current_line = lines[i]
            # Skip lines that start with '*' as they may have missing information
            if current_line.lstrip().startswith("*"):
                continue

            # Try matching with the first pattern
            full_info_match = full_info_pattern_1.search(current_line)

            # If the first pattern doesn't match, try the second pattern
            if not full_info_match:
                full_info_match = full_info_pattern_2.search(current_line)

            if full_info_match:
                # Get the fifth number (Best Integer value) from the match
                new_value = float(full_info_match.group(0).split()[4])

                # Record only if the value changes
                if current_value is None or new_value != current_value:
                    # Calculate interpolated time
                    if end_line - start_line > 0:
                        interpolated_time = last_elapsed_time + \
                            (current_elapsed_time - last_elapsed_time) * (i - start_line) / (end_line - start_line)
                    else:
                        interpolated_time = last_elapsed_time

                    results.append((new_value, interpolated_time))
                    current_value = new_value

    # Iterate through lines to identify relevant information
    last_elapsed_time = 0.0
    for line_num, line in enumerate(lines):
        # Identify the start header line
        if header_pattern.search(line):
            start_reading = True
            continue

        if start_reading:
            # Look for "Elapsed time" lines and record line numbers and solution counts
            elapsed_match = elapsed_time_pattern.search(line)
            if elapsed_match:
                current_elapsed_time = float(elapsed_match.group(1))
                solutions = int(elapsed_match.group(2))

                # Record the line number and corresponding information
                elapsed_time_lines.append((line_num, current_elapsed_time))
                solution_count.append(solutions)

                # Process lines before the first "Elapsed time" line
                if len(elapsed_time_lines) == 1:
                    process_lines(0, line_num, 0, current_elapsed_time)

                # Process lines between the last two "Elapsed time" lines
                if len(elapsed_time_lines) > 1:
                    if solution_count[-1] != solution_count[-2]:
                        process_lines(elapsed_time_lines[-2][0] + 1, line_num,
                                      elapsed_time_lines[-2][1],
                                      current_elapsed_time)

                last_elapsed_time = current_elapsed_time

    return results


# Usage
egraphs = [
    'tensat_nasneta', 'tensat_nasrnn', 'tensat_bert', 'tensat_vgg',
    'rover_fir_8_tap_5iteration_egraph', 'rover_fir_8_tap_6iteration_egraph',
    'rover_fir_8_tap_7iteration_egraph', 'rover_fir_8_tap_8iteration_egraph'
]
key_mapping = {
    'nasneta': 'NASNetA',
    'nasrnn': 'NASRNN',
    'bert': 'BERT',
    'vgg': 'VGG',
    'fir_8_tap_5iteration_egraph': 'fir_5',
    'fir_8_tap_6iteration_egraph': 'fir_6',
    'fir_8_tap_7iteration_egraph': 'fir_7',
    'fir_8_tap_8iteration_egraph': 'fir_8',
}
data = defaultdict(dict)
for egraph in egraphs:
    file_path = f"logs/{egraph}.json_cplex.log"
    result_list = parse_cplex_log(file_path)
    if 'tensat' in egraph:
        egraph = egraph[7:]
    else:
        egraph = egraph[6:]
    data[egraph]['CPLEX'] = result_list

smoothe_result = json.load(
    open(os.path.join('logs', 'tensat_linear_0_v.json'), 'r'))
smoothe_result.update(
    json.load(open(os.path.join('logs', 'rover_linear_0_v.json'), 'r')))


def update_data(result, name):
    for key in result:
        cost = result[key]['inference_loss']
        time = result[key]['time'][1:]

        new_cost = []
        new_time = []
        min_cost = float(10000)
        for i in range(len(cost)):
            if cost[i] < min_cost:
                min_cost = cost[i]
                new_cost.append(cost[i])
                new_time.append(time[i])

        list = [(c, t) for c, t in zip(new_cost, new_time)]
        index = key[:-5]
        data[index][name] = list


update_data(smoothe_result, 'SmoothE')

color_map = {
    "CPLEX": plt.colormaps.get_cmap("Blues")(0.7),
    "SmoothE (ours)": plt.colormaps.get_cmap("Oranges")(0.6),
    "SmoothE": plt.colormaps.get_cmap("Oranges")(0.6),
    "ILP": plt.colormaps.get_cmap("Purples")(0.6),
    r"ILP$^*$": plt.colormaps.get_cmap("Purples")(0.6),
    "Genetic": plt.colormaps.get_cmap("Greens")(0.6),  # A moderate green
}


def plot(data):
    fig, axs = plt.subplots(2, 4, figsize=(16, 5), dpi=300)
    fig.subplots_adjust(wspace=0.3, hspace=0.5)  # Adjust this value as needed
    for key, ax in zip(data, axs.flatten()):
        if len(data[key]) == 1:
            continue
        for solver in data[key]:
            if solver == 'SCIP':
                continue
            if solver == 'SmoothE (Greedy)':
                continue
            result = data[key][solver]
            if isinstance(result, float):
                # ax.axhline(result, label=solver, color=color_map[solver], linestyle='--')
                pass
            elif result is None:
                pass
            else:
                cost = ([r[0] for r in result])
                time = ([r[1] for r in result])
                time.append(900)
                cost.append(cost[-1])
                time = np.array(time)
                cost = np.array(cost)
                mask = cost < 10000
                time = time[mask]
                cost = cost[mask]
                label = solver
                ax.plot(time, cost, label=label, color=color_map[solver])
            ax.set_xlim(3, 800)
            egraph = key_mapping[key]
            if egraph == 'NASRNN':
                ax.set_ylim(0.949, .98)
            elif egraph == 'BERT':
                ax.set_ylim(0.0, 3)
                ax.set_xlim(0, 200)
            elif egraph == 'NASNetA':
                ax.set_ylim(10, 20)
                ax.set_xlim(0, 30)
            elif egraph == 'VGG':
                ax.set_ylim(4.7, 8)
                ax.set_xlim(0, 40)
            elif egraph == 'fir_5':
                ax.set_ylim(5800, 7000)
                ax.set_xlim(0, 80)
            elif egraph == 'fir_6':
                ax.set_ylim(5800, 7000)
                ax.set_xlim(0, 90)
            elif egraph == 'fir_7':
                ax.set_ylim(5800, 7000)
                ax.set_xlim(0, 150)
            elif egraph == 'fir_8':
                ax.set_ylim(5800, 7000)
                ax.set_xlim(0, 240)

            ax.set_title(key, size=14)
            ax.xaxis.set_minor_locator(plt.NullLocator())
            for label in ax.get_yticklabels():
                label.set_rotation(45)
            ax.grid(True, alpha=0.3)

    ax.legend(loc=(-2.2, 2.8), ncol=3)
    fig.supxlabel('Time (sec.)', fontsize=18, x=0.5, y=-0.03)
    fig.supylabel('Cost', fontsize=18, x=0.08, y=0.5)
    plt.savefig(f'fig/Figure 6.pdf', bbox_inches='tight')


plot(data)

# Figure 5
tensat = json.load(open(os.path.join('logs', 'tensat_linear_0_v.json'), 'r'))
rover = json.load(open(os.path.join('logs', 'rover_linear_0_v.json'), 'r'))

loss = {
    'Tensat/NASNet-A': tensat['nasneta.json'],
    'Tensat/NASRNN': tensat['nasrnn.json'],
    'ROVER/box_3': rover['box_filter_3iteration_egraph.json'],
    'ROVER/box_4': rover['box_filter_4iteration_egraph.json'],
}

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 14


def loss_plot(data):
    fig, axs = plt.subplots(1, len(data), figsize=(14, 1.8), dpi=300)
    fig.subplots_adjust(wspace=0.3)  # Adjust this value as needed

    for i, (name, d) in enumerate(data.items()):
        train_loss = d['loss'][:-1]
        inf_loss = d['inference_loss']

        l = len(train_loss)
        axs[i].scatter(np.arange(l),
                       train_loss,
                       label=r'Optimization Loss $f(p)$',
                       s=4)
        axs[i].scatter(np.arange(l),
                       inf_loss,
                       label=r'Sampling Loss $f_b(s)$',
                       s=4)
        axs[i].set_title(name, fontsize=14)
        if name == 'Tensat/NASNet-A':
            axs[i].set_ylim(10, 20)
        elif name == 'Tensat/NASRNN':
            axs[i].set_ylim(.8, 1.5)
            pass
        elif name == 'ResNet-50':
            axs[i].set_ylim(3, 5)
        elif name == 'ROVER/box_3':
            axs[i].set_ylim(1500, 3000)
        elif name == 'ROVER/box_4':
            axs[i].set_ylim(1500, 3000)
    plt.legend(loc=(1.1, 0.6), ncol=1, fontsize=12)

    fig.supxlabel('# Optimization Steps', fontsize=14, x=0.5, y=-0.15)
    fig.supylabel('Cost', fontsize=14, x=0.08, y=0.55)
    plt.savefig(f'fig/Figure 4.pdf', bbox_inches='tight', pad_inches=0)


loss_plot(loss)

# Figure 4

# load smoothe nonlinear
dataset_names = ['diospyros', 'flexc', 'impress', 'rover', 'tensat']
data = defaultdict(list)
for dataset in dataset_names:
    for egraph in os.listdir(os.path.join('dataset', dataset)):
        if egraph.endswith('dot') or egraph.endswith('json'):
            data[dataset].append(egraph)

raw_smoothe_quad = {}
raw_smoothe_mlp = {}
smoothe_quad = defaultdict(dict)
smoothe_mlp = defaultdict(dict)
for dataset in dataset_names:
    quad_time = []
    mlp_time = []
    raw_smoothe_quad[dataset] = json.load(
        open(os.path.join('logs', dataset + '_quad_0_v.json'), 'r'))
    raw_smoothe_mlp[dataset] = json.load(
        open(os.path.join('logs', dataset + '_mlp_0_v.json'), 'r'))
    for egraph_name in data[dataset]:
        if egraph_name.endswith('.json'):
            key = egraph_name[:-5]
        elif egraph_name.endswith('.dot'):
            key = egraph_name[:-4]
        else:
            raise ValueError
        quad = raw_smoothe_quad[dataset][egraph_name]
        mlp = raw_smoothe_mlp[dataset][egraph_name]
        smoothe_quad[dataset][key] = get_results(quad)[0]
        smoothe_mlp[dataset][key] = get_results(mlp)[0]
        quad_time.append(get_results(quad)[1])
        mlp_time.append(get_results(mlp)[1])

# load genetic nonlinear
genetic_quad = [defaultdict(dict), defaultdict(dict), defaultdict(dict)]
genetic_mlp = [defaultdict(dict), defaultdict(dict), defaultdict(dict)]
for dataset in dataset_names:
    for egraph in data[dataset]:
        if egraph.endswith('.json'):
            egraph = egraph[:-5]
        elif egraph.endswith('.dot'):
            egraph = egraph[:-4]
        else:
            raise ValueError

        for i in range(3):
            quad_path = os.path.join('logs/genetic', dataset,
                                     f'{egraph}_quad_cost_{i}.json')
            mlp_path = os.path.join('logs/genetic', dataset,
                                    f'{egraph}_mlp_cost_{i}.json')

            raw_quad = json.load(open(quad_path, 'r'))
            raw_mlp = json.load(open(mlp_path, 'r'))

            genetic_quad[i][dataset][egraph] = raw_quad['Best Cost']
            genetic_mlp[i][dataset][egraph] = raw_mlp['Best Cost']

# CPLEX optimal
cplex_optimal = defaultdict(dict)
for dataset in data:
    for egraph in data[dataset]:
        if egraph.endswith('.json'):
            egraph_name = egraph[:-5]
        elif egraph.endswith('.dot'):
            egraph_name = egraph[:-4]
        else:
            raise ValueError
        cplex_10h = json.load(
            open(
                os.path.join(
                    'logs/ilp_log',
                    f'{egraph_name}_ilp_solver_cplex_time_{36000}.json')))

        if cplex_10h['status'] == 'Infeasible':
            cplex_optimal[dataset][egraph_name] = float('nan')
        else:
            cplex_optimal[dataset][egraph_name] = cplex_10h['cost']

# quad results
norm_quad = defaultdict(list)
for dataset in sorted(dataset_names):
    smoothe = []
    greedy = []
    genetic = [[], [], []]
    for egraph in smoothe_quad[dataset]:
        optimal = smoothe_quad[dataset][egraph]
        if optimal == 0:
            continue
        abs_optimal = abs(optimal)
        greedy.append((cplex_optimal[dataset][egraph] - optimal) / abs_optimal)
        for i in range(3):
            genetic[i].append(
                (genetic_quad[i][dataset][egraph] - optimal) / abs_optimal)
    norm_quad['Greedy'].append(new_mean(np.array(greedy) + 1))
    for i in range(3):
        norm_quad[f'Genetic{i}'].append(new_mean(np.array(genetic[i]) + 1))

for method in norm_quad:
    norm_quad[method] = np.array(norm_quad[method])

genetic = np.stack([norm_quad[f'Genetic{i}'] for i in range(3)], axis=0)
quad_avg = {
    'ILP': norm_quad['Greedy'] - 1,
    'Genetic': np.mean(genetic, axis=0) - 1,
    'SmoothE (ours)': np.zeros_like(norm_quad['Greedy'])
}
quad_diff = {'Genetic': (np.min(genetic, axis=0), np.max(genetic, axis=0))}

# mlp results
norm_mlp = defaultdict(list)
for dataset in sorted(dataset_names):
    smoothe = []
    greedy = []
    genetic = [[], [], []]
    for egraph in smoothe_mlp[dataset]:
        optimal = smoothe_mlp[dataset][egraph]
        if optimal == 0:
            continue
        abs_optimal = abs(optimal)
        greedy.append((cplex_optimal[dataset][egraph] - optimal) / abs_optimal)
        for i in range(3):
            genetic[i].append(
                (genetic_mlp[i][dataset][egraph] - optimal) / abs_optimal)
    norm_mlp['Greedy'].append(new_mean(np.array(greedy) + 1))
    for i in range(3):
        norm_mlp[f'Genetic{i}'].append(new_mean(np.array(genetic[i]) + 1))

for method in norm_mlp:
    norm_mlp[method] = np.array(norm_mlp[method])

genetic = np.stack([norm_mlp[f'Genetic{i}'] for i in range(3)], axis=0)
mlp_avg = {
    r'ILP$^*$': norm_mlp['Greedy'] - 1,
    'Genetic': np.mean(genetic, axis=0) - 1,
    'SmoothE (ours)': np.zeros_like(norm_mlp['Greedy'])
}
mlp_diff = {'Genetic': (np.min(genetic, axis=0), np.max(genetic, axis=0))}


def bar_plot(ax, data, diff_data, title, x_labels, ylim=None, legend=False):
    multiplier = 0
    width = 0.22
    if ylim == [1, 4]:
        ax.set_ylim(1, 4)
        ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels(['Best', r'$2\times$', r'$3\times$', r'Infeasible'])
    elif ylim == [1, 1.75]:
        ax.set_ylim(1, 1.75)
        ax.set_yticks([1, 1.25, 1.50, 1.75])
        ax.set_yticklabels(['Best', r'+$25\%$', r'+$50\%$', r'Infeasible'])
    ax.spines.top.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.right.set_visible(False)

    x = np.arange(len(list(data.values())[0]))
    mmax = 0
    mmin = np.inf

    mmin, mmax = ylim
    cross = False
    for solver, value in data.items():

        nan_pos = np.isnan(value)
        value = np.array(value).round(2)
        offset = width * multiplier

        value[nan_pos] = 1024
        kwargs = {
            'label': solver,
            'bottom': 1,
            'color': color_map[solver],
            'alpha': 0.7
        }
        rects = ax.bar(x + offset, value, width, **kwargs)
        if solver in diff_data:
            # yerr = ax.errorbar(x+offset, value+1, diff_data[solver])
            # add to the following line
            yerr = np.abs(np.array(diff_data[solver]) - (value + 1))
            ax.errorbar(x + offset,
                        value + 1,
                        yerr=yerr,
                        fmt='none',
                        ecolor=color_map[solver],
                        capsize=3)

        for i in range(len(value)):
            if nan_pos[i]:
                ax.scatter(x[i] + offset,
                           1024,
                           marker='x',
                           color=color_map[solver],
                           clip_on=False,
                           s=70)
            elif value[i] >= 4:
                ax.scatter(x[i] + offset,
                           ylim[1],
                           marker='x',
                           color=color_map[solver],
                           clip_on=False,
                           s=70)
                cross = True
                # ax.annotate(f'{value[i]+1:.1f}'+r'$\times$', (x[i], 1.75), rotation=45)
                # pass
            elif value[i] <= 0:
                ax.scatter(x[i] + offset,
                           1.000,
                           marker='*',
                           color=color_map[solver],
                           clip_on=False,
                           s=70)
            else:
                # ax.annotate(f'+{value[i]*100:.0f}%', (x[i]+offset-width*1.2, value[i]+0.99), rotation=45)
                pass

        # ax1.bar_label(rects, rotation=45, fmt=custom_fmt, fontsize=12)
        multiplier += 1

    ax.grid(axis='y', linestyle='--')
    ax.set_title(title)
    if legend:
        na_legend = ax.legend(handles=ax.get_legend_handles_labels()[0],
                              loc=(-0.05, 1.08),
                              ncol=4)

        handles = []
        scatter_star = plt.scatter([], [],
                                   marker='*',
                                   color=color_map['SmoothE (ours)'],
                                   s=70,
                                   label='Best')
        handles.append(scatter_star)
        scatter_cross = plt.scatter([], [],
                                    marker='x',
                                    color=color_map['Genetic'],
                                    s=70,
                                    label='Infeasible')
        handles.append(scatter_cross)
        handles = ax.get_legend_handles_labels()[0]
        handles = handles[-3:] + handles[0:-3]
        na_legend = ax.legend(handles=handles, loc=(-1.0, 1.20), ncol=5)

    if x_labels:
        ax.set_xticks(x + width * (len(data) / 2 - 0.5))
        ax.set_xticklabels(x_labels)
        ax.xaxis.set_ticks_position('none')

    if not legend:
        ax.set_ylabel('Normalized Cost')


fig, axs = plt.subplots(1, 2, figsize=(13, 2.0), dpi=300)
fig.subplots_adjust(wspace=0.2)  # Adjust this value as needed

bar_plot(axs[0],
         quad_avg,
         quad_diff,
         'Quadratic model',
         sorted(dataset_names),
         ylim=[1, 1.75])
bar_plot(axs[1],
         mlp_avg,
         mlp_diff,
         r'MLP model',
         sorted(dataset_names),
         ylim=[1, 4],
         legend=True)

# plt.tight_layout()
plt.savefig('fig/Figure 4.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
