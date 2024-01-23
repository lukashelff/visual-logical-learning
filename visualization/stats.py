import json
import seaborn as sns
from matplotlib import pyplot as plt


def plot_aleph_sys_stats():
    tr_sizes = [100, 1000, 10000]
    # load json file
    mem = []
    time = []
    for t in tr_sizes:
        p = f'output/neuro-symbolic/system_stats/Trains_aleph_custom_{t}smpl_0noise_0.json'
        with open(p, 'r') as f:
            data = json.load(f)
            mem.append(round(max(data['memory_usage']) / 1024, 2))
            time.append(data['time'] / 3600)
    # create two subplots
    sns.set_theme(style="whitegrid")

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)
    sns.lineplot(x=tr_sizes, y=mem, marker='o', ax=ax1)
    sns.lineplot(x=tr_sizes, y=time, marker='o', ax=ax2)
    ax1.set_xlabel('Number of training samples')
    ax1.set_ylabel('Memory [GB]')
    ax1.set_yscale('log')
    # ax1.set_xlim(0, 10000)
    ax2.set_xlabel('Number of training samples')
    ax2.set_ylabel('Time [h]')
    ax2.set_yscale('log')
    # sns.set_context("paper")
    sns.set(font_scale=2)
    plt.tight_layout()
    plt.savefig('output/neuro-symbolic/system_stats/stats.png')
    plt.close()


    noises = [0, 0.1, 0.3]
    # load json file
    mem = []
    time = []
    for n in noises:
        p = f'output/neuro-symbolic/system_stats/Trains_aleph_custom_1000smpl_{n}noise_0.json'
        with open(p, 'r') as f:
            data = json.load(f)
            mem.append(round(max(data['memory_usage']), 2))
            time.append(data['time'])
    sns.set_theme(style="whitegrid")

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)
    sns.lineplot(x=tr_sizes, y=mem, marker='o', ax=ax1)
    sns.lineplot(x=tr_sizes, y=time, marker='o', ax=ax2)
    ax1.set_xlabel('Number of training samples')
    ax1.set_ylabel('Memory [MB]')
    # ax1.set_yscale('log')
    # ax1.set_xlim(0, 10000)
    ax2.set_xlabel('Number of training samples')
    ax2.set_ylabel('Time [s]')
    # ax2.set_yscale('log')
    # sns.set_context("paper")
    sns.set(font_scale=2)
    plt.tight_layout()
    plt.savefig('output/neuro-symbolic/system_stats/stats_noise.png')
    plt.close()