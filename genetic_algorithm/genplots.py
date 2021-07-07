import os
import json

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn


def read_file(file_name):
    
    f = open(file_name)
    return json.load(f)


def get_maze(file):

    name = file.split('_')[1:2]
    name = [x.capitalize() for x in name]
    name = ''.join(name)
    return name


def read_folder(pop_size):

    folder = {}
    files = os.listdir(f'output_{pop_size}')

    for file in files:
        f = get_maze(file)

        if f not in folder:
            folder[f] = []

        folder[f].append(read_file(f'output_{pop_size}/{file}'))

    return folder


def read_data():

    data = {}
    
    data['10'] = read_folder(10)
    data['100'] = read_folder(100)

    return data


def save_or_load(data=None):

    if data is not None:
        with open(f'data.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open('data.pickle', 'rb') as handle:
            data = pickle.load(handle)
    
    return data


def get_means(data):
    
    minis, maxis, means, scores = np.array([0]*100), np.array([0]*100), np.array([0]*100), np.array([0]*100)

    for d in data:

        minis = np.add(minis, np.array(d['mini']))
        maxis = np.add(maxis, np.array(d['maxi']))
        means = np.add(means, np.array(d['mean']))
        scores = np.add(scores, np.array(np.sum(d['scores'])))

    simpli = {'minis': minis / len(data),
              'maxis': maxis / len(data),
              'means': means / len(data),
              'scores': scores / len(data)}

    return simpli


def simplify(data):

    result = {}

    for pop_size in ['10', '100']:

        result[pop_size] = {}

        for maze in data[pop_size].keys():

            result[pop_size][maze] = get_means(data[pop_size][maze])

    return result


def plot(data, pop_size):
    
    fig, axs = plt.subplots(3)
    fig.suptitle(f'Comparação entre Labirintos ({pop_size})', fontweight="bold")

    for i, m in enumerate(['Smallclassic', 'Mediumclassic', 'Originalclassic']):

        axs[i].set_ylabel(m, fontweight='bold')

        axs[i].plot(np.arange(100), data[pop_size][m]['minis'], c='#b7094c', label='Min')
        axs[i].plot(np.arange(100), data[pop_size][m]['means'], c='#5c4d7d', label='Mean')
        axs[i].plot(np.arange(100), data[pop_size][m]['maxis'], c='#0091ad', label='Max')
        
        axs[i].plot(np.arange(100), [0]*100, '--', c='gray')

        axs[i].set_ylim([-650, 200])
        axs[i].set_yticks(np.arange(-600, 201, 200))

    output_dir = 'plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.xlabel('Gerações', fontweight='bold')
    plt.legend(bbox_to_anchor=(0.18, 3.7), loc='upper left', ncol=3)
    # plt.tight_layout()
    plt.savefig(f'{output_dir}/plot_{pop_size}')


def main():

    # print('Read All')
    data = read_data()

    # print('Simplify')
    simpli = simplify(data)

    print('Save or Load')
    save_or_load(simpli)
    # simpli = save_or_load()

    print('Plot')
    plot(simpli, '10')
    plot(simpli, '100')


if __name__ == '__main__':
    main()