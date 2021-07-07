import os
import random

import numpy as np
from tqdm import tqdm
import json

from search.game import Directions
from search.pacman import runGames
from search.ghostAgents import RandomGhost
from search import layout, textDisplay

from agents import SuperAgent

# random.seed(42)
# np.random.seed(42)

MOVES = [Directions.NORTH, Directions.EAST, Directions.SOUTH, Directions.WEST]

# Parameters
BEST_RATE = 0.4
CROSSOVER_RATE = 0.2
MUTATION_RATE = 0.2
REPLACEMENT_RATE = 0.2

def generate_gene(size:int = 200):

    return np.random.choice(MOVES, size)


def generate_population(size:int = 100):

    population = []

    for i in range(size):
        population.append(generate_gene())
    
    return population


def gen_args(layout_name, agent):

    args = {}

    args['layout'] = layout.getLayout(layout_name)
    args['pacman'] = agent
    args['ghosts'] = [RandomGhost(i+1) for i in range(4)]

    args['display'] = textDisplay.NullGraphics()
    args['numGames'] = 1
    args['record'] = False
    args['catchExceptions'] = False
    args['timeout'] = 1

    return args


def select(population, scores):
    
    # Get how many elements will be selected
    qty = int(len(population)*BEST_RATE)
    
    # Get top qty elements from list
    # Ref: https://stackoverflow.com/questions/13070461/get-indices-of-the-top-n-values-of-a-list
    indexes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:qty]
    selected = np.array(population)[indexes].tolist()

    return selected


def crossover(population):

    # Get how many elements will be selected
    qty = int(len(population)*CROSSOVER_RATE)

    crossed = []

    # Get two random elements and apply crossover
    for i in range(qty//2):

        # Get two random indexes
        x = random.randrange(len(population))
        y = random.randrange(len(population))

        # Get random genes
        x = population[x]
        y = population[y]

        # Get crossover split position
        split = random.randrange(len(x))
        new_x = np.array(x)[:split].tolist() + np.array(y)[split:].tolist()
        new_y = np.array(y)[:split].tolist() + np.array(x)[split:].tolist()

        crossed.append(new_x)
        crossed.append(new_y)

    return crossed


def mutation(population):
    
    # Get how many elements will be selected
    qty = int(len(population)*MUTATION_RATE)

    mutated = []

    for i in range(qty):
        
        # Get a random index
        x = random.randrange(len(population))

        # Get random gene
        x = population[x]

        sel_move = np.random.choice(MOVES, 1)[0]
        sel_pos = random.randrange(len(x))

        x[sel_pos] = sel_move

        mutated.append(x)

    return mutated


def replacement(population, psel, pcro, pmut):
    
    replaced = []

    replaced.extend(psel)
    replaced.extend(pcro)
    replaced.extend(pmut)

    qty = int(len(population)*REPLACEMENT_RATE)
    rep = np.random.choice(len(population), qty)

    rep = np.array(population)[rep].tolist()

    replaced.extend(rep)

    return replaced


def evaluate(population, maze='tinyMaze'):
    
    scores = []

    for p in population:

        args = gen_args(maze, SuperAgent(p))
        games = runGames(**args)
        score = games[-1].state.getScore()
        scores.append(score)
    
    return scores


def get_stats(history, scores):

    history['mini'].append(np.amin(scores))
    history['maxi'].append(np.amax(scores))
    history['mean'].append(np.mean(scores))
    history['scores'].append(scores)

    return history


def save_history(history, maze, iteration, pop_size):

    output_dir = f'output_{pop_size}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(f'{output_dir}{os.sep}history_{maze}_{iteration}.json', 'w') as fp:
        json.dump(history, fp, indent=4)


def get_best(population, scores):

    # Get top qty elements from list
    # Ref: https://stackoverflow.com/questions/13070461/get-indices-of-the-top-n-values-of-a-list
    index = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[0]
    best = population[index]
    return best


def run(maze, iteration, pop_size):

    population = generate_population(pop_size)
    history = {
        'mini': [],
        'maxi': [],
        'mean': [],
        'scores': []
    }
    
    for i in tqdm(range(100)):

        scores = evaluate(population, maze)
        history = get_stats(history, scores)

        psel = select(population, scores)
        pcro = crossover(population)
        pmut = mutation(population)
        prep = replacement(population, psel, pcro, pmut)
        
        population = prep

    history['best'] = get_best(population, scores)
    save_history(history, maze, iteration, pop_size)


if __name__ == '__main__':

    for pop_size in 10, 100:

        for maze in ['smallClassic', 'mediumClassic', 'originalClassic']:

            for i in range(10):

                print(f'#> Running: {maze} / {i}')
            
                run(maze, i, pop_size)