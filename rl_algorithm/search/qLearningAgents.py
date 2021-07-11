from pacman import Directions
from game import Agent, Actions
import random
import util
import numpy as np
from scipy.spatial import distance


class QLearningAgent(Agent):

    def __init__(self, alpha=0.2, epsilon=0.1, gamma=0.8, numTraining=1000, **args):
        super().__init__()
        # alpha: learning rate
        self.alpha = float(alpha)
        # epsilon: exploration rate
        self.epsilon = float(epsilon)
        # gamma: discount factor
        self.gamma = float(gamma)
        # num_training: number of training episodes
        self.num_training = int(numTraining)
        # number of games played
        self.games_played = 0
        self.last_100_scores = []
        self.games_won_last_100 = 0
        self.max_games_won_overall_100 = 0
        if 'discretization' not in args:
            args['discretization'] = 'small'
        self.discrete_os_win_size = self.get_discretization(args['discretization'])
        # Q-values
        self.q_value = util.Counter()
        # current score
        self.score = 0
        # last state
        self.lastState = []
        # last action
        self.lastAction = []
        self.direction_map = {'East': 0, 'West': 1, 'North': 2, 'South': 3, 'Stop': 4}

    def get_discretization(self, opt):
        if opt == 'small':
            DISCRETE_OS_SIZE = [17, 4, 17, 4, 17, 4]
            discrete_os_win_size = np.array([17, 4, 17, 4, 17, 4]) / DISCRETE_OS_SIZE
        elif opt == 'medium':
            DISCRETE_OS_SIZE = [12, 6, 12, 6, 12, 6]
            discrete_os_win_size = np.array([17, 8, 17, 8, 17, 8]) / DISCRETE_OS_SIZE
        else:
            DISCRETE_OS_SIZE = [8, 8, 8, 8, 8, 8]
            discrete_os_win_size = np.array([25, 24, 25, 24, 25, 24]) / DISCRETE_OS_SIZE
        return discrete_os_win_size

    def increment_games_played(self):
        self.games_played += 1

    def get_games_played(self):
        return self.games_played

    def get_num_training(self):
        return self.num_training

    def set_epsilon(self, value):
        self.epsilon = value

    def update_epsilon(self):
        new_epsilon = (1 - float(self.get_games_played()) / self.get_num_training()) / 10.0
        self.set_epsilon(new_epsilon)

    def set_alpha(self, value):
        self.alpha = value

    def getQValue(self, state, action):
        discrete_state = self.get_discrete_state(state)
        index = discrete_state + tuple([self.direction_map[action]])
        return self.q_value[index]

    def get_discrete_state(self, state):
        f_state = state.getPacmanPosition()
        distances = {}
        for position in state.getGhostPositions():
            distances[position] = [distance.euclidean(state.getPacmanPosition(), position)]

        f_state += min(distances, key=distances.get)
        f_state += self.getClosestFood(state)
        discrete_state = f_state / self.discrete_os_win_size
        ret_value = tuple(discrete_state.astype(np.int))
        return ret_value

    # return the maximum Q of state
    def getMaxQ(self, state):
        q_list = []
        for action in state.getLegalPacmanActions():
            q = self.getQValue(state, action)
            q_list.append(q)
        if len(q_list) == 0:
            return 0
        return max(q_list)

    def updateQ(self, state, action, reward, qmax):
        discrete_state = self.get_discrete_state(state)
        q = self.getQValue(state, action)
        index = discrete_state + tuple([self.direction_map[action]])
        self.q_value[index] = q + self.alpha * (reward + self.gamma * qmax - q)

    def getAction(self, state):
        reward = state.getScore() - self.score
        if len(self.lastState) > 0:
            last_state = self.lastState[-1]
            last_action = self.lastAction[-1]
            max_q = self.getMaxQ(state)
            self.updateQ(last_state, last_action, reward, max_q)

        if np.random.random() > self.epsilon:
            legal_actions = state.getLegalPacmanActions()
            actions = util.Counter()
            for action in legal_actions:
                actions[action] = self.getQValue(state, action)
            action = actions.argMax()
        else:
            legal_actions = state.getLegalPacmanActions()
            if Directions.STOP in legal_actions:
                legal_actions.remove(Directions.STOP)
            action = random.choice(legal_actions)

        # update attributes
        self.score = state.getScore()
        self.lastState.append(state)
        self.lastAction.append(action)

        return action

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):

        if state.isWin():
            self.games_won_last_100 += 1

        # update Q-values
        reward = state.getScore() - self.score
        last_state = self.lastState[-1]
        last_action = self.lastAction[-1]
        self.updateQ(last_state, last_action, reward, 0)
        self.print_stats(reward, state)

        self.last_100_scores.append(state.getScore())

        # reset attributes
        self.score = 0
        self.lastState = []
        self.lastAction = []

        self.update_epsilon()

        self.increment_games_played()
        if self.get_games_played() % 100 == 0:
            if self.games_won_last_100 >= self.max_games_won_overall_100:
                self.max_games_won_overall_100 = self.games_won_last_100
            self.print_stats_100()
            self.last_100_scores = []
            self.games_won_last_100 = 0

        if self.get_games_played() == self.get_num_training():
            print('Training completed')
            self.set_alpha(0)
            self.set_epsilon(0)

    def print_stats(self, reward, state):
        print('Training episode: {0} Reward: {1} Number of actions: {2} Score: {3}'
              .format(self.games_played, reward, len(self.lastAction), state.getScore()))

    def print_stats_100(self):
        print("\nCompleted %s runs of training" % self.get_games_played())
        print("Average Score: {}".format(np.average(self.last_100_scores)))
        print("Winning percentage: {}%".format(self.games_won_last_100))
        print("Max winning percentage so far: {}%\n".format(self.max_games_won_overall_100))

    def getClosestFood(self, state):
        food = state.getFood()
        walls = state.getWalls()

        x, y = state.getPacmanPosition()

        fringe = [(x, y, 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                # return dist
                return pos_x, pos_y
            # otherwise spread out from the location to its neighbours
            neighbours = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in neighbours:
                fringe.append((nbr_x, nbr_y, dist + 1))
        # no food found
        return None