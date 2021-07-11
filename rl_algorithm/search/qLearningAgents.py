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
        # number of episodes played
        self.games_played = 0
        # scores for the last 100 episodes
        self.last_100_scores = []
        # number of games won in the last 100 episodes
        self.games_won_last_100 = 0
        # maximum number of games won in a 100 episode interval
        self.max_games_won_overall_100 = 0
        if 'discretization' not in args:
            args['discretization'] = 'small'
        self.discrete_os_win_size = self.get_discretization(args['discretization'])
        # Q Table
        self.q_table = util.Counter()
        # current score
        self.score = 0
        # last state
        self.lastState = []
        # last action
        self.lastAction = []
        # direction mapping to numbers
        self.direction_map = {'East': 0, 'West': 1, 'North': 2, 'South': 3, 'Stop': 4}

    # return a discretization window
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

    # update epsilon after each episode
    def update_epsilon(self):
        new_epsilon = (1 - float(self.get_games_played()) / self.get_num_training()) / 10.0
        self.set_epsilon(new_epsilon)

    def set_alpha(self, value):
        self.alpha = value

    # return a value from the Q table
    def get_q_value(self, state, action):
        discrete_state = self.get_discrete_state(state)
        index = discrete_state + tuple([self.direction_map[action]])
        return self.q_table[index]

    # return a discrete state given a state
    def get_discrete_state(self, state):
        # pacman position
        f_state = state.getPacmanPosition()
        distances = {}
        for position in state.getGhostPositions():
            distances[position] = [distance.euclidean(state.getPacmanPosition(), position)]

        # position of the closest ghost
        f_state += min(distances, key=distances.get)
        # position of the closest food
        f_state += self.get_closest_food(state)
        # discretization function
        discrete_state = f_state / self.discrete_os_win_size
        ret_value = tuple(discrete_state.astype(np.int))
        return ret_value

    # return the optimal action
    def get_max_q_value(self, state):
        q_list = []
        for action in state.getLegalPacmanActions():
            q = self.get_q_value(state, action)
            q_list.append(q)
        if len(q_list) == 0:
            return 0
        return max(q_list)

    # update the Q table values
    def update_q_table(self, state, action, reward, qmax):
        discrete_state = self.get_discrete_state(state)
        q = self.get_q_value(state, action)
        index = discrete_state + tuple([self.direction_map[action]])
        self.q_table[index] = q + self.alpha * (reward + self.gamma * qmax - q)

    # return the action that pacman must take
    def getAction(self, state):
        reward = state.getScore() - self.score
        if len(self.lastState) > 0:
            last_state = self.lastState[-1]
            last_action = self.lastAction[-1]
            max_q = self.get_max_q_value(state)
            self.update_q_table(last_state, last_action, reward, max_q)

        # epsilon-greedy
        if np.random.random() > self.epsilon:
            legal_actions = state.getLegalPacmanActions()
            actions = util.Counter()
            for action in legal_actions:
                actions[action] = self.get_q_value(state, action)
            action = actions.argMax()
        else:
            # prevent stop action when exploring
            legal_actions = state.getLegalPacmanActions()
            if Directions.STOP in legal_actions:
                legal_actions.remove(Directions.STOP)
            action = random.choice(legal_actions)

        # update attributes
        self.score = state.getScore()
        self.lastState.append(state)
        self.lastAction.append(action)

        return action

    # called after each episode
    def final(self, state):
        if state.isWin():
            self.games_won_last_100 += 1

        # update Q Table
        reward = state.getScore() - self.score
        last_state = self.lastState[-1]
        last_action = self.lastAction[-1]
        self.update_q_table(last_state, last_action, reward, 0)
        self.print_stats(reward, state)
        self.last_100_scores.append(state.getScore())

        # reset attributes
        self.score = 0
        self.lastState = []
        self.lastAction = []

        # update epsilon
        self.update_epsilon()

        # increment games played
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

    # print stats per episode
    def print_stats(self, reward, state):
        print('Training episode: {0} Reward: {1} Number of actions: {2} Score: {3}'
              .format(self.games_played, reward, len(self.lastAction), state.getScore()))

    # print state per 100 episodes
    def print_stats_100(self):
        print("\nCompleted %s runs of training" % self.get_games_played())
        print("Average Score: {}".format(np.average(self.last_100_scores)))
        print("Winning percentage: {}%".format(self.games_won_last_100))
        print("Max winning percentage so far: {}%\n".format(self.max_games_won_overall_100))

    # find the position of the closest food
    def get_closest_food(self, state):
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