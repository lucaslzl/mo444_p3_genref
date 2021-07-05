from search.game import Agent, Directions


class SuperAgent(Agent):


    def __init__(self, moves=None):

        self.moves = moves
        self.iteration = 0


    def getAction(self, state):

        self.iteration += 1

        if  self.iteration < len(self.moves) and self.moves[self.iteration-1] in state.getLegalPacmanActions():
            return self.moves[self.iteration-1]
        else:
            return Directions.STOP