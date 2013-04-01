# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        if action == 'Stop':
            return -999999999

        foodDistances = []
        foodList = currentGameState.getFood().asList()
        pacmanPosition = list(successorGameState.getPacmanPosition())
        
        # If this state is on a ghost, return negative infinity and RUN AWAY!
        for ghostState in newGhostStates:
            if ghostState.getPosition() == tuple(pacmanPosition) and ghostState.scaredTimer is 0:
                return -999999999

        # Calculate the Manhattan Distance to each of the remaining food pellets
        for food in foodList:
            x = -1 * abs(food[0] - pacmanPosition[0])
            y = -1 * abs(food[1] - pacmanPosition[1])
            foodDistances.append(x + y)

        return max(foodDistances)

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.pacmanIndex = 0

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          Directions.STOP:
            The stop direction, which is always legal

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        curDepth = 0
        currentAgentIndex = 0
        val = self.value(gameState, currentAgentIndex, curDepth)

        return val[0]

    # Retruns the value value of actions given a game state and tree depth
    def value(self, gameState, currentAgentIndex, curDepth): 
        if currentAgentIndex >= gameState.getNumAgents():
            currentAgentIndex = 0
            curDepth += 1

        if curDepth == self.depth:
            return self.evaluationFunction(gameState)

        # If the current agent is Pacman, maximize. If not, minimize
        if currentAgentIndex == self.pacmanIndex:
            return self.maxValue(gameState, currentAgentIndex, curDepth)
        else:
            return self.minValue(gameState, currentAgentIndex, curDepth)
    
    # Returns the minimum action value for a given agent
    def minValue(self, gameState, currentAgentIndex, curDepth):
        value = ("unknown", 999999999)
        
        if not gameState.getLegalActions(currentAgentIndex):
            return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(currentAgentIndex):
            if action == "Stop":
                continue
            
            retVal = self.value(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex + 1, curDepth)
            if type(retVal) is tuple:
                retVal = retVal[1] 

            vNew = min(value[1], retVal)

            if vNew is not value[1]:
                value = (action, vNew) 

        return value
    # Returns the maximum action value for a given agent
    def maxValue(self, gameState, currentAgentIndex, curDepth):
        value = ("unknown", -999999999)
        
        if not gameState.getLegalActions(currentAgentIndex):
            return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(currentAgentIndex):
            if action == "Stop":
                continue
            
            retVal = self.value(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex + 1, curDepth)
            if type(retVal) is tuple:
                retVal = retVal[1] 

            vNew = max(value[1], retVal)

            if vNew is not value[1]:
                value = (action, vNew) 
        
        return value

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        curDepth = 0
        currentAgentIndex = 0
        alpha = -999999999
        beta = 999999999
        val = self.value(gameState, currentAgentIndex, curDepth, alpha, beta)
        # print "Returning %s" % str(val)
        return val[0]

    def value(self, gameState, currentAgentIndex, curDepth, alpha, beta): 
        if currentAgentIndex >= gameState.getNumAgents():
            currentAgentIndex = 0
            curDepth += 1

        if curDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if currentAgentIndex == self.pacmanIndex:
            return self.maxValue(gameState, currentAgentIndex, curDepth, alpha, beta)
        else:
            return self.minValue(gameState, currentAgentIndex, curDepth, alpha, beta)
        
    def minValue(self, gameState, currentAgentIndex, curDepth, alpha, beta):
        value = ("unknown", 999999999)
        
        if not gameState.getLegalActions(currentAgentIndex):
            return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(currentAgentIndex):
            if action == "Stop":
                continue
            
            retVal = self.value(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex + 1, curDepth, alpha, beta)
            if type(retVal) is tuple:
                retVal = retVal[1] 

            vNew = min(value[1], retVal)

            if vNew is not value[1]:
                value = (action, vNew) 
            
            if value[1] < alpha:
                return value
            
            beta = min(beta, value[1])

        return value

    def maxValue(self, gameState, currentAgentIndex, curDepth, alpha, beta):
        value = ("unknown", -999999999)
        
        if not gameState.getLegalActions(currentAgentIndex):
            return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(currentAgentIndex):
            if action == "Stop":
                continue
            
            retVal = self.value(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex + 1, curDepth, alpha, beta)
            if type(retVal) is tuple:
                retVal = retVal[1] 

            vNew = max(value[1], retVal)

            if vNew is not value[1]:
                value = (action, vNew) 
            
            if value[1] > beta:
                return value

            alpha = max(alpha, value[1])

        return value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        curDepth = 0
        currentAgentIndex = 0
        val = self.value(gameState, currentAgentIndex, curDepth)
        # print "Returning %s" % str(val)
        return val[0]

    def value(self, gameState, currentAgentIndex, curDepth): 
        if currentAgentIndex >= gameState.getNumAgents():
            currentAgentIndex = 0
            curDepth += 1

        if curDepth == self.depth:
            return self.evaluationFunction(gameState)

        if currentAgentIndex == self.pacmanIndex:
            return self.maxValue(gameState, currentAgentIndex, curDepth)
        else:
            return self.expectimaxValue(gameState, currentAgentIndex, curDepth)
        
    def expectimaxValue(self, gameState, currentAgentIndex, curDepth):
        value = ["unknown", 0]
        
        if not gameState.getLegalActions(currentAgentIndex):
            return self.evaluationFunction(gameState)
        
        # Used to determine what value to multiply our evaluation function values by
        # We should utilize the reciprocal of the number of child nodes here since we want an average
        probFactor = 1.0 / len(gameState.getLegalActions(currentAgentIndex))
        
        for action in gameState.getLegalActions(currentAgentIndex):
            if action == "Stop":
                continue
            
            retVal = self.value(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex + 1, curDepth)
            if type(retVal) is tuple:
                retVal = retVal[1] 

            # Calculate the probability
            value[1] += retVal * probFactor
            value[0] = action

        return tuple(value)

    def maxValue(self, gameState, currentAgentIndex, curDepth):
        value = ("unknown", -1 * 999999999)
        
        if not gameState.getLegalActions(currentAgentIndex):
            return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(currentAgentIndex):
            if action == "Stop":
                continue
            
            retVal = self.value(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex + 1, curDepth)
            if type(retVal) is tuple:
                retVal = retVal[1] 

            vNew = max(value[1], retVal)

            if vNew is not value[1]:
                value = (action, vNew) 
        
        return value

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # What is important to determine the score of state?
    # We'll go with the following:
    distanceToFood = []
    distanceToNearestGhost = []
    distanceToCapsules = []
    numOfScaredGhosts = 0
        
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsuleList = currentGameState.getCapsules()
    pacmanPos = list(currentGameState.getPacmanPosition())

    # Gather data about the ghosts
    for ghostState in ghostStates:
        if ghostState.scaredTimer > 0:
            numOfScaredGhosts += 1
            distanceToNearestGhost.append(0)
            continue

        # Calculate the Manhattan Distance to this ghost
        ghostCoord = ghostState.getPosition()
        x = abs(ghostCoord[0] - pacmanPos[0])
        y = abs(ghostCoord[1] - pacmanPos[1])
        if (x + y) == 0:
            distanceToNearestGhost.append(0)
        else:
            distanceToNearestGhost.append(-1.0 / (x + y))

    # Gather data about all of the remaining food
    for food in foodList:
        x = abs(food[0] - pacmanPos[0])
        y = abs(food[1] - pacmanPos[1])
        distanceToFood.append(-1 * (x + y))
        
    # No food left?
    if not distanceToFood:
        distanceToFood.append(0)

    return currentGameState.getScore() - 1000 * numOfScaredGhosts
    # return max(distanceToFood) + min(distanceToNearestGhost) + currentGameState.getScore() + 100 * len(capsuleList) - len(ghostStates) + 1000 * numOfScaredGhosts

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        
        curDepth = 0
        currentAgentIndex = 0
        val = self.value(gameState, currentAgentIndex, curDepth)

        return val[0]

    # Retruns the value value of actions given a game state and tree depth
    def value(self, gameState, currentAgentIndex, curDepth): 
        if currentAgentIndex >= gameState.getNumAgents():
            currentAgentIndex = 0
            curDepth += 1

        if curDepth == self.depth:
            return betterEvaluationFunction(gameState)
            # return self.evaluationFunction(gameState)

        # If the current agent is Pacman, maximize. If not, minimize
        if currentAgentIndex == self.pacmanIndex:
            return self.maxValue(gameState, currentAgentIndex, curDepth)
        else:
            return self.minValue(gameState, currentAgentIndex, curDepth)
    
    # Returns the minimum action value for a given agent
    def minValue(self, gameState, currentAgentIndex, curDepth):
        value = ("unknown", 999999999)
        
        if not gameState.getLegalActions(currentAgentIndex):
            return betterEvaluationFunction(gameState)
            # return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(currentAgentIndex):
            if action == "Stop":
                continue
            
            retVal = self.value(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex + 1, curDepth)
            if type(retVal) is tuple:
                retVal = retVal[1] 

            vNew = min(value[1], retVal)

            if vNew is not value[1]:
                value = (action, vNew) 

        return value
    # Returns the maximum action value for a given agent
    def maxValue(self, gameState, currentAgentIndex, curDepth):
        value = ("unknown", -999999999)
        
        if not gameState.getLegalActions(currentAgentIndex):
            return betterEvaluationFunction(gameState)
            # return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(currentAgentIndex):
            if action == "Stop":
                continue
            
            retVal = self.value(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex + 1, curDepth)
            if type(retVal) is tuple:
                retVal = retVal[1] 

            vNew = max(value[1], retVal)

            if vNew is not value[1]:
                value = (action, vNew) 
        
        return value