# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.qValues = {}

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """

        # Check to see if we've been in this state before and have a Q-value
        if self.qValues.has_key(state) and len(self.qValues[state]) > 0:
            # Now check to see if we have a Q-value for this action
            qValuesForState = self.qValues[state]
            if qValuesForState.has_key(action):
                return qValuesForState[action]
            else:
                return 0.0
        else:
            return 0.0


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """

        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return 0.0

        maxActionValue = -999999999
        for action in legalActions:
            qValue = self.getQValue(state, action)
            if qValue > maxActionValue:
                maxActionValue = qValue

        return maxActionValue

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = self.getLegalActions(state)

        if len(legalActions) == 0:
            return None

        bestAction = None
        bestActionValue = -999999999
        for action in legalActions:
            qValue = self.getQValue(state, action)
            if qValue == bestActionValue:
                # Break ties!
                randomAction = random.choice([bestAction, action])
                if randomAction == action:
                    bestActionValue = qValue
                    bestAction = action
            elif qValue > bestActionValue:
                bestActionValue = qValue
                bestAction = action

        return bestAction

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None

        if len(legalActions) == 0:
            return None

        probability = self.epsilon
        if util.flipCoin(probability):
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """

        # Calculate our new Q value
        alpha = self.alpha
        gamma = self.discount
        oldQValue = self.getQValue(state, action)
        valueSPrimeAPrime = self.getValue(nextState)
        newQValue = (1 - alpha) * oldQValue + alpha * (reward + gamma * valueSPrimeAPrime)

        # Set it
        if not self.qValues.has_key(state):
            self.qValues[state] = {}

        if not self.qValues[state].has_key(action):
            self.qValues[state][action] = 0

        self.qValues[state][action] = newQValue

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)
        
    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """

        qValue = 0

        features = self.featExtractor.getFeatures(state, action)
        weights = self.getWeights()
        for key in features.keys():
            qValue += features[key] * weights[key]

        return qValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """

        # First calculate the difference
        gamma = self.discount
        alpha = self.alpha
        valueSPrimeAPrime = self.getValue(nextState)
        qValueSA = self.getQValue(state, action)

        difference = (reward + (gamma * valueSPrimeAPrime)) - qValueSA

        # Now update the weights
        features = self.featExtractor.getFeatures(state, action)
        weights = self.getWeights()
        for key in features.keys():
            self.weights[key] = weights[key] + alpha * difference * features[key]


    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
