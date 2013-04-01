# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
 
"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""
 
import util
 
class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).
 
    You do not need to change anything in this class, ever.
    """
 
    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()
 
    def isGoalState(self, state):
        """
         state: Search state
 
        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()
 
    def getSuccessors(self, state):
        """
          state: Search state
 
        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
       cost of expanding to that successor
        """
        util.raiseNotDefined()
 
    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take
 
        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()
 
 
def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]
 
def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first
 
    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm
 
    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
 
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
   
    fringe = util.Stack()
    closed = set()
    resultPath = {}
   
    for successor in problem.getSuccessors(problem.getStartState()):
        fringe.push(successor)
        closed.add(problem.getStartState())
        resultPath[successor] = "START"
       
    while not fringe.isEmpty():
        current = fringe.pop()
 
        if current[0] in closed: continue
        closed.add(current[0])
               
        if problem.isGoalState(current[0]):
            goal = current
            break
           
        successors = problem.getSuccessors(current[0])
        for successor in successors:
            fringe.push(successor)
            if not successor[0] in closed:
                resultPath[successor] = current
       
        if len(successors) == 0:
            while resultPath[-1] != fringe[-1]:
                resultPath.pop()
   
    actions = []
    current = goal
    while current != "START":
        actions.append(current[1])
        current = resultPath[current]
   
    actions.reverse()
    return actions
 
def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
   
    fringe = util.Queue()
    closed = set()
    resultPath = {}
    goal = "START"
   
    for successor in problem.getSuccessors(problem.getStartState()):
        fringe.push(successor)
        closed.add(problem.getStartState())
        resultPath[successor] = "START"

    while not fringe.isEmpty():
        current = fringe.pop()
 
        if current[0] in closed: continue
        closed.add(current[0])
               
        if problem.isGoalState(current[0]):
            goal = current
            break
           
        successors = problem.getSuccessors(current[0])
        for successor in successors:           
            if not successor in fringe.list:
                fringe.push(successor)
            if not successor[0] in closed:
                resultPath[successor] = current
       
        if len(successors) == 0:
            while resultPath[-1] != fringe[-1]:
                resultPath.pop()
   
    actions = []
    current = goal
    while current != "START":
        actions.append(current[1])
        current = resultPath[current]
   
    actions.reverse()
    return actions
 
def uniformCostSearch(problem):
    "Search the node of least total cost first. "
 
    fringe = util.PriorityQueue()
    closed = set()
    resultPath = {}
   
    for successor in problem.getSuccessors(problem.getStartState()):
        fringe.push(successor, successor[2])
        closed.add(problem.getStartState())
        resultPath[successor] = "START"
       
    while not fringe.isEmpty():
        current = fringe.pop()
 
        if current[0] in closed: continue
        closed.add(current[0])
               
        if problem.isGoalState(current[0]):
            goal = current
            break
        
        # Get the total cost of the actions up until this node
        actions = getActionsFromResultPath(current, "START", resultPath)
        totalCost = problem.getCostOfActions(actions)
        
        successors = problem.getSuccessors(current[0])
        for successor in successors:
            fringe.push(successor, totalCost + successor[2])
            if not successor[0] in closed:
                resultPath[successor] = current

    actions = getActionsFromResultPath(goal, "START", resultPath)
    return actions
 
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0
 
def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    
    fringe = util.PriorityQueue()
    closed = set()
    resultPath = {}
   
    for successor in problem.getSuccessors(problem.getStartState()):
        heuristicCost = heuristic(successor[0], problem)
        fringe.push(successor, heuristicCost + successor[2])
        closed.add(problem.getStartState())
        resultPath[successor] = "START"
       
    while not fringe.isEmpty():
        current = fringe.pop()
 
        if current[0] in closed: continue
        closed.add(current[0])
               
        if problem.isGoalState(current[0]):
            goal = current
            break
        
        # Get the total cost of the actions up until this node
        actions = getActionsFromResultPath(current, "START", resultPath)        
        totalCost = problem.getCostOfActions(actions)
        
        successors = problem.getSuccessors(current[0])
        for successor in successors:
            heuristicCost = heuristic(successor[0], problem)
            fringe.push(successor, totalCost + heuristicCost + successor[2])
            if not successor[0] in closed:
                resultPath[successor] = current

    actions = getActionsFromResultPath(goal, "START", resultPath)
    return actions
    
def getActionsFromResultPath(goalNode, startValue, resultPath):
    actions = []
    current = goalNode
    while current != startValue:
        actions.append(current[1])
        current = resultPath[current]
   
    actions.reverse()
    return actions
 
 
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch