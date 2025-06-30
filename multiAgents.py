# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from util import Queue
from game import Directions
import random, util
from game import Agent

import heapq

FOOD_ENCORAGEMENT = 10.0
DANGER_GHOST_TOLERANCE = 2
DANGER_GHOST_PANELTY = 1000
NORMAL_GHOST_PANELTY = 10.0
from util import Queue



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

        getAction takes a GameState and returns some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        # Generate the successor game state after taking the given action
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        total_scared_time = sum(newScaredTimes)

        # Initialize the score with the current game score
        score = successorGameState.getScore()

        # 获取迷宫布局以计算真实路径
        walls = currentGameState.getWalls()

        pacpos = (int(newPos[0]), int(newPos[1]))
        point_distance = {(pacpos): 0}

        # BFS来计算实际路径距离
        def getTrueDistance(start, goal, walls):
            from util import Queue
            
            if start == goal:
                return 0
            
            queue = Queue()
            queue.push((start, 0))  # (点坐标, 距离)
            visited = set()

            while not queue.isEmpty():
                current_position, distance = queue.pop()

                if current_position in visited:
                    continue
                visited.add(current_position)

                if current_position == goal:
                    point_distance[current_position] = distance
                    return distance

                # 添加相邻合法走位（东南西北）
                x, y = current_position
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    next_position = (x + dx, y + dy)
                    if not walls[int(next_position[0])][int(next_position[1])]:  # 非墙
                        queue.push((next_position, distance + 1))

            return float('inf')  # 默认

        s1 = score
        # 计算与所有糖豆的实际路径距离
        foodDistances = []
        for food in newFood.asList():
            # 使用字典缓存之前计算过的距离
            if food in point_distance:
                foodDistances.append(point_distance[food])
            else:
                distance = getTrueDistance(newPos, food, walls)
                point_distance[food] = distance  # 缓存当前食物的距离
                foodDistances.append(distance)
        # print(f"{action}: {newFood.asList()}, {foodDistances}")
        if foodDistances:
            score += 20.0 / (min(foodDistances) + 1)  # Encourage moving towards the closest food
        
        s2 = score
        # print(s2 - s1)
        # 靠近幽灵的惩罚项（非胶囊状态）
        # 小于2惩罚急剧增加。
        for i, ghostState in enumerate(newGhostStates):
            position = ghostState.getPosition()
            position_int = (int(position[0]), int(position[1]))
            if position_int in point_distance:
                ghostDistance = point_distance[position_int]
            else:
                ghostDistance = getTrueDistance(newPos, position_int, walls)
            if newScaredTimes[i] == 0:  # If the ghost is not scared
                if ghostDistance < 2:
                    score -= 1000 
                else:
                    score -= 8.0 / ghostDistance


        s3 = score
        # 胶囊激励
        capsules = currentGameState.getCapsules()
        capsuleDistances = [manhattanDistance(newPos, capsule) for capsule in capsules]
        if capsuleDistances and total_scared_time == 0:
            capsule_encoragement = 15.0 / (min(capsuleDistances) + 0.1)
            score += capsule_encoragement


        s4 = score
        # 追击幽灵激励
        for i, ghostState in enumerate(newGhostStates):
            if newScaredTimes[i] > 0:
                position = ghostState.getPosition()
                position_int = (int(position[0]), int(position[1]))
                if position_int in point_distance:
                    ghostDistance = point_distance[position_int]
                else:
                    ghostDistance = getTrueDistance(newPos, position_int, walls)
                score += 500.0 / (ghostDistance + 1)
                # print(f"ghost {i} position is {position_int}, distance is {ghostDistance}")
        s5 = score
        # print(s5 - s4)

        return score

    
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
    该类为所有多智能体搜索器提供了一些通用元素。这里定义的任何方法都可以在 MinimaxPacmanAgent、AlphaBetaPacmanAgent 和 ExpectimaxPacmanAgent 中使用。

    你不需要在这里进行任何更改，但如果你想为所有对抗性搜索代理添加功能，可以在此添加。不过，请不要删除任何内容。

    注意:这是一个抽象类,不能直接实例化。它只是部分定义,旨在被扩展。Agent(在 game.py 中)也是一个抽象类。
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 4).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
