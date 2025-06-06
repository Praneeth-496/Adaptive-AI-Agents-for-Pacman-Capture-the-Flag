from captureAgents import CaptureAgent
import random, util
from game import Directions

####################
# Team creation    #
####################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'HeuristicAttacker', second = 'HeuristicDefender'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

####################
# Heuristic Attacker #
####################

class HeuristicAttacker(CaptureAgent):
    def chooseAction(self, gameState):
        my_state = gameState.getAgentState(self.index)
        my_pos = my_state.getPosition()
        actions = [a for a in gameState.getLegalActions(self.index) if a != Directions.STOP]
        
        # Threat detection
        threats = []
        for i in self.getOpponents(gameState):
            enemy = gameState.getAgentState(i)
            if not enemy.isPacman and enemy.getPosition():
                dist = self.getMazeDistance(my_pos, enemy.getPosition())
                if dist <= 5 and enemy.scaredTimer == 0:
                    threats.append(enemy.getPosition())

        best_action = Directions.STOP
        best_score = -float('inf')
        
        for action in actions:
            score = 0
            successor = gameState.generateSuccessor(self.index, action)
            new_pos = successor.getAgentState(self.index).getPosition()
            
            # Food collection
            food_list = self.getFood(successor).asList()
            if food_list:
                closest_food = min(self.getMazeDistance(new_pos, f) for f in food_list)
                score += 20 / (closest_food + 1)
                if new_pos in self.getFood(gameState).asList():
                    score += 100

            # Threat avoidance
            if threats:
                closest_threat = min(self.getMazeDistance(new_pos, t) for t in threats)
                score -= 50 / (closest_threat + 1)
                if closest_threat <= 1:
                    score -= 1000

            # Return strategy
            if my_state.numCarrying > 3:
                home_dist = min(self.getMazeDistance(new_pos, edge) 
                            for edge in self.getHomeEdges(gameState))
                score += 40 / (home_dist + 1)

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def getHomeEdges(self, gameState):
        width = gameState.data.layout.width
        border_x = (width // 2) - 1 if self.red else width // 2
        return [(border_x, y) for y in range(gameState.data.layout.height) 
                if not gameState.hasWall(border_x, y)]

####################
# Heuristic Defender #
####################

class HeuristicDefender(CaptureAgent):
    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)
        self.initialFood = set(self.getFoodYouAreDefending(gameState).asList())
        self.patrol_points = self.createPatrolRoute(gameState)
        self.current_target = 0

    def chooseAction(self, gameState):
        my_pos = gameState.getAgentState(self.index).getPosition()
        actions = [a for a in gameState.getLegalActions(self.index) if a != Directions.STOP]

        # Detect invaders and food theft
        invaders = []
        for i in self.getOpponents(gameState):
            enemy = gameState.getAgentState(i)
            if enemy.isPacman and enemy.getPosition():
                invaders.append(enemy.getPosition())
                
        current_food = set(self.getFoodYouAreDefending(gameState).asList())
        missing_food = self.initialFood - current_food

        # Determine target
        target = self.patrol_points[self.current_target]
        if invaders:
            target = min(invaders, key=lambda x: self.getMazeDistance(my_pos, x))
        elif missing_food:
            target = min(missing_food, key=lambda x: self.getMazeDistance(my_pos, x))

        # Update patrol index
        if my_pos == self.patrol_points[self.current_target]:
            self.current_target = (self.current_target + 1) % len(self.patrol_points)

        # Move towards target
        return min(actions, 
                   key=lambda a: self.getMazeDistance(
                       gameState.generateSuccessor(self.index, a).getAgentState(self.index).getPosition(),
                       target
                   ), 
                   default=Directions.STOP)

    def createPatrolRoute(self, gameState):
        width = gameState.data.layout.width
        height = gameState.data.layout.height
        border_x = (width // 2) - 1 if self.red else width // 2
        vertical = [y for y in range(1, height-1) if not gameState.hasWall(border_x, y)]
        return [(border_x, y) for y in vertical + vertical[-2:0:-1]]