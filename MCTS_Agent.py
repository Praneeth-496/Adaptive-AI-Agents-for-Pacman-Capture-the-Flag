from captureAgents import CaptureAgent
import random, time, util, math
from game import Directions

####################
# Flags            #
####################

VANILLA = False

####################
# Hyperparameters  #
####################
DISCOUNT_FACTOR = 0.9
RAVE_CONSTANT = 300.0

####################
# Team creation    #
####################


def createTeam(
    firstIndex,
    secondIndex,
    isRed,
    first="AggressiveAttacker",
    second="ProactiveDefender",
    exploration=1.5,
    simulations=200,
    max_depth=12,
):
    return [
        eval(first)(firstIndex, exploration, simulations, max_depth),
        eval(second)(secondIndex),
    ]


####################
# MCTS Node Class  #
####################


class MCTSNode:
    def __init__(
        self,
        gameState,
        agent,
        action,
        parent,
        enemy_positions,
        border_positions,
        exploration=1.5,
        simulations=200,
        max_depth=12,
    ):
        self.gameState = gameState
        self.agent = agent
        self.action = action
        self.parent = parent
        self.enemy_positions = enemy_positions
        self.border_positions = border_positions
        self.child_nodes = []
        self.untried_actions = [
            a for a in gameState.getLegalActions(agent.index) if a != Directions.STOP
        ]
        random.shuffle(self.untried_actions)
        self.visits = 0
        self.q_value = 0.0

        # RAVE stats
        self.rave_visits = {}  # action -> count
        self.rave_q_values = {}  # action -> cumulative reward

        self.depth = 0 if parent is None else parent.depth + 1

        self.exploration = exploration
        self.simulations = simulations
        self.max_depth = max_depth

    def is_terminal(self):
        return self.depth >= self.max_depth or (
            not self.untried_actions and not self.child_nodes
        )

    def expand(self):
        if self.untried_actions:
            action = self.untried_actions.pop()
            next_state = self.gameState.generateSuccessor(self.agent.index, action)
            child = MCTSNode(
                next_state,
                self.agent,
                action,
                self,
                self.enemy_positions,
                self.border_positions,
                exploration=self.exploration,
                simulations=self.simulations,
                max_depth=self.max_depth,
            )
            self.child_nodes.append(child)
            return child
        return None

    def select_child(self):
        best_score = -float("inf")
        best_child = None
        k = RAVE_CONSTANT  # RAVE constant

        for child in self.child_nodes:
            if child.visits == 0:
                return child  # Prioritize unexplored nodes

            # RAVE selection formula
            q_value = child.q_value / child.visits
            rave_visits = self.rave_visits.get(child.action, 0)
            rave_q = self.rave_q_values.get(child.action, 0.0)
            rave_q_value = (rave_q / rave_visits) if rave_visits > 0 else q_value

            beta = math.sqrt(k / (3.0 * self.visits + k)) if self.visits > 0 else 1.0

            # UCT + RAVE selection formula
            exploitation = (1 - beta) * q_value + beta * rave_q_value
            exploration = self.exploration * math.sqrt(
                math.log(self.visits) / child.visits
            )
            score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def simulate_reward(self):
        current_pos = self.gameState.getAgentPosition(self.agent.index)
        if current_pos == self.gameState.getInitialAgentPosition(self.agent.index):
            return -1000.0

        reward = 0.0
        if self.enemy_positions and current_pos in self.enemy_positions:
            return -1000.0

        if self.border_positions:
            dist_home = min(
                self.agent.getMazeDistance(current_pos, bp)
                for bp in self.border_positions
            )
            reward += -dist_home

        if self.parent:
            prev_food = len(self.agent.getFood(self.parent.gameState).asList())
            curr_food = len(self.agent.getFood(self.gameState).asList())
            reward += 100.0 * (prev_food - curr_food)

        return reward

    def backpropagate(self, reward, actions_seen):
        self.visits += 1
        self.q_value += reward

        for action in actions_seen:
            self.rave_visits[action] = self.rave_visits.get(action, 0) + 1
            self.rave_q_values[action] = self.rave_q_values.get(action, 0.0) + reward

        if self.parent:
            self.parent.backpropagate(reward * DISCOUNT_FACTOR, actions_seen)

    def mcts_search(self):
        if VANILLA:
            # Use a purely random legal action
            legal_actions = [
                a
                for a in self.gameState.getLegalActions(self.agent.index)
                if a != Directions.STOP
            ]
            return random.choice(legal_actions) if legal_actions else Directions.STOP

        for _ in range(self.simulations):
            node = self
            actions_taken = []

            # Selection & Expansion
            while not node.is_terminal():
                if node.untried_actions:
                    node = node.expand()
                    actions_taken.append(node.action)
                    break
                else:
                    node = node.select_child()
                    actions_taken.append(node.action)

            # Simulation
            reward = node.simulate_reward()

            # Backpropagation
            node.backpropagate(reward, actions_seen=actions_taken)

        # Select most visited action
        if not self.child_nodes:
            return Directions.STOP
        return max(self.child_nodes, key=lambda c: c.visits).action


####################
# Aggressive Attacker #
####################


class AggressiveAttacker(CaptureAgent):

    def __init__(self, index, exploration=1.5, simulations=200, max_depth=12):
        super().__init__(index)
        self.exploration = float(exploration)
        self.simulations = int(simulations)
        self.max_depth = float(max_depth)

    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)
        width = gameState.data.layout.width
        height = gameState.data.layout.height
        border_x = (width // 2) - 1 if self.red else width // 2
        self.homeBoundary = [
            (border_x, y) for y in range(height) if not gameState.hasWall(border_x, y)
        ]

    def chooseAction(self, gameState):
        my_state = gameState.getAgentState(self.index)
        my_pos = my_state.getPosition()
        close_ghosts = []

        # Detect nearby enemy ghosts
        for i in self.getOpponents(gameState):
            opp_state = gameState.getAgentState(i)
            if not opp_state.isPacman and opp_state.getPosition() is not None:
                dist = self.getMazeDistance(my_pos, opp_state.getPosition())
                if dist <= 5:
                    close_ghosts.append(opp_state.getPosition())

        if my_state.isPacman:
            # Fixed food check using .asList()
            food_left = len(self.getFood(gameState).asList())
            if close_ghosts or my_state.numCarrying > 7 or food_left <= 2:
                root = MCTSNode(
                    gameState,
                    self,
                    None,
                    None,
                    close_ghosts,
                    self.homeBoundary,
                    exploration=self.exploration,
                    simulations=self.simulations,
                    max_depth=self.max_depth,
                )
                return root.mcts_search()
            else:
                return self.greedy_food_action(gameState)
        else:
            return self.greedy_food_action(gameState)

    def greedy_food_action(self, gameState):
        actions = [
            a for a in gameState.getLegalActions(self.index) if a != Directions.STOP
        ]
        best_action = None
        best_score = -float("inf")

        for action in actions:
            successor = gameState.generateSuccessor(self.index, action)
            new_pos = successor.getAgentState(self.index).getPosition()
            food_list = self.getFood(successor).asList()
            if food_list:
                min_dist = min(self.getMazeDistance(new_pos, f) for f in food_list)
                score = -min_dist
                if new_pos in self.getFood(gameState).asList():
                    score += 100
                if score > best_score:
                    best_score = score
                    best_action = action
        return best_action if best_action else Directions.STOP


####################
# Proactive Defender #
####################


class ProactiveDefender(CaptureAgent):

    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)
        width = gameState.data.layout.width
        height = gameState.data.layout.height
        border_x = (width // 2) - 1 if self.red else width // 2
        self.borderPositions = [
            (border_x, y) for y in range(height) if not gameState.hasWall(border_x, y)
        ]
        self.patrolTarget = None
        self.lastIntruderPos = None

    def chooseAction(self, gameState):
        my_pos = gameState.getAgentState(self.index).getPosition()
        invaders = []

        # Proper invader detection
        for i in self.getOpponents(gameState):
            opp_state = gameState.getAgentState(i)
            if opp_state.isPacman and opp_state.getPosition() is not None:
                invaders.append(opp_state.getPosition())

        if invaders:
            # Chase closest invader
            self.patrolTarget = min(
                invaders, key=lambda x: self.getMazeDistance(my_pos, x)
            )
            self.lastIntruderPos = self.patrolTarget
        else:
            # Patrol middle border position
            if self.borderPositions:
                mid_idx = len(self.borderPositions) // 2
                self.patrolTarget = self.borderPositions[mid_idx]
            else:
                self.patrolTarget = gameState.getInitialAgentPosition(self.index)

        actions = [
            a for a in gameState.getLegalActions(self.index) if a != Directions.STOP
        ]
        best_action = None
        best_dist = float("inf")

        if self.patrolTarget:
            for action in actions:
                successor = gameState.generateSuccessor(self.index, action)
                new_pos = successor.getAgentState(self.index).getPosition()
                dist = self.getMazeDistance(new_pos, self.patrolTarget)
                if dist < best_dist:
                    best_dist = dist
                    best_action = action

        return (
            best_action
            if best_action
            else random.choice(actions) if actions else Directions.STOP
        )