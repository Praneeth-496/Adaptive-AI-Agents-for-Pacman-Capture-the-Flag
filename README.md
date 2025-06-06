# Adaptive AI Agents for Pacman Capture the Flag

This repository contains the implementation of two adaptive AI agents for the Pacman Capture the Flag game: one based on Monte Carlo Tree Search (MCTS) with Rapid Action Value Estimation (RAVE) and another using heuristic strategies.

## Project Overview

In this project, we developed and evaluated two different approaches to creating adaptive agents for the Pacman Capture the Flag game:

1. **MCTS Agent**: A simulation-based planning agent that uses Monte Carlo Tree Search enhanced with RAVE for efficient action evaluation under real-time constraints.

2. **Heuristic Agent**: A rule-based agent that relies on domain-specific strategies including greedy food selection when attacking and patrol-based policies when defending.

Both agents dynamically switch between attacking as Pacman and defending as a ghost based on the game state, rather than adhering to fixed roles.

## Features

### MCTS Agent
- Simulation-based planning with configurable parameters
- Rapid Action Value Estimation (RAVE) for faster value learning
- Dynamic role switching between offense and defense
- Real-time decision making within one-second constraints
- Adaptive exploration based on game state

### Heuristic Agent
- Greedy food selection strategy when on offense
- Patrol-based policy when on defense
- Threat-aware escape maneuvers to avoid capture
- Dynamic role switching based on position and game state
- Rule-based decision making with handcrafted evaluation functions

## Experimental Results

Our comprehensive evaluation included three main experiment sets, each consisting of 108 matches:

1. **MCTS Self-Play**: Two MCTS agents competing against each other
2. **Heuristic Self-Play**: Two heuristic agents competing against each other
3. **MCTS vs. Heuristic**: Direct comparison between the two agent types

Key findings:
- The MCTS agent consistently outperformed the heuristic agent, achieving a 100% win rate in head-to-head matches
- MCTS performance improved with higher exploration constants
- ELO ratings showed a significant skill gap between MCTS and heuristic approaches
- Simulation-based planning demonstrated superior adaptability compared to rule-based strategies

## Implementation Details

### MCTS Configuration
- 200 simulations per move
- Maximum search depth of 12
- Exploration constants tested: 1.0, 1.5, 2.0
- RAVE implementation for faster convergence

### Evaluation Metrics
- Win/loss/tie rates
- Average normalized score
- ELO rating system for skill progression tracking

## Requirements

- Python 3.6+
- Pacman Capture the Flag environment
- NumPy
- Matplotlib (for visualization)

## Usage

1. Clone the repository:
```
git clone https://github.com/yourusername/pacman-capture-flag-ai.git
```

2. Navigate to the project directory:
```
cd pacman-capture-flag-ai
```

3. Run a match between MCTS and heuristic agents:
```
python capture.py -r mcts_agent.py -b heuristic_agent.py
```

4. Run a tournament with multiple configurations:
```
python tournament.py --config tournament_config.json
```

## Project Structure

```
├── agents/
│   ├── mcts_agent.py       # MCTS agent implementation
│   ├── heuristic_agent.py  # Heuristic agent implementation
│   └── base_agent.py       # Common agent functionality
├── capture.py              # Main game environment
├── tournament.py           # Tournament runner
├── analysis/
│   ├── elo_calculator.py   # ELO rating calculation
│   └── visualize.py        # Result visualization
├── configs/                # Configuration files
└── results/                # Experiment results
```

## Authors

- Praneeth Dathu 
- Sai Krishna Reddy 
- Gaurisankar Jayadas 

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- University of Leiden, Modern Gaming AI course
- Berkeley AI Pacman projects for the base environment

