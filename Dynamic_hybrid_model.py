# Dynamic_hybrid_model.py
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

class PathAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        colors = ['blue','green','purple','orange','cyan','magenta','yellow','pink','brown','olive']
        self.color = colors[unique_id % len(colors)]
        # path_history will be initialized after the agent is placed on the grid
        self.path_history = []
        self.steps_taken = 0
        self.start_time = datetime.now()
        # DON'T hardcode goal here — use model grid size in model init
        self.position = None   # will be synced with self.pos
        self.pheromone_strength = 1.0
        self.velocity = np.zeros(2)
        self.best_position = None
        self.best_fitness = float('inf')
        self.network = NeuralNetwork(4, 32, 4)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)

    def sync_position(self):
        # Keep self.position in sync with mesa's internal .pos
        if getattr(self, "pos", None) is not None:
            self.position = tuple(self.pos)
        # ensure path_history initialized
        if not self.path_history and self.position is not None:
            self.path_history = [self.position]

    def move(self):
        # keep position synced at start of decision
        self.sync_position()
        if self.position is None:
            return

        x, y = self.position
        if (x, y) == self.model.goal:
            return

        state = torch.FloatTensor(self.get_state())
        action_probs = self.network(state).detach().numpy()

        goal_direction = np.array([self.model.goal[0] - x, self.model.goal[1] - y])
        goal_direction = goal_direction / (np.linalg.norm(goal_direction) + 1e-9)

        moves = [(0,1),(1,0),(0,-1),(-1,0)]
        move_scores = []
        for i, (dx, dy) in enumerate(moves):
            new_pos = (x + dx, y + dy)
            move_direction = np.array([dx, dy])
            is_valid = (0 <= new_pos[0] < self.model.grid.width and
                        0 <= new_pos[1] < self.model.grid.height and
                        (self.model.grid.is_cell_empty(new_pos) or new_pos == self.model.goal))
            goal_alignment = np.dot(move_direction, goal_direction)
            move_score = 0.95 * goal_alignment + 0.1 * float(action_probs[i])

            # Pheromone map uses (x,y) indexing
            pheromone_weight = 0.0
            if is_valid:
                pheromone_weight = float(self.model.pheromone_map[new_pos[0], new_pos[1]])
            move_score += 0.2 * pheromone_weight

            velocity_alignment = np.dot(move_direction, self.velocity)
            move_score += 0.1 * velocity_alignment

            if not is_valid:
                move_score = -float('inf')

            move_scores.append(move_score)

        best_move_idx = int(np.argmax(move_scores))
        dx, dy = moves[best_move_idx]
        new_pos = (x + dx, y + dy)

        # move if valid
        if (0 <= new_pos[0] < self.model.grid.width and
            0 <= new_pos[1] < self.model.grid.height and
            (self.model.grid.is_cell_empty(new_pos) or new_pos == self.model.goal)):
            # use mesa's API
            try:
                self.model.grid.move_agent(self, new_pos)
            except Exception:
                # fallback: set pos directly (rare)
                self.pos = new_pos
            # sync internal position
            self.sync_position()
            # deposit pheromone at the updated location
            self.deposit_pheromone()

            reward = self.calculate_reward()
            self.train_network(state, best_move_idx, reward)

        # append the new position to path history (ensures collector sees path)
        if self.position is not None:
            if not self.path_history or self.path_history[-1] != self.position:
                self.path_history.append(self.position)

        # increment steps if not yet at goal
        if self.position != self.model.goal:
            self.steps_taken += 1

    def calculate_reward(self):
        if self.position is None:
            return 0.0
        dx = self.position[0] - self.model.goal[0]
        dy = self.position[1] - self.model.goal[1]
        distance_to_goal = np.sqrt(dx*dx + dy*dy)
        reward = 10.0 / (distance_to_goal + 1.0)
        if self.position == self.model.goal:
            reward += 50.0
        return reward

    def train_network(self, state, action, reward):
        try:
            self.optimizer.zero_grad()
            action_probs = self.network(state)
            loss = -torch.log(action_probs[action] + 1e-9) * reward
            loss.backward()
            self.optimizer.step()
        except Exception as e:
            print("Training error:", e)
            # reinit network
            self.network = NeuralNetwork(4, 32, 4)
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)

    def step(self):
        # run move (syncs position and records path)
        self.move()

    def get_state(self):
        # normalized by grid size (not hard-coded)
        w = max(1, self.model.grid.width)
        h = max(1, self.model.grid.height)
        if self.position is None:
            self.sync_position()
        x, y = (self.position if self.position is not None else (0,0))
        gx, gy = self.model.goal
        return [x / w, y / h, gx / w, gy / h]

    def update_velocity(self):
        global_best = self.model.get_global_best() or np.array(self.position)
        try:
            cognitive = 2.0 * np.random.random() * (np.array(self.best_position) - np.array(self.position))
        except Exception:
            cognitive = 0.0
        social = 2.0 * np.random.random() * (np.array(global_best) - np.array(self.position))
        self.velocity = 0.7 * self.velocity + cognitive + social

    def deposit_pheromone(self):
        if self.position is None:
            return
        x, y = self.position
        # increment pheromone at index (x,y)
        try:
            self.model.pheromone_map[x, y] += self.pheromone_strength
        except Exception:
            pass

    def calculate_fitness(self):
        if self.position is None:
            return float('inf')
        return np.sqrt((self.position[0] - self.model.goal[0])**2 + (self.position[1] - self.model.goal[1])**2)

class ObstacleAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
    def step(self):
        # Random move obstacles to a valid free cell (not start or goal)
        current_pos = self.pos
        possible_steps = [(current_pos[0] + dx, current_pos[1] + dy)
                          for dx, dy in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)]]
        valid_steps = [pos for pos in possible_steps
                       if 0 <= pos[0] < self.model.grid.width
                       and 0 <= pos[1] < self.model.grid.height
                       and pos != (0,0)
                       and pos != self.model.goal
                       and self.model.grid.is_cell_empty(pos)]
        if valid_steps:
            new_position = self.random.choice(valid_steps)
            try:
                self.model.grid.move_agent(self, new_position)
            except Exception:
                self.pos = new_position

class PathFindingModel(Model):
    def __init__(self, width=20, height=20, n_agents=10, n_obstacles=40):
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.pheromone_map = np.zeros((width, height), dtype=float)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.accuracy_file = "simulation_accuracy_log.csv"
        self.global_best = None
        self.global_best_fitness = float('inf')
        self.num_obstacles = n_obstacles
        self.goal = (width - 1, height - 1)
        self.running = True
        self.all_paths = {}
        self.time_steps = {}

        # create obstacles
        for i in range(n_obstacles):
            x = self.random.randrange(width)
            y = self.random.randrange(height)
            if (x,y) != (0,0) and (x,y) != self.goal and self.grid.is_cell_empty((x,y)):
                obstacle = ObstacleAgent(i + 1000, self)
                self.grid.place_agent(obstacle, (x,y))
                self.schedule.add(obstacle)

        # create path agents
        for i in range(n_agents):
            agent = PathAgent(i, self)
            self.schedule.add(agent)
            # place agent on grid (sets agent.pos)
            self.grid.place_agent(agent, (0,0))
            # sync internal position and ensure history starts
            agent.sync_position()

        self.datacollector = DataCollector(model_reporters={"Average_Fitness": lambda m: self.get_average_fitness()})

    def step(self):
        # run one schedule step (agents will call step)
        self.schedule.step()
        current_step = self.schedule.steps
        if current_step % 50 == 0:
            print(f"\nStep {current_step}")

        # sync path history & time steps for path agents
        path_agents = [a for a in self.schedule.agents if isinstance(a, PathAgent)]
        all_finished = all((a.position == self.goal) for a in path_agents if a.position is not None)

        for agent in path_agents:
            if agent.unique_id not in self.all_paths:
                self.all_paths[agent.unique_id] = []
                self.time_steps[agent.unique_id] = []
            # ensure sync
            agent.sync_position()
            self.all_paths[agent.unique_id].append(agent.position)
            self.time_steps[agent.unique_id].append(current_step)

        if all_finished:
            self.running = False
            # plotting calls: they may try to show; in tests we monkeypatch plt.show
            self.plot_time_graphs()
            self.plot_paths()
            self.plot_path_efficiency()
            return

        # move obstacles explicitly if needed (they already move in schedule.step())
        # apply pheromone evaporation
        self.pheromone_map *= 0.95
        self.datacollector.collect(self)

    def save_accuracy_record(self, overall_accuracy):
        record = {
            'Run ID': self.run_id,
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Overall Accuracy': overall_accuracy,
            'Num Agents': len([a for a in self.schedule.agents if isinstance(a, PathAgent)]),
            'Num Obstacles': self.num_obstacles
        }
        try:
            df = pd.read_csv(self.accuracy_file)
        except Exception:
            df = pd.DataFrame(columns=record.keys())
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        try:
            df.to_csv(self.accuracy_file, index=False)
        except Exception:
            pass

        print("\nAccuracy History:")
        print(df.tail(20))
        print(f"\nAverage Accuracy across all runs: {df['Overall Accuracy'].mean():.2f}%")
        print(f"Best Accuracy so far: {df['Overall Accuracy'].max():.2f}%")

    def get_average_fitness(self):
        active = [a for a in self.schedule.agents if isinstance(a, PathAgent) and a.position != self.goal]
        if not active:
            return 0.0
        fitnesses = [a.calculate_fitness() for a in active]
        return float(sum(fitnesses) / len(fitnesses))

    def get_global_best(self):
        current_best = None
        current_best_fitness = float('inf')
        for agent in self.schedule.agents:
            if isinstance(agent, PathAgent) and agent.best_fitness < current_best_fitness:
                current_best_fitness = agent.best_fitness
                current_best = agent.best_position
        if current_best is not None and current_best_fitness < self.global_best_fitness:
            self.global_best_fitness = current_best_fitness
            self.global_best = current_best
        return self.global_best

    def plot_time_graphs(self):
        plt.figure(figsize=(12,6))
        plt.title("Steps Taken by Each Agent to Reach the Goal")
        agent_steps = {}
        agent_colors = {}
        for agent in self.schedule.agents:
            if isinstance(agent, PathAgent):
                agent_id = agent.unique_id
                agent_colors[agent_id] = agent.color
                path = list(filter(lambda p: p is not None, self.all_paths.get(agent_id, [])))
                goal_pos = self.goal
                try:
                    steps = path.index(goal_pos) + 1
                except ValueError:
                    steps = len(path)
                agent_steps[agent_id] = steps
        agent_ids = list(agent_steps.keys())
        steps_taken = list(agent_steps.values())
        colors = [agent_colors[i] for i in agent_ids]
        bars = plt.bar(agent_ids, steps_taken, color=colors)
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., h, f'{int(h)}', ha='center', va='bottom')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_path_efficiency(self):
        path_agents = [a for a in self.schedule.agents if isinstance(a, PathAgent)]
        expected_optimal_steps = (self.grid.width - 1) + (self.grid.height - 1)  # Manhattan-like baseline
        agent_ids = []
        eff_vals = []
        for agent in path_agents:
            if agent.steps_taken > 0:
                eff = (expected_optimal_steps / agent.steps_taken) * 100
                agent_ids.append(agent.unique_id)
                eff_vals.append(min(eff, 100.0))
        if agent_ids:
            plt.figure(figsize=(10,6))
            plt.plot(agent_ids, eff_vals, marker='o')
            overall_accuracy = float(sum(eff_vals) / len(eff_vals))
            self.save_accuracy_record(overall_accuracy)
            plt.xlabel("Agent ID"); plt.ylabel("Path Efficiency (%)")
            plt.title("Path Efficiency of Agents")
            plt.tight_layout()
            plt.show()

    def plot_paths(self):
        plt.figure(figsize=(10,10))
        plt.title("Agent Paths to the Goal")
        plt.xlim(-1, self.grid.width)
        plt.ylim(-1, self.grid.height)
        plt.grid(True, linestyle='--', alpha=0.6)
        for agent_id, path in self.all_paths.items():
            path = [p for p in path if p is not None]
            if not path:
                continue
            arr = np.array(path)
            x, y = arr.T
            color = plt.cm.tab10(agent_id % 10)
            plt.plot(x, y, '-', color=color, alpha=0.6, linewidth=2)
            plt.plot(x, y, 'o', color=color, markersize=4)
            plt.plot(x[0], y[0], 'o', color=color, markersize=8)
            plt.plot(x[-1], y[-1], 's', color=color, markersize=8)
        obs_positions = [tuple(a.pos) for a in self.schedule.agents if isinstance(a, ObstacleAgent)]
        if obs_positions:
            xs, ys = zip(*obs_positions)
            plt.scatter(xs, ys, c='black', marker='s', s=60, label='Obstacles')
        plt.scatter([0],[0], color='green', marker='*', s=150, label='Start')
        gx, gy = self.goal
        plt.scatter([gx],[gy], color='red', marker='*', s=150, label='Goal')
        plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
        plt.tight_layout()
        plt.show()

# portrayal helper used by your server
def agent_portrayal(agent):
    pos = getattr(agent, "pos", None)
    # Base portrayal for any agent
    if isinstance(agent, PathAgent):
        color = getattr(agent, "color", "blue")
        return {"Shape":"circle", "Color": color, "Filled":"true", "Layer":2, "r":0.5}
    if isinstance(agent, ObstacleAgent):
        return {"Shape":"rect", "Color":"black", "Filled":"true", "Layer":1, "w":1, "h":1}
    # fallback
    return {"Shape":"circle", "Color":"gray", "Filled":"true", "Layer":0, "r":0.4}

# If you want to run as a server, keep your ModularServer code outside this file or import PathFindingModel externally.
