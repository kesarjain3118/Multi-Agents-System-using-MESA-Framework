from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.modules import ChartModule
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
        # Assign unique color to each agent
        colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'pink', 'brown', 'olive']
        self.color = colors[unique_id % len(colors)]
        self.path_history = [(0,0)] # Track path
        self.steps_taken = 0
        self.start_time = datetime.now()
        self.position = (0, 0)
        self.goal = (19, 19)
        self.pheromone_strength = 1.0
        self.velocity = np.zeros(2)
        self.best_position = None
        self.best_fitness = float('inf')
        self.network = NeuralNetwork(4, 32, 4)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        
    def move(self):
        # Get current position
        x, y = self.position
    
        if (x, y) == self.goal:
            return
        
        # Get state and NN prediction
        state = torch.FloatTensor(self.get_state())
        action_probs = self.network(state)
    
        # Combine NN output with goal direction bias
        goal_direction = np.array([
            self.goal[0] - x,
            self.goal[1] - y
        ])
        goal_direction = goal_direction / (np.linalg.norm(goal_direction) + 1e-6)
    
        # Available moves
        moves = [(0,1), (1,0), (0,-1), (-1,0)]
    
        # Score each move combining NN prediction and goal bias
        move_scores = []
        for i, (dx, dy) in enumerate(moves):
            new_pos = (x + dx, y + dy)
            move_direction = np.array([dx, dy])
        
            # Check if move leads to obstacle
            is_valid = (0 <= new_pos[0] < self.model.grid.width and 
                   0 <= new_pos[1] < self.model.grid.height and 
                   (self.model.grid.is_cell_empty(new_pos) or new_pos == self.goal))
        
            goal_alignment = np.dot(move_direction, goal_direction)
            move_score = 0.95 * goal_alignment + 0.1 * action_probs[i].item()
        
            # Add ACO pheromone bias
            pheromone_weight = self.model.pheromone_map[new_pos] if is_valid else 0
            pheromone_bias = 0.2 * pheromone_weight
            move_score += pheromone_bias

            # Add PSO velocity alignment
            velocity_alignment = np.dot(move_direction, self.velocity)
            move_score += 0.1 * velocity_alignment
        
            if not is_valid:
                move_score = -float('inf')  # Heavily penalize invalid moves
            
            move_scores.append(move_score)
    
        best_move_idx = np.argmax(move_scores)
        dx, dy = moves[best_move_idx]
        new_pos = (x + dx, y + dy)
    
        if (0 <= new_pos[0] < self.model.grid.width and 0 <= new_pos[1] < self.model.grid.height and (self.model.grid.is_cell_empty(new_pos) or new_pos == self.goal)):
            self.model.grid.move_agent(self, new_pos)
            self.position = new_pos
            self.deposit_pheromone()
        
            reward = self.calculate_reward()
            self.train_network(state, best_move_idx, reward)

        if self.position != self.goal:
            self.steps_taken += 1

    def calculate_reward(self):
        # Calculate distance-based reward
        distance_to_goal = np.sqrt((self.position[0] - self.goal[0])**2 + (self.position[1] - self.goal[1])**2)
    
        # Higher reward for being closer to goal
        reward = 10.0 / (distance_to_goal + 1)
    
        # Bonus reward for reaching goal
        if self.position == self.goal:
            reward += 50.0
        
        return reward

    def train_network(self, state, action, reward):
        try:
            self.optimizer.zero_grad()
            action_probs = self.network(state)
            loss = -torch.log(action_probs[action]) * reward
            loss.backward()
            self.optimizer.step()
        except RuntimeError as e:
            print(f"Training error: {e}")
            # Reinitialize network if needed
            self.network = NeuralNetwork(4, 32, 4)
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)

    def step(self):
        self.move()

    def get_state(self):
        x, y = self.position
        goal_x, goal_y = self.goal
        return [x/20, y/20, goal_x/20, goal_y/20]
        
    def get_pheromone_weights(self):
        weights = np.zeros(4)
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            new_pos = (self.position[0] + dx, self.position[1] + dy)
            if self.model.grid.is_cell_empty(new_pos):
                weights[self.direction_to_index(dx, dy)] = self.model.pheromone_map[new_pos]
        return weights
        
    def update_velocity(self):
        global_best = self.model.get_global_best()
        cognitive = 2.0 * np.random.random() * (np.array(self.best_position) - np.array(self.position))
        social = 2.0 * np.random.random() * (np.array(global_best) - np.array(self.position))
        self.velocity = 0.7 * self.velocity + cognitive + social
        
    def deposit_pheromone(self):
        self.model.pheromone_map[self.position] += self.pheromone_strength
        
    def calculate_fitness(self):
        return np.sqrt((self.position[0] - self.goal[0])**2 + (self.position[1] - self.goal[1])**2)
        
    @staticmethod
    def direction_to_index(dx, dy):
        return {(0,1): 0, (1,0): 1, (0,-1): 2, (-1,0): 3}[(dx, dy)]
        
    def get_new_position(self, direction):
        moves = [(0,1), (1,0), (0,-1), (-1,0)]
        dx, dy = moves[direction]
        return (self.position[0] + dx, self.position[1] + dy)
    
class ObstacleAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
    
    def step(self):
        # Get current position
        current_pos = self.pos
        # Random movement
        possible_steps = [
            (current_pos[0] + dx, current_pos[1] + dy)
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
        ]
        # Filter valid positions
        valid_steps = [
            pos for pos in possible_steps
            if 0 <= pos[0] < self.model.grid.width
            and 0 <= pos[1] < self.model.grid.height
            and pos != (0,0)
            and pos != (19,19)
            and self.model.grid.is_cell_empty(pos)
        ]
        
        if valid_steps:
            new_position = self.random.choice(valid_steps)
            self.model.grid.move_agent(self, new_position)

class PathFindingModel(Model):
    def __init__(self, width, height, n_agents, n_obstacles):
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.pheromone_map = np.zeros((width, height))
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.accuracy_file = "simulation_accuracy_log.csv"
        self.global_best = None
        self.global_best_fitness = float('inf')
        self.num_obstacles = n_obstacles
        self.goal = (width - 1, height - 1)
        self.running = True
        self.all_paths = {}
        self.time_steps = {}
        
        # Create obstacle agents
        for i in range(n_obstacles):
            x = self.random.randrange(width)
            y = self.random.randrange(height)
            if (x,y) != (0,0) and (x,y) != (width-1, height-1):
                obstacle = ObstacleAgent(i + 1000, self)
                self.grid.place_agent(obstacle, (x,y))
                self.schedule.add(obstacle)

        # Create path agents
        for i in range(n_agents):
            agent = PathAgent(i, self)
            self.schedule.add(agent)
            self.grid.place_agent(agent, (0, 0))  
     
        self.datacollector = DataCollector(
            model_reporters={"Average_Fitness": lambda m: self.get_average_fitness()}
        )
        
    def step(self):
        self.schedule.step()
        # Check if all agents reached the goal
        current_step = self.schedule.steps

        if current_step % 50 == 0:
            print(f"\nStep {current_step}")
        
        path_agents = [agent for agent in self.schedule.agents if isinstance(agent, PathAgent)]
        all_agents_finished = all(agent.position == self.goal for agent in path_agents)
        
        # Update paths dictionary
        for agent in path_agents:
            if agent.unique_id not in self.all_paths:
                self.all_paths[agent.unique_id] = []
                self.time_steps[agent.unique_id] = []
            self.all_paths[agent.unique_id].append(agent.position)
            self.time_steps[agent.unique_id].append(current_step)
        
        if all_agents_finished:
            self.running = False
            self.plot_time_graphs()
            self.plot_paths()
            self.plot_path_efficiency()
            return

        # Existing step code continues here
        obstacle_agents = [agent for agent in self.schedule.agents if isinstance(agent, ObstacleAgent)]
        for obstacle in obstacle_agents:
            obstacle.step()
        
        for agent in self.schedule.agents:
            if isinstance(agent, PathAgent):
                agent.step()
        
        self.pheromone_map *= 0.95
        self.datacollector.collect(self)

    def save_accuracy_record(self, overall_accuracy):
        record = {
            'Run ID': self.run_id,
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Overall Accuracy': overall_accuracy,
            'Num Agents': len([agent for agent in self.schedule.agents if isinstance(agent, PathAgent)]),
            'Num Obstacles': self.num_obstacles
        }
        
        try:
            df = pd.read_csv(self.accuracy_file)
        except FileNotFoundError:
            df = pd.DataFrame(columns=record.keys())
        
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        df.to_csv(self.accuracy_file, index=False)
        
        print("\nAccuracy History:")
        print(df.tail(50))  # Show last 50 runs
        print(f"\nAverage Accuracy across all runs: {df['Overall Accuracy'].mean():.2f}%")
        print(f"Best Accuracy so far: {df['Overall Accuracy'].max():.2f}%")

    def get_average_fitness(self):
        # Only consider PathAgents that haven't reached the goal yet
        active_agents = [
            agent for agent in self.schedule.agents 
            if isinstance(agent, PathAgent) and agent.position != self.goal
        ]
    
        if not active_agents:  # If all agents reached goal
            return 0
    
        # Calculate fitness only for active agents
        fitnesses = [agent.calculate_fitness() for agent in active_agents]
        return sum(fitnesses) / len(fitnesses)
        
    def get_global_best(self):
        current_best = None
        current_best_fitness = float('inf')
        for agent in self.schedule.agents:
            if agent.best_fitness < current_best_fitness:
                current_best_fitness = agent.best_fitness
                current_best = agent.best_position
        if current_best_fitness < self.global_best_fitness:
            self.global_best_fitness = current_best_fitness
            self.global_best = current_best
        return self.global_best
    
    def plot_time_graphs(self):
        plt.figure(figsize=(12, 6))
        plt.title("Steps Taken by Each Agent to Reach the Goal", fontsize=14, pad=20)
        plt.xlabel("Agent ID", fontsize=12)
        plt.ylabel("Number of Steps", fontsize=12)
        # Calculate actual steps for each agent
        agent_steps = {}
        agent_colors= {}

        # Get agent colors and steps
        for agent in self.schedule.agents:
            if isinstance(agent, PathAgent):
                agent_id = agent.unique_id
                agent_colors[agent_id] = agent.color
                path = self.all_paths.get(agent_id, [])
            
                # Find the first occurrence of goal position in the path
                goal_pos = (self.grid.width-1, self.grid.height-1)
                try:
                    steps = path.index(goal_pos) + 1
                except ValueError:
                    steps = len(path)
                agent_steps[agent_id] = steps

        # Create bar plot
        agent_ids = list(agent_steps.keys())
        steps_taken = list(agent_steps.values())
        colors = [agent_colors[agent_id] for agent_id in agent_ids]
    
        bars = plt.bar(agent_ids, steps_taken, color=colors)

        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height)}',
                     ha='center', va='bottom')

        # Customize grid and spines
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        plt.tight_layout()
        plt.show()
    
    def plot_path_efficiency(self):
        path_agents = [agent for agent in self.schedule.agents if isinstance(agent, PathAgent)]
        expected_optimal_steps = 38
        agent_ids = []
        efficiency_values = []

        for agent in path_agents:
            if agent.steps_taken > 0:
                efficiency = (expected_optimal_steps / agent.steps_taken) * 100
                agent_ids.append(agent.unique_id)
                efficiency_values.append(min(efficiency, 100))  # Cap at 100%

        if agent_ids:  # Only plot if we have data
            plt.figure(figsize=(10, 6))
        
            # Plot line graph
            line = plt.plot(agent_ids, efficiency_values, marker='o', linestyle='-', color='b', label="Path Efficiency (%)")
        
            # Add percentage labels above each point
            for i, efficiency in enumerate(efficiency_values):
                plt.annotate(f'{efficiency:.1f}%', 
                        (agent_ids[i], efficiency_values[i]),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center',
                        fontweight='bold')
                
            # Calculate and display overall accuracy
            overall_accuracy = sum(efficiency_values) / len(efficiency_values)
            self.save_accuracy_record(overall_accuracy)
        
            plt.text(0.02, 0.98, f'Overall Accuracy: {overall_accuracy:.1f}%',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                fontsize=12,
                fontweight='bold')

            plt.xlabel("Agent ID", fontsize=12)
            plt.ylabel("Path Efficiency (%)", fontsize=12)
            plt.title("Path Efficiency of Agents with Accuracy Percentages", fontsize=14)
            plt.ylim(0, 110)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
        
            # Add average efficiency
            avg_efficiency = sum(efficiency_values) / len(efficiency_values)
            plt.axhline(y=avg_efficiency, color='r', linestyle='--', label=f'Avg: {avg_efficiency:.1f}%')
            plt.axhline(y=overall_accuracy, color='r', linestyle='--', label=f'Avg: {overall_accuracy:.1f}%')
            plt.tight_layout()
            plt.show()

    def plot_paths(self):
        plt.figure(figsize=(10, 10))
        plt.title("Agent Paths to the Goal", fontsize=14, pad=20)
        plt.xlabel("X Coordinate", fontsize=12)
        plt.ylabel("Y Coordinate", fontsize=12)

        # Set proper axis limits
        plt.xlim(-1, self.grid.width)
        plt.ylim(-1, self.grid.height)

        # Plot grid lines
        plt.grid(True, linestyle='--', alpha=0.6)

        # Plot paths with improved visibility
        for agent_id, path in self.all_paths.items():
            if path:
                path_array = np.array(path)
                x, y = path_array.T
            
                # Generate unique color for each agent
                color = plt.cm.tab10(agent_id / len(self.all_paths))
            
                # Plot path line
                plt.plot(x, y, '-', color=color, alpha=0.6, linewidth=2)
                # Plot points along path
                plt.plot(x, y, 'o', color=color, markersize=4, label=f'Agent {agent_id}')
            
                # Mark start and end points of this agent's path
                plt.plot(x[0], y[0], 'o', color=color, markersize=8)
                plt.plot(x[-1], y[-1], 's', color=color, markersize=8)

        # Plot obstacles
        obstacle_positions = [(agent.pos[0], agent.pos[1]) for agent in self.schedule.agents 
                             if isinstance(agent, ObstacleAgent)]
        if obstacle_positions:
            x_obs, y_obs = zip(*obstacle_positions)
            plt.scatter(x_obs, y_obs, c='black', marker='s', s=100, label='Obstacles')

        # Highlight start and goal positions
        plt.scatter([0], [0], color='green', marker='*', s=200, label='Start', zorder=5)
        plt.scatter([self.grid.width-1], [self.grid.height-1], color='red', 
                   marker='*', s=200, label='Goal', zorder=5)

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

def agent_portrayal(agent):
    pos = agent.pos
    if pos == agent.model.goal:
        portrayal = {
            "Shape": "rect", 
            "Color": "red", 
            "Filled": "true", 
            "Layer": 0, 
            "w": 1,
            "h": 1
        }
        return portrayal
    
    if isinstance(agent, PathAgent):
            return {
                "Shape": "circle", 
                "Color": agent.color, 
                "Filled": "true", 
                "Layer": 2, 
                "r": 0.5
            }
        
    elif isinstance(agent, ObstacleAgent):
        return {
            "Shape": "rect",
            "Color": "black",
            "Filled": "true",
            "Layer": 1,
            "w": 1,
            "h": 1
        }

    # Highlight starting position (0,0) as yellow
    if pos == (0, 0):
        portrayal["Color"] = "yellow"

    if pos == agent.model.goal:
        portrayal["Color"] = "red"

    return portrayal
   

grid = CanvasGrid(agent_portrayal, 20, 20, 500, 500)
chart = ChartModule([
    { "Label": "Average_Fitness", 
     "Color": "Black", 
     "min_value": 0,
     "max_value":np.sqrt(2*20**2)
    }
], data_collector_name='datacollector')

if __name__ == "__main__":
    server = ModularServer(PathFindingModel, [grid, chart], "Hybrid Pathfinding Model", {"width": 20, "height": 20, "n_agents": 10, "n_obstacles": 100})
    server.launch()