# comparative_pathfinding.py
"""
Baselines for pathfinding comparison: greedy, astar, pso_only, rl_only, hybrid.
Provides run_single_trial() and run_experiment().
"""
import heapq
import numpy as np
import random
import pandas as pd
from collections import defaultdict
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import torch
import torch.nn as nn

# ---------- A* helper ----------
def a_star_grid(start, goal, width, height, obstacle_set):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))
    came_from = {}
    gscore = {start: 0}
    visited = set()
    while open_set:
        f, g, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        if current in visited:
            continue
        visited.add(current)
        x, y = current
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)]:
            nx, ny = x+dx, y+dy
            npos = (nx, ny)
            if 0 <= nx < width and 0 <= ny < height and npos not in obstacle_set:
                cost = g + (1 if abs(dx)+abs(dy)==1 else 1.4)
                if npos not in gscore or cost < gscore[npos]:
                    gscore[npos] = cost
                    came_from[npos] = current
                    heapq.heappush(open_set, (cost + heuristic(npos, goal), cost, npos))
    return []

# ---------- small NN for RL/hybrid ----------
class NeuralNetwork(nn.Module):
    def __init__(self, input_size=4, hidden=32, output=4):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, output)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

# ---------- Agent able to run in multiple modes ----------
class ComparisonAgent(Agent):
    def __init__(self, unique_id, model, mode='hybrid'):
        super().__init__(unique_id, model)
        self.mode = mode
        self.position = (0,0)
        self.goal = model.goal
        self.steps_taken = 0
        self.path = [self.position]
        self.reached = False
        self.velocity = np.zeros(2)
        self.network = NeuralNetwork()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        if self.mode == 'astar':
            self.astar_path = a_star_grid(self.position, self.goal, model.grid.width, model.grid.height, model.obstacle_positions)
            if self.astar_path and self.astar_path[0] == self.position:
                self.astar_path = self.astar_path[1:]

    def step(self):
        if self.reached: return
        if self.position == self.goal:
            self.reached = True
            return
        if self.mode == 'greedy':
            self.greedy_move()
        elif self.mode == 'astar':
            self.astar_move()
        elif self.mode == 'pso_only':
            self.pso_move()
        elif self.mode == 'rl_only':
            self.rl_move()
        elif self.mode == 'hybrid':
            self.hybrid_move()
        self.steps_taken += 1
        self.path.append(self.position)
        if self.position == self.goal:
            self.reached = True

    def valid(self, pos):
        x,y = pos
        return 0<=x<self.model.grid.width and 0<=y<self.model.grid.height and pos not in self.model.obstacle_positions

    def neighbors4(self, pos):
        x,y = pos
        return [(x,y+1),(x+1,y),(x,y-1),(x-1,y)]

    def greedy_move(self):
        x,y = self.position
        gx,gy = self.goal
        best = None; best_score = -1e9
        for nx,ny in self.neighbors4(self.position):
            if not self.valid((nx,ny)): continue
            dirv = np.array([nx-x, ny-y]); goalv = np.array([gx-x, gy-y])
            score = 0 if np.linalg.norm(goalv)==0 else np.dot(dirv, goalv) / (np.linalg.norm(dirv)+1e-6)
            if score>best_score:
                best_score=score; best=(nx,ny)
        if best:
            self.model.grid.move_agent(self, best); self.position = best

    def astar_move(self):
        if hasattr(self,'astar_path') and self.astar_path:
            next_pos = self.astar_path.pop(0)
            if self.valid(next_pos) and (self.model.grid.is_cell_empty(next_pos) or next_pos==self.goal):
                self.model.grid.move_agent(self, next_pos); self.position = next_pos

    def pso_move(self):
        gb = self.model.get_global_best_pos()
        if gb is None:
            self.greedy_move(); return
        dir_to_gb = np.array(gb) - np.array(self.position)
        if np.linalg.norm(dir_to_gb)>0:
            dir_to_gb = dir_to_gb / (np.linalg.norm(dir_to_gb)+1e-6)
        self.velocity = 0.6*self.velocity + 0.8*(np.random.rand(2)-0.5) + 0.5*dir_to_gb
        dx = int(np.sign(self.velocity[0])); dy = int(np.sign(self.velocity[1]))
        candidates = [(self.position[0]+dx, self.position[1]+dy),(self.position[0]+dx,self.position[1]),(self.position[0],self.position[1]+dy)]
        for cand in candidates:
            if self.valid(cand) and (self.model.grid.is_cell_empty(cand) or cand==self.goal):
                self.model.grid.move_agent(self, cand); self.position = cand
                self.model.pheromone_map[self.position] = self.model.pheromone_map.get(self.position,0) + 1
                return
        self.greedy_move()

    def rl_move(self):
        state = torch.FloatTensor(self.get_state())
        probs = self.network(state).detach().numpy()
        moves = self.neighbors4(self.position)
        idxs = np.argsort(-probs)
        for i in idxs:
            if i < len(moves):
                cand = moves[i]
                if self.valid(cand) and (self.model.grid.is_cell_empty(cand) or cand==self.goal):
                    self.model.grid.move_agent(self, cand); self.position = cand; return
        self.greedy_move()

    def hybrid_move(self):
        x,y = self.position; gx,gy = self.goal
        state = torch.FloatTensor(self.get_state())
        probs = self.network(state).detach().numpy()
        moves = [(0,1),(1,0),(0,-1),(-1,0)]
        best_score = -1e9; best_pos = None
        goal_direction = np.array([gx-x, gy-y])
        if np.linalg.norm(goal_direction)>0:
            goal_direction = goal_direction / (np.linalg.norm(goal_direction)+1e-6)
        for i,(dx,dy) in enumerate(moves):
            new_pos = (x+dx,y+dy)
            if not self.valid(new_pos): continue
            move_dir = np.array([dx,dy])
            goal_align = np.dot(move_dir, goal_direction)
            pher = self.model.pheromone_map.get(new_pos,0)
            vel_align = np.dot(move_dir, self.velocity)
            score = 0.8*goal_align + 0.15*probs[i] + 0.2*pher + 0.1*vel_align
            if score>best_score:
                best_score=score; best_pos=new_pos
        if best_pos:
            self.model.grid.move_agent(self, best_pos); self.position = best_pos
            self.model.pheromone_map[self.position] = self.model.pheromone_map.get(self.position,0) + 1
        else:
            self.greedy_move()

    def get_state(self):
        x,y = self.position; gx,gy = self.goal
        return [x/self.model.grid.width, y/self.model.grid.height, gx/self.model.grid.width, gy/self.model.grid.height]

# ---------- Model ----------
class ComparativeModel(Model):
    def __init__(self, width=20, height=20, n_agents=10, n_obstacles=40, agent_mode='hybrid', seed=None):
        super().__init__()
        if seed is not None:
            random.seed(seed); np.random.seed(seed)
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.goal = (width-1, height-1)
        self.pheromone_map = {}
        self.agent_mode = agent_mode
        self.obstacle_positions = set()
        for _ in range(n_obstacles):
            x = random.randrange(width); y = random.randrange(height)
            if (x,y) not in [(0,0), self.goal]:
                self.obstacle_positions.add((x,y))
        for i in range(n_agents):
            a = ComparisonAgent(i, self, mode=self.agent_mode)
            self.schedule.add(a)
            self.grid.place_agent(a, (0,0))
        self.datacollector = DataCollector(model_reporters={"avg_fitness": lambda m: self.get_avg_fitness()})

    def step(self):
        self.schedule.step()
        for k in list(self.pheromone_map.keys()):
            self.pheromone_map[k] *= 0.95
            if self.pheromone_map[k] < 1e-3:
                del self.pheromone_map[k]

    def get_avg_fitness(self):
        agents = [a for a in self.schedule.agents if isinstance(a, ComparisonAgent) and not a.reached]
        if not agents: return 0.0
        return float(np.mean([a.steps_taken for a in agents]) if agents else 0.0)

    def get_global_best_pos(self):
        best=None; best_steps=float('inf')
        for a in self.schedule.agents:
            if isinstance(a, ComparisonAgent):
                if a.steps_taken < best_steps:
                    best_steps=a.steps_taken; best=a.position
        return best

# ---------- runner ----------
def run_single_trial(mode='hybrid', width=20, height=20, n_agents=10, n_obstacles=40, max_steps=500, seed=None):
    model = ComparativeModel(width=width, height=height, n_agents=n_agents, n_obstacles=n_obstacles, agent_mode=mode, seed=seed)
    for step in range(max_steps):
        model.step()
        agents = [a for a in model.schedule.agents if isinstance(a, ComparisonAgent)]
        if all(a.reached for a in agents):
            break
    metrics = []
    for a in model.schedule.agents:
        metrics.append({'agent_id': a.unique_id, 'mode': mode, 'reached': bool(a.reached), 'steps': int(a.steps_taken), 'path_length': int(len(a.path))})
    return metrics

def run_experiment(mode='hybrid', n_runs=5, **kwargs):
    rows=[]
    for i in range(n_runs):
        seed = random.randint(0,10**6)
        trial = run_single_trial(mode=mode, seed=seed, **kwargs)
        for r in trial:
            r['run'] = i+1
            rows.append(r)
    df = pd.DataFrame(rows)
    summary = df.groupby('mode').agg({'reached':['mean'], 'steps':['mean','median'], 'path_length':['mean']})
    return df, summary
