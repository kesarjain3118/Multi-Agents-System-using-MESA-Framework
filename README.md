# Hybrid Pathfinding Simulation (Multi-Agent System with Mesa)

An intelligent multi-agent pathfinding simulation built with Python, Mesa, and PyTorch.
This project benchmarks a hybrid algorithm (A* + PSO + RL + ACO) against traditional baseline algorithms under dynamic obstacle environments, providing reproducible results with CSV logging and visualizations.

# Features

-   *Hybrid Pathfinding Model:*
    1. A* (efficiency in shortest pathfinding)
    2. Particle Swarm Optimization (global search)
    3. Reinforcement Learning with Neural Network (adaptability)
    4. Ant Colony Optimization pheromone bias (collaborative intelligence)
-   *Baseline Comparisons:* Greedy, A*, PSO-only, RL-only, and Hybrid baseline.
-   *Dynamic Obstacles:* Agents navigate a grid with moving obstacles.
-   *Performance Logging:* Results stored in CSV files for reproducibility.
-   *Data Visualization:* Success rates, steps, and path efficiency plotted automatically.
-   *Interactive Simulation:* Mesa server-based visualization with grid and charts.

# Results

-    Achieved 100% success rate in reaching goals (matching A*).
-    Outperformed Greedy (50%), PSO-only (50%), RL-only (0%), and Hybrid baseline (0%).
-    While A* was more step-efficient (~22 steps), the hybrid model demonstrated greater robustness in dynamic environments.

# Screenshots


*Simulation Grid with Agents & Obstacles*
<img width="2830" height="1316" alt="image" src="https://github.com/user-attachments/assets/4142acff-d513-41e2-8628-964ecc21456e" />

*Success Rate Comparision*
<img width="640" height="480" alt="compare_plot_success_rate" src="https://github.com/user-attachments/assets/79e1a7dc-cfee-41e9-a10b-613f8f2360a7" />

*Path Efficiency Visualization*
<img width="800" height="400" alt="compare_plot_avg_path_length" src="https://github.com/user-attachments/assets/45e11964-7476-4748-83c9-073845a9f5a4" />

# Tech Stack

-   *Backend/Simulation:* Python, Mesa, PyTorch
-   *Data Handling:* Pandas, NumPy
-   *Visualization:* Matplotlib
-   *Version Control:* Git & Github
-   Team Mates : kesar jain ,abhishek reddy , sai suvan 
