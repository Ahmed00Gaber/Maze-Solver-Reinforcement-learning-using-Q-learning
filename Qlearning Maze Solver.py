import numpy as np
import random
import matplotlib.pyplot as plt

# ==================== MAZE CLASS ====================
class Maze:
    def __init__(self, rows, cols, wall_percent=0.2):
        self.rows, self.cols = rows, cols
        self.start = (0, 0)          # Start position (top-left)
        self.goal = (rows-1, cols-1) # Goal position (bottom-right)
        self.wall_percent = wall_percent  # Probability of a cell being a wall
        self.grid = self.generate_maze()
    
    def generate_maze(self):
        grid = np.zeros((self.rows, self.cols))  # Initialize empty grid
        grid[self.start] = 2  # Start (value 2)
        grid[self.goal] = 3   # Goal (value 3)
        
        # Add random walls (value 1)
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in [self.start, self.goal]:  # Don't overwrite start/goal
                    if random.random() < self.wall_percent:
                        grid[r, c] = 1  # Mark as wall
        return grid
    
    def step(self, state, action):
        r, c = state
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left
        dr, dc = moves[action]  # Get movement direction
        new_state = (r + dr, c + dc)
        
        # Check boundaries and walls
        if not (0 <= new_state[0] < self.rows and 0 <= new_state[1] < self.cols):
            new_state = state  # Hit boundary, stay put
        elif self.grid[new_state] == 1:
            new_state = state  # Hit wall, stay put
        
        # Calculate reward
        if new_state == self.goal:
            reward = 10    # Big reward for reaching goal
            done = True    # Episode ends
        elif self.grid[new_state] == 1:
            reward = -5    # Penalty for hitting wall
            done = False
        else:
            reward = -0.1  # Small penalty for each move (encourage efficiency)
            done = False
        
        return new_state, reward, done

# ==================== Q-LEARNING AGENT ====================
class QAgent:
    def __init__(self, maze, alpha=0.1, gamma=0.9, epsilon=1.0, decay=0.995, min_epsilon=0.01):
        self.maze = maze
        # Q-learning parameters
        self.alpha = alpha      # Learning rate: how much to update Q-values
        self.gamma = gamma      # Discount factor: importance of future rewards
        self.epsilon = epsilon  # Exploration rate: prob of random action
        self.decay = decay      # How much to reduce epsilon each episode
        self.min_epsilon = min_epsilon  # Minimum exploration rate
        
        # Q-table: 3D array [rows][cols][actions] storing learned values
        self.Q = np.zeros((maze.rows, maze.cols, 4))
        
        # Training history
        self.rewards = []      # Total reward per episode
        self.steps_per_episode = []  # Steps taken per episode
    
    def choose_action(self, state):
        """Epsilon-greedy action selection: balance exploration vs exploitation"""
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # Explore: random action
        r, c = state
        return np.argmax(self.Q[r, c])   # Exploit: best known action
    
    def update(self, state, action, reward, next_state):
        """Q-learning update equation: Q(s,a) += α * (r + γ*maxQ(s') - Q(s,a))"""
        r, c = state
        nr, nc = next_state
        
        current_q = self.Q[r, c, action]          # Current Q-value
        next_max_q = np.max(self.Q[nr, nc])       # Best future Q-value
        
        # Q-LEARNING CORE EQUATION
        # Temporal Difference error: (r + γ*maxQ(s') - Q(s,a))
        td_error = reward + self.gamma * next_max_q - current_q
        self.Q[r, c, action] += self.alpha * td_error  # Update Q-value
    
    def train(self, episodes=1000, max_steps=100):
        print(f"Training for {episodes} episodes...")
        
        for episode in range(episodes):
            state = self.maze.start
            total_reward = 0
            steps = 0  # Track steps taken this episode
            
            for step in range(max_steps):
                action = self.choose_action(state)                    # 1. Choose action
                next_state, reward, done = self.maze.step(state, action) # 2. Take action
                self.update(state, action, reward, next_state)       # 3. Update Q-table
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break  # Goal reached, end episode
            
            # Reduce exploration over time (epsilon decay)
            self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)
            
            # Record training statistics
            self.rewards.append(total_reward)
            self.steps_per_episode.append(steps)  # Track steps per episode
            
            # Print progress every 100 episodes
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.rewards[-100:])
                print(f"Episode {episode+1}: Avg Reward = {avg_reward:.2f}, Epsilon = {self.epsilon:.3f}")
    
    def get_path(self):
        """Follow greedy policy from start to goal (no exploration)"""
        path = [self.maze.start]
        state = self.maze.start
        
        # Follow best actions until goal reached or stuck
        while state != self.maze.goal and len(path) < self.maze.rows * self.maze.cols:
            r, c = state
            action = np.argmax(self.Q[r, c])  # Always choose best action
            
            next_state, _, _ = self.maze.step(state, action)
            
            if next_state == state:  # Stuck (can't move)
                break
            
            state = next_state
            path.append(state)
        
        return path if state == self.maze.goal else []  # Return path if goal reached

# ==================== VISUALIZATION ====================
def visualize(maze, agent, path=None):
    """Plot maze, training history, and path - NOW WITH 4 SUBPLOTS"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # Changed to 2x2 grid for 4 plots
    
    # 1. Plot Maze (top-left)
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'green', 'red'])
    axes[0, 0].imshow(maze.grid, cmap=cmap, vmin=0, vmax=3)
    axes[0, 0].set_title(f"Maze ({maze.rows}x{maze.cols})")
    axes[0, 0].set_xticks(range(maze.cols))
    axes[0, 0].set_yticks(range(maze.rows))
    axes[0, 0].grid(color='gray', linestyle='-', linewidth=0.5)
    
    # Plot path if exists (blue circles)
    if path:
        for (r, c) in path:
            axes[0, 0].add_patch(plt.Circle((c, r), 0.3, color='blue', alpha=0.5))
    
    # 2. Plot Q-values heatmap (top-right)
    max_q = np.max(np.abs(agent.Q))
    if max_q > 0:
        q_heatmap = np.max(agent.Q, axis=2)  # Max Q-value for each state
        im = axes[0, 1].imshow(q_heatmap, cmap='hot')
        axes[0, 1].set_title("Q-values Heatmap")
        plt.colorbar(im, ax=axes[0, 1])
    
    # 3. Plot Training Rewards (bottom-left)
    axes[1, 0].plot(agent.rewards, alpha=0.7)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Total Reward')
    axes[1, 0].set_title('Training Progress (Rewards)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Moving average for rewards
    if len(agent.rewards) > 50:
        window = 50
        moving_avg = np.convolve(agent.rewards, np.ones(window)/window, mode='valid')
        axes[1, 0].plot(range(window-1, len(agent.rewards)), moving_avg, 'r-', linewidth=2)
    
    # 4. NEW: Plot Steps to Goal Moving Average (bottom-right)
    axes[1, 1].plot(agent.steps_per_episode, alpha=0.7, label='Steps per Episode')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Steps to Goal')
    axes[1, 1].set_title('Learning Efficiency (Steps to Goal)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Moving average for steps
    if len(agent.steps_per_episode) > 50:
        window_steps = 50
        steps_ma = np.convolve(agent.steps_per_episode, np.ones(window_steps)/window_steps, mode='valid')
        axes[1, 1].plot(range(window_steps-1, len(agent.steps_per_episode)), steps_ma, 'g-', linewidth=2, label=f'MA ({window_steps} episodes)')
    
    axes[1, 1].legend()
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n{'='*50}")
    print("TRAINING SUMMARY")
    print(f"{'='*50}")
    print(f"Maze Size: {maze.rows} x {maze.cols}")
    print(f"Start: {maze.start}, Goal: {maze.goal}")
    print(f"Episodes: {len(agent.rewards)}")
    print(f"Final Epsilon: {agent.epsilon:.4f}")
    
    # Steps analysis
    if agent.steps_per_episode:
        avg_steps = np.mean(agent.steps_per_episode[-100:]) if len(agent.steps_per_episode) >= 100 else np.mean(agent.steps_per_episode)
        print(f"Avg Steps (last 100): {avg_steps:.1f}")
    
    if path:
        print(f"Path Found: {len(path)} steps")
    else:
        print("No complete path found")

# ==================== MAIN FUNCTION ====================
def main():
    # ========== PARAMETERS YOU CAN CHANGE ==========
    # Maze parameters
    ROWS = 15                   # Maze height
    COLS = 15                # Maze width  
    WALL_PERCENT = 0.30         # Wall density (0-1)
    
    # Q-learning parameters
    ALPHA = 0.1                 # Learning rate (0-1)
    GAMMA = 0.9                 # Discount factor (0-1)
    EPSILON = 1.0               # Initial exploration rate
    DECAY = 0.995               # Epsilon decay
    MIN_EPSILON = 0.01          # Minimum exploration rate
    
    # Training parameters
    EPISODES = 1000             # Number of training episodes
    MAX_STEPS = 200             # Max steps per episode
    # ===============================================
    
    # Create maze and agent
    maze = Maze(rows=ROWS, cols=COLS, wall_percent=WALL_PERCENT)
    agent = QAgent(maze, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON, decay=DECAY, min_epsilon=MIN_EPSILON)
    
    # Train agent
    agent.train(episodes=EPISODES, max_steps=MAX_STEPS)
    
    # Get optimal path
    path = agent.get_path()
    
    # Visualize results
    visualize(maze, agent, path)
    
    # Show policy at start position
    print(f"\nQ-values at start position {maze.start}:")
    actions = ['Up', 'Right', 'Down', 'Left']
    r, c = maze.start
    for i, action in enumerate(actions):
        print(f"  {action}: {agent.Q[r, c, i]:.4f}")
    
# Run the program
if __name__ == "__main__":
    main()