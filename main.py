import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, state_bins, action_space_size, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        """
        Q-Learning Agent for CartPole
        
        Args:
            state_bins: Number of bins for discretizing each state dimension
            action_space_size: Number of possible actions (2 for CartPole: left/right)
            learning_rate: How fast the agent learns (alpha)
            discount_factor: How much future rewards matter (gamma)
            epsilon: Exploration rate (1.0 = always explore, 0.0 = always exploit)
            epsilon_decay: How much epsilon decreases each episode
            min_epsilon: Minimum exploration rate
        """
        self.state_bins = state_bins
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Initialize Q-table with zeros
        # Q-table shape: (cart_pos_bins, cart_vel_bins, pole_angle_bins, pole_vel_bins, actions)
        self.q_table = np.zeros(state_bins + (action_space_size,))
        
        # Define state space bounds for discretization
        self.state_bounds = [
            [-2.4, 2.4],        # Cart position
            [-3.0, 3.0],        # Cart velocity  
            [-0.5, 0.5],        # Pole angle
            [-2.0, 2.0]         # Pole velocity
        ]
    
    def discretize_state(self, state):
        """Convert continuous state to discrete state for Q-table lookup"""
        discrete_state = []
        for i, val in enumerate(state):
            # Clip values to bounds
            val = max(self.state_bounds[i][0], min(self.state_bounds[i][1], val))
            # Convert to discrete bin
            bin_size = (self.state_bounds[i][1] - self.state_bounds[i][0]) / self.state_bins[i]
            discrete_val = int((val - self.state_bounds[i][0]) / bin_size)
            # Ensure we don't exceed array bounds
            discrete_val = min(self.state_bins[i] - 1, discrete_val)
            discrete_state.append(discrete_val)
        return tuple(discrete_state)
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy strategy"""
        if np.random.random() < self.epsilon:
            # Explore: choose random action
            return np.random.randint(self.action_space_size)
        else:
            # Exploit: choose best known action
            discrete_state = self.discretize_state(state)
            return np.argmax(self.q_table[discrete_state])
    
    def learn(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning formula"""
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        # Current Q-value
        current_q = self.q_table[discrete_state + (action,)]
        
        # Calculate target Q-value
        if done:
            target_q = reward  # No future rewards if episode ended
        else:
            max_next_q = np.max(self.q_table[discrete_next_state])
            target_q = reward + self.discount_factor * max_next_q
        
        # Update Q-value using Q-learning formula
        # Q(s,a) = Q(s,a) + α * (reward + γ * max(Q(s',a')) - Q(s,a))
        self.q_table[discrete_state + (action,)] = current_q + self.learning_rate * (target_q - current_q)
    
    def decay_epsilon(self):
        """Reduce exploration rate over time"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

def train_agent():
    """Train the Q-learning agent on CartPole"""
    # Create environment
    env = gym.make("CartPole-v1")
    
    # Create agent
    state_bins = (10, 10, 10, 10)  # Discretize each state dimension into 10 bins
    agent = QLearningAgent(state_bins, env.action_space.n)
    
    # Training parameters
    num_episodes = 1000
    episode_rewards = []
    
    print("🚀 Starting Q-Learning Training...")
    print("📊 The agent will learn to balance the pole better over time!")
    print()
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Agent chooses action
            action = agent.choose_action(state)
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Agent learns from experience
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
        
        # Decay exploration rate
        agent.decay_epsilon()
        
        # Track performance
        episode_rewards.append(total_reward)
        
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1:4d} | Avg Reward: {avg_reward:6.2f} | Epsilon: {agent.epsilon:.3f}")
    
    env.close()
    return agent, episode_rewards

def test_agent(agent, render=True):
    """Test the trained agent"""
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    
    print("\n🎯 Testing Trained Agent...")
    print("💡 The agent should now balance the pole much better!")
    
    # Test for 5 episodes
    test_rewards = []
    for episode in range(5):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        # Set epsilon to 0 for testing (no exploration, only exploitation)
        agent.epsilon = 0
        
        while not done:
            action = agent.choose_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        test_rewards.append(total_reward)
        print(f"Test Episode {episode + 1}: Reward = {total_reward}")
    
    env.close()
    return test_rewards

def plot_learning_progress(episode_rewards):
    """Plot the learning progress"""
    plt.figure(figsize=(12, 5))
    
    # Plot raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.3, color='blue')
    
    # Plot moving average
    window = 50
    if len(episode_rewards) >= window:
        moving_avg = [np.mean(episode_rewards[i:i+window]) for i in range(len(episode_rewards)-window+1)]
        plt.plot(range(window-1, len(episode_rewards)), moving_avg, color='red', linewidth=2, label=f'{window}-episode moving average')
        plt.legend()
    
    plt.title('Learning Progress: Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    
    # Plot recent performance
    plt.subplot(1, 2, 2)
    recent_rewards = episode_rewards[-200:] if len(episode_rewards) > 200 else episode_rewards
    plt.plot(recent_rewards, color='green', linewidth=1)
    plt.title('Recent Performance (Last 200 Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Train the agent
    trained_agent, rewards = train_agent()
    
    # Plot learning progress
    plot_learning_progress(rewards)
    
    # Test the trained agent
    test_rewards = test_agent(trained_agent)
    
    print(f"\n📈 Training Results:")
    print(f"   First 100 episodes average: {np.mean(rewards[:100]):.2f}")
    print(f"   Last 100 episodes average: {np.mean(rewards[-100:]):.2f}")
    print(f"   Test episodes average: {np.mean(test_rewards):.2f}")
    print(f"   Maximum episode reward: {np.max(rewards):.2f}")
    
    print(f"\n🎉 Training complete! The agent learned to balance the pole better over time.")