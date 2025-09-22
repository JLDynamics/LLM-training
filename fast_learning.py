import gymnasium as gym
import numpy as np

class SimpleQLearningAgent:
    def __init__(self):
        """Simple Q-Learning Agent with smaller state space for faster learning"""
        # Smaller state space for faster learning
        self.state_bins = (6, 6, 12, 12)  # Reduced bins
        self.action_space_size = 2
        
        # Learning parameters
        self.learning_rate = 0.2      # Higher learning rate
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        
        # Initialize Q-table
        self.q_table = np.zeros(self.state_bins + (self.action_space_size,))
        
        # State bounds for CartPole
        self.state_bounds = [
            [-2.4, 2.4],    # Cart position
            [-2.0, 2.0],    # Cart velocity  
            [-0.2, 0.2],    # Pole angle (smaller range)
            [-2.0, 2.0]     # Pole velocity
        ]
    
    def discretize_state(self, state):
        """Convert continuous state to discrete bins"""
        discrete_state = []
        for i, val in enumerate(state):
            # Clip to bounds
            val = np.clip(val, self.state_bounds[i][0], self.state_bounds[i][1])
            # Convert to bin
            bin_size = (self.state_bounds[i][1] - self.state_bounds[i][0]) / self.state_bins[i]
            discrete_val = int((val - self.state_bounds[i][0]) / bin_size)
            discrete_val = min(self.state_bins[i] - 1, max(0, discrete_val))
            discrete_state.append(discrete_val)
        return tuple(discrete_state)
    
    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space_size)
        else:
            discrete_state = self.discretize_state(state)
            return np.argmax(self.q_table[discrete_state])
    
    def learn(self, state, action, reward, next_state, done):
        """Update Q-table"""
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        current_q = self.q_table[discrete_state + (action,)]
        
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[discrete_next_state])
            target_q = reward + self.discount_factor * max_next_q
        
        # Q-learning update
        self.q_table[discrete_state + (action,)] = current_q + self.learning_rate * (target_q - current_q)
    
    def decay_epsilon(self):
        """Reduce exploration rate"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

def quick_train():
    """Fast training with progress updates"""
    env = gym.make("CartPole-v1")
    agent = SimpleQLearningAgent()
    
    # Shorter training for quicker results
    num_episodes = 500
    episode_rewards = []
    
    print("🚀 Starting Fast Q-Learning Training...")
    print("📊 Watch the agent learn to balance the pole!")
    print("=" * 50)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 500:  # Limit steps per episode
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        
        # Show progress every 50 episodes
        if (episode + 1) % 50 == 0:
            recent_avg = np.mean(episode_rewards[-50:])
            print(f"Episode {episode + 1:3d} | Recent Avg: {recent_avg:6.1f} | Best: {max(episode_rewards):3.0f} | Epsilon: {agent.epsilon:.3f}")
    
    env.close()
    
    # Show learning progress
    print("\n📈 Learning Progress:")
    print(f"   Episodes 1-50:   Average = {np.mean(episode_rewards[:50]):.1f}")
    print(f"   Episodes 51-100: Average = {np.mean(episode_rewards[50:100]):.1f}")
    if len(episode_rewards) > 150:
        print(f"   Episodes 151-200: Average = {np.mean(episode_rewards[150:200]):.1f}")
    if len(episode_rewards) > 400:
        print(f"   Episodes 401-450: Average = {np.mean(episode_rewards[400:450]):.1f}")
    print(f"   Last 50 episodes: Average = {np.mean(episode_rewards[-50:]):.1f}")
    
    return agent, episode_rewards

def test_trained_agent(agent):
    """Test the trained agent with visual feedback"""
    env = gym.make("CartPole-v1", render_mode="human")
    
    print("\n🎯 Testing Trained Agent (Watch the window!)...")
    print("💡 The agent should now balance much better than random actions!")
    
    agent.epsilon = 0  # No exploration during testing
    
    test_rewards = []
    for episode in range(3):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        print(f"\nTest Episode {episode + 1} starting...")
        
        while not done:
            action = agent.choose_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        test_rewards.append(total_reward)
        print(f"Test Episode {episode + 1}: Reward = {total_reward}")
    
    env.close()
    
    print(f"\n🎉 Test Results: Average = {np.mean(test_rewards):.1f}")
    return test_rewards

if __name__ == "__main__":
    # Quick training
    trained_agent, rewards = quick_train()
    
    # Test the agent
    test_rewards = test_trained_agent(trained_agent)
    
    print(f"\n✅ Training Complete!")
    print(f"   The agent improved from {np.mean(rewards[:50]):.1f} to {np.mean(rewards[-50:]):.1f} average reward")
    print(f"   That's a {((np.mean(rewards[-50:]) / np.mean(rewards[:50])) - 1) * 100:.1f}% improvement!")