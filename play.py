import gymnasium as gym
#import gymnasium.envs.atari  
import ale_py
import time
from stable_baselines3 import DQN

def main():
    # Create the Atari Pong environment in human render mode
    env = gym.make("ALE/Pong-v5", render_mode="human")
    
    # Load the trained DQN model from the Models folder
    model = DQN.load("./models/dqn_model.zip")
    
    num_episodes = 5  # Number of episodes to run for demonstration
    for episode in range(num_episodes):
        obs, info = env.reset()  # Reset the environment
        done = False
        total_reward = 0
        
        while not done:
            # Predict the action using the greedy (deterministic) policy
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            # Render the environment for visualization
            env.render()
            time.sleep(0.02)
            
            if done or truncated:
                break
        
        print(f"Episode {episode+1} finished with total reward: {total_reward}")
    
    env.close()

if __name__ == "__main__":
    main()
