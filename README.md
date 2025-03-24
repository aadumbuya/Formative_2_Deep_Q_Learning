# Formative_2_Deep_Q_Learning
# Deep Q-Network (DQN) Training and Evaluation on Atari Pong

## Overview
This project trains a Deep Q-Network (DQN) agent to play Atari Pong using Stable-Baselines3 and Gymnasium. The training process is optimized to reduce memory usage while maintaining good performance.

## Installation
Ensure you have Python 3.8+ and install the required dependencies:

```bash
pip install stable-baselines3[extra] gymnasium[atari] torch tensorboard
```

If you encounter ALE NamespaceNotFound, install Atari dependencies:

```bash
pip install gymnasium[accept-rom-license]
```

## Training the DQN Agent
To train the agent, run:

```bash
python train.py
```

### Features:
- Uses CnnPolicy for deep learning-based vision processing.
- Saves models every 100K steps to avoid memory overflow.
- Reduces memory usage by:
  - Lowering buffer size to 200K.
  - Reducing batch size to 16.
  - Using float16 precision to save GPU memory.
  - Training in smaller steps 50K.

### Key Hyperparameters:
| Parameter         | Value |
|------------------|-------|
| Learning Rate    | 1e-4  |
| Buffer Size      | 200K  |
| Batch Size       | 16    |
| Train Steps      | 500K  |
| Target Update    | 2000  |



Summary of Hyperparameters in our Code
1. MLP Model (First Training Block)
Learning Rate (lr): 1e-4

Gamma (γ): 0.99

Batch Size: 32 (default, not explicitly set).

Epsilon Settings:

Start: 1.0 (default).

End: 0.05 (default).

Decay Period: 20,000 steps (10% of total 200k steps).

Observed Behavior:

Reward plateaued around -20.7 to -21.0.

Likely due to insufficient exploration (fast ε-decay) or MLP’s inability to process pixel inputs effectively.

2. CNN Model (Second Training Block)
Learning Rate (lr): 1e-4

Gamma (γ): 0.99

Batch Size: 16 (explicitly set).

Epsilon Settings:

Start: 1.0 (default).

End: 0.01 (explicitly set).

Decay Period: 5,000 steps (10% of total 50k steps).

Observed Behavior:

Reward stuck at -21.0 ± 0.0 after 50k steps.

Aggressive ε-decay (to 0.01) and small batch size likely caused unstable training.

Key Issues Identified
Stagnant Rewards: Both models failed to improve beyond the baseline performance (~-21.0).

Exploration-Exploitation Mismatch:

MLP: ε decayed too slowly but ended at 0.05 (still some exploration).

CNN: ε decayed too quickly to 0.01, leading to premature exploitation.

Architecture Limitations:

MLP struggles with pixel-based environments like Pong.

CNN’s small batch size (16) might have caused noisy updates.

| Hyperparameter Set                                                                 | Noted Behavior                                                                 |
|------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| `lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=200k` | MLP Model: Reward stagnated at **-20.7 to -21.0** after 200k steps. No improvement observed. |
| `lr=1e-4, gamma=0.99, batch=16, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=50k`  | CNN Model: Reward stuck at **-21.0 ± 0.0**. Likely due to aggressive ε-decay and small batch size. |
| `lr=1e-4, gamma=0.99, batch=16, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=100k` |Mean reward (ep_rew_mean) remains negative, suggesting the agent struggles to improve significantly.|




## Playing with the Trained Model
Once training is complete, evaluate the model by running:

```bash
python play.py
```

### Features:
- Loads the trained model dqn_model.zip.
- Uses GreedyQPolicy for deterministic actions.
- Displays the Atari Pong game as the agent plays.

## TensorBoard Logging
Monitor training progress with TensorBoard:

```bash
tensorboard --logdir=./tensorboard_logs/
```

## Troubleshooting
- If you get ALE NamespaceNotFound, run:
  ```bash
  pip install gymnasium[accept-rom-license]
  ```
- If training crashes due to memory, reduce buffer_size further (e.g., 100K).
- If you get torch out of memory error, try reducing batch_size or train in smaller steps.


  CONTRIBUTORS
  1. Alhassan A Dumbuya - train.py
  2. Florent Hirwa - hyperparameters and Play.py
  3. Adaobi Stella Ibeh - hyperparameter tuning documentation
