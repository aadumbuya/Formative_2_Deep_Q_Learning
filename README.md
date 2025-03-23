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
  2. Florent Hirwa -Play.py
  3. Adaobi Stella Ibeh - hyperparameter tuning documentation
