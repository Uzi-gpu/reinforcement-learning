# 🎮 Reinforcement Learning Projects

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red.svg)](https://pytorch.org/)
[![Gym](https://img.shields.io/badge/OpenAI-Gym-green.svg)](https://gym.openai.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive collection of **Reinforcement Learning** projects demonstrating mastery of Q-Learning, Policy Gradients, and Actor-Critic methods using PyTorch and OpenAI Gym environments.

---

## 📋 Table of Contents
- [Projects Overview](#-projects-overview)
- [Technologies Used](#️-technologies-used)
- [Installation](#-installation)
- [Project Details](#-project-details)
- [Key RL Concepts](#-key-rl-concepts)
- [Results](#-results)
- [Contact](#-contact)

---

## 🚀 Projects Overview

| # | Project | Algorithm | Notebook | Environment |
|---|---------|-----------|----------|-------------|
| 1 | **Q-Learning (Tabular)** | Q-Table | [`01_q_learning_tabular.ipynb`](01_q_learning_tabular.ipynb) | Discrete State Space |
| 2 | **Actor-Critic** | A2C | [`02_actor_critic_cartpole.ipynb`](02_actor_critic_cartpole.ipynb) | CartPole-v1 |
| 3 | **REINFORCE** | Policy Gradient | [`03_reinforce_policy_gradient.ipynb`](03_reinforce_policy_gradient.ipynb) | Various Gym Envs |

---

## 🛠️ Technologies Used

### Core Libraries
- **PyTorch** - Deep RL neural networks
- **OpenAI Gym** - RL environments
- **NumPy** - Numerical computing
- **Matplotlib** - Visualization

### RL Techniques
- **Value-Based Methods** - Q-Learning, DQN
- **Policy-Based Methods** - REINFORCE, Policy Gradients
- **Actor-Critic Methods** - A2C, A3C variants
- **Exploration Strategies** - ε-greedy, entropy regularization

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- GPU recommended (but not required)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/uzi-gpu/reinforcement-learning.git
   cd reinforcement-learning
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

---

## 📊 Project Details

### 1. 📊 Q-Learning (Tabular Method)

**File:** [`01_q_learning_tabular.ipynb`](01_q_learning_tabular.ipynb)

**Objective:** Implement tabular Q-Learning from scratch for discrete state-action spaces

**Algorithm:** Q-Learning
- Off-policy temporal difference learning
- Updates Q-table using Bellman equation
- ε-greedy exploration strategy

**Key Concepts:**
- ✅ Q-Table initialization and updates
- ✅ Exploration vs Exploitation trade-off
- ✅ Learning rate (α) and discount factor (γ)
- ✅ Episode-based training
- ✅ Convergence analysis
- ✅ Performance visualization

**Implementation Highlights:**
```python
Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
```

**Use Cases:**
- Small discrete state spaces
- Grid world problems
- Simple game environments

---

### 2. 🎯 Actor-Critic for CartPole

**File:** [`02_actor_critic_cartpole.ipynb`](02_actor_critic_cartpole.ipynb)

**Objective:** Solve the CartPole balancing problem using Actor-Critic method

**Environment:** CartPole-v1
- **Goal:** Balance pole on cart by moving left/right
- **State Space:** 4 continuous values (position, velocity, angle, angular velocity)
- **Action Space:** 2 discrete actions (left, right)
- **Reward:** +1 for each timestep pole remains upright

**Architecture:**

**Actor Network (Policy):**
- Input: State (4 dimensions)
- Hidden: Fully connected layers with ReLU
- Output: Action probabilities (softmax)

**Critic Network (Value Function):**
- Input: State (4 dimensions)
- Hidden: Fully connected layers with ReLU
- Output: State value V(s)

**Training Process:**
- ✅ Actor learns optimal policy π(a|s)
- ✅ Critic estimates value function V(s)
- ✅ Advantage function: A(s,a) = R + γV(s') - V(s)
- ✅ Policy gradient with baseline reduction
- ✅ Simultaneous actor-critic updates

**Key Features:**
- ✅ Continuous state space handling
- ✅ On-policy learning
- ✅ Variance reduction through baseline
- ✅ Episode reward tracking
- ✅ Training visualization

---

### 3. 🚀 REINFORCE Policy Gradient

**File:** [`03_reinforce_policy_gradient.ipynb`](03_reinforce_policy_gradient.ipynb)

**Objective:** Implement REINFORCE algorithm for policy optimization

**Algorithm:** REINFORCE (Monte Carlo Policy Gradient)
- Pure policy-based method
- No value function approximation
- Learn policy parameters directly

**Mathematical Foundation:**
```python
∇J(θ) = E[∑ ∇log π(a|s,θ) · G_t]
```
Where G_t = cumulative discounted reward

**Implementation:**
- ✅ Policy network with softmax output
- ✅ Monte Carlo return estimation
- ✅ Policy gradient calculation
- ✅ Gradient ascent optimization
- ✅ Baseline subtraction (optional)
- ✅ Entropy regularization

**Advantages:**
- Works well with continuous action spaces
- Can learn stochastic policies
- Effective for high-dimensional problems

**Challenges:**
- High variance in gradient estimates
- Requires complete episodes
- Sample inefficient

**Solutions Implemented:**
- Baseline subtraction to reduce variance
- Reward normalization
- Adaptive learning rates

---

## 📚 Key RL Concepts Demonstrated

### Fundamental RL Components
1. **Agent-Environment Interaction**
   - State observation
   - Action selection
   - Reward signals
   - State transitions

2. **Exploration vs Exploitation**
   - ε-greedy strategy
   - Entropy-based exploration
   - Decaying exploration rates

3. **Value Functions**
   - State-value function V(s)
   - Action-value function Q(s,a)
   - Advantage function A(s,a)

4. **Policy Optimization**
   - Policy gradients
   - Actor-critic methods
   - On-policy vs off-policy learning

### Advanced Techniques
- **Temporal Difference Learning** - Bootstrapping updates
- **Eligibility Traces** - Credit assignment
- **Function Approximation** - Neural network values/policies
- **Variance Reduction** - Baselines, advantage estimates
- **Reward Shaping** - Engineering reward signals

---

## 🏆 Results

### Q-Learning Performance
- **Convergence:** Successfully learns optimal policy
- **Stability:** Stable Q-table after sufficient episodes
- **Exploration:** ε-greedy ensures thorough state coverage

### Actor-Critic on CartPole
- **Training Episodes:** Typically solves in 200-500 episodes
- **Max Timesteps:** Achieves 200+ timesteps (environment maximum)
- **Stability:** Reliable convergence with proper hyperparameters
- **Model Saved:** Trained weights available for inference

### REINFORCE Algorithm
- **Policy Learning:** Successfully optimizes stochastic policies
- **Sample Efficiency:** Improved with baseline subtraction
- **Generalization:** Adapts to various Gym environments

---

## 🎓 Learning Outcomes

Through these projects, I have demonstrated expertise in:

1. **RL Foundations**
   - Markov Decision Processes (MDPs)
   - Bellman equations
   - Value iteration and policy iteration

2. **Deep RL**
   - Neural network function approximators
   - Policy gradient methods
   - Actor-critic architectures

3. **Practical RL**
   - Environment setup and interaction
   - Training loop implementation
   - Hyperparameter tuning
   - Performance evaluation and visualization

4. **Advanced Topics**
   - On-policy vs off-policy methods
   - Variance reduction techniques
   - Exploration strategies
   - Continuous vs discrete action spaces

---

## 📧 Contact

**Uzair Mubasher** - BSAI Graduate

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](www.linkedin.com/in/uzair-bin-mubasher-208ba5164/)
[![Email](https://img.shields.io/badge/Email-uzairmubasher5@gmail.com-red)](mailto:uzairmubasher5@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-uzi--gpu-black)](https://github.com/uzi-gpu)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- OpenAI Gym team for excellent RL environments
- PyTorch community for deep learning framework
- RL course instructors and resources

---

**⭐ If you found this repository helpful, please consider giving it a star!**

