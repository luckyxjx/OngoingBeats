# Reinforcement Learning Model - Deep Technical Analysis

## Table of Contents
1. [Overview](#overview)
2. [RL Problem Formulation](#rl-problem-formulation)
3. [REINFORCE Algorithm](#reinforce-algorithm)
4. [Model Architectures](#model-architectures)
5. [Reward Function Design](#reward-function-design)
6. [Training Process](#training-process)
7. [Mathematical Foundations](#mathematical-foundations)
8. [Implementation Details](#implementation-details)
9. [Hyperparameters](#hyperparameters)
10. [Training Workflow](#training-workflow)

---

## 1. Overview

The RL system implements **Policy Gradient (REINFORCE)** to fine-tune a pre-trained music generator for better emotion alignment and musical quality.

### Why Reinforcement Learning?

**Limitations of Supervised Learning:**
- Only learns from training data patterns
- No explicit optimization for emotion accuracy
- Cannot incorporate human preferences
- Fixed objective (cross-entropy loss)

**Advantages of RL:**
- Optimizes for specific goals (emotion alignment, coherence, diversity)
- Can incorporate human feedback through reward design
- Improves beyond training data distribution
- Flexible reward function design


---

## 2. RL Problem Formulation

### 2.1 Markov Decision Process (MDP)

The music generation task is formulated as an MDP: **M = (S, A, P, R, γ)**

#### State Space (S)
**State at timestep t:**
```
s_t = {
    token_sequence: [t₁, t₂, ..., t_t],
    hidden_state: h_t ∈ ℝ^d_model,
    emotion: e ∈ {0, 1, 2, 3, 4, 5},
    duration: d ∈ ℝ⁺
}
```

**Components:**
- **Token Sequence**: All tokens generated so far
- **Hidden State**: Last layer output from Transformer (512-dim vector)
- **Emotion**: Target emotion index (joy=0, sadness=1, anger=2, calm=3, surprise=4, fear=5)
- **Duration**: Target duration in minutes

**State Space Size**: Effectively infinite (continuous hidden states + variable-length sequences)

#### Action Space (A)
**Action at timestep t:**
```
a_t = next_token ∈ {0, 1, ..., vocab_size-1}
```

**Action Space Size**: ~2,500 tokens
- 256 note tokens (128 NOTE_ON + 128 NOTE_OFF)
- 32 time shift tokens
- Velocity buckets
- Instrument tokens
- Special tokens (<BOS>, <EOS>, <PAD>)

**Action Type**: Discrete


#### Transition Function (P)
**Deterministic transition:**
```
P(s_{t+1} | s_t, a_t) = 1 if s_{t+1} = append(s_t, a_t)
                      = 0 otherwise
```

The next state is deterministically the current state with the action (token) appended.

#### Reward Function (R)
**Sparse reward structure:**
```
R(s_t, a_t) = 0                    for t < T (intermediate steps)
R(s_T, a_T) = reward_function(τ)  for t = T (terminal step)
```

Where τ is the complete trajectory (full music sequence).

**Reward computed at episode end based on:**
1. Emotion alignment (60% weight)
2. Musical coherence (25% weight)
3. Output diversity (15% weight)

#### Discount Factor (γ)
```
γ = 0.99
```

High discount factor because:
- Long episodes (100-3000 tokens)
- Future tokens affect overall music quality
- Sparse reward at end requires credit assignment

### 2.2 Policy

**Policy Definition:**
```
π_θ(a_t | s_t) = P(next_token | previous_tokens, emotion, duration; θ)
```

**Policy Network**: Pre-trained Transformer music generator
- Parameters θ: All Transformer weights (~50M parameters)
- Stochastic policy: Samples from probability distribution
- Parameterized by neural network


### 2.3 Episode Structure

**Episode = One complete music generation**

```
Episode τ:
    t=0: s₀ = {[<BOS>], h₀, emotion, duration}
         a₀ ~ π_θ(·|s₀)
    
    t=1: s₁ = {[<BOS>, a₀], h₁, emotion, duration}
         a₁ ~ π_θ(·|s₁)
    
    ...
    
    t=T: s_T = {[<BOS>, a₀, ..., a_{T-1}], h_T, emotion, duration}
         a_T = <EOS>
         
    Reward: R(τ) = reward_function(complete_sequence)
```

**Episode Length**: Variable (100-3000 tokens typically)

**Termination Conditions:**
1. Model generates <EOS> token
2. Maximum token limit reached (3072)
3. Minimum note requirement met (100 notes)

---

## 3. REINFORCE Algorithm

### 3.1 Algorithm Overview

**REINFORCE** (Williams, 1992) is a Monte Carlo policy gradient algorithm.

**Key Idea**: 
- Sample complete episodes from current policy
- Compute returns (cumulative rewards)
- Update policy to increase probability of high-reward actions

**Algorithm Type**: On-policy, model-free, policy gradient


### 3.2 Policy Gradient Theorem

**Objective**: Maximize expected return
```
J(θ) = E_{τ~π_θ}[R(τ)]
     = E_{τ~π_θ}[∑_{t=0}^T γ^t r_t]
```

**Policy Gradient**:
```
∇_θ J(θ) = E_{τ~π_θ}[∑_{t=0}^T ∇_θ log π_θ(a_t|s_t) · G_t]
```

Where:
- **G_t** = Return from timestep t = ∑_{t'=t}^T γ^{t'-t} r_{t'}
- **∇_θ log π_θ(a_t|s_t)** = Score function (gradient of log probability)

**Intuition**:
- If action led to high return (G_t > 0) → increase its probability
- If action led to low return (G_t < 0) → decrease its probability
- Magnitude of update proportional to return

### 3.3 REINFORCE with Baseline

**Problem**: High variance in gradient estimates
- Returns can vary significantly between episodes
- Noisy gradients → slow, unstable learning

**Solution**: Subtract baseline b(s_t) from returns
```
∇_θ J(θ) = E_{τ~π_θ}[∑_{t=0}^T ∇_θ log π_θ(a_t|s_t) · (G_t - b(s_t))]
```

**Baseline Choice**: Value function V_φ(s_t)
```
b(s_t) = V_φ(s_t) ≈ E[G_t | s_t]
```

**Advantage Function**:
```
A(s_t, a_t) = G_t - V_φ(s_t)
```

**Interpretation**:
- A(s_t, a_t) > 0: Action better than average → increase probability
- A(s_t, a_t) < 0: Action worse than average → decrease probability
- A(s_t, a_t) = 0: Action average → no change

**Variance Reduction**: Baseline doesn't change expected gradient but reduces variance


### 3.4 REINFORCE Algorithm Pseudocode

```python
# Initialize
policy_network = PretrainedTransformer()
baseline_network = ValueFunction()
policy_optimizer = Adam(policy_network.parameters(), lr=1e-5)
baseline_optimizer = Adam(baseline_network.parameters(), lr=1e-4)

for episode in range(num_episodes):
    # 1. Generate episode using current policy
    states, actions, log_probs = [], [], []
    state = initial_state()
    
    for t in range(max_steps):
        # Sample action from policy
        action_probs = policy_network(state)
        action = sample(action_probs)
        log_prob = log(action_probs[action])
        
        # Store trajectory
        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        
        # Transition
        state = next_state(state, action)
        
        if action == EOS:
            break
    
    # 2. Compute reward for complete episode
    reward = reward_function(states, actions)
    
    # 3. Compute returns (discounted cumulative rewards)
    returns = []
    G = 0
    for t in reversed(range(len(states))):
        G = reward + gamma * G  # Sparse reward: reward only at end
        returns.insert(0, G)
    
    # 4. Compute advantages using baseline
    baselines = baseline_network(states)
    advantages = returns - baselines
    
    # 5. Update policy (REINFORCE gradient)
    policy_loss = -sum(log_probs[t] * advantages[t] for t in range(len(states)))
    policy_loss.backward()
    clip_grad_norm_(policy_network.parameters(), max_norm=1.0)
    policy_optimizer.step()
    
    # 6. Update baseline (MSE loss)
    baseline_loss = MSE(baselines, returns)
    baseline_loss.backward()
    baseline_optimizer.step()
```


---

## 4. Model Architectures

### 4.1 Policy Network (Music Generator)

**Architecture**: Transformer Decoder (same as pre-trained model)

```
Input Layer:
├── Token Embedding: vocab_size → 512
├── Emotion Embedding: 6 → 64
└── Duration Embedding: 1 → 32
    ↓
Concatenate: [512 + 64 + 32] = 608
    ↓
Input Projection: 608 → 512
    ↓
Positional Encoding: Add sinusoidal position embeddings
    ↓
Transformer Decoder (8 layers):
│   ├── Multi-Head Self-Attention (8 heads, 64 dims each)
│   ├── Layer Normalization
│   ├── Feed-Forward Network (512 → 2048 → 512)
│   ├── Layer Normalization
│   └── Residual Connections
    ↓
Output Projection: 512 → vocab_size (~2500)
    ↓
Softmax: Probability distribution over tokens
```

**Parameters**: ~50 million
**Input**: Token sequence + emotion + duration
**Output**: Probability distribution π_θ(a|s) over next token

**Key Features**:
- **Causal masking**: Can only attend to previous tokens
- **Emotion conditioning**: Emotion embedding added to every token
- **Duration control**: Duration embedding guides sequence length


### 4.2 Baseline Network (Value Function)

**Architecture**: 3-layer MLP

```python
class Baseline(nn.Module):
    def __init__(self, d_model=512):
        self.network = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
```

**Architecture Diagram**:
```
Hidden State (from Transformer): 512 dims
    ↓
Linear Layer: 512 → 128
    ↓
ReLU Activation
    ↓
Dropout (p=0.2)
    ↓
Linear Layer: 128 → 64
    ↓
ReLU Activation
    ↓
Linear Layer: 64 → 1
    ↓
Scalar Value: V_φ(s_t) ∈ ℝ
```

**Parameters**: ~67,000
**Input**: Hidden state from Transformer (512-dim vector)
**Output**: Scalar value V_φ(s_t) = expected future return

**Purpose**:
- Estimate expected return from state s_t
- Reduce variance in policy gradient
- Trained with MSE loss against actual returns

**Training**:
```
Loss = E[(G_t - V_φ(s_t))²]
```


### 4.3 Emotion Classifier (for Reward Computation)

**Architecture**: BiLSTM + MLP Classifier

```python
class EmotionClassifier(nn.Module):
    def __init__(self, vocab_size=2500, d_model=256, num_emotions=6):
        self.embedding = nn.Embedding(vocab_size, 256)
        self.lstm = nn.LSTM(256, 256, num_layers=2, 
                           batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),  # 512 = 256*2 (bidirectional)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 6)
        )
```

**Architecture Diagram**:
```
Token Sequence: [batch, seq_len]
    ↓
Embedding Layer: vocab_size → 256
    ↓ [batch, seq_len, 256]
Bidirectional LSTM (2 layers):
│   Forward LSTM: 256 → 256
│   Backward LSTM: 256 → 256
    ↓
Concatenate Final States: [h_forward, h_backward]
    ↓ [batch, 512]
Linear Layer: 512 → 128
    ↓
ReLU Activation
    ↓
Dropout (p=0.3)
    ↓
Linear Layer: 128 → 6
    ↓
Softmax: Probability distribution over emotions
    ↓ [batch, 6]
Output: P(emotion | music_sequence)
```

**Parameters**: ~2 million
**Input**: Complete token sequence
**Output**: Emotion probabilities [joy, sadness, anger, calm, surprise, fear]

**Training**:
- Supervised learning on labeled EMOPIA data
- Cross-entropy loss
- 10 epochs, ~80% validation accuracy

**Usage in RL**:
- Frozen during RL training (not updated)
- Used to compute emotion alignment reward
- Reward = confidence in target emotion


---

## 5. Reward Function Design

### 5.1 Multi-Component Reward Structure

**Total Reward Formula**:
```
R_total = w₁ · R_emotion + w₂ · R_coherence + w₃ · R_diversity

Default weights:
w₁ = 0.6  (emotion alignment - most important)
w₂ = 0.25 (musical coherence)
w₃ = 0.15 (diversity)
```

**Design Rationale**:
- **Emotion (60%)**: Primary objective - music must match target emotion
- **Coherence (25%)**: Music must be musically valid and pleasant
- **Diversity (15%)**: Avoid repetitive, boring outputs

### 5.2 Component 1: Emotion Alignment Reward

**Objective**: Generated music should match target emotion

**Computation**:
```python
def compute_emotion_reward(tokens, target_emotion):
    # 1. Pass tokens through emotion classifier
    logits = emotion_classifier(tokens)  # [batch, 6]
    probs = softmax(logits, dim=-1)      # [batch, 6]
    
    # 2. Extract confidence in target emotion
    emotion_reward = probs[:, target_emotion]  # [batch]
    
    return emotion_reward
```

**Mathematical Formulation**:
```
R_emotion(τ, e_target) = P(e_target | τ; θ_classifier)
```

Where:
- τ = generated token sequence
- e_target = target emotion index
- θ_classifier = emotion classifier parameters (frozen)

**Range**: [0, 1]
- 0 = No confidence in target emotion
- 1 = Perfect confidence in target emotion

**Example**:
```
Target: joy (emotion 0)
Classifier output: [0.82, 0.05, 0.03, 0.04, 0.03, 0.03]
Emotion reward: 0.82
```

