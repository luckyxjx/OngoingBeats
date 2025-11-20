# Reinforcement Learning System - Complete Explanation

## Overview
The RL system fine-tunes the pre-trained music generator using **human feedback** to improve emotion alignment and music quality.

---

## 1. THE RL PROBLEM SETUP

### 1.1 Why RL?
**Problem**: Pre-trained model generates music, but:
- Emotion alignment might be weak
- Musical quality varies
- Outputs can be repetitive

**Solution**: Use RL to optimize for:
- Better emotion matching
- Higher musical coherence
- More diverse outputs

### 1.2 RL Formulation

**Agent**: Music Generator (Transformer model)

**Environment**: Music generation task

**State (s_t)**: 
- Current token sequence: [t₁, t₂, ..., t_t]
- Hidden state from Transformer
- Target emotion embedding

**Action (a_t)**: 
- Select next token from vocabulary
- Action space = vocab_size (~2,500 tokens)

**Reward (r)**: 
- Computed after full sequence generation
- Multi-component: emotion + coherence + diversity

**Episode**: 
- Generate one complete music piece
- Sequence of actions until <EOS> token

---

## 2. RL ALGORITHM: REINFORCE (Policy Gradient)

### 2.1 Algorithm Choice
**REINFORCE** = Classic policy gradient algorithm

**Why REINFORCE?**
- Simple and effective for discrete action spaces
- Works well with pre-trained models
- Suitable for sparse rewards (reward at end of episode)

### 2.2 Policy Gradient Theorem

**Objective**: Maximize expected reward
```
J(θ) = E[R(τ)] where τ is a trajectory
```

**Gradient**:
```
∇_θ J(θ) = E[∑_t ∇_θ log π_θ(a_t|s_t) * G_t]

where:
- π_θ = policy (our generator model)
- a_t = action (token) at time t
- s_t = state at time t
- G_t = return (cumulative reward from time t)
```

**In plain English**:
- If action led to high reward → increase its probability
- If action led to low reward → decrease its probability

### 2.3 REINFORCE with Baseline

**Problem**: High variance in gradient estimates

**Solution**: Subtract baseline (value function)
```
∇_θ J(θ) = E[∑_t ∇_θ log π_θ(a_t|s_t) * (G_t - V(s_t))]

where:
- V(s_t) = baseline value function
- (G_t - V(s_t)) = advantage
```

**Advantage**: How much better is this action compared to average?

---

## 3. MODEL ARCHITECTURE

### 3.1 Policy Network (Generator)
**Same Transformer model from supervised training**

```
Input: [tokens, emotion, duration]
    ↓
Transformer Encoder-Decoder (8 layers)
    ↓
Output: Probability distribution over next token
    ↓
Sample action: a_t ~ π_θ(·|s_t)
```

**Key**: Model outputs probabilities, we sample from them

### 3.2 Baseline Network (Value Function)

```python
class Baseline(nn.Module):
    def __init__(self, d_model=256):
        self.network = nn.Sequential(
            Linear(d_model → 128),
            ReLU(),
            Dropout(0.2),
            Linear(128 → 64),
            ReLU(),
            Linear(64 → 1)  # Predict expected reward
        )
```

**Input**: Hidden state from Transformer (d_model dims)
**Output**: Scalar value V(s_t) = expected future reward

**Purpose**: Reduce variance in policy gradient

### 3.3 Emotion Classifier (for Reward)

```python
class EmotionClassifier(nn.Module):
    def __init__(self):
        self.embedding = Embedding(vocab_size → 256)
        self.lstm = BiLSTM(256 → 256, 2 layers)
        self.classifier = Linear(512 → 6 emotions)
```

**Architecture**:
```
Token Sequence
    ↓
Embedding Layer (vocab_size → 256)
    ↓
Bidirectional LSTM (2 layers)
    ↓
Concatenate final forward + backward states
    ↓
MLP Classifier (512 → 128 → 6)
    ↓
Softmax → Emotion probabilities
```

**Training**: Supervised learning on labeled EMOPIA data
**Purpose**: Evaluate emotion alignment for reward

---

## 4. REWARD FUNCTION (Multi-Component)

### 4.1 Total Reward Formula
```
R_total = w₁ * R_emotion + w₂ * R_coherence + w₃ * R_diversity

Default weights:
- w₁ = 0.6 (emotion alignment)
- w₂ = 0.25 (musical coherence)
- w₃ = 0.15 (diversity)
```

### 4.2 Component 1: Emotion Reward

**Goal**: Generated music should match target emotion

**Computation**:
```python
def compute_emotion_reward(tokens, target_emotion):
    # 1. Pass tokens through emotion classifier
    logits = emotion_classifier(tokens)  # [batch, 6]
    probs = softmax(logits)              # [batch, 6]
    
    # 2. Reward = confidence in target emotion
    reward = probs[:, target_emotion]    # [batch]
    
    return reward
```

**Example**:
- Target: joy (emotion 0)
- Classifier output: [0.8, 0.05, 0.05, 0.05, 0.03, 0.02]
- Reward: 0.8 (high confidence in joy)

**Range**: [0, 1]

### 4.3 Component 2: Coherence Reward

**Goal**: Music should be musically coherent (not random)

**Metrics**:

**A. Pitch Coherence**:
```python
# Extract pitch tokens (NOTE_ON tokens)
pitches = [token for token in tokens if is_note_on(token)]

# Compute average pitch jump
pitch_diffs = abs(diff(pitches))
avg_jump = mean(pitch_diffs)

# Reward smaller jumps (more melodic)
pitch_score = exp(-avg_jump / 12.0)  # Normalize by octave
```

**B. Rhythm Consistency**:
```python
# Measure token distribution entropy
token_counts = Counter(tokens)
entropy = -sum((count/total) * log2(count/total) 
               for count in token_counts)
max_entropy = log2(len(tokens))

# Normalized entropy (0 = repetitive, 1 = random)
rhythm_score = entropy / max_entropy
```

**Combined**:
```python
coherence = 0.6 * pitch_score + 0.4 * rhythm_score
```

**Range**: [0, 1]

### 4.4 Component 3: Diversity Reward

**Goal**: Avoid generating same music repeatedly

**Metrics**:

**A. Token Entropy** (internal diversity):
```python
token_counts = Counter(tokens)
entropy = -sum((count/total) * log2(count/total))
entropy_score = entropy / max_entropy
```

**B. Uniqueness** (compared to previous samples):
```python
# Jaccard similarity with previous 10 samples
for prev_sample in previous_samples[-10:]:
    intersection = len(set(tokens) & set(prev_sample))
    union = len(set(tokens) | set(prev_sample))
    similarity = intersection / union

uniqueness_score = 1.0 - mean(similarities)
```

**Combined**:
```python
diversity = 0.7 * entropy_score + 0.3 * uniqueness_score
```

**Range**: [0, 1]

### 4.5 Reward Normalization

**Problem**: Rewards have different scales

**Solution**: Normalize to zero mean, unit variance
```python
# Running statistics
reward_mean = mean(last_1000_rewards)
reward_std = std(last_1000_rewards)

# Normalize
normalized_reward = (reward - reward_mean) / reward_std
```

---

## 5. TRAINING ALGORITHM (Step-by-Step)

### 5.1 Episode Generation

```python
def generate_episode(emotion, duration):
    tokens = [<BOS>]
    log_probs = []
    states = []
    
    for t in range(max_tokens):
        # 1. Get current state
        input_ids = torch.tensor([tokens])
        
        # 2. Forward pass through policy
        outputs = generator(input_ids, emotion=emotion)
        logits = outputs['logits'][0, -1, :]  # Last position
        hidden_state = outputs['hidden_states'][-1][0, -1, :]
        
        # 3. Sample action from policy
        probs = softmax(logits / temperature)
        next_token = sample(probs)
        
        # 4. Store log probability and state
        log_prob = log(probs[next_token])
        log_probs.append(log_prob)
        states.append(hidden_state)
        
        # 5. Add token to sequence
        tokens.append(next_token)
        
        # 6. Check for termination
        if next_token == <EOS>:
            break
    
    return tokens, log_probs, states
```

### 5.2 Advantage Computation

```python
def compute_advantages(rewards, states):
    # 1. Compute returns (discounted cumulative rewards)
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G  # gamma = 0.99
        returns.insert(0, G)
    
    # 2. Predict baselines
    baselines = baseline_network(states)  # V(s_t)
    
    # 3. Compute advantages
    advantages = returns - baselines  # A(s_t, a_t)
    
    return advantages
```

**Example**:
```
Rewards:     [0.5, 0.5, 0.5, 0.8]
Returns:     [2.13, 1.64, 1.15, 0.8]  (with gamma=0.99)
Baselines:   [1.8, 1.5, 1.0, 0.7]
Advantages:  [0.33, 0.14, 0.15, 0.1]
```

### 5.3 Policy Update

```python
def update_policy(log_probs, advantages):
    # 1. Compute policy gradient loss
    policy_loss = 0
    for log_prob, advantage in zip(log_probs, advantages):
        policy_loss -= log_prob * advantage
    
    policy_loss = policy_loss / len(log_probs)
    
    # 2. Backward pass
    policy_loss.backward()
    
    # 3. Gradient clipping (prevent instability)
    clip_grad_norm_(generator.parameters(), max_norm=1.0)
    
    # 4. Update weights
    policy_optimizer.step()
```

**Intuition**:
- If advantage > 0: Action was better than expected → increase probability
- If advantage < 0: Action was worse than expected → decrease probability

### 5.4 Baseline Update

```python
def update_baseline(states, returns):
    # 1. Predict values
    predicted_values = baseline_network(states)
    
    # 2. MSE loss
    baseline_loss = MSE(predicted_values, returns)
    
    # 3. Update baseline
    baseline_loss.backward()
    baseline_optimizer.step()
```

**Goal**: Train baseline to predict returns accurately

### 5.5 Complete Training Loop

```python
for episode in range(num_episodes):
    # 1. Sample emotion
    emotion = random.choice([0, 1, 2, 3, 4, 5])
    
    # 2. Generate episode
    tokens, log_probs, states = generate_episode(emotion)
    
    # 3. Compute reward
    total_reward, components = reward_function(tokens, emotion)
    
    # 4. Assign reward to all timesteps (sparse reward)
    rewards = [total_reward] * len(log_probs)
    
    # 5. Compute advantages
    advantages = compute_advantages(rewards, states)
    
    # 6. Update policy
    policy_loss = update_policy(log_probs, advantages)
    
    # 7. Update baseline
    baseline_loss = update_baseline(states, returns)
    
    # 8. Log metrics
    print(f"Episode {episode}: Reward={total_reward:.4f}")
```

---

## 6. HYPERPARAMETERS

### 6.1 Learning Rates
```python
policy_lr = 1e-5      # Very small (fine-tuning pre-trained model)
baseline_lr = 1e-4    # Larger (training from scratch)
```

**Why different?**
- Policy: Already trained, small updates
- Baseline: New network, needs faster learning

### 6.2 Discount Factor
```python
gamma = 0.99  # High discount (long-term rewards matter)
```

### 6.3 Gradient Clipping
```python
max_grad_norm = 1.0  # Prevent exploding gradients
```

### 6.4 Reward Weights
```python
emotion_weight = 0.6      # Most important
coherence_weight = 0.25   # Musical quality
diversity_weight = 0.15   # Avoid repetition
```

### 6.5 Training Schedule
```python
num_episodes = 1000       # Total episodes
log_interval = 10         # Print every 10 episodes
save_interval = 100       # Save checkpoint every 100
```

---

## 7. TRAINING PHASES

### Phase 1: Train Emotion Classifier
```
1. Use labeled EMOPIA data
2. Train BiLSTM classifier
3. Achieve ~80% accuracy
4. Save best model
```

### Phase 2: RL Fine-tuning
```
1. Load pre-trained generator
2. Initialize baseline network
3. Train with REINFORCE for 1000 episodes
4. Save best model based on average reward
```

---

## 8. MATHEMATICAL FORMULATION

### 8.1 Policy Gradient Objective
```
J(θ) = E_τ~π_θ [R(τ)]

where:
- τ = (s₀, a₀, r₀, s₁, a₁, r₁, ..., s_T)
- π_θ = policy parameterized by θ
- R(τ) = ∑_t γ^t r_t
```

### 8.2 REINFORCE Gradient
```
∇_θ J(θ) = E_τ [∑_t ∇_θ log π_θ(a_t|s_t) * (G_t - b_t)]

where:
- G_t = ∑_{t'=t}^T γ^(t'-t) r_t'  (return)
- b_t = V_φ(s_t)  (baseline)
```

### 8.3 Advantage Function
```
A(s_t, a_t) = Q(s_t, a_t) - V(s_t)
            = G_t - V_φ(s_t)

where:
- Q(s_t, a_t) = expected return after taking action a_t
- V(s_t) = expected return from state s_t
```

### 8.4 Baseline Loss
```
L_baseline = E[(G_t - V_φ(s_t))²]
```

---

## 9. COMPARISON: SUPERVISED vs RL

### Supervised Learning (Stage 1)
```
Objective: Minimize cross-entropy loss
Loss = -log P(next_token | previous_tokens)

Pros:
- Stable training
- Fast convergence
- Good general music generation

Cons:
- No explicit emotion optimization
- Can't incorporate human preferences
- Fixed training data
```

### Reinforcement Learning (Stage 2)
```
Objective: Maximize reward
Reward = emotion_accuracy + coherence + diversity

Pros:
- Optimizes for specific goals (emotion)
- Can incorporate human feedback
- Improves beyond training data

Cons:
- Unstable training (high variance)
- Slower convergence
- Requires reward function design
```

---

## 10. KEY INSIGHTS

### 10.1 Why Policy Gradient?
- **Discrete actions**: Can't use value-based methods easily
- **Large action space**: 2,500 tokens
- **Pre-trained model**: Fine-tuning works better than training from scratch

### 10.2 Why Baseline?
- **Variance reduction**: Gradients are more stable
- **Faster convergence**: Less noisy updates
- **Better performance**: Higher final reward

### 10.3 Why Multi-Component Reward?
- **Emotion alone**: Might sacrifice music quality
- **Coherence alone**: Might ignore emotion
- **Diversity alone**: Might generate random noise
- **Combined**: Balanced optimization

---

## 11. TRAINING WORKFLOW

```
1. Pre-train Generator (Supervised)
   ├── EMOPIA dataset (400 files)
   ├── Cross-entropy loss
   └── 150 epochs
   
2. Train Emotion Classifier
   ├── Same EMOPIA data
   ├── Classification loss
   └── 10 epochs
   
3. RL Fine-tuning (REINFORCE)
   ├── Pre-trained generator
   ├── Trained emotion classifier
   ├── Multi-component reward
   └── 1000 episodes
   
4. Evaluation
   ├── Generate samples
   ├── Human evaluation
   └── Compare with baseline
```

---

## 12. EXPECTED IMPROVEMENTS

### Before RL (Supervised only):
- Emotion accuracy: ~60-70%
- Musical coherence: Variable
- Diversity: Low (repetitive)

### After RL (Fine-tuned):
- Emotion accuracy: ~75-85%
- Musical coherence: Higher
- Diversity: Improved

---

## SUMMARY

**Algorithm**: REINFORCE (Policy Gradient with Baseline)

**Models**:
1. **Policy Network**: Pre-trained Transformer (music generator)
2. **Baseline Network**: 3-layer MLP (value function)
3. **Emotion Classifier**: BiLSTM (for reward computation)

**Reward**: Multi-component (emotion + coherence + diversity)

**Training**: 
- Generate episode → Compute reward → Calculate advantages → Update policy & baseline

**Goal**: Fine-tune generator to produce emotion-aligned, coherent, diverse music

This is essentially **RLHF (Reinforcement Learning from Human Feedback)** for music generation!
