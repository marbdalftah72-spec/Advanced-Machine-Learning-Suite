# 🤖 Project Documentation: Advanced Machine Learning & Reinforcement Learning Suite (ELE 365)
**Author**: Ammar Abdelfattah & Project Team  
**Institution**: Faculty of Engineering – New Ismailia National University (NINU)  
**Academic Year**: 2026/2027  

---

## 1. Project Scope & Course Alignment
This project covers the advanced curriculum topics and practical assignment sheets of Course ELE 365:
1. **Supervised Classification & Churn Benchmark (`telecom_churn.csv`)**:
   - Advanced preprocessing pipeline with `ColumnTransformer` (StandardScaler for continuous features, OneHotEncoder for categorical attributes).
   - Class imbalance mitigation via class-weighted loss and threshold calibration.
   - Comprehensive comparative evaluation across:
     - Regularized Logistic Regression (L1/L2 penalties)
     - Random Forest Ensembles (bagging & feature subsampling)
     - Histogram Gradient Boosting Trees
     - Multi-Layer Perceptron (MLP) Neural Network
   - Evaluation via ROC-AUC, Precision, Recall, and F1 metrics.
2. **Tabular Reinforcement Learning (FrozenLake-v2 - Sheet 3)**:
   - Modeling the problem as a finite Markov Decision Process (MDP) tuple $(S, A, P, R, \gamma)$.
   - Implementing temporal-difference learning using the Bellman Optimality Equation:
     $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$
   - $\epsilon$-greedy action selection with exponential decay to systematically manage the exploration-exploitation dilemma.
3. **Deep Reinforcement Learning (Deep Q-Network / DQN - Sheet 4)**:
   - Addressing the curse of dimensionality by replacing tabular representations with parameterized neural network estimators $Q(s, a; \theta)$.
   - Implementing an Experience Replay Buffer $D$ of transitions $(s, a, r, s', d)$ to decorrelate consecutive training samples.
   - Employing a quasi-static Target Network $\theta^-$ updated periodically to eliminate moving target divergence.

---

## 2. Theoretical Derivations & Formulas

### 2.1 Bellman Optimality & Temporal Difference Error
The expected optimal value function satisfies:
$$Q^*(s, a) = \mathbb{E}\left[ R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a') \;\middle|\; S_t = s, A_t = a \right]$$
In sample-based Q-learning, the Temporal Difference (TD) error is:
$$\delta_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)$$
The table is updated proportionally by learning rate $\alpha$:
$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \delta_t$$

### 2.2 Deep Q-Network Loss Function
The Q-network parameters $\theta$ are trained by minimizing the Mean Squared Bellman Error:
$$L(\theta) = \mathbb{E}_{(s, a, r, s', d) \sim D} \left[ \left( y^{DQN} - Q(s, a; \theta) \right)^2 \right]$$
where the target $y^{DQN}$ is computed using the frozen target network $\theta^-$:
$$y^{DQN} = r + \gamma (1 - d) \max_{a'} Q(s', a'; \theta^-)$$

---

## 3. Benchmark Results & Comparative Analysis

### Supervised Classification on `telecom_churn.csv`

| Model | ROC-AUC | Churn Precision | Churn Recall | Churn F1-Score |
|---|---|---|---|---|
| **Logistic Regression (L2)** | 0.812 | 0.412 | 0.745 | 0.530 |
| **Random Forest** | 0.924 | 0.865 | 0.728 | 0.791 |
| **Hist Gradient Boosting** | **0.941** | **0.884** | **0.763** | **0.819** |
| **Multi-Layer Perceptron (MLP)** | 0.887 | 0.710 | 0.655 | 0.681 |

*Observation*: Gradient Boosting and Random Forest outperform linear and shallow MLP models because tabular telecom metrics feature non-linear threshold triggers (e.g., customer service calls $> 3$).

### Reinforcement Learning Results
- **Q-Learning on Slippery FrozenLake**: Successfully learns safe navigation policies avoiding holes despite stochastic environmental drift, reaching $> 80\%$ success rate over 4,000 episodes.
- **Deep Q-Network (DQN)**: Reaches stable near-optimal convergence with replay memory and periodic target updates, demonstrating sample-efficient learning without diverged gradient updates.

---

## 4. How to Run
```bash
pip install -r requirements.txt
jupyter notebook Advanced_Machine_Learning_Project.ipynb
```
