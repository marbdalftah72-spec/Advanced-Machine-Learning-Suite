# 🤖 Advanced Machine Learning & Reinforcement Learning Suite

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Libraries-Scikit--Learn%20%7C%20PyTorch%20%7C%20Gymnasium-orange.svg)](https://gymnasium.farama.org/)

An end-to-end repository implementing rigorous supervised benchmarking pipelines and reinforcement learning architectures (tabular and deep variants), built as part of the advanced Machine Learning curriculum at **Faculty of Engineering – New Ismailia National University (NINU)**.

---

## 📌 Project Overview & Structure

The repository tackles three fundamental machine learning domains:
1. **Supervised Customer Churn Classification (`telecom_churn.csv`)**:
   - Production-grade scikit-learn `Pipeline` and `ColumnTransformer` (scaling continuous attributes and one-hot encoding categoricals).
   - Class-weight penalization to address data imbalance.
   - Comprehensive model benchmark:
     - Regularized Logistic Regression (L1/L2 penalties)
     - Random Forest Ensembles (bagging & feature subsampling)
     - Histogram-based Gradient Boosting Trees (LightGBM equivalent)
     - Multi-Layer Perceptron (MLP) Neural Network
   - Evaluation metrics: ROC-AUC, Precision, Recall, and F1-Score.

2. **Tabular Reinforcement Learning (`FrozenLake-v2`)**:
   - Finite Markov Decision Process (MDP) modeling: $(S, A, P, R, \gamma)$.
   - Temporal-Difference learning via the Bellman Optimality Equation:
     $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$
   - Adaptive $\epsilon$-greedy exploration with exponential decay.

3. **Deep Reinforcement Learning (Deep Q-Network / DQN)**:
   - Overcoming state space curse of dimensionality with parameterized neural function approximators $Q(s, a; \theta)$.
   - Experience Replay Buffer $D$ storing $(s, a, r, s', d)$ transitions to remove temporal sample correlation.
   - Separate quasi-static Target Network $\theta^-$ updated periodically to eliminate moving target divergence.

---

## 📁 Repository Structure

```text
├── Advanced_Machine_Learning_Project.ipynb  # Full reproducible Jupyter Notebook
├── Documentation.md                        # Theoretical analysis & mathematical proofs
├── telecom_churn.csv                       # Benchmark dataset
├── requirements.txt                        # Dependency specifications
└── README.md                               # Project documentation
```

---

## 🚀 Quickstart & Installation

```bash
# Clone the repository
git clone https://github.com/marbdalftah72-spec/ML-task.git
cd ML-task

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook Advanced_Machine_Learning_Project.ipynb
```

---

## 👨‍💻 Author & Attribution

- **Ammar Abdelfattah & Project Team**
- Department of Computer / Systems Engineering, Faculty of Engineering
- **New Ismailia National University (NINU)**
