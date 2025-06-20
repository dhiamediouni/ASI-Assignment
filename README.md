# Bayesian Deep Learning with MC Dropout

*Project by Sarra Gharsallah and Mohamed Dhia Mediouni*  
*For the ASI (Advanced Statistical Inference) lecture at EURECOM, Spring 2025*

This repository contains a unified notebook reproducing the experiments from the seminal paper  
**"Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"**  
by Yarin Gal and Zoubin Ghahramani.

---

## Project Overview

The notebook implements and analyzes three distinct settings demonstrating Monte Carlo (MC) Dropout as a Bayesian approximation technique:

1. **Regression on UCI Dataset (Boston Housing)**  
2. **Classification on MNIST**  
3. **Reinforcement Learning with MC Dropout Deep Q-Network (DQN)**

Each part links theoretical concepts from the paper to practical code implementations and includes detailed explanations.

---

## Part 1: Regression with MC Dropout on Boston Housing Dataset

- A fully connected neural network with dropout is trained on the Boston Housing dataset for regression.  
- The training objective combines mean squared error with weight decay, corresponding to a variational inference interpretation.  
- At test time, multiple stochastic forward passes with dropout active are performed to estimate the predictive mean and uncertainty (variance).  
- The model’s performance is evaluated using Root Mean Squared Error (RMSE) and predictive log-likelihood, quantifying both accuracy and uncertainty.  
- Visualization shows predictions with uncertainty intervals, demonstrating that MC Dropout effectively captures model confidence, especially in regions with less data.

---

## Part 2: Classification with MC Dropout on MNIST

- A convolutional neural network with dropout layers approximates a Bayesian classifier on the MNIST handwritten digits dataset.  
- MC Dropout is used at test time by performing multiple stochastic forward passes to estimate predictive class probabilities and uncertainty.  
- Model performance is measured by accuracy and predictive log-likelihood.  
- Uncertainty is visualized via predictive entropy, highlighting samples where the model is less confident due to ambiguity or noise.  
- Additional visualization shows the distribution of predictive confidence across the test set, confirming that the model expresses calibrated uncertainty.

---

## Part 3: Reinforcement Learning Exploration with Bayesian DQN

- The MC Dropout technique is applied to Deep Q-Networks (DQN) to incorporate model uncertainty into reinforcement learning.  
- Dropout is applied during both training and action selection, enabling stochastic Q-value estimates that capture epistemic uncertainty.  
- Action selection uses MC sampling to approximate the expected Q-values, favoring exploration of uncertain actions rather than relying on traditional ε-greedy strategies.  
- The agent is trained on the CartPole environment, with a replay buffer and target network updates.  
- Training results show that uncertainty-driven exploration improves sample efficiency and leads to stable learning performance over episodes.

---

## Dependencies

- Python 3.8+  
- PyTorch  
- NumPy, SciPy, Scikit-learn  
- Matplotlib, Plotly  
- Torchvision (for MNIST)  
- Gym (for RL experiments)  
