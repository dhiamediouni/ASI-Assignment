# 🧠 Bayesian Dropout for Time-Series Regression

This project is a reproduction and extension of the seminal work:

**"Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"**  
*Yarin Gal and Zoubin Ghahramani (2016)*  
📄 [arXiv:1506.02142](https://arxiv.org/abs/1506.02142)

It demonstrates how **MC Dropout** can be used for uncertainty estimation in deep learning, applied to the **Mauna Loa CO₂ time series** dataset.

---

## 🎯 Objective

This project was created as part of the **ASI (Advanced Statistical Inference)** course, with goals to:

- Understand and implement a Bayesian approach to deep learning
- Reproduce and visualize key results from the Gal & Ghahramani paper
- Evaluate uncertainty-aware predictions on real-world time-series data

---

## 📊 Dataset

The model is trained and evaluated on the **Mauna Loa CO₂ concentration data** (`co2-mm-mlo.csv`), which contains monthly atmospheric CO₂ measurements from 1958 to present.

---

## 📁 Project Structure

- `mc_dropout_full_notebook.ipynb` – Main implementation notebook
- `co2-mm-mlo.csv` – Cleaned CO₂ dataset
- `README.md` – Project overview and documentation

---

## 🧠 Key Methods and Concepts

- **MC Dropout**: Dropout is used during both training and inference to approximate Bayesian posterior predictive distribution
- **Forward Passes**: Multiple stochastic forward passes used at test time
- **Uncertainty Estimation**:
  \[
  \text{Var}(y^*) = \text{Var}_{\text{MC}}(y^*) + \tau^{-1}
  \]
- **Loss Function**:
  \[
  \text{MSE Loss} + \lambda \sum ||w||^2 \quad \text{(L2 regularization)}
  \]
- **Precision Estimation**:
  \[
  \tau = \frac{p l^2}{2N\lambda}
  \]
  where \( p \) = dropout probability, \( l \) = length scale, \( N \) = number of data points, \( \lambda \) = weight decay

---

## 🧪 Model Architecture

- Fully connected neural network with ReLU activations
- Dropout layers placed before every linear layer (including inference time)
- Tunable hyperparameters:
  - Hidden dimensions
  - Dropout probability
  - Learning rate
  - L2 weight decay (regularization)
  - Number of MC samples

---

## 📈 Results & Visualization

- Captures temporal trends and expresses meaningful uncertainty
- Uncertainty is high in extrapolated regions and sparse data zones
- Evaluation Metrics:
  - **Root Mean Squared Error (RMSE)**
  - **Predictive Log-Likelihood**
- Visualizations include:
  - Predictions with confidence intervals
  - Comparison of deterministic vs Bayesian predictions
  - Error bands and model variance

---

## ⚙️ Dependencies

Tested with Python 3.8+. Required packages:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- torch

To install dependencies:

```bash
pip install -r requirements.txt
