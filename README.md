# Bayesian Dropout Reproduction

This repository contains the final project for the **ASI (Applied Statistical Inference)** course. It reproduces core results from the following foundational paper in Bayesian deep learning:

**Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning**  
by *Yarin Gal and Zoubin Ghahramani (2016)*  
[arXiv link](https://arxiv.org/abs/1506.02142)

---

## 🎯 Project Objective (ASI Course)

As part of the ASI course, students were required to:

- Select and understand a research paper in probabilistic machine learning  
- Reproduce key results using real or synthetic data  
- Implement mathematical ideas in code  
- Explain both theoretical and empirical results in a structured report

This project satisfies those goals by implementing **MC Dropout**, a Bayesian approximation technique, and applying it to the **Mauna Loa CO₂ dataset** to estimate predictive uncertainty in a time-series regression setting.

---

## 📊 Dataset

The model is evaluated on the **Mauna Loa CO₂ concentration data** (`co2-mm-mlo.csv`), a real-world dataset measuring monthly atmospheric CO₂ levels since 1958.

---

## 📁 Repository Structure

- `Bayesian_Dropout_Reproduction.ipynb` – Full implementation notebook
- `co2-mm-mlo.csv` – Monthly CO₂ data (cleaned)
- `README.md` – Project overview (this file)

---

## 🧠 Methods Implemented

- Dropout as approximate Bayesian inference (per Gal & Ghahramani)
- Stochastic forward passes during test time (MC Dropout)
- Variance estimation using:
  \[
  \text{Var}(y^*) = \text{Var}_{\text{MC}} + \tau^{-1}
  \]
- Loss = Mean Squared Error + L2 regularization (weight decay)
- Estimation of τ⁻¹ using the formula:
  \[
  \tau = \frac{pl^2}{2N\lambda}
  \]

---

## 📈 Results

- The model captures temporal trends and expresses predictive uncertainty
- Wider uncertainty in areas with less training data
- Metrics:
  - **Root Mean Squared Error (RMSE)**
  - **Predictive Log-Likelihood**

---

## ⚙️ Dependencies

- Python 3.8+
- PyTorch
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn

```bash
pip install -r requirements.txt
