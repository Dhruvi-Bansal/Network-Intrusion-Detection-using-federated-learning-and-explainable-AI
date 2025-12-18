### Privacy-Preserving Network Intrusion Detection using Federated Learning & XAI
## Project Overview
This project implements a privacy-preserving Intrusion Detection System (IDS) using Federated Learning (FL) and Explainable AI (XAI).<br>
The system enables multiple organizations to collaboratively train an intrusion detection model without sharing raw network traffic data, addressing privacy and compliance concerns in cybersecurity.
Two heterogeneous real-world datasets—UNSW-NB15 and NF-UQ-NIDS—are used to simulate Non-IID client environments.<br><br>
## Key Features<br>
-	*Federated Learning from Scratch*:
Manual implementation of FedAvg and FedProx algorithms for collaborative model training.
-	*Privacy Quantification*:
-Privacy leakage measured using Membership Inference Attacks (MIA) with<br>
Privacy Score = 1 − Attack Success Rate (ASR).
-	*Privacy-Enhanced FedProx*:
Applied confidence clipping, Gaussian noise, Top-K filtering, and rounding, achieving significantly improved privacy with minimal accuracy loss.
-	*Explainable AI (SHAP)*:
Integrated global and local SHAP explanations to interpret intrusion predictions.
-*Interactive Dashboard*:
Built a Plotly Dash dashboard to visualize accuracy, privacy scores, confusion matrices, and SHAP results.
<br><br>
## Architecture Overview
-	*Clients*:
Perform local preprocessing and training on private network datasets.
-	*Central Server*:
Aggregates model updates using FedAvg or FedProx without accessing raw data.
-	*Privacy Module*:
Evaluates model vulnerability using Membership Inference Attacks.
-	*Explainability Module*:
Provides SHAP-based insights for model transparency.
-	*Dashboard*:
Displays training metrics, comparisons, and explanations.
## Results Summary

### Model Performance & Privacy Comparison

| Model | Training Setup | Global Accuracy (%) | Privacy Score (1 − ASR) | Key Observations |
|------|---------------|---------------------|--------------------------|------------------|
| Decision Tree (Baseline) | Centralized | 88 | 0.28 | Overconfident predictions; highly vulnerable to membership inference |
| FedAvg | Federated (No Privacy Controls) | 73 | 0.40 | Improved privacy via decentralization but weaker convergence |
| FedProx (Proposed) | Federated + Privacy Enhancements | 85 | 0.48 | Best privacy–utility trade-off under Non-IID data |

## How to Run
### 1. Clone the Repository
```bash
git clone https://github.com/Dhruvi-Bansal/Network-Intrusion-Detection-using-federated-learning-and-explainable-AI.git
cd Network-Intrusion-Detection-using-federated-learning-and-explainable-AI
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run Baseline Decision Tree
### 4. Run fedavg, fedprox models
### 5. Run MIA jupyter notebook for evaluating Privacy Score
### 6. Launch Interactive Dashboard using command prompt
```bash
streamlit run app.py
```

## Technologies Used
-	Python
-	TensorFlow / Keras
-	Scikit-learn
-	Federated Learning (FedAvg, FedProx)
-	SHAP (Explainable AI)
-	Plotly Dash<br>

## Future Enhancements
- Support for multiple federated clients
- Integration of secure aggregation techniques
- Real-time IDS deployment
- Evaluation against additional privacy attacks



