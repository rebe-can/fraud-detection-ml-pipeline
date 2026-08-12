# 🛡️ Detecção de Fraudes em Cartão de Crédito com Machine Learning

Este repositório contém uma solução completa e robusta de Machine Learning para identificar transações fraudulentas em dados de cartão de crédito sob condições de desbalanceamento extremo.

## 📌 Destaques do Projeto
* **Tratamento de Dados Desbalanceados:** Modelagem focada em métricas adequadas para cenários onde fraudes representam menos de 0.2% das transações (foco em PR-AUC/Average Precision em vez de Acurácia).
* **Engenharia de Features:** Aplicação de transformação logarítmica (`log1p`) nos valores das transações para suavizar a assimetria da distribuição e remoção de variáveis irrelevantes.
* **Modelagem e Hiperparametrização:** Validação cruzada estratificada (`StratifiedKFold`) e otimização via `GridSearchCV` avaliando **Regressão Logística**, **Random Forest** e **XGBoost**.
* **Ajuste Fino de Threshold:** Calibração do limiar de decisão no conjunto de validação para maximizar o F1-Score antes da avaliação final no conjunto de teste.
* **Explicabilidade (XAI):** Utilização da biblioteca **SHAP** (*SHapley Additive exPlanations*) para interpretar os atributos que mais impactam nas decisões do modelo campeão.

## 🛠️ Tecnologias Utilizadas
* **Linguagem:** Python
* **Manipulação e Análise:** Pandas, NumPy
* **Visualização:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, XGBoost
* **Explicabilidade:** SHAP

## 🚀 Como Executar
1. Clone o repositório:
   ```bash
   git clone [https://github.com/rebe-can/fraud-detection-ml-pipeline.git](https://github.com/rebe-can/fraud-detection-ml-pipeline.git)
