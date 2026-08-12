import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    ParameterGrid,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier


# >>> Dataset :::

URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
)

print("\n[+] Baixando dataset de fraudes...")
df_raw = pd.read_csv(URL)

print(f"\n[+] Primeiros registros:\n{df_raw.head()}")
print(f"\n[+] Estrutura inicial: {df_raw.shape}")
df_raw.info()

print("\n[+] Checagem de nulos:")
print(df_raw.isnull().sum().sum(), "valores nulos encontrados.")

# Tratamento de duplicatas
n_duplicates = df_raw.duplicated().sum()
print(f"\n[!] Registros duplicados identificados: {n_duplicates}")

if n_duplicates > 0:
    df_clean = df_raw.drop_duplicates().reset_index(drop=True)
    print(f"[+] Dimensão pós-limpeza: {df_clean.shape}")
else:
    df_clean = df_raw.copy()

print("\n[+] Balanço da variável-alvo ('Class'):")
print(df_clean["Class"].value_counts())
print(df_clean["Class"].value_counts(normalize=True).map("{:.4%}".format))

plt.figure(figsize=(6, 4))
sns.countplot(data=df_clean, x="Class", palette="viridis")
plt.title("Proporção: Transações Legítimas vs Fraudes")
plt.xticks([0, 1], ["Normal (0)", "Fraude (1)"])
plt.tight_layout()
plt.show()

fraud_ratio = df_clean["Class"].mean()
print(f"\n[i] Fraudes representam ~{fraud_ratio:.4%} da base total.")
print(
    "[i] Devido ao desbalanceamento extremo, usaremos o PR-AUC (Average Precision)"
    " como métrica principal, descartando Acurácia simples."
)



# >>> Feature Engineering

df_clean["Amount_log"] = np.log1p(df_clean["Amount"])
df_clean = df_clean.drop(columns=["Amount", "Time"])

print("\n[+] Feature 'Amount' transformada via log1p. 'Time' removido com sucesso.")




# >>> Divisão dos Dados (Train / Validation / Test)

X_data = df_clean.drop(columns=["Class"])
y_data = df_clean["Class"]

# Split 60/20/20 estratificado
X_train, X_temp, y_train, y_temp = train_test_split(
    X_data, y_data, test_size=0.40, stratify=y_data, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print(f"\n[+] Particionamento executado com sucesso:")
print(f"    - Treino: {X_train.shape} | Fraudes: {y_train.sum()}")
print(f"    - Validação: {X_val.shape}  | Fraudes: {y_val.sum()}")
print(f"    - Teste: {X_test.shape}     | Fraudes: {y_test.sum()}")


#>>> Definição das Pipelines e Espaços de Busca

cv_strat = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Fator para desbalanceamento no XGBoost
n_negatives = (y_train == 0).sum()
n_positives = (y_train == 1).sum()

if n_positives == 0:
    raise ValueError("Nenhuma fraude encontrada na partição de treino.")

pos_scale_factor = n_negatives / n_positives
print(f"\n[+] Scale_pos_weight estimado (Treino): {pos_scale_factor:.2f}")


# Modelos definidos::
pipe_logreg = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        class_weight="balanced", 
        max_iter=1000, 
        random_state=42
    ))
])

rf_base = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1)

xgb_base = XGBClassifier(
    random_state=42,
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=1
)


# Parâmetros definidos::
params_logreg = [
    {"model__C": [0.1, 1], "model__solver": ["lbfgs"]}
]

params_rf = {
    "n_estimators": [100],
    "max_depth": [15],
}

params_xgb = {
    "n_estimators": [100],
    "max_depth": [3],
    "learning_rate": [0.1],
    "scale_pos_weight": [pos_scale_factor]
}


# >>> Otimização dos Modelos via GridSearchCV
candidates = {
    "Logistic Regression": (pipe_logreg, params_logreg),
    "Random Forest": (rf_base, params_rf),
    "XGBoost": (xgb_base, params_xgb)
}

fitted_grids = {}

for name, (estimator, param_grid) in candidates.items():
    print(f"\n[>] Executando busca em grade para: {name}...")
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="average_precision",
        cv=cv_strat,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    fitted_grids[name] = grid
    print(f"    Best PR-AUC (CV): {grid.best_score_:.4f}")
    print(f"    Melhores Parâmetros: {grid.best_params_}")

# >>> Avaliação e Comparação no Conjunto de Validação

val_metrics = []
val_predictions = {}

for name, grid in fitted_grids.items():
    best_m = grid.best_estimator_
    y_probs = best_m.predict_proba(X_val)[:, 1]
    y_preds_default = (y_probs >= 0.5).astype(int)
    
    val_predictions[name] = {
        "model": best_m,
        "probs": y_probs,
        "params": grid.best_params_
    }
    
    val_metrics.append({
        "Modelo": name,
        "PR-AUC": average_precision_score(y_val, y_probs),
        "Precision@0.5": precision_score(y_val, y_preds_default, zero_division=0),
        "Recall@0.5": recall_score(y_val, y_preds_default, zero_division=0),
        "F1-score@0.5": f1_score(y_val, y_preds_default, zero_division=0)
    })

df_comparison = pd.DataFrame(val_metrics).sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)
print("\n[+] Comparativo no Conjunto de Validação (Threshold = 0.5):")
print(df_comparison.to_string(index=False))

# Plot comparativo
df_melted = df_comparison.melt(id_vars="Modelo", var_name="Métrica", value_name="Valor")
plt.figure(figsize=(10, 5))
sns.barplot(data=df_melted, x="Modelo", y="Valor", hue="Métrica", palette="magma")
plt.title("Performance na Validação por Modelo")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# Seleção do Campeão
champion_name = df_comparison.iloc[0]["Modelo"]
champion_info = val_predictions[champion_name]

print(f"\n[*] Modelo Campeão Escolhido: {champion_name}")
print(f"[*] PR-AUC na Validação: {champion_info['model']}")



# >>> Curvas Precision-Recall e Ajuste Fino de Threshold

plt.figure(figsize=(8, 5))
for name, info in val_predictions.items():
    p_curve, r_curve, _ = precision_recall_curve(y_val, info["probs"])
    score = average_precision_score(y_val, info["probs"])
    plt.plot(r_curve, p_curve, label=f"{name} (PR-AUC = {score:.4f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curvas Precision-Recall (Validação)")
plt.legend()
plt.tight_layout()
plt.show()

# Tuning do Limite de Decisão (Threshold) no Campeão
p_vals, r_vals, thresh_vals = precision_recall_curve(y_val, champion_info["probs"])

p_arr, r_arr = p_vals[:-1], r_vals[:-1]
f1_arr = 2 * (p_arr * r_arr) / (p_arr + r_arr + 1e-10)

if len(thresh_vals) > 0:
    max_f1 = np.max(f1_arr)
    tied_f1_idx = np.where(np.isclose(f1_arr, max_f1))[0]
    
    # 1: Maior Precision
    max_p_in_ties = np.max(p_arr[tied_f1_idx])
    tied_p_idx = tied_f1_idx[np.isclose(p_arr[tied_f1_idx], max_p_in_ties)]
    
    # 2: Maior Threshold
    opt_idx = tied_p_idx[np.argmax(thresh_vals[tied_p_idx])]
    
    opt_threshold = thresh_vals[opt_idx]
    best_val_p = p_arr[opt_idx]
    best_val_r = r_arr[opt_idx]
    best_val_f1 = f1_arr[opt_idx]
else:
    opt_threshold = 0.5
    y_p_temp = (champion_info["probs"] >= opt_threshold).astype(int)
    best_val_p = precision_score(y_val, y_p_temp, zero_division=0)
    best_val_r = recall_score(y_val, y_p_temp, zero_division=0)
    best_val_f1 = f1_score(y_val, y_p_temp, zero_division=0)

print(f"\n[+] Limiar Operacional Otimizado via Validação: {opt_threshold:.4f}")
print(f"    - Precision (Fraude): {best_val_p:.4f}")
print(f"    - Recall (Fraude):    {best_val_r:.4f}")
print(f"    - F1-Score (Fraude):  {best_val_f1:.4f}")

if len(thresh_vals) > 0:
    plt.figure(figsize=(9, 5))
    plt.plot(thresh_vals, p_arr, label="Precision", color="teal")
    plt.plot(thresh_vals, r_arr, label="Recall", color="crimson")
    plt.plot(thresh_vals, f1_arr, label="F1-Score", color="purple")
    plt.axvline(x=opt_threshold, linestyle="--", color="black", label=f"Opt Thresh = {opt_threshold:.4f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Otimização do Threshold de Decisão")
    plt.legend()
    plt.tight_layout()
    plt.show()

# >>> Hold-out Test

print("\n" + "="*60)
print("     AVALIAÇÃO FINAL INDEPENDENTE (CONJUNTO DE TESTE)")
print("="*60)

champion_model = champion_info["model"]
y_test_probs = champion_model.predict_proba(X_test)[:, 1]
y_test_preds = (y_test_probs >= opt_threshold).astype(int)

test_pr_auc = average_precision_score(y_test, y_test_probs)
test_precision = precision_score(y_test, y_test_preds, zero_division=0)
test_recall = recall_score(y_test, y_test_preds, zero_division=0)
test_f1 = f1_score(y_test, y_test_preds, zero_division=0)

print(f"\n[RESULTS] Modelo Evaluated: {champion_name}")
print(f"  - Limiar Utilizado: {opt_threshold:.4f}")
print(f"  - PR-AUC:           {test_pr_auc:.4f}")
print(f"  - Precision:        {test_precision:.4f}")
print(f"  - Recall:           {test_recall:.4f}")
print(f"  - F1-Score:         {test_f1:.4f}")

print("\nRelatório de Classificação Completo:")
print(classification_report(y_test, y_test_preds, target_names=["Normal", "Fraude"], digits=4, zero_division=0))

conf_matrix = confusion_matrix(y_test, y_test_preds)
tn, fp, fn, tp = conf_matrix.ravel()

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens", xticklabels=["Normal", "Fraude"], yticklabels=["Normal", "Fraude"])
plt.title("Matriz de Confusão Final (Teste)")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.tight_layout()
plt.show()


# >>> Explicabilidade dos Resultados via SHAP
print("\n[+] Gerando interpretabilidade de modelo com SHAP...")

try:
    sample_size = min(100, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)

    if champion_name == "Logistic Regression":
        scaler_step = champion_model.named_steps["scaler"]
        core_model = champion_model.named_steps["model"]
        
        X_sample_norm = pd.DataFrame(scaler_step.transform(X_sample), columns=X_sample.columns, index=X_sample.index)
        bg_sample = pd.DataFrame(
            scaler_step.transform(X_train.sample(n=min(500, len(X_train)), random_state=42)),
            columns=X_train.columns
        )
        explainer = shap.LinearExplainer(core_model, bg_sample)
        shap_values_raw = explainer(X_sample_norm)
        X_plot = X_sample_norm
    else:
        explainer = shap.TreeExplainer(champion_model)
        shap_values_raw = explainer(X_sample)
        X_plot = X_sample

    # Processamento universal da estrutura do SHAP
    if isinstance(shap_values_raw, shap.Explanation):
        vals = shap_values_raw.values
        if vals.ndim == 3:
            shap_pos = vals[:, :, 1] if vals.shape[2] > 1 else vals[:, :, 0]
        else:
            shap_pos = vals
    elif isinstance(shap_values_raw, list):
        shap_pos = np.asarray(shap_values_raw[1] if len(shap_values_raw) > 1 else shap_values_raw[0])
    else:
        shap_pos = np.asarray(shap_values_raw)
        if shap_pos.ndim == 3:
            shap_pos = shap_pos[:, :, 1] if shap_pos.shape[2] > 1 else shap_pos[:, :, 0]

    # Ranking das mais importantes
    shap_importance = pd.DataFrame({
        "Feature": X_plot.columns,
        "Mean_Abs_SHAP": np.abs(shap_pos).mean(axis=0)
    }).sort_values(by="Mean_Abs_SHAP", ascending=False)

    print("\nTop 10 Atributos de Maior Impacto na Decisão:")
    print(shap_importance.head(10).to_string(index=False))

    shap.summary_plot(shap_pos, X_plot, plot_type="bar", show=False)
    plt.tight_layout()
    plt.show()

    shap.summary_plot(shap_pos, X_plot, show=False)
    plt.tight_layout()
    plt.show()

except Exception as err:
    print(f"\n[!] Falha ao calcular SHAP values: {err}")



# >>> Resumo Executivo // Métricas Finais
print("\n" + "="*60)
print("                 RESUMO EXECUTIVO DO PROJETO")
print("="*60)
print(f"• Modelo Selecionado: {champion_name}")
print(f"• Hiperparâmetros Otimizados: {champion_info['params']}")
print(f"• Cut-off Optimizou F1 na Validação: {opt_threshold:.4f}")
print("• Desempenho no Teste (Inédito):")
print(f"    - PR-AUC: {test_pr_auc:.4f}")
print(f"    - F1-Score: {test_f1:.4f}")
print(f"    - Fraudes Identificadas (TP): {tp} de {tp + fn} ({test_recall:.2%})")
print(f"    - Falsos Alarmes (FP): {fp}")
print("="*60)
