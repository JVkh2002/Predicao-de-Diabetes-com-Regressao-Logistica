
#Este projeto consiste em construir um modelo preditivo para diagnosticar diabetes em pacientes do sexo feminino com base em dados médicos.
#Utilizando a linguagem Python e bibliotecas populares como pandas, scikit-learn, e Logistic Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# URL do conjunto de dados
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
# Carregar os dados em um DataFrame
df = pd.read_csv(url)

# Visualizar as primeiras linhas do DataFrame
print(df.head())
# Obter informações sobre o DataFrame
print(df.info())
# Estatísticas descritivas
print(df.describe())

print(df.isnull().sum())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LogisticRegression(random_state=42)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy:.2f}")

print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
