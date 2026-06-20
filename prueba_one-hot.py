import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import numpy as np

df = pd.read_csv('vector_feature.csv')

def preparar_datos():
    print(df.head())
    print("----------------------------------------------------")
    print(f"Prefijos identficaos:  {df['prefijo'].unique()}")
    print("----------------------------------------------------")
    print(f"Verbos raíz: {df['raiz'].unique()}")
    print("----------------------------------------------------")
    print(f"Componente sufijo identificados: {df['sufijo'].unique()}")
    print("----------------------------------------------------")
    print(f"Personas verbales identificadas: {df['persona'].unique()}")
    print("----------------------------------------------------")
    print(f"Grupos de frases: {df['error'].unique()}")
    print("----------------------------------------------------")

    #incluyendo ahora la categoria persona.
    columnas_categoricas = ['prefijo', 'raiz', 'sufijo', 'persona']
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
    ohetransform= ohe.fit_transform(df[columnas_categoricas])

    df_final = pd.concat([df, ohetransform], axis=1).drop(columns= columnas_categoricas)
    df_final.to_csv(path_or_buf='resultados/one_hot_vector.csv', index=False, encoding='utf-8')

    return df_final

def entrenar_modeloRegresionBalanced(X_train, X_test, y_train, y_test):

    md_rl_bal = LogisticRegression(max_iter=1000, random_state=45, class_weight='balanced')
    md_rl_bal.fit(X_train, y_train)
    y_pred_bal = md_rl_bal.predict(X_test)
    score = md_rl_bal.score(X_test,y_test)

    print('-------------------------------')
    print(f"Score (balanced): {score:.4f}")
    print('-------------------------------')
    print('Matriz de confusión (balanced):')
    print(confusion_matrix(y_test, y_pred_bal))
    print('-------------------------------')
    print('Reporte de clasificación (balanced):')
    print(classification_report(y_test, y_pred_bal, target_names=['No válido', 'Válido']))
    graficar_countplot(y_pred_bal, "Regresión Logística")
    graficar_matriz_confusion(y_test, y_pred_bal, score, "Regresión Logística")
    generar_reporte_predicciones(df_preparado, X_test, y_test, y_pred_bal, "Regresion Logistica")

    return md_rl_bal

def entrenar_modeloArbolDesicion(X_train, X_test, y_train, y_test):
    
    md_ad = DecisionTreeClassifier(random_state=45, class_weight='balanced')
    md_ad.fit(X_train, y_train)
    y_pred= md_ad.predict(X_test)
    score = md_ad.score(X_test, y_test)

    print('-------------------------------')
    print(f"Score árbol de decisión: {score:.4f}")
    print('-------------------------------')
    print('Matriz de confusión:')
    print(confusion_matrix(y_test, y_pred))
    print('-------------------------------')
    print('Reporte de clasificación:')
    print(classification_report(y_test, y_pred, target_names=['No válido', 'Válido']))
    graficar_countplot(y_pred, "Árbol de Decisión")
    plt.figure(figsize=(20, 10))
    plot_tree(md_ad, feature_names=X_train.columns.tolist(), class_names=['No Válido', 'Válido'], filled=True, rounded=True)
    plt.title("Estructura del Árbol de Decisión")
    plt.tight_layout()
    plt.show()

    #Gráfica con los primeros tres niveles
    plt.figure(figsize=(20, 10))
    plot_tree(md_ad, max_depth=3, feature_names=X_train.columns.tolist(), class_names=['No Válido', 'Válido'], filled=True, rounded=True)
    plt.title("Estructura del Árbol de Decisión (3 niveles)")
    plt.tight_layout()
    plt.show()

    graficar_matriz_confusion(y_test, y_pred, score, "Árbol de Decisión")
    generar_reporte_predicciones(df_preparado, X_test, y_test, y_pred, "Arbol de Decisión")

    return md_ad

def entrenar_modeloArbolRndom(X_train, X_test, y_train, y_test):
    md_rf = RandomForestClassifier(n_estimators=100, random_state=45, class_weight='balanced')
    md_rf.fit(X_train, y_train)
    y_pred = md_rf.predict(X_test)
    score =md_rf.score(X_test, y_test)

    print('-------------------------------')
    print(f"Score Árboles aleatorios: {score:.4f}")
    print('-------------------------------')
    print('Matriz de confusión:')
    print(confusion_matrix(y_test, y_pred))
    print('-------------------------------')
    print('Reporte de clasificación:')
    print(classification_report(y_test, y_pred, target_names=['No válido', 'Válido']))
    graficar_countplot(y_pred, "Árboles aleatorios")
    importancias = md_rf.feature_importances_
    indices = np.argsort(importancias)[-20:]
    nombres_features = X_train.columns

    plt.figure(figsize=(10,6))
    plt.title("Carácteristicas importantes en el clasificador Árboles aleatorios")
    plt.barh(range(len(indices)), importancias[indices], color='teal', align='center')
    plt.yticks(range(len(indices)), [nombres_features[i] for i in indices], fontsize=9)
    plt.xlabel("Importancia de la característica")
    plt.tight_layout()
    plt.show()

    graficar_matriz_confusion(y_test, y_pred, score, "Árboles aleatorios")
    generar_reporte_predicciones(df_preparado, X_test, y_test, y_pred, "Árboles aleatorios")

    return md_rf

def graficar_matriz_confusion(y_real, y_pred, score, nombre_modelo):
    cm = confusion_matrix(y_real, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['No Válido', 'Válido'], yticklabels=['No Válido', 'Válido'])
    plt.title(f"Matriz de Confusión - {nombre_modelo}\nAccuracy (Score): {score:.4f}", fontsize=12, pad=15)
    plt.xlabel("Predicción del Modelo", fontsize=11)
    plt.ylabel("Clase Real", fontsize=11)
    plt.tight_layout()
    plt.show()

def graficar_countplot(y_predic, nombre_modelo):
    plt.figure(figsize=(10, 5))
    sns.countplot(x=y_predic) 
    plt.title(f"Distribución de Predicciones - {nombre_modelo}", fontsize=12, pad=12)
    plt.xlabel("0: No Válido | 1: Válido", fontsize=11)
    plt.ylabel("Número de Palabras", fontsize=11)
    plt.tight_layout()
    plt.show()

#Funcion para obtener los reportes de cuales frases esta asignando a train y cuales a test
def reporte_asignacion_datos(df_original, X_train, X_test, y_train, y_test):
    reporte_train = pd.DataFrame({'Indice': X_train.index, 'Frase': df_original.loc[X_train.index, 'frase'], 'Valor label': y_train
    })
    reporte_test = pd.DataFrame({'Indice': X_test.index, 'Frase': df_original.loc[X_test.index, 'frase'], 'Valor label': y_test
    })
    reporte_train.to_csv('resultados/registro_entrenamiento_train.csv', index=False, encoding='utf-8')
    reporte_test.to_csv('resultados/registro_pruebas_test.csv', index=False, encoding='utf-8')

#Funcion para obtener el dataframe de cuales fueron las predicciones por modelo
def generar_reporte_predicciones(df_original, X_test, y_test, y_pred, nombre_modelo):
    reporte = pd.DataFrame({
        'Frase': df_original.loc[X_test.index, 'frase'],
        'Label Real': y_test.values,
        'Prediccion': y_pred,
        'Correcto': (y_test.values == y_pred)
    })
    ruta = f'resultados/predicciones_{nombre_modelo.replace(" ", "_")}.csv'
    reporte.to_csv(ruta, index=False, encoding='utf-8')

df_preparado = preparar_datos()
X = df_preparado.drop(columns=['frase', 'label', 'error'])  #dropee apostrofe y errror
y = df_preparado['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39, stratify=y)
print("\n--- Registros train (80%) ---")
print(y_train.value_counts())
print("\n--- Registros test (20%) ---")
print(y_test.value_counts())

reporte_asignacion_datos(df_preparado, X_train, X_test, y_train, y_test)

entrenar_modeloRegresionBalanced(X_train, X_test, y_train, y_test)
entrenar_modeloArbolDesicion(X_train, X_test, y_train, y_test)
entrenar_modeloArbolRndom(X_train, X_test, y_train, y_test)