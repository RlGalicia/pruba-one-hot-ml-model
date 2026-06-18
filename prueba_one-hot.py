import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

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

    #incluyendo ahora la categoria persona.
    columnas_categoricas = ['prefijo', 'raiz', 'sufijo', 'persona']
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
    ohetransform= ohe.fit_transform(df[columnas_categoricas])

    df_final = pd.concat([df, ohetransform], axis=1).drop(columns= columnas_categoricas)
    df_final.to_csv(path_or_buf='resultados/one_hot_vector.csv', index=False, encoding='utf-8')

    return df_final

def entrenar_modeloRegresion(X_train, X_test, y_train, y_test):

    md_rl = LogisticRegression(max_iter=1000, random_state=45)
    md_rl.fit(X_train, y_train)
    y_pred = md_rl.predict(X_test)

    print('-------------------------------')
    print(f"Score: {md_rl.score(X_test, y_test)}")
    print('-------------------------------')
    print(confusion_matrix(y_test, y_pred))
    print('-------------------------------')
    print(classification_report(y_test, y_pred, target_names=['No válido', 'Válido'] ))

    plt.figure(figsize=(10, 5))
    sns.countplot(x=y_pred)
    plt.title("Modelo Regresión Logística")
    plt.xlabel("0: No válido | 1: válido")
    plt.ylabel("No. de palabras")
    plt.tight_layout()
    plt.show()

 # class_weight='balanced' -> Otra predicición considerando este parametro 
def entrenar_modeloRegresionBalanced(X_train, X_test, y_train, y_test):

    md_rl_bal = LogisticRegression(max_iter=1000, random_state=45, class_weight='balanced')
    md_rl_bal.fit(X_train, y_train)
    y_pred_bal = md_rl_bal.predict(X_test)

    print('-------------------------------')
    print(f"Score (balanced): {md_rl_bal.score(X_test, y_test):.4f}")
    print('-------------------------------')
    print('Matriz de confusión (balanced):')
    print(confusion_matrix(y_test, y_pred_bal))
    print('-------------------------------')
    print('Reporte de clasificación (balanced):')
    print(classification_report(y_test, y_pred_bal, target_names=['No válido', 'Válido']))

    plt.figure(figsize=(9, 4))
    sns.countplot(x=y_pred_bal)
    plt.title("Modelo Regresión Logística(con class_weight)")
    plt.xlabel("0: No válido | 1: válido")
    plt.ylabel("No. de palabras")
    plt.tight_layout()
    plt.show()

def entrenar_modeloArbolDesicion():
    ...

def entrenar_modeloArbolRndom():
    ...


df_preparado = preparar_datos()
X = df_preparado.drop(columns=['frase', 'label'])
y = df_preparado['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39, stratify=y)

#entrenar_modeloRegresion(X_train, X_test, y_train, y_test)
entrenar_modeloRegresionBalanced(X_train, X_test, y_train, y_test)