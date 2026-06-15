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

    columnas_categoricas = ['prefijo', 'raiz', 'sufijo']
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

def entrenar_modeloArbolDesicion():
    ...

def entrenar_modeloArbolRndom():
    ...


df_preparado = preparar_datos()
X = df_preparado.drop(columns=['frase', 'label'])
y = df_preparado['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39, stratify=y)

entrenar_modeloRegresion(X_train, X_test, y_train, y_test)