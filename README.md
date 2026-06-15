# Progreso de los modelos de machile learning para frases diidxazá

Realización de feature engineering y clasificación binaria supervisada para validar la concordancia morfológica del pronombre (prefijo), la raíz verbal, el sufijo gramatical y la presencia del apóstrofe en la variante lingüística del Zapoteco de La Mata, Ixtaltepec.

##Progreso

Carga de datos estructurados desde vector_feature.csv.
Feature Engineering: Variables categóricas (prefijo, raiz, sufijo) mediante One-Hot Encoding mapeando las dependencias en un dataFrame numérico.
Guardado de la matriz resultante en resultados/one_hot_vector.csv.
Evaluación gramatical (label) utilizando Regresión Logística.

## Instalación

### Entorno virtual (venv)

Crear y activar el entorno virtual:

```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate
**Instalar dependencias:**
pip install -r requirements.txt
**Ejecución**
python prueba_one-hot.py
