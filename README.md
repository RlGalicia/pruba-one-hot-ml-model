# Progreso de los modelos de machile learning para frases diidxazá

Realización de feature engineering y clasificación binaria supervisada para validar la concordancia morfológica del pronombre (prefijo), la raíz verbal, el sufijo gramatical y la presencia del apóstrofe en la variante lingüística del Zapoteco de La Mata, Ixtaltepec.

## Progreso

Carga de datos estructurados desde vector_feature.csv.
Feature Engineering: Variables categóricas (prefijo, raiz, sufijo) mediante One-Hot Encoding mapeando los registros en un dataFrame numérico.
Guardado de la matriz resultante en resultados/one_hot_vector.csv.
Evaluación gramatical (label) utilizando Regresión Logística.

## Estructura del csv
| Bloque | Grupo de filas | apostrofe = 0 | apostrofe = 1 | label = 1 | label = 0 |
|--------|----------------|---------------|---------------|-----------|-----------|
correctas	| 2-349 =348	| 58	| 290 | 	348 | 	0
error_apostrofe	| 350 - 697 = 348	| 348	| 0	| 58	| 290
error_concordoncia	| 698 - 1045 = 348| 	58	| 290	| 0	| 348
Totales	| 1044	| 464	| 580	| 406	| 638
					
| Matriz	| label*	| apostrofe |
|---------|-------|-----------|
correctas	| 1	| 1			
error_apostrofe	| 0	| 0			
error_concordancia	| 0	| 1			

*Considerando el caso la 1ps 'naa'

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
