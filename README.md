# Prototipo de Sistema de Gestión del Conocimiento con IA — NovaTech Soluciones

Avance de proyecto: clasificación automática de documentos (TF-IDF + Naive Bayes / Regresión Logística), búsqueda por similitud coseno y ontología OWL, sobre el caso de estudio **NovaTech Soluciones S.A. de C.V.** proporcionado por el profesor.

**Integrantes:** Eduardo Alcántara González (3074833), Ángel Auster Agüero Ávila, Mauricio González González, Patricio Diddier Ramírez

## El caso: NovaTech Soluciones S.A. de C.V.

Empresa mexicana de tecnología con sede en Monterrey, Nuevo León (~180 colaboradores, esquema híbrido). Desarrolla software a la medida, opera una plataforma SaaS de facturación electrónica e inventarios para PyMEs y ofrece consultoría de TI.

**Problema:** los documentos están dispersos y los colaboradores tardan en encontrar la política o el procedimiento correcto; soporte resuelve tickets repetidos sin aprovechar la base de conocimiento y las lecciones aprendidas de los proyectos no se reutilizan.

Las cinco áreas de conocimiento son las categorías del clasificador (nombre de carpeta entre paréntesis):

| Área | Carpeta | Documentos típicos |
|---|---|---|
| Recursos Humanos | `recursos_humanos` | Políticas (vacaciones, home office), manual de inducción, evaluación del desempeño, plan de capacitación |
| Desarrollo de Software | `desarrollo_software` | Estándares de código, documentación de API, arquitectura, manual de despliegue CI/CD, estrategia de pruebas |
| Seguridad de la Información | `seguridad_informacion` | Política de contraseñas, respuesta a incidentes, respaldos, control de accesos, auditorías |
| Soporte a Clientes | `soporte_clientes` | Procedimiento de tickets, SLA, preguntas frecuentes, manuales de usuario, artículos de base de conocimiento |
| Gestión de Proyectos | `gestion_proyectos` | Metodología Scrum, actas de constitución, reportes de avance, minutas, lecciones aprendidas |

## Estructura

```
├── README.md
├── requirements.txt
├── ejecutar_en_colab.ipynb      # ejecuta todo en Google Colab
├── data/
│   ├── documentos/<categoria>/  # 50 documentos etiquetados (10 por área, .txt UTF-8)
│   ├── nuevos/                  # 6 documentos sin etiqueta para la demostración
│   └── consultas_prueba.csv     # 6 consultas de la búsqueda (consulta,categoria_esperada)
├── src/
│   ├── preprocesamiento.py      # limpieza compartida (+ stemming opcional)
│   ├── clasificador.py          # Paso 4: clasificación automática
│   ├── busqueda_semantica.py    # Paso 5: búsqueda por similitud coseno
│   └── ontologia.py             # Paso 3: ontología OWL (owlready2) + grafo
├── resultados/                  # métricas, matrices de confusión, CSV, ontología (.owl y .png)
└── docs/                        # PDF de avance y diagramas (ciclo de vida)
```

## Instalación (local)

Requiere Python 3.10 o superior.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución

Todos los comandos se ejecutan desde la raíz del proyecto:

```bash
python src/clasificador.py                # entrena, evalúa y clasifica data/nuevos/
python src/clasificador.py --stemming     # misma corrida con stemming
python src/busqueda_semantica.py --consulta "qué hago si recibo un correo sospechoso"
python src/busqueda_semantica.py --pruebas --comparar-stemming
python src/busqueda_semantica.py --interactivo
python src/ontologia.py                   # genera resultados/ontologia.owl y ontologia.png
```

En Google Colab, abre `ejecutar_en_colab.ipynb`, clona el repositorio y ejecuta las celdas.

## Cómo funciona

1. **Preprocesamiento** (`preprocesamiento.py`): minúsculas, sin acentos, números ni signos, y sin *stopwords* en español. Se usa la misma función para entrenar y para consultar. Opcionalmente reduce plurales y la última vocal (*stemming* simple).
2. **Clasificador** (`clasificador.py`): `Pipeline` con `TfidfVectorizer` (unigramas y bigramas) y Naive Bayes o Regresión Logística, con división estratificada 70/30 y validación cruzada de 5 particiones. Elige el mejor modelo, analiza los errores, clasifica los documentos de `data/nuevos/` con su confianza y guarda el modelo con `joblib`.
3. **Búsqueda** (`busqueda_semantica.py`): indexa los documentos con TF-IDF (`fit_transform`), convierte la consulta con `transform`, calcula `cosine_similarity` y muestra el ranking top-k.
4. **Ontología** (`ontologia.py`): clases Organización, Área de conocimiento, Concepto, Tipo de documento y Documento; 25 conceptos (5 por área) y relaciones con nombre entre conceptos (6 de ellas cruzan áreas) y entre documentos y conceptos (`trata_sobre`, `es_de_tipo`, `clasificado_en`).

## Resultados (corrida con los documentos incluidos)

| Modelo | Exactitud (prueba) | Validación cruzada |
|---|---|---|
| Naive Bayes | 0.800 | 0.900 ± 0.063 |
| Regresión Logística | 0.867 | 0.920 ± 0.075 |

Los resultados completos están en `resultados/`. Con pocos documentos, las métricas cambian ligeramente si se cambia la semilla o los textos.

**Limitación principal de la búsqueda:** TF-IDF compara palabras, no significados. La consulta «mi computadora se infectó con un virus» no encuentra nada porque los documentos usan «malware».

## Origen de los documentos

Los 56 documentos de `data/` son ficticios. Fueron redactados como borradores con asistencia de IA (Claude) a partir del caso de estudio de NovaTech y revisados por el equipo.
