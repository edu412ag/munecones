"""Paso 4: clasificador automático de documentos con TF-IDF.

Flujo: cargar -> limpiar -> dividir -> TF-IDF + modelo (Pipeline) -> evaluar
-> analizar errores -> clasificar documentos nuevos -> guardar modelo.

Se comparan dos modelos: Naive Bayes y Regresión Logística.

Uso:
    python src/clasificador.py
    python src/clasificador.py --stemming
"""
import argparse
from functools import partial

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from preprocesamiento import (
    RUTA_RESULTADOS,
    cargar_documentos,
    cargar_nuevos,
    limpiar_texto,
)


SLUGS = {"Naive Bayes": "naive_bayes", "Regresión Logística": "regresion_logistica"}


def construir_pipeline(nombre: str, stemming: bool) -> Pipeline:
    """TF-IDF (con la misma limpieza del proyecto) + modelo."""
    vectorizador = TfidfVectorizer(
        preprocessor=partial(limpiar_texto, usar_stemming=stemming),
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    if nombre == "Naive Bayes":
        modelo = MultinomialNB()
    else:
        modelo = LogisticRegression(max_iter=1000)
    return Pipeline([("tfidf", vectorizador), ("modelo", modelo)])


def terminos_principales(pipeline: Pipeline, texto: str, n: int = 6) -> str:
    """Los n términos con mayor peso TF-IDF en el documento."""
    vec = pipeline.named_steps["tfidf"]
    fila = vec.transform([texto]).toarray()[0]
    nombres = vec.get_feature_names_out()
    idx = np.argsort(fila)[::-1][:n]
    return ", ".join(nombres[i] for i in idx if fila[i] > 0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Clasificador TF-IDF de documentos")
    ap.add_argument("--stemming", action="store_true", help="reduce palabras a su raíz")
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sufijo = "_stemming" if args.stemming else ""
    RUTA_RESULTADOS.mkdir(exist_ok=True)

    # 1. Cargar
    df = cargar_documentos()
    conteo = df["categoria"].value_counts().sort_index()
    print("=== Documentos por categoría ===")
    print(conteo.to_string(), f"\nTotal: {len(df)}\n")
    if conteo.min() < 5:
        raise SystemExit("Cada categoría necesita al menos 5 documentos (ideal 10).")

    X, y = df["texto"], df["categoria"]
    categorias = sorted(y.unique())

    # 2. Dividir (estratificado)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    print(f"Entrenamiento: {len(X_tr)} | Prueba: {len(X_te)} | stemming: {args.stemming}\n")

    n_folds = int(min(5, conteo.min()))
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)

    # 3-4. Entrenar y evaluar los dos modelos
    resumen = []
    for nombre in ["Naive Bayes", "Regresión Logística"]:
        pipe = construir_pipeline(nombre, args.stemming)
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        exactitud = accuracy_score(y_te, y_pred)
        cv_scores = cross_val_score(
            construir_pipeline(nombre, args.stemming), X, y, cv=cv, scoring="accuracy"
        )

        print(f"=== {nombre} ===")
        print(f"Exactitud (prueba): {exactitud:.3f}")
        print(f"Validación cruzada ({n_folds} particiones): "
              f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(classification_report(y_te, y_pred, zero_division=0))

        cm = confusion_matrix(y_te, y_pred, labels=categorias)
        pd.DataFrame(cm, index=categorias, columns=categorias).to_csv(
            RUTA_RESULTADOS / f"matriz_confusion_{SLUGS[nombre]}{sufijo}.csv"
        )
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay(cm, display_labels=categorias).plot(
            ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False
        )
        ax.set_title(f"Matriz de confusión - {nombre}{' (stemming)' if args.stemming else ''}")
        fig.tight_layout()
        fig.savefig(
            RUTA_RESULTADOS / f"matriz_confusion_{SLUGS[nombre]}{sufijo}.png", dpi=150
        )
        plt.close(fig)

        reporte = pd.DataFrame(
            classification_report(y_te, y_pred, output_dict=True, zero_division=0)
        ).T
        reporte.to_csv(RUTA_RESULTADOS / f"reporte_{SLUGS[nombre]}{sufijo}.csv")

        resumen.append(
            {
                "modelo": nombre,
                "exactitud_prueba": round(exactitud, 3),
                "cv_media": round(cv_scores.mean(), 3),
                "cv_desv": round(cv_scores.std(), 3),
            }
        )

    tabla = pd.DataFrame(resumen)
    tabla.to_csv(RUTA_RESULTADOS / f"comparacion_modelos{sufijo}.csv", index=False)
    print("=== Comparación de modelos ===")
    print(tabla.to_string(index=False), "\n")

    # Mejor modelo: mayor validación cruzada, desempata la exactitud de prueba
    mejor = tabla.sort_values(["cv_media", "exactitud_prueba"], ascending=False).iloc[0]["modelo"]
    print(f"Modelo elegido: {mejor}\n")

    # 5. Análisis de errores con validación cruzada (usa todos los documentos,
    #    así hay más oportunidades de ver errores que con un solo split pequeño)
    pipe_cv = construir_pipeline(mejor, args.stemming)
    pred_cv = cross_val_predict(pipe_cv, X, y, cv=cv)
    pipe_exp = construir_pipeline(mejor, args.stemming).fit(X, y)
    errores = []
    for i in np.where(pred_cv != y.values)[0]:
        errores.append(
            {
                "id": df.loc[i, "id"],
                "titulo": df.loc[i, "titulo"],
                "categoria_real": y.iloc[i],
                "categoria_predicha": pred_cv[i],
                "terminos_principales": terminos_principales(pipe_exp, df.loc[i, "texto"]),
            }
        )
    errores_df = pd.DataFrame(
        errores,
        columns=["id", "titulo", "categoria_real", "categoria_predicha", "terminos_principales"],
    )
    errores_df.to_csv(RUTA_RESULTADOS / f"errores_clasificacion{sufijo}.csv", index=False)
    print(f"=== Errores en validación cruzada: {len(errores_df)} de {len(df)} ===")
    if len(errores_df):
        print(errores_df.to_string(index=False))
        print("\nPara el reporte: revisa qué términos comparten estos documentos con la "
              "categoría predicha (suele ser vocabulario común entre dos áreas).\n")
    else:
        print("Sin errores. Revisa que los documentos no sean casi idénticos entre sí.\n")

    # 6. Clasificar documentos nuevos con el modelo entrenado con todos los datos
    nuevos = cargar_nuevos()
    if len(nuevos):
        proba = pipe_exp.predict_proba(nuevos["texto"])
        clases = pipe_exp.classes_
        nuevos["categoria_predicha"] = clases[proba.argmax(axis=1)]
        nuevos["confianza"] = proba.max(axis=1).round(3)
        nuevos["terminos_principales"] = [
            terminos_principales(pipe_exp, t) for t in nuevos["texto"]
        ]
        salida = nuevos[["id", "titulo", "categoria_predicha", "confianza", "terminos_principales"]]
        salida.to_csv(RUTA_RESULTADOS / f"clasificacion_nuevos{sufijo}.csv", index=False)
        print("=== Clasificación de documentos nuevos ===")
        print(salida.to_string(index=False), "\n")
    else:
        print("No hay documentos en data/nuevos/.\n")

    # 7. Guardar el modelo
    joblib.dump(pipe_exp, RUTA_RESULTADOS / f"modelo_clasificador{sufijo}.joblib")
    print(f"Modelo guardado en resultados/modelo_clasificador{sufijo}.joblib")


if __name__ == "__main__":
    main()
