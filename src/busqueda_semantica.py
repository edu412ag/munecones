"""Paso 5: búsqueda por similitud coseno sobre vectores TF-IDF.

Flujo: indexar (fit_transform) -> consulta -> limpiar igual -> transform
-> cosine_similarity -> ranking top-k.

Uso:
    python src/busqueda_semantica.py --consulta "como reporto un correo sospechoso"
    python src/busqueda_semantica.py --interactivo
    python src/busqueda_semantica.py --pruebas                  # consultas de prueba
    python src/busqueda_semantica.py --pruebas --comparar-stemming
"""
import argparse
from functools import partial

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocesamiento import (
    RAIZ,
    RUTA_RESULTADOS,
    cargar_documentos,
    cargar_nuevos,
    limpiar_texto,
)

RUTA_CONSULTAS = RAIZ / "data" / "consultas_prueba.csv"


def construir_indice(df: pd.DataFrame, stemming: bool):
    """Ajusta TF-IDF sobre todos los documentos: matriz documentos x términos."""
    vectorizador = TfidfVectorizer(
        preprocessor=partial(limpiar_texto, usar_stemming=stemming),
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matriz = vectorizador.fit_transform(df["texto"])
    return vectorizador, matriz


def buscar(consulta: str, vectorizador, matriz, df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Devuelve los k documentos más parecidos a la consulta."""
    vector_consulta = vectorizador.transform([consulta])  # transform, NO fit_transform
    similitudes = cosine_similarity(vector_consulta, matriz)[0]
    orden = np.argsort(similitudes)[::-1][:k]
    return pd.DataFrame(
        {
            "posicion": range(1, len(orden) + 1),
            "id": df["id"].values[orden],
            "titulo": df["titulo"].values[orden],
            "categoria": df["categoria"].values[orden],
            "similitud": similitudes[orden].round(4),
        }
    )


def cargar_indice_df(incluir_nuevos: bool) -> pd.DataFrame:
    df = cargar_documentos()
    if incluir_nuevos:
        nuevos = cargar_nuevos()
        nuevos["categoria"] = "sin_etiqueta"
        df = pd.concat([df, nuevos], ignore_index=True)
    return df


def correr_pruebas(df, stemming: bool, k: int) -> pd.DataFrame:
    """Ejecuta data/consultas_prueba.csv (columnas: consulta, categoria_esperada)."""
    if not RUTA_CONSULTAS.exists():
        raise SystemExit(f"Falta {RUTA_CONSULTAS} con columnas: consulta,categoria_esperada")
    consultas = pd.read_csv(RUTA_CONSULTAS, encoding="utf-8")
    vec, matriz = construir_indice(df, stemming)
    filas = []
    for _, c in consultas.iterrows():
        res = buscar(c["consulta"], vec, matriz, df, k)
        top = res.iloc[0]
        if top["similitud"] == 0:  # la consulta no comparte ninguna palabra con los documentos
            filas.append(
                {
                    "consulta": c["consulta"],
                    "categoria_esperada": c["categoria_esperada"],
                    "top1_id": "(sin coincidencias)",
                    "top1_titulo": "",
                    "top1_categoria": "",
                    "similitud": 0.0,
                    "acierto": False,
                }
            )
            continue
        filas.append(
            {
                "consulta": c["consulta"],
                "categoria_esperada": c["categoria_esperada"],
                "top1_id": top["id"],
                "top1_titulo": top["titulo"],
                "top1_categoria": top["categoria"],
                "similitud": top["similitud"],
                "acierto": top["categoria"] == c["categoria_esperada"],
            }
        )
    return pd.DataFrame(filas)


def main() -> None:
    ap = argparse.ArgumentParser(description="Búsqueda por similitud coseno (TF-IDF)")
    ap.add_argument("--consulta", help="pregunta en lenguaje natural")
    ap.add_argument("--k", type=int, default=5, help="cuántos resultados mostrar")
    ap.add_argument("--stemming", action="store_true")
    ap.add_argument("--interactivo", action="store_true")
    ap.add_argument("--pruebas", action="store_true", help="corre data/consultas_prueba.csv")
    ap.add_argument("--comparar-stemming", action="store_true",
                    help="con --pruebas: compara sin y con stemming")
    ap.add_argument("--incluir-nuevos", action="store_true",
                    help="indexa también data/nuevos/")
    args = ap.parse_args()

    df = cargar_indice_df(args.incluir_nuevos)
    RUTA_RESULTADOS.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)

    if args.pruebas:
        modos = [False, True] if args.comparar_stemming else [args.stemming]
        resultados = {}
        for stem in modos:
            res = correr_pruebas(df, stem, args.k)
            resultados[stem] = res
            etiqueta = "con stemming" if stem else "sin stemming"
            print(f"=== Consultas de prueba ({etiqueta}) ===")
            print(res.to_string(index=False))
            print(f"Aciertos en el primer resultado: {res['acierto'].sum()}/{len(res)}\n")
            res.to_csv(
                RUTA_RESULTADOS / f"busqueda_pruebas{'_stemming' if stem else ''}.csv",
                index=False,
            )
        if args.comparar_stemming:
            comp = resultados[False][["consulta", "similitud", "acierto"]].merge(
                resultados[True][["consulta", "similitud", "acierto"]],
                on="consulta",
                suffixes=("_sin", "_con"),
            )
            comp.to_csv(RUTA_RESULTADOS / "busqueda_comparacion_stemming.csv", index=False)
            print("=== Comparación sin vs. con stemming ===")
            print(comp.to_string(index=False))
        return

    vec, matriz = construir_indice(df, args.stemming)

    def mostrar(consulta: str) -> None:
        res = buscar(consulta, vec, matriz, df, args.k)
        print(f"\nConsulta: {consulta}")
        if res["similitud"].max() == 0:
            print("Sin coincidencias: la consulta no comparte palabras con los documentos.")
        print(res.to_string(index=False))

    if args.consulta:
        mostrar(args.consulta)
        res = buscar(args.consulta, vec, matriz, df, args.k)
        res.to_csv(RUTA_RESULTADOS / "busqueda_ultima_consulta.csv", index=False)
    elif args.interactivo:
        print("Escribe una consulta (vacío para salir).")
        while True:
            consulta = input("\n> ").strip()
            if not consulta:
                break
            mostrar(consulta)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
