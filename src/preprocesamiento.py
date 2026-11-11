"""Preprocesamiento compartido por el clasificador y la búsqueda semántica.

Ambos flujos usan EXACTAMENTE la misma función de limpieza (`limpiar_texto`),
de modo que la consulta y los documentos quedan en el mismo vocabulario.
"""
import re
import unicodedata
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RUTA_DOCUMENTOS = RAIZ / "data" / "documentos"
RUTA_NUEVOS = RAIZ / "data" / "nuevos"
RUTA_RESULTADOS = RAIZ / "resultados"


def quitar_acentos(texto: str) -> str:
    """Quita acentos y diéresis (la ñ también pasa a n)."""
    normal = unicodedata.normalize("NFD", texto)
    return "".join(c for c in normal if unicodedata.category(c) != "Mn")


_STOPWORDS_BASE = """
a al algo algun alguna algunas alguno algunos ante antes aqui asi aun aunque
cada como con contra cual cuales cuando cuyo de del desde donde dos e el ella
ellas ello ellos en entre era eran es esa esas ese eso esos esta estaba estan
estar este esto estos fue fueron ha haber habia han hasta hay la las le les lo
los mas me mi mis mucho muy nada ni no nos nosotros nuestra nuestro o os otra
otras otro otros para pero poco por porque puede pueden que quien quienes se
sea sean segun ser si sin sobre su sus tambien tan tanto te tenemos tiene
tienen todo todos tu tus un una unas uno unos usted ustedes va van y ya yo
cuando donde ese esa estos estas debe deben deberan sera seran sido ser son
asi cada cual cuyos mediante misma mismo mismos mismas parte partes tras
"""
STOPWORDS = {quitar_acentos(p) for p in _STOPWORDS_BASE.split()}


def raiz_simple(palabra: str) -> str:
    """Stemming muy simple: quita plural (s) y la última vocal (a, e, o).

    Así reporto/reporta -> report y correo/correos -> corre.
    """
    if len(palabra) > 3 and palabra.endswith("s"):
        palabra = palabra[:-1]
    if len(palabra) > 4 and palabra[-1] in "aeo":
        palabra = palabra[:-1]
    return palabra


def limpiar_texto(texto: str, usar_stemming: bool = False) -> str:
    """Minúsculas, sin acentos, sin números ni signos, sin stopwords."""
    texto = quitar_acentos(str(texto).lower())
    texto = re.sub(r"[^a-z\s]", " ", texto)
    tokens = [t for t in texto.split() if len(t) > 1 and t not in STOPWORDS]
    if usar_stemming:
        tokens = [raiz_simple(t) for t in tokens]
    return " ".join(tokens)


def _leer_txt(ruta: Path) -> tuple[str, str]:
    contenido = ruta.read_text(encoding="utf-8").strip()
    titulo = contenido.splitlines()[0].strip() if contenido else ruta.stem
    return titulo, contenido


def cargar_documentos(ruta: Path = RUTA_DOCUMENTOS) -> pd.DataFrame:
    """Recorre data/documentos/<categoria>/*.txt y arma un DataFrame.

    La categoría es el nombre de la carpeta; la primera línea es el título.
    """
    filas = []
    for carpeta in sorted(p for p in Path(ruta).iterdir() if p.is_dir()):
        for archivo in sorted(carpeta.glob("*.txt")):
            titulo, texto = _leer_txt(archivo)
            filas.append(
                {
                    "id": f"{carpeta.name}/{archivo.name}",
                    "titulo": titulo,
                    "texto": texto,
                    "categoria": carpeta.name,
                }
            )
    if not filas:
        raise FileNotFoundError(f"No se encontraron .txt en {ruta}")
    return pd.DataFrame(filas)


def cargar_nuevos(ruta: Path = RUTA_NUEVOS) -> pd.DataFrame:
    """Carga los documentos sin etiqueta de data/nuevos/*.txt."""
    filas = []
    for archivo in sorted(Path(ruta).glob("*.txt")):
        titulo, texto = _leer_txt(archivo)
        filas.append({"id": archivo.name, "titulo": titulo, "texto": texto})
    return pd.DataFrame(filas, columns=["id", "titulo", "texto"])
