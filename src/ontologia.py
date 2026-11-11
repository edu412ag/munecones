"""Paso 3: ontología OWL (owlready2) + grafo conceptual (matplotlib).

Caso: NovaTech Soluciones S.A. de C.V. (empresa de tecnología, Monterrey).

Genera en resultados/:
  ontologia.owl             ontología OWL (se abre en Protégé)
  ontologia.png             grafo con un color por clase y leyenda
  ontologia_clases.csv      tabla de clases
  ontologia_relaciones.csv  tabla de relaciones entre clases y entre conceptos
  ontologia_documentos.csv  relaciones documento-concepto y documento-tipo

Los conceptos se detectan en cada documento con listas de palabras clave: si el
documento menciona una de ellas, se crea la relación `trata_sobre`.

Uso:
    python src/ontologia.py
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from owlready2 import ObjectProperty, Thing, get_ontology

from preprocesamiento import RUTA_RESULTADOS, cargar_documentos, limpiar_texto, quitar_acentos

# ---------------------------------------------------------------------------
# CONFIGURACIÓN DEL CASO NOVATECH
# ---------------------------------------------------------------------------
ORGANIZACION = "NovaTech Soluciones"

# área -> {concepto: [palabras clave (una sola palabra) que lo detectan]}
AREAS = {
    "recursos_humanos": {
        "Vacaciones": ["vacaciones", "permisos", "incapacidad", "incapacidades"],
        "Teletrabajo": ["home", "remoto", "hibrido", "teletrabajo"],
        "Capacitacion": ["capacitacion", "curso", "cursos", "induccion"],
        "Talento": ["desempeno", "evaluacion", "objetivos", "candidato", "vacante", "entrevista"],
        "Nomina": ["nomina", "prestaciones", "aguinaldo"],
    },
    "seguridad_informacion": {
        "Contrasena": ["contrasena", "contrasenas", "credenciales", "multifactor"],
        "ControlAccesos": ["accesos", "privilegio", "roles", "vpn"],
        "Auditoria": ["auditoria", "auditorias", "auditores", "hallazgos"],
        "Incidente": ["incidente", "incidentes", "phishing", "malware"],
        "Respaldo": ["respaldo", "respaldos", "restauracion", "restaura"],
    },
    "soporte_clientes": {
        "Ticket": ["ticket", "tickets"],
        "Cliente": ["cliente", "clientes", "usuario", "usuarios"],
        "SLA": ["sla"],
        "Escalamiento": ["escalamiento", "escala", "escalar"],
        "BaseConocimiento": ["conocimiento", "articulo", "articulos", "frecuentes"],
    },
    "gestion_proyectos": {
        "LeccionAprendida": ["lecciones", "aprendidas", "retrospectiva"],
        "Proyecto": ["proyecto", "proyectos", "cronograma", "alcance"],
        "Riesgo": ["riesgo", "riesgos", "contingencia"],
        "Entregable": ["entregable", "entregables", "hitos"],
        "Scrum": ["scrum", "sprint", "sprints", "backlog"],
    },
    "desarrollo_software": {
        "Codigo": ["codigo", "commit", "commits", "linter"],
        "API": ["api", "apis", "endpoint", "openapi"],
        "Arquitectura": ["arquitectura", "microservicios", "contenedores"],
        "Despliegue": ["despliegue", "pipeline", "rollback"],
        "Pruebas": ["pruebas", "cobertura", "regresion", "defectos"],
    },
}

TIPOS_DOCUMENTO = [
    "Politica", "Manual", "Procedimiento", "Guia", "Reporte",
    "Acta", "Plantilla", "Estandar", "Plan", "Articulo",
]

# palabra del TÍTULO (sin acentos) -> tipo de documento
TIPO_POR_TITULO = {
    "politica": "Politica", "manual": "Manual", "procedimiento": "Procedimiento",
    "guia": "Guia", "reporte": "Reporte", "informe": "Reporte", "acta": "Acta",
    "plantilla": "Plantilla", "estandar": "Estandar", "plan": "Plan",
    "articulo": "Articulo",
}

# (sujeto, relación, objeto) entre conceptos. Las de color rojo cruzan áreas.
RELACIONES_CONCEPTOS = [
    # Recursos Humanos
    ("Nomina", "paga", "Vacaciones"),
    ("Talento", "orienta", "Capacitacion"),
    # Seguridad de la Información
    ("Contrasena", "refuerza", "ControlAccesos"),
    ("Auditoria", "revisa", "ControlAccesos"),
    ("ControlAccesos", "previene", "Incidente"),
    ("Respaldo", "mitiga", "Incidente"),
    # Soporte a Clientes
    ("Cliente", "levanta", "Ticket"),
    ("Ticket", "se_rige_por", "SLA"),
    ("Ticket", "se_escala_mediante", "Escalamiento"),
    ("BaseConocimiento", "resuelve", "Ticket"),
    # Gestión de Proyectos
    ("Proyecto", "enfrenta", "Riesgo"),
    ("Proyecto", "produce", "Entregable"),
    ("LeccionAprendida", "mejora", "Proyecto"),
    # Desarrollo de Software
    ("API", "forma_parte_de", "Arquitectura"),
    ("Codigo", "se_publica_mediante", "Despliegue"),
    ("Pruebas", "valida", "Despliegue"),
    # --- Relaciones que cruzan áreas ---
    ("Incidente", "genera", "Ticket"),                      # Seguridad -> Soporte
    ("Capacitacion", "sensibiliza_sobre", "Incidente"),     # RR. HH. -> Seguridad
    ("Teletrabajo", "requiere", "ControlAccesos"),          # RR. HH. -> Seguridad
    ("Vacaciones", "reduce_capacidad_de", "Scrum"),         # RR. HH. -> Gestión
    ("LeccionAprendida", "enriquece", "BaseConocimiento"),  # Gestión -> Soporte
    ("Proyecto", "se_construye_con", "Codigo"),             # Gestión -> Desarrollo
]

# documentos que se dibujan en el grafo (id = carpeta/archivo.txt); mínimo 5
DOCS_DIAGRAMA = [
    "recursos_humanos/politica_home_office.txt",
    "recursos_humanos/procedimiento_evaluacion_desempeno.txt",
    "seguridad_informacion/procedimiento_respuesta_incidentes.txt",
    "seguridad_informacion/politica_control_accesos.txt",
    "soporte_clientes/procedimiento_tickets.txt",
    "soporte_clientes/politica_sla.txt",
    "gestion_proyectos/reporte_lecciones_aprendidas.txt",
    "desarrollo_software/manual_despliegue_cicd.txt",
    "desarrollo_software/guia_documentacion_api.txt",
]
CONCEPTOS_POR_DOC_EN_GRAFO = 2

# ---------------------------------------------------------------------------


def ident(texto: str) -> str:
    """Convierte un texto en un identificador válido para OWL."""
    limpio = re.sub(r"[^0-9a-zA-Z_]", "_", quitar_acentos(str(texto)))
    return limpio if limpio and not limpio[0].isdigit() else f"n_{limpio}"


NOMBRES_AREA = {
    "recursos_humanos": "Recursos Humanos",
    "seguridad_informacion": "Seguridad de la Información",
    "soporte_clientes": "Soporte a Clientes",
    "gestion_proyectos": "Gestión de Proyectos",
    "desarrollo_software": "Desarrollo de Software",
}


def etiqueta_area(area: str) -> str:
    return NOMBRES_AREA.get(area, area.replace("_", " ").title())


def area_de_concepto() -> dict[str, str]:
    return {c: a for a, cs in AREAS.items() for c in cs}


def detectar_conceptos(texto: str, area: str | None = None) -> list[str]:
    """Conceptos cuyas palabras clave aparecen en el documento.

    Si se da `area`, los conceptos de esa área van primero.
    """
    palabras = set(limpiar_texto(texto).split())
    encontrados = []
    for a, conceptos in AREAS.items():
        for concepto, claves in conceptos.items():
            if palabras & {limpiar_texto(c) for c in claves}:
                encontrados.append((0 if a == area else 1, concepto))
    return [c for _, c in sorted(encontrados, key=lambda t: t[0])]


def detectar_tipo(titulo: str) -> str | None:
    for p in limpiar_texto(titulo).split():
        if p in TIPO_POR_TITULO:
            return TIPO_POR_TITULO[p]
    return None


def construir_ontologia(df: pd.DataFrame):
    onto = get_ontology("http://novatech.example.org/ontologia.owl")
    with onto:
        class Organizacion(Thing): pass
        class AreaConocimiento(Thing): pass
        class Concepto(Thing): pass
        class TipoDocumento(Thing): pass
        class Documento(Thing): pass

        class tiene_area(ObjectProperty):
            domain, range = [Organizacion], [AreaConocimiento]

        class tiene_concepto(ObjectProperty):
            domain, range = [AreaConocimiento], [Concepto]

        class trata_sobre(ObjectProperty):
            domain, range = [Documento], [Concepto]

        class es_de_tipo(ObjectProperty):
            domain, range = [Documento], [TipoDocumento]

        class clasificado_en(ObjectProperty):
            domain, range = [Documento], [AreaConocimiento]

        # relaciones nombradas entre conceptos
        propiedades = {}
        for _, rel, _ in RELACIONES_CONCEPTOS:
            if rel not in propiedades:
                propiedades[rel] = type(rel, (ObjectProperty,), {"domain": [Concepto], "range": [Concepto]})

        org = Organizacion(ident(ORGANIZACION))
        areas, conceptos = {}, {}
        for area, lista in AREAS.items():
            areas[area] = AreaConocimiento(ident(area))
            org.tiene_area.append(areas[area])
            for concepto in lista:
                conceptos[concepto] = Concepto(ident(concepto))
                areas[area].tiene_concepto.append(conceptos[concepto])

        for suj, rel, obj in RELACIONES_CONCEPTOS:
            getattr(conceptos[suj], rel).append(conceptos[obj])

        tipos = {t: TipoDocumento(ident(t)) for t in TIPOS_DOCUMENTO}

        for _, fila in df.iterrows():
            doc = Documento(ident(fila["id"].replace(".txt", "")))
            doc.clasificado_en.append(areas[fila["categoria"]])
            for c in detectar_conceptos(fila["texto"], fila["categoria"]):
                doc.trata_sobre.append(conceptos[c])
            tipo = detectar_tipo(fila["titulo"])
            if tipo:
                doc.es_de_tipo.append(tipos[tipo])
    return onto


def tablas(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clases = pd.DataFrame(
        [
            ("Organizacion", "La empresa", ORGANIZACION),
            ("AreaConocimiento", "Área que agrupa el conocimiento (categoría del modelo)",
             ", ".join(etiqueta_area(a) for a in AREAS)),
            ("Concepto", "Sustantivo clave del negocio (5 por área)",
             ", ".join(c for a in AREAS.values() for c in a)),
            ("TipoDocumento", "Forma del documento", ", ".join(TIPOS_DOCUMENTO)),
            ("Documento", "Archivo de conocimiento (.txt)", f"{len(df)} documentos"),
        ],
        columns=["clase", "descripcion", "ejemplos"],
    )
    filas = [("Organizacion", "tiene_area", "AreaConocimiento", False),
             ("AreaConocimiento", "tiene_concepto", "Concepto", False),
             ("Documento", "trata_sobre", "Concepto", False),
             ("Documento", "es_de_tipo", "TipoDocumento", False),
             ("Documento", "clasificado_en", "AreaConocimiento", False)]
    area_de = area_de_concepto()
    filas += [(s, r, o, area_de[s] != area_de[o]) for s, r, o in RELACIONES_CONCEPTOS]
    relaciones = pd.DataFrame(filas, columns=["sujeto", "relacion", "objeto", "cruza_areas"])

    docs = []
    for _, f in df.iterrows():
        docs.append({
            "documento": f["id"],
            "area": f["categoria"],
            "tipo": detectar_tipo(f["titulo"]) or "",
            "conceptos": "; ".join(detectar_conceptos(f["texto"], f["categoria"])),
        })
    return clases, relaciones, pd.DataFrame(docs)


COLORES = {
    "Organizacion": "#6a3d9a",
    "AreaConocimiento": "#1f5fa8",
    "Concepto": "#2a8a3e",
    "TipoDocumento": "#c75d00",
    "Documento": "#5f5f5f",
}
ROJO = "#d62728"


def dibujar_grafo(df: pd.DataFrame, ruta: Path) -> None:
    """Grafo por columnas: conceptos | áreas | documentos | tipos (organización arriba)."""
    DPI = 150
    fig, ax = plt.subplots(figsize=(22, 16), dpi=DPI)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
    area_de = area_de_concepto()

    # --- posiciones -------------------------------------------------------
    X_CON, X_ARE, X_DOC, X_TIP = 0.0, 2.6, 5.2, 7.6
    pos, clase, area_nodo = {}, {}, {}
    y = 0.0
    for area, lista in AREAS.items():
        ys = []
        for c in lista:
            pos[c], clase[c] = (X_CON, y), "Concepto"
            ys.append(y)
            y -= 0.55
        pos[etiqueta_area(area)] = (X_ARE, float(np.mean(ys)))
        clase[etiqueta_area(area)] = "AreaConocimiento"
        y -= 0.45
    y_min = y
    ys_areas = [pos[etiqueta_area(a)][1] for a in AREAS]
    pos[ORGANIZACION] = (X_ARE, max(ys_areas) + 1.0)
    clase[ORGANIZACION] = "Organizacion"

    elegidos = df[df["id"].isin(DOCS_DIAGRAMA)]
    por_area = {a: elegidos[elegidos["categoria"] == a] for a in AREAS}
    aristas_doc = []   # (doc, area, [conceptos], tipo)
    tipo_ys: dict[str, list[float]] = {}
    for area, sub in por_area.items():
        k = len(sub)
        for j, (_, fila) in enumerate(sub.iterrows()):
            nombre = fila["id"].split("/")[-1].replace(".txt", "")
            yy = pos[etiqueta_area(area)][1] + (j - (k - 1) / 2) * 1.05
            pos[nombre], clase[nombre] = (X_DOC, yy), "Documento"
            tipo = detectar_tipo(fila["titulo"])
            if tipo:
                tipo_ys.setdefault(tipo, []).append(yy)
            aristas_doc.append(
                (nombre, etiqueta_area(area),
                 detectar_conceptos(fila["texto"], area)[:CONCEPTOS_POR_DOC_EN_GRAFO], tipo)
            )
    tipos_ordenados = sorted(tipo_ys, key=lambda t: np.mean(tipo_ys[t]), reverse=True)
    y_prev = None
    for t in tipos_ordenados:
        yy = float(np.mean(tipo_ys[t]))
        if y_prev is not None and y_prev - yy < 0.6:
            yy = y_prev - 0.6
        pos[t], clase[t] = (X_TIP, yy), "TipoDocumento"
        y_prev = yy

    ax.set_xlim(-3.4, 8.8)
    ax.set_ylim(y_min - 0.2, pos[ORGANIZACION][1] + 0.6)
    ax.axis("off")

    # --- nodos (cajas de texto) ---------------------------------------------
    cajas = {}
    for n, (x, yy) in pos.items():
        cajas[n] = ax.text(
            x, yy, n, ha="center", va="center", fontsize=8.5, color="white",
            fontweight="bold", zorder=3,
            bbox=dict(boxstyle="round,pad=0.35", fc=COLORES[clase[n]], ec="none"),
        )

    def etiqueta(p, texto, color, fs=6.5):
        ax.text(p[0], p[1], texto, ha="center", va="center", fontsize=fs, color=color,
                style="italic", zorder=4,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.88))

    def flecha(a, b, color, lw, ls="-", rad=0.0, zorder=5):
        ax.annotate(
            "", xy=pos[b], xytext=pos[a], xycoords="data", textcoords="data", zorder=zorder,
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, mutation_scale=11,
                            connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1,
                            patchA=cajas[a].get_bbox_patch(), patchB=cajas[b].get_bbox_patch()),
        )

    def en_segmento(a, b, f):
        (x1, y1), (x2, y2) = pos[a], pos[b]
        return (x1 + f * (x2 - x1), y1 + f * (y2 - y1))

    # --- aristas estructurales (gris claro) ---------------------------------
    GRIS = "#9a9a9a"
    inv = ax.transData.inverted()
    for area, lista in AREAS.items():
        # organización -> área: arcos hacia la derecha para no cruzar las cajas de las áreas
        a = np.array(ax.transData.transform(pos[ORGANIZACION]))
        b = np.array(ax.transData.transform(pos[etiqueta_area(area)]))
        d = b - a
        dist_in = float(np.hypot(*d)) / DPI
        rad = float(np.sign(d[1]) * 2 * (0.9 + 0.1 * dist_in) / dist_in)
        flecha(ORGANIZACION, etiqueta_area(area), GRIS, 0.9, rad=rad)
        ctrl = (a + b) / 2 + rad * np.array([d[1], -d[0]])
        etiqueta(inv.transform(0.25 * a + 0.5 * ctrl + 0.25 * b), "tiene_area", "#555555")
        for c in lista:
            flecha(etiqueta_area(area), c, GRIS, 0.7)
            etiqueta(en_segmento(etiqueta_area(area), c, 0.5), "tiene_concepto", "#666666", 5.5)
    for doc, area, conceptos, tipo in aristas_doc:
        flecha(doc, area, "#c9a227", 1.0)
        etiqueta(en_segmento(doc, area, 0.5), "clasificado_en", "#8a6d00", 6)
        for i, c in enumerate(conceptos):
            flecha(doc, c, "#7a9cc6", 0.9)
            etiqueta(en_segmento(doc, c, 0.16 + 0.2 * i), "trata_sobre", "#2f5d8a", 6)
        if tipo:
            flecha(doc, tipo, "#d9a066", 0.9)
            etiqueta(en_segmento(doc, tipo, 0.5), "es_de_tipo", "#9a5a1a", 6)

    # --- relaciones entre conceptos (arcos a la izquierda) --------------------
    for s, r, o in RELACIONES_CONCEPTOS:
        cruza = area_de[s] != area_de[o]
        a = np.array(ax.transData.transform(pos[s]))
        b = np.array(ax.transData.transform(pos[o]))
        d = b - a
        dist_in = float(np.hypot(*d)) / DPI
        bulge_in = 1.5 + 0.12 * dist_in
        rad = 2 * bulge_in / dist_in
        rad = -np.sign(d[1]) * rad          # curva hacia la izquierda
        flecha(s, o, ROJO if cruza else "#333333", 1.5 if cruza else 1.2,
               "--" if cruza else "-", rad=float(rad))
        ctrl = (a + b) / 2 + rad * np.array([d[1], -d[0]])
        apex = 0.25 * a + 0.5 * ctrl + 0.25 * b
        etiqueta(inv.transform(apex), r, ROJO if cruza else "#222222", 6.8)

    ax.legend(
        handles=[Patch(fc=c, label=k) for k, c in COLORES.items()] + [
            Line2D([0], [0], color="#333333", lw=1.4, label="relación entre conceptos (misma área)"),
            Line2D([0], [0], color=ROJO, lw=1.6, ls="--", label="relación entre conceptos que cruza áreas"),
            Line2D([0], [0], color=GRIS, lw=1, label="relación estructural (área, documento, tipo)"),
        ],
        loc="upper right", title="Clases y relaciones", fontsize=9, title_fontsize=10,
        framealpha=0.95,
    )
    ax.set_title(f"Ontología del conocimiento de {ORGANIZACION}", fontsize=15, pad=6)
    fig.savefig(ruta, dpi=DPI)
    plt.close(fig)


def main() -> None:
    RUTA_RESULTADOS.mkdir(exist_ok=True)
    df = cargar_documentos()
    onto = construir_ontologia(df)
    onto.save(file=str(RUTA_RESULTADOS / "ontologia.owl"), format="rdfxml")
    clases, relaciones, docs = tablas(df)
    clases.to_csv(RUTA_RESULTADOS / "ontologia_clases.csv", index=False)
    relaciones.to_csv(RUTA_RESULTADOS / "ontologia_relaciones.csv", index=False)
    docs.to_csv(RUTA_RESULTADOS / "ontologia_documentos.csv", index=False)
    dibujar_grafo(df, RUTA_RESULTADOS / "ontologia.png")
    print("Clases:", len(list(onto.classes())), "| Propiedades:", len(list(onto.object_properties())),
          "| Individuos:", len(list(onto.individuals())))
    print("Documentos sin tipo detectado:", int((docs["tipo"] == "").sum()),
          "| sin conceptos:", int((docs["conceptos"] == "").sum()))
    print("Archivos en resultados/: ontologia.owl, ontologia.png, ontologia_clases.csv, "
          "ontologia_relaciones.csv, ontologia_documentos.csv")


if __name__ == "__main__":
    main()
