
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

HOY = pd.Timestamp.now(tz=timezone.utc).tz_convert(None)

def _norm(s: str) -> str:
    import unicodedata, re
    if s is None: return ""
    s = str(s)
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    return re.sub(r'\s+', ' ', s.strip()).lower()

def _to_local_datetime(s):
    try:
        dt = pd.to_datetime(s, errors='coerce', dayfirst=True, utc=False)
        try:
            dt = dt.dt.tz_localize(None)
        except Exception:
            pass
        return dt
    except Exception:
        return pd.to_datetime(pd.Series([None]*len(s)))

def build_role_index(diccionario_df: Optional[pd.DataFrame]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    idx: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if diccionario_df is None or diccionario_df.empty:
        return idx
    cols = [c for c in diccionario_df.columns]
    # Esperamos columnas con al menos: hoja, columna, rol
    hoja_col = next((c for c in cols if "hoja" in _norm(c)), None)
    col_col  = next((c for c in cols if "colum" in _norm(c)), None)
    rol_col  = next((c for c in cols if "rol" in _norm(c)), None)
    if not (hoja_col and col_col and rol_col):
        return idx
    for _, r in diccionario_df.iterrows():
        hoja = str(r.get(hoja_col, "")).strip()
        col  = str(r.get(col_col, "")).strip()
        rol  = str(r.get(rol_col, ""))
        if not hoja or not col: 
            continue
        idx.setdefault(hoja, {})[col] = {"rol": rol}
    return idx

# ===== KPIs base
ING_HINTS = ["ingreso", "venta", "monto", "neto", "total", "abono", "bruto"]
COST_HINTS = ["costo", "coste", "gasto", "egreso", "compra"]

def _find_numeric_cols(df: pd.DataFrame, hints: List[str]) -> List[str]:
    res = []
    for c in df.columns:
        n = _norm(c)
        if any(h in n for h in hints) and pd.api.types.is_numeric_dtype(df[c]):
            res.append(c)
    if not res:
        # última chance: permitir object convertible
        for c in df.columns:
            n = _norm(c)
            if any(h in n for h in hints):
                try:
                    pd.to_numeric(df[c], errors='raise')
                    res.append(c)
                except Exception:
                    pass
    return list(dict.fromkeys(res))

def analizar_datos_taller(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    facts: Dict[str, Any] = {}
    df = data.get("MODELO_BOT")
    if df is not None and not df.empty:
        facts["ot_total"] = int(len(df))
        # Días en planta (si hay), promedio
        dias_cols = [c for c in df.columns if "dia" in _norm(c) and "plant" in _norm(" ".join(df.columns)) or "planta" in _norm(" ".join(df.columns))]
        if dias_cols:
            try:
                d = pd.to_numeric(df[dias_cols[0]], errors="coerce")
                facts["dias_promedio"] = float(d.mean(skipna=True))
            except Exception:
                pass
    fz = data.get("FINANZAS")
    if fz is not None and not fz.empty:
        ing_cols = _find_numeric_cols(fz, ING_HINTS)
        cost_cols= _find_numeric_cols(fz, COST_HINTS)
        ingresos = float(pd.DataFrame({c: pd.to_numeric(fz[c], errors='coerce') for c in ing_cols}).sum(axis=1).sum()) if ing_cols else 0.0
        costos   = float(pd.DataFrame({c: pd.to_numeric(fz[c], errors='coerce') for c in cost_cols}).sum(axis=1).sum()) if cost_cols else 0.0
        facts["ingresos_acum"] = ingresos
        facts["costos_acum"]   = costos
        facts["margen_acum"]   = ingresos - costos
    return facts

def narrativa_resumen(facts: Dict[str,Any]) -> str:
    ot = facts.get("ot_total", 0)
    dias = facts.get("dias_promedio")
    ing = facts.get("ingresos_acum", 0.0)
    cos = facts.get("costos_acum", 0.0)
    mar = facts.get("margen_acum", ing - cos)
    partes = [
        "### Resumen ejecutivo",
        f"- OT (COUNT): {ot}.",
        f"- Resultado global (margen): ${int(round(mar)):,}".replace(",", "."),
        "",
        "### Diagnóstico",
        f"- Ingresos acumulados: ${int(round(ing)):,}".replace(",", "."),
        f"- Costos acumulados: ${int(round(cos)):,}".replace(",", "."),
    ]
    if dias is not None:
        partes.append(f"- Días promedio en planta: {dias:.1f}.")
    partes += [
        "",
        "### Recomendaciones",
        "- Diseñar campañas sobre las 3 categorías/clientes más frecuentes.",
        "- Acelerar ventas con mejor conversión y márgenes.",
        "- Apalancar referidos en categorías con mayor recurrencia.",
        "",
        "### Estimaciones y proyecciones",
        "- Base (siguiente periodo): tendencia simple.",
        "- Optimista: +10%.",
        "- Conservador: -10%.",
        "",
        "### Riesgos y alertas",
        "- Concentración en pocas categorías/clientes.",
        "- Subida de costos impacta margen si no se ajusta precio.",
        "- Demoras operativas elevan lead time y reducen capacidad efectiva.",
    ]
    return "\n".join(partes)

# ===== Entregados no facturados (cruce) — placeholder robusto
def entregados_no_facturados(data: Dict[str,pd.DataFrame], role_idx: Dict[str,Dict[str,Dict[str,Any]]], days_back: int = 180) -> pd.DataFrame:
    mb = data.get("MODELO_BOT")
    if mb is None or mb.empty:
        return pd.DataFrame()
    df = mb.copy()
    # Heurística de columnas
    entrega_col = next((c for c in df.columns if "entreg" in _norm(c) or ("fecha" in _norm(c) and "entreg" in _norm(c))), None)
    ot_col = next((c for c in df.columns if _norm(c) in {"ot","# ot","n° ot","numero ot","num ot","nro ot"} or " ot" in _norm(c)), None)
    factura_cols = [c for c in df.columns if "fact" in _norm(c) or "folio" in _norm(c) or "document" in _norm(c)]
    if not entrega_col:
        # nada que filtrar
        return pd.DataFrame()
    d = _to_local_datetime(df[entrega_col])
    mask_deliv = d.notna()
    if days_back:
        piso = HOY - pd.Timedelta(days=days_back)
        mask_deliv &= (d >= piso)
    # No facturado: si no hay columnas, devolvemos vacío (mejor que falso positivo)
    if not factura_cols:
        mask_nf = pd.Series([False]*len(df))
    else:
        mask_nf = pd.Series([False]*len(df))
        for c in factura_cols:
            s = df[c].astype(str).map(_norm)
            mask_nf = mask_nf | s.isin({"","nan","none","0","no","false","pendiente","sin","s/factura"}) | s.str.contains("pendient|sin fact|no fact", na=False)
    res = df.loc[mask_deliv & mask_nf].copy()
    # Selección de columnas amigables
    cols = []
    for pref in [["patente","placa","matric"],["marca"],["modelo"],["cliente","aseguradora"],[entrega_col]]:
        for c in df.columns:
            n = _norm(c)
            if any(p in n for p in pref):
                cols.append(c)
                break
    if ot_col: cols = [ot_col] + cols
    cols = [c for c in cols if c in df.columns]
    return res[cols].drop_duplicates()

# ========= Nuevas utilidades para finanzas y proyecciones =========

CLIENTE_HINTS = ["cliente", "razon social", "razón social", "nombre cliente", "asegurado"]
ASEGURADORA_HINTS = ["aseguradora", "compania", "compañia", "compañía", "cia", "cia."]
FECHA_HINTS = ["fecha", "fecha doc", "f. doc", "f. emisión", "f. emision", "emision", "emisión", "fecha factura", "fecha venta", "fecha comprobante"]

def _first_date_col(df: pd.DataFrame, role_idx, hoja: str):
    for col, meta in role_idx.get(hoja, {}).items():
        r = _norm(meta.get("rol",""))
        if "fecha" in r or "date" in r:
            if col in df.columns: return col
    for c in df.columns:
        if any(h in _norm(c) for h in FECHA_HINTS):
            return c
    return None

def _group_col(df: pd.DataFrame, role_idx, hoja: str, group: str):
    hints = ASEGURADORA_HINTS if group=="aseguradora" else CLIENTE_HINTS
    for col, meta in role_idx.get(hoja, {}).items():
        r = _norm(meta.get("rol",""))
        if group in r and col in df.columns:
            return col
    for c in df.columns:
        if any(h in _norm(c) for h in hints):
            return c
    return None

def series_mensual_finanzas(data: Dict[str,pd.DataFrame], role_idx) -> pd.DataFrame:
    df = data.get("FINANZAS")
    if df is None or df.empty:
        return pd.DataFrame(columns=["mes","ingresos","costos","margen"])
    date_col = _first_date_col(df, role_idx, "FINANZAS")
    ing_cols  = _find_numeric_cols(df, ING_HINTS)
    cost_cols = _find_numeric_cols(df, COST_HINTS)
    if not date_col:
        return pd.DataFrame(columns=["mes","ingresos","costos","margen"])
    d = _to_local_datetime(df[date_col])
    x = pd.DataFrame({
        "fecha": d,
        "ingresos": pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in ing_cols}).sum(axis=1) if ing_cols else 0.0,
        "costos": pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in cost_cols}).sum(axis=1) if cost_cols else 0.0,
    })
    x["mes"] = x["fecha"].dt.to_period("M").astype(str)
    g = x.groupby("mes", dropna=True).agg({"ingresos":"sum","costos":"sum"}).reset_index()
    g["margen"] = g["ingresos"] - g["costos"]
    return g

def proyeccion_mes_siguiente(data: Dict[str,pd.DataFrame], role_idx) -> Dict[str,Any]:
    s = series_mensual_finanzas(data, role_idx)
    if s.empty:
        return {"serie": s, "forecast": None}
    k = 3 if len(s)>=3 else len(s)
    f_ing = float(s["ingresos"].tail(k).mean())
    f_cos = float(s["costos"].tail(k).mean())
    f_mar = f_ing - f_cos
    last_mes = s["mes"].iloc[-1]
    y, m = map(int, last_mes.split("-"))
    if m == 12:
        y2, m2 = y+1, 1
    else:
        y2, m2 = y, m+1
    prox = f"{y2:04d}-{m2:02d}"
    fc = {"mes": prox, "ingresos": f_ing, "costos": f_cos, "margen": f_mar}
    return {"serie": s, "forecast": fc}

def ingresos_por_agrupador(data: Dict[str,pd.DataFrame], role_idx, group="aseguradora", months_back: int = 6) -> pd.DataFrame:
    df = data.get("FINANZAS")
    if df is None or df.empty:
        return pd.DataFrame(columns=[group,"ingresos"])
    date_col = _first_date_col(df, role_idx, "FINANZAS")
    group_col = _group_col(df, role_idx, "FINANZAS", group)
    ing_cols  = _find_numeric_cols(df, ING_HINTS)
    if not group_col:
        return pd.DataFrame(columns=[group,"ingresos"])
    d = _to_local_datetime(df[date_col]) if date_col else None
    if months_back and d is not None:
        piso = (HOY.replace(day=1) - pd.offsets.MonthBegin(months_back)).date()
        mask = d.dt.date >= piso
        df = df.loc[mask]
        if d is not None:
            d = d.loc[mask]
    ing = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in ing_cols}).sum(axis=1) if ing_cols else pd.Series([0.0]*len(df))
    g = pd.DataFrame({group: df[group_col].astype(str), "ingresos": ing})
    g = g.groupby(group, dropna=False)["ingresos"].sum().reset_index().sort_values("ingresos", ascending=False)
    return g

def narrativa_controller(title: str, facts: Dict[str,Any], bullet_insights: List[str], forecast: Optional[Dict[str,Any]]=None) -> str:
    base = narrativa_resumen(facts)
    extra = ["", f"### {title}"]
    if bullet_insights:
        extra += [f"- {b}" for b in bullet_insights]
    if forecast is not None:
        try:
            extra.append("")
            extra.append("### Proyección (próximo mes)")
            extra.append(f"- Mes: {forecast.get('mes')}")
            extra.append(f"- Ingresos esperados: ${int(round(forecast.get('ingresos',0))):,}".replace(",", "."))
            extra.append(f"- Costos esperados: ${int(round(forecast.get('costos',0))):,}".replace(",", "."))
            extra.append(f"- Margen esperado: ${int(round(forecast.get('margen',0))):,}".replace(",", "."))
        except Exception:
            pass
    extra.append("")
    extra.append("### Recomendaciones")
    extra.append("- Concentrar esfuerzo comercial en las cuentas de mayor ticket y conversión.")
    extra.append("- En operaciones, acelerar cuellos de botella y seguimiento de atrasos.")
    extra.append("- Finanzas: priorizar cobranza temprana y evitar stockear gastos sin respaldo.")
    return "\n".join([base] + extra)
