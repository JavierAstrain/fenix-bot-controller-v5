
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import timezone

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
        idx.setdefault(hoja.upper(), {})[col] = {"rol": rol}
    return idx

# KPIs base
ING_HINTS = ["ingreso", "venta", "monto", "neto", "total", "abono", "bruto"]
COST_HINTS = ["costo", "coste", "gasto", "egreso", "compra"]

def _find_numeric_cols(df: pd.DataFrame, hints: List[str]) -> List[str]:
    res = []
    for c in df.columns:
        n = _norm(c)
        if any(h in n for h in hints) and pd.api.types.is_numeric_dtype(df[c]):
            res.append(c)
    if not res:
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
        "- Demoras operativas elevan lead time y reduen capacidad efectiva.",
    ]
    return "\n".join(partes)

def series_mensual_finanzas(data: Dict[str,pd.DataFrame], role_idx) -> pd.DataFrame:
    df = data.get("FINANZAS")
    if df is None or df.empty:
        return pd.DataFrame(columns=["mes","ingresos","costos","margen"])
    # Intentar fecha por rol o heurística
    date_col = None
    for col, meta in role_idx.get("FINANZAS", {}).items():
        r = _norm(meta.get("rol", ""))
        if "fecha" in r and col in df.columns:
            date_col = col
            break
    if not date_col:
        for c in df.columns:
            if "fecha" in _norm(c) or "emision" in _norm(c) or "emisión" in _norm(c):
                date_col = c; break
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
    y2, m2 = (y+1, 1) if m==12 else (y, m+1)
    prox = f"{y2:04d}-{m2:02d}"
    fc = {"mes": prox, "ingresos": f_ing, "costos": f_cos, "margen": f_mar}
    return {"serie": s, "forecast": fc}

def ingresos_por_agrupador(data: Dict[str,pd.DataFrame], role_idx, group="aseguradora", months_back: int = 6) -> pd.DataFrame:
    df = data.get("FINANZAS")
    if df is None or df.empty:
        return pd.DataFrame(columns=[group,"ingresos"])
    # group_col heurístico/rol
    group_col = None
    candidates = ["aseguradora","compania","compañia","compañía","cliente","razon social","razón social","nombre cliente","asegurado"]
    for col, meta in role_idx.get("FINANZAS", {}).items():
        r = _norm(meta.get("rol",""))
        if group in r and col in df.columns:
            group_col = col; break
    if not group_col:
        for c in df.columns:
            if any(h in _norm(c) for h in candidates if h==group or group in h):
                group_col = c; break
    ing_cols  = _find_numeric_cols(df, ING_HINTS)
    # fecha y filtro últimos meses
    date_col = None
    for col, meta in role_idx.get("FINANZAS", {}).items():
        r = _norm(meta.get("rol",""))
        if "fecha" in r and col in df.columns:
            date_col = col; break
    if not date_col:
        for c in df.columns:
            if "fecha" in _norm(c) or "emision" in _norm(c) or "emisión" in _norm(c):
                date_col = c; break
    if months_back and date_col:
        d = _to_local_datetime(df[date_col])
        piso = (HOY.replace(day=1) - pd.offsets.MonthBegin(months_back)).date()
        df = df.loc[d.dt.date >= piso]
    if not group_col:
        return pd.DataFrame(columns=[group,"ingresos"])
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

# Determinista: entregados sin facturar
def delivered_not_invoiced(data: dict, roles_index: dict, days_back: int = 180) -> pd.DataFrame:
    mb = data.get("MODELO_BOT")
    if mb is None or mb.empty:
        return pd.DataFrame(columns=["OT","PATENTE","CLIENTE","MARCA","MODELO","FECHA_ENTREGA","MOTIVO"])

    def _col_by_role(df: pd.DataFrame, role_names, hoja: str):
        role_names = {r.lower() for r in role_names}
        for col, meta in roles_index.get(hoja, {}).items():
            rol = _norm(meta.get("rol",""))
            if any(r in rol for r in role_names) and col in df.columns:
                return col
        for c in df.columns:
            n = _norm(c)
            if any(r in n for r in role_names):
                return c
        return None

    def _to_dt(s):
        try:
            dt = pd.to_datetime(s, errors="coerce", dayfirst=True, utc=False)
            try: dt = dt.dt.tz_localize(None)
            except Exception: pass
            return dt
        except Exception:
            return pd.to_datetime(pd.Series([None]*len(s)))

    entrega_col = _col_by_role(mb, ["fecha_entrega","entrega","salida"], "MODELO_BOT")
    ot_col      = _col_by_role(mb, ["ot","ot_id","# ot","nro ot","numero ot"], "MODELO_BOT")
    plate_col   = _col_by_role(mb, ["patente","placa","matricula"], "MODELO_BOT")
    cliente_col = _col_by_role(mb, ["cliente","aseguradora"], "MODELO_BOT")
    marca_col   = _col_by_role(mb, ["marca"], "MODELO_BOT")
    modelo_col  = _col_by_role(mb, ["modelo"], "MODELO_BOT")

    if not entrega_col:
        out = pd.DataFrame(columns=["OT","PATENTE","CLIENTE","MARCA","MODELO","FECHA_ENTREGA","MOTIVO"])
        out.loc[0,"MOTIVO"] = "No se encontró columna de fecha de entrega."
        return out

    d_entrega = _to_dt(mb[entrega_col])
    mask_entregado = d_entrega.notna()
    if days_back:
        piso = pd.Timestamp.now().tz_localize(None) - pd.Timedelta(days=days_back)
        mask_entregado &= (d_entrega >= piso)

    entregados = mb.loc[mask_entregado, [c for c in [ot_col, plate_col, cliente_col, marca_col, modelo_col, entrega_col] if c]].copy()
    entregados.rename(columns={
        ot_col: "OT", plate_col: "PATENTE", cliente_col: "CLIENTE",
        marca_col: "MARCA", modelo_col: "MODELO", entrega_col: "FECHA_ENTREGA"
    }, inplace=True, errors="ignore")
    entregados["FECHA_ENTREGA"] = _to_dt(entregados.get("FECHA_ENTREGA"))

    fin = data.get("FINANZAS")
    if fin is not None and not fin.empty:
        fin_ot    = _col_by_role(fin, ["ot","ot_id"], "FINANZAS")
        fin_plate = _col_by_role(fin, ["patente","placa","matricula"], "FINANZAS")
        fin_doc   = _col_by_role(fin, ["factura","folio","documento","boleta","nro documento","numero documento"], "FINANZAS")
        fin_fecha = _col_by_role(fin, ["fecha","fecha doc","fecha factura","emision","emisión"], "FINANZAS")
        if fin_doc:
            s_norm = fin[fin_doc].astype(str).str.strip().str.lower()
            mask_doc = ~s_norm.isin({"", "nan", "none", "0", "false"})
            fact = fin.loc[mask_doc, [c for c in [fin_ot, fin_plate, fin_doc, fin_fecha] if c]].copy()
            fact["OT"]      = fact[fin_ot]    if fin_ot in fact.columns else None
            fact["PATENTE"] = fact[fin_plate] if fin_plate in fact.columns else None
            fact["DOC"]     = fact[fin_doc].astype(str)
            if fin_fecha in fact.columns:
                fact["FECHA_DOC"] = _to_dt(fact[fin_fecha])
        else:
            fact = pd.DataFrame(columns=["OT","PATENTE","DOC","FECHA_DOC"])
    else:
        fact = pd.DataFrame(columns=["OT","PATENTE","DOC","FECHA_DOC"])

    def _clean_key(s):
        return s.astype(str).str.strip().str.lower()

    if "OT" in entregados.columns and "OT" in fact.columns and fact["OT"].notna().any():
        en = entregados.copy(); fa = fact.copy()
        en["__key"] = _clean_key(en["OT"]); fa["__key"] = _clean_key(fa["OT"])
    elif "PATENTE" in entregados.columns and "PATENTE" in fact.columns and fact["PATENTE"].notna().any():
        en = entregados.copy(); fa = fact.copy()
        en["__key"] = _clean_key(en["PATENTE"]); fa["__key"] = _clean_key(fa["PATENTE"])
    else:
        out = entregados.copy()
        out["MOTIVO"] = "No se encontró llave común (OT o PATENTE) entre MODELO_BOT y FINANZAS."
        cols = ["OT","PATENTE","CLIENTE","MARCA","MODELO","FECHA_ENTREGA","MOTIVO"]
        return out[[c for c in cols if c in out.columns]].sort_values("FECHA_ENTREGA", ascending=False)

    fact_keys = set(fa.loc[fa["__key"].notna(), "__key"])
    mask_no_fact = ~en["__key"].isin(fact_keys)

    out = en.loc[mask_no_fact, ["OT","PATENTE","CLIENTE","MARCA","MODELO","FECHA_ENTREGA"]].copy()
    out["MOTIVO"] = "Entregado sin documento en FINANZAS."
    cols = ["OT","PATENTE","CLIENTE","MARCA","MODELO","FECHA_ENTREGA","MOTIVO"]
    out = out[[c for c in cols if c in out.columns]].drop_duplicates()
    return out.sort_values("FECHA_ENTREGA", ascending=False)
