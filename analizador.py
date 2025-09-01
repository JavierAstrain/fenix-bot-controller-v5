# analizador.py — v6 (fechas robustas + atrasos y pendientes)
import pandas as pd
from typing import Dict, Any
from datetime import datetime
from zoneinfo import ZoneInfo

_TZ = ZoneInfo("America/Santiago")
HOY = datetime.now(_TZ).date()  # fecha actual del sistema

def _norm(s: str) -> str:
    return str(s).strip().lower()

def _to_local_datetime(s):
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True, utc=False)
    try:
        if getattr(dt.dt, "tz", None) is None:
            return dt.dt.tz_localize(_TZ, nonexistent='shift_forward', ambiguous='NaT').dt.tz_convert(_TZ)
        return dt.dt.tz_convert(_TZ)
    except Exception:
        return dt

# -------------------------------
# NUEVAS FUNCIONES DE FECHAS
# -------------------------------
def calcular_atrasos(df: pd.DataFrame, fecha_col: str) -> pd.DataFrame:
    """Agrega columnas de atraso y estado en base a la fecha de entrega/pago."""
    fechas = _to_local_datetime(df[fecha_col]).dt.date
    df["_dias_atraso"] = (HOY - fechas).dt.days
    df["_estado_fecha"] = df["_dias_atraso"].apply(
        lambda d: "pendiente (futuro)" if d < 0 else ("cumplida" if d >= 0 else "sin fecha")
    )
    return df

def facturas_pendientes(df: pd.DataFrame, fecha_pago_col: str, id_col: str):
    """Retorna IDs de facturas con fecha de pago > hoy."""
    fechas = _to_local_datetime(df[fecha_pago_col]).dt.date
    pendientes = df[fechas > HOY]
    return pendientes[id_col].astype(str).tolist()

# -------------------------------
# FUNCIONES EXISTENTES (ajustadas)
# -------------------------------
ING_HINTS = ["monto", "neto", "total", "importe", "facturacion", "facturación", "ingreso", "venta", "principal"]
COST_HINTS = [
    "costo", "costos", "gasto", "gastos", "insumo", "insumos",
    "repuesto", "repuestos", "pintura", "material", "materiales",
    "mano de obra", "mo", "imposicion", "imposiciones",
    "arriendo", "admin", "administracion", "administración", "equipo de planta", "generales",
    "financiero", "financieros", "interes", "intereses"
]

def _find_numeric_cols(df: pd.DataFrame, keywords):
    cols = []
    for c in df.columns:
        c2 = _norm(c)
        if any(k in c2 for k in keywords):
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0:
                cols.append(c)
    return cols

def _count_services(df: pd.DataFrame) -> int:
    keys = ["patente", "orden", "oc", "folio", "documento", "presupuesto", "id", "nro", "número", "num"]
    for k in keys:
        for c in df.columns:
            if k in _norm(c):
                return int(df[c].nunique(dropna=True))
    ing_cols = _find_numeric_cols(df, ING_HINTS)
    if ing_cols:
        s = pd.to_numeric(df[ing_cols[0]], errors="coerce")
        return int((s > 0).sum())
    return 0

def _lead_time_days(df: pd.DataFrame):
    start_keys = ["fecha ingreso", "ingreso", "recepcion", "recepción", "entrada"]
    end_keys   = ["fecha salida", "salida", "entrega", "egreso", "termino", "término"]
    start_col = end_col = None
    for c in df.columns:
        cc = _norm(c)
        if not start_col and any(k in cc for k in start_keys): start_col = c
        if not end_col and any(k in cc for k in end_keys):     end_col = c
    if start_col and end_col:
        s = _to_local_datetime(df[start_col])
        e = _to_local_datetime(df[end_col])
        d = (e - s).dt.days
        d = d[(d.notna()) & (d >= 0) & (d < 365)]
        if len(d) >= 3:
            return float(d.median())
    return None

def _apply_client_filter(df: pd.DataFrame, client_substr: str) -> pd.DataFrame:
    if not client_substr:
        return df
    cliente_col = None
    for c in df.columns:
        if "cliente" in _norm(c):
            cliente_col = c
            break
    if cliente_col:
        out = df.copy()
        return out[out[cliente_col].astype(str).str.contains(client_substr, case=False, na=False)]
    return df

def analizar_datos_taller(data: Dict[str, pd.DataFrame], cliente_contiene: str = "") -> Dict[str, Any]:
    total_ing = 0.0
    total_cost = 0.0
    total_services = 0
    conversion = None
    lead_time = None
    hojas = {}

    for hoja, df in data.items():
        if df is None or df.empty:
            continue

        df2 = _apply_client_filter(df, cliente_contiene)

        ing_cols = _find_numeric_cols(df2, ING_HINTS)
        cost_cols = _find_numeric_cols(df2, COST_HINTS)

        ing_sum = float(pd.DataFrame({c: pd.to_numeric(df2[c], errors="coerce") for c in ing_cols}).sum().sum()) if ing_cols else 0.0
        cost_sum = float(pd.DataFrame({c: pd.to_numeric(df2[c], errors="coerce") for c in cost_cols}).sum().sum()) if cost_cols else 0.0

        total_ing += ing_sum
        total_cost += cost_sum
        total_services += _count_services(df2)

        lt = _lead_time_days(df2)
        if lt is not None:
            lead_time = lt

        hojas[hoja] = {
            "filas": int(len(df2)),
            "ing_cols": ing_cols,
            "cost_cols": cost_cols,
            "ingresos_hoja": ing_sum,
            "costos_hoja": cost_sum
        }

    margen = total_ing - total_cost
    margen_pct = (margen / total_ing * 100.0) if total_ing else 0.0
    ticket = (total_ing / total_services) if total_services else None

    return {
        "fecha_actual": HOY.strftime("%d/%m/%Y"),
        "ingresos": total_ing,
        "costos": total_cost,
        "margen": margen,
        "margen_pct": round(margen_pct, 2),
        "servicios": total_services,
        "ticket_promedio": ticket,
        "conversion_pct": conversion,
        "lead_time_mediano_dias": lead_time,
        "hojas": hojas
    }

