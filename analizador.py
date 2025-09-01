# analizador.py — v7 (fechas robustas + atrasos/pendientes + DICCIONARIO)
from __future__ import annotations
import pandas as pd
from typing import Dict, Any, List, Optional
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

# --- Hints existentes (no toco tus nombres) ---
ING_HINTS = ["monto", "neto", "total", "importe", "facturacion", "facturación", "ingreso", "venta", "principal"]
COST_HINTS = [
    "costo","costos","gasto","gastos","insumo","insumos","repuesto","repuestos","pintura","material","materiales",
    "mano de obra","mo","imposicion","imposiciones","arriendo","admin","administracion","administración","equipo de planta",
    "generales","financiero","financieros","interes","intereses"
]
ID_HINTS = ["id","folio","factura","documento","nro","número","num","ot","orden","oc","patente","presupuesto"]
ENTREGA_HINTS = ["fecha entrega","entrega","salida","egreso","término","termino","compromiso"]
PAGO_HINTS = ["fecha pago","pago","venc","vencimiento","fecha venc","fecha de pago"]

def _find_numeric_cols(df: pd.DataFrame, keywords: List[str]):
    cols = []
    for c in df.columns:
        c2 = _norm(c)
        if any(k in c2 for k in keywords):
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0:
                cols.append(c)
    return cols

def _count_services(df: pd.DataFrame) -> int:
    for k in ID_HINTS:
        for c in df.columns:
            if k in _norm(c):
                return int(df[c].nunique(dropna=True))
    ing_cols = _find_numeric_cols(df, ING_HINTS)
    if ing_cols:
        s = pd.to_numeric(df[ing_cols[0]], errors="coerce")
        return int((s > 0).sum())
    return 0

def _lead_time_days(df: pd.DataFrame):
    start_keys = ["fecha ingreso","ingreso","recepcion","recepción","entrada"]
    end_keys   = ["fecha salida","salida","entrega","egreso","termino","término"]
    start_col = end_col = None
    for c in df.columns:
        cc = _norm(c)
        if not start_col and any(k in cc for k in start_keys): start_col = c
        if not end_col   and any(k in cc for k in end_keys):   end_col = c
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
    total_ing = 0.0; total_cost = 0.0; total_services = 0; lead_time=None
    hojas = {}
    for hoja, df in data.items():
        if df is None or df.empty: continue
        df2 = _apply_client_filter(df, cliente_contiene)
        ing_cols  = _find_numeric_cols(df2, ING_HINTS)
        cost_cols = _find_numeric_cols(df2, COST_HINTS)
        ing_sum  = float(pd.DataFrame({c: pd.to_numeric(df2[c], errors="coerce") for c in ing_cols }).sum().sum()) if ing_cols else 0.0
        cost_sum = float(pd.DataFrame({c: pd.to_numeric(df2[c], errors="coerce") for c in cost_cols}).sum().sum()) if cost_cols else 0.0
        total_ing += ing_sum; total_cost += cost_sum; total_services += _count_services(df2)
        lt = _lead_time_days(df2)
        if lt is not None: lead_time = lt
        hojas[hoja] = {"filas": int(len(df2)), "ing_cols": ing_cols, "cost_cols": cost_cols,
                       "ingresos_hoja": ing_sum, "costos_hoja": cost_sum}
    margen = total_ing - total_cost
    margen_pct = (margen / total_ing * 100.0) if total_ing else 0.0
    ticket = (total_ing / total_services) if total_services else None
    return {"fecha_actual": HOY.strftime("%d/%m/%Y"), "ingresos": total_ing, "costos": total_cost,
            "margen": margen, "margen_pct": round(margen_pct, 2), "servicios": total_services,
            "ticket_promedio": ticket, "lead_time_mediano_dias": lead_time, "hojas": hojas}

# =============== NUEVO: soporte DICCIONARIO + búsquedas de columnas por rol ===============
def build_role_index(diccionario: pd.DataFrame) -> Dict[str, Dict[str, Dict[str,str]]]:
    """ { hoja: { col: {rol, descripcion} } } — usa columnas: hoja, columna, rol, descripción """
    role_idx: Dict[str, Dict[str, Dict[str,str]]] = {}
    if diccionario is None or diccionario.empty:
        return role_idx
    cols = { _norm(c): c for c in diccionario.columns }
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None
    hoja_c = pick("hoja")
    col_c  = pick("columna")
    rol_c  = pick("rol")
    des_c  = pick("descripcion","descripción")
    for _, row in diccionario.iterrows():
        hoja = str(row.get(hoja_c,"")).strip()
        col  = str(row.get(col_c,"")).strip()
        rol  = str(row.get(rol_c,"")).strip()
        des  = str(row.get(des_c,"")).strip() if des_c else ""
        if hoja and col:
            role_idx.setdefault(hoja, {})[col] = {"rol": _norm(rol), "descripcion": des}
    return role_idx

def _first_col_by_role(df: pd.DataFrame, role_idx: Dict[str, Dict[str, Dict[str,str]]], hoja: str,
                       role_substrings: List[str], fallback_hints: List[str]) -> Optional[str]:
    # 1) por DICCIONARIO
    for col, meta in role_idx.get(hoja, {}).items():
        r = _norm(meta.get("rol",""))
        if any(rs in r for rs in role_substrings):
            if col in df.columns:
                return col
    # 2) heurística por nombre
    for c in df.columns:
        cc = _norm(c)
        if any(h in cc for h in fallback_hints):
            return c
    return None

# =============== NUEVO: funciones determinísticas para fechas =============================
def entregas_status(data: Dict[str, pd.DataFrame], dic_idx: Dict[str, Dict[str, Dict[str,str]]]) -> pd.DataFrame:
    filas = []
    for hoja, df in data.items():
        if df is None or df.empty: continue
        fecha_col = _first_col_by_role(df, dic_idx, hoja, ["entrega","salida","egreso"], ENTREGA_HINTS)
        id_col    = _first_col_by_role(df, dic_idx, hoja, ["id","orden","ot","folio","patente"], ID_HINTS)
        if not fecha_col or not id_col: continue
        fechas = _to_local_datetime(df[fecha_col]).dt.date
        dias = (pd.Series(HOY, index=fechas.index) - fechas).dt.days
        estado = dias.apply(lambda d: "pendiente (futuro)" if pd.notna(d) and d < 0 else ("cumplida" if pd.notna(d) else "sin fecha"))
        tmp = pd.DataFrame({
            "hoja": hoja,
            "id": df[id_col].astype(str),
            "fecha_entrega": fechas,
            "dias_atraso": dias,
            "estado": estado
        })
        filas.append(tmp)
    return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame(columns=["hoja","id","fecha_entrega","dias_atraso","estado"])

def facturas_pendientes(data: Dict[str, pd.DataFrame], dic_idx: Dict[str, Dict[str, Dict[str,str]]]) -> pd.DataFrame:
    filas = []
    for hoja, df in data.items():
        if df is None or df.empty: continue
        pago_col = _first_col_by_role(df, dic_idx, hoja, ["pago","venc"], PAGO_HINTS)
        id_col   = _first_col_by_role(df, dic_idx, hoja, ["factura","id","folio","documento"], ID_HINTS)
        if not pago_col or not id_col: continue
        fechas = _to_local_datetime(df[pago_col]).dt.date
        pendientes = df[fechas > HOY]
        if not pendientes.empty:
            tmp = pendientes[[id_col]].copy()
            tmp["hoja"] = hoja
            tmp["fecha_pago"] = fechas.loc[pendientes.index]
            filas.append(tmp.rename(columns={id_col: "id"}))
    return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame(columns=["id","hoja","fecha_pago"])


