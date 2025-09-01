from __future__ import annotations
import pandas as pd
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

_TZ = ZoneInfo("America/Santiago")
HOY = datetime.now(_TZ).date()

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

# --- Hints ---
ING_HINTS = ["monto","neto","total","importe","facturacion","facturación","ingreso","venta","principal"]
COST_HINTS = ["costo","costos","gasto","gastos","insumo","insumos","repuesto","repuestos","pintura","material","materiales","mano de obra","mo","imposicion","imposiciones","arriendo","admin","administracion","administración","equipo de planta","generales","financiero","financieros","interes","intereses"]
ID_HINTS = ["id","folio","factura","documento","nro","número","num","ot","orden","oc","patente","presupuesto"]
ENTREGA_HINTS = ["fecha entrega","entrega","salida","egreso","término","termino","compromiso"]
RECEPCION_HINTS = ["fecha rece", "recepción", "recepcion", "ingreso", "entrada"]
SERVICIO_ESTADO_HINTS = ["estado servicio","servicio","estado"]
PRESUP_ESTADO_HINTS = ["estado presupuesto","presupuesto","aprob","enviado","ganado","perdido"]
PAGO_HINTS = ["fecha pago","pago","venc","vencimiento","fecha venc","fecha de pago"]
FACTURA_HINTS = ["factura","n° factura","nro factura","folio","doc"]

def _find_numeric_cols(df: pd.DataFrame, keywords: List[str]):
    cols = []
    for c in df.columns:
        c2 = _norm(c)
        if any(k in c2 for k in keywords):
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0:
                cols.append(c)
    return cols

def _first_col_by_role_or_hint(df: pd.DataFrame, role_idx, hoja: str, role_subs: List[str], hints: List[str]) -> Optional[str]:
    # Prefer dictionary
    for col, meta in role_idx.get(hoja, {}).items():
        r = _norm(meta.get("rol",""))
        if any(rs in r for rs in role_subs):
            if col in df.columns: return col
    # Fallback by header
    for c in df.columns:
        cc = _norm(c)
        if any(h in cc for h in hints):
            return c
    return None

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

def analizar_datos_taller(data: Dict[str, pd.DataFrame], cliente_contiene: str = "") -> Dict[str, Any]:
    total_ing = 0.0; total_cost = 0.0; total_services = 0; lead_time=None
    hojas = {}
    for hoja, df in (data or {}).items():
        if df is None or df.empty: continue
        ing_cols  = _find_numeric_cols(df, ING_HINTS)
        cost_cols = _find_numeric_cols(df, COST_HINTS)
        ing_sum  = float(pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in ing_cols }).sum().sum()) if ing_cols else 0.0
        cost_sum = float(pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in cost_cols}).sum().sum()) if cost_cols else 0.0
        total_ing += ing_sum; total_cost += cost_sum; total_services += _count_services(df)
        lt = _lead_time_days(df); 
        if lt is not None: lead_time = lt
        hojas[hoja] = {"filas": int(len(df)), "ing_cols": ing_cols, "cost_cols": cost_cols,
                       "ingresos_hoja": ing_sum, "costos_hoja": cost_sum}
    margen = total_ing - total_cost
    margen_pct = (margen / total_ing * 100.0) if total_ing else 0.0
    ticket = (total_ing / total_services) if total_services else None
    return {"fecha_actual": HOY.strftime("%d/%m/%Y"), "ingresos": total_ing, "costos": total_cost,
            "margen": margen, "margen_pct": round(margen_pct,2), "servicios": total_services,
            "ticket_promedio": ticket, "lead_time_mediano_dias": lead_time, "hojas": hojas}

def build_role_index(diccionario: pd.DataFrame):
    role_idx = {}
    if diccionario is None or diccionario.empty:
        return role_idx
    cols = { _norm(c): c for c in diccionario.columns }
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None
    hoja_c = pick("hoja"); col_c = pick("columna"); rol_c = pick("rol"); des_c = pick("descripcion","descripción")
    for _, row in diccionario.iterrows():
        hoja = str(row.get(hoja_c,"")).strip(); col = str(row.get(col_c,"")).strip()
        rol  = str(row.get(rol_c,"")).strip(); des = str(row.get(des_c,"")).strip() if des_c else ""
        if hoja and col: role_idx.setdefault(hoja, {})[col] = {"rol": _norm(rol), "descripcion": des}
    return role_idx

def _collect_invoice_keys(data: Dict[str,pd.DataFrame], role_idx) -> set:
    keys = set()
    for hoja, df in (data or {}).items():
        if df is None or df.empty: continue
        has_fact = any(h in _norm(c) for c in df.columns for h in FACTURA_HINTS)
        if not has_fact:
            for col, meta in role_idx.get(hoja, {}).items():
                if "fact" in _norm(meta.get("rol","")): has_fact = True; break
        if not has_fact: continue
        id_col = _first_col_by_role_or_hint(df, role_idx, hoja, ["ot","id","orden","patente"], ID_HINTS)
        if not id_col: continue
        keys.update(df[id_col].astype(str).str.strip().str.upper().tolist())
    return keys

def _guess_key_cols(df: pd.DataFrame, role_idx, hoja: str) -> Optional[str]:
    return _first_col_by_role_or_hint(df, role_idx, hoja, ["ot","id","orden","patente"], ID_HINTS)

def entregas_status(data: Dict[str, pd.DataFrame], dic_idx):
    filas = []
    for hoja, df in (data or {}).items():
        if df is None or df.empty: continue
        fecha_col = _first_col_by_role_or_hint(df, dic_idx, hoja, ["entrega","salida","egreso"], ENTREGA_HINTS)
        id_col    = _guess_key_cols(df, dic_idx, hoja)
        if not fecha_col or not id_col: continue
        fechas = _to_local_datetime(df[fecha_col]).dt.date
        dias = (pd.Series(HOY, index=fechas.index) - fechas).dt.days
        estado = dias.apply(lambda d: "pendiente (futuro)" if pd.notna(d) and d < 0 else ("cumplida" if pd.notna(d) else "sin fecha"))
        filas.append(pd.DataFrame({"hoja": hoja, "id": df[id_col].astype(str), "fecha_entrega": fechas, "dias_atraso": dias, "estado": estado}))
    return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame(columns=["hoja","id","fecha_entrega","dias_atraso","estado"])

def facturas_pendientes(data: Dict[str, pd.DataFrame], dic_idx):
    filas = []
    for hoja, df in (data or {}).items():
        if df is None or df.empty: continue
        pago_col = _first_col_by_role_or_hint(df, dic_idx, hoja, ["pago","venc"], PAGO_HINTS)
        id_col   = _first_col_by_role_or_hint(df, dic_idx, hoja, ["factura","id","folio","documento"], ID_HINTS)
        if not pago_col or not id_col: continue
        fechas = _to_local_datetime(df[pago_col]).dt.date
        pendientes = df[fechas > HOY]
        if not pendientes.empty:
            tmp = pendientes[[id_col]].copy()
            tmp["hoja"] = hoja; tmp["fecha_pago"] = fechas.loc[pendientes.index]
            filas.append(tmp.rename(columns={id_col: "id"}))
    return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame(columns=["id","hoja","fecha_pago"])

def facturas_por_pagar_en_dias(data, dic_idx, dias=30):
    try:
        dias = int(dias)
    except Exception:
        dias = 30
    if dias < 1: dias = 1
    horizon = HOY + timedelta(days=dias)
    filas = []
    for hoja, df in (data or {}).items():
        if df is None or getattr(df, "empty", True): continue
        pago_col = _first_col_by_role_or_hint(df, dic_idx, hoja, ["pago","venc"], PAGO_HINTS)
        id_col   = _first_col_by_role_or_hint(df, dic_idx, hoja, ["factura","id","folio","documento"], ID_HINTS)
        if not pago_col or not id_col: continue
        fechas = _to_local_datetime(df[pago_col]).dt.date
        mask = (fechas > HOY) & (fechas <= horizon)
        sub = df[mask]
        if sub.empty: continue
        tmp = sub[[id_col]].copy()
        tmp["hoja"] = hoja; tmp["fecha_pago"] = fechas.loc[sub.index]
        filas.append(tmp.rename(columns={id_col: "id"}))
    return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame(columns=["id","hoja","fecha_pago"])

def entregados_no_facturados(data, dic_idx, days_back: int = 120):
    fact_keys = _collect_invoice_keys(data, dic_idx)
    filas = []
    min_date = HOY - timedelta(days=days_back) if days_back else None
    for hoja, df in (data or {}).items():
        if df is None or df.empty: continue
        fecha_col = _first_col_by_role_or_hint(df, dic_idx, hoja, ["entrega","salida","egreso"], ENTREGA_HINTS)
        id_col    = _guess_key_cols(df, dic_idx, hoja)
        if not fecha_col or not id_col: continue
        fechas = _to_local_datetime(df[fecha_col]).dt.date
        mask_base = fechas.notna() & (fechas <= HOY)
        if min_date is not None: mask_base &= (fechas >= min_date)
        ids = df[id_col].astype(str).str.strip().str.upper()
        mask_no_fact = ~ids.isin(fact_keys)
        sub = df[mask_base & mask_no_fact].copy()
        if sub.empty: continue
        tmp = pd.DataFrame({"hoja": hoja, "id": ids[mask_base & mask_no_fact], "fecha_entrega": fechas[mask_base & mask_no_fact]})
        filas.append(tmp)
    return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame(columns=["hoja","id","fecha_entrega"])

def en_taller_con_dias(data, dic_idx, top_n: int = 10):
    filas = []
    for hoja, df in (data or {}).items():
        if df is None or df.empty: continue
        rec_col   = _first_col_by_role_or_hint(df, dic_idx, hoja, ["recepcion","ingreso","entrada"], RECEPCION_HINTS)
        ent_col   = _first_col_by_role_or_hint(df, dic_idx, hoja, ["entrega","salida","egreso"], ENTREGA_HINTS)
        estado_sv = _first_col_by_role_or_hint(df, dic_idx, hoja, ["estado servicio","servicio"], SERVICIO_ESTADO_HINTS)
        id_col    = _guess_key_cols(df, dic_idx, hoja)
        if not rec_col or not id_col: continue
        rec = _to_local_datetime(df[rec_col]).dt.date
        entregada = _to_local_datetime(df[ent_col]).dt.date if ent_col else pd.Series([pd.NaT]*len(df), index=df.index)
        mask_no_entrega = entregada.isna()
        if estado_sv and estado_sv in df.columns:
            mask_no_entrega |= ~df[estado_sv].astype(str).str.lower().str.contains("entreg", na=False)
        dias = (pd.Series(HOY, index=rec.index) - rec).dt.days
        sub = pd.DataFrame({"hoja": hoja, "id": df[id_col].astype(str), "fecha_recepcion": rec, "dias_en_taller": dias})[mask_no_entrega]
        sub = sub[sub["fecha_recepcion"].notna()]
        if not sub.empty: filas.append(sub)
    out = pd.concat(filas, ignore_index=True) if filas else pd.DataFrame(columns=["hoja","id","fecha_recepcion","dias_en_taller"])
    if out.empty: return out
    return out.sort_values("dias_en_taller", ascending=False).head(top_n)

def entregas_proximos_dias_sin_factura(data, dic_idx, dias: int = 30):
    fact_keys = _collect_invoice_keys(data, dic_idx)
    filas = []
    horizon = HOY + timedelta(days=max(1,int(dias or 1)))
    for hoja, df in (data or {}).items():
        if df is None or df.empty: continue
        fecha_col = _first_col_by_role_or_hint(df, dic_idx, hoja, ["entrega","salida","egreso"], ENTREGA_HINTS)
        id_col    = _guess_key_cols(df, dic_idx, hoja)
        if not fecha_col or not id_col: continue
        fechas = _to_local_datetime(df[fecha_col]).dt.date
        ids = df[id_col].astype(str).str.strip().str.upper()
        mask = (fechas > HOY) & (fechas <= horizon) & (~ids.isin(fact_keys))
        sub = pd.DataFrame({"hoja": hoja, "id": ids[mask], "fecha_entrega": fechas[mask], "dias_para_entrega": (fechas[mask] - pd.Series(HOY, index=fechas.index)[mask]).dt.days})
        if not sub.empty: filas.append(sub)
    out = pd.concat(filas, ignore_index=True) if filas else pd.DataFrame(columns=["hoja","id","fecha_entrega","dias_para_entrega"])
    if out.empty: return out
    return out.sort_values(["fecha_entrega","id"])

def sin_aprobacion_presupuesto(data, dic_idx):
    fact_keys = _collect_invoice_keys(data, dic_idx)
    filas = []
    for hoja, df in (data or {}).items():
        if df is None or df.empty: continue
        estado_p_col = _first_col_by_role_or_hint(df, dic_idx, hoja, ["presupuesto","aprob"], PRESUP_ESTADO_HINTS)
        id_col       = _guess_key_cols(df, dic_idx, hoja)
        if not estado_p_col or not id_col: continue
        estados = df[estado_p_col].astype(str).str.lower()
        ids = df[id_col].astype(str).str.strip().str.upper()
        mask_enviado = estados.str.contains("envi", na=False)
        mask_no_gan = ~estados.str.contains("ganad", na=False)
        mask_no_per = ~estados.str.contains("perdid", na=False)
        mask_no_fact = ~ids.isin(fact_keys)
        sub = df[mask_enviado & mask_no_gan & mask_no_per & mask_no_fact]
        if not sub.empty:
            filas.append(pd.DataFrame({"hoja": hoja, "id": ids.loc[sub.index], "estado_presupuesto": estados.loc[sub.index]}))
    return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame(columns=["hoja","id","estado_presupuesto"])

