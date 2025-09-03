import re, json, datetime as dt
from typing import Dict, Any, List

# ===============================
# Planificador por reglas (determinístico)
# ===============================

MESES = {
    "enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,
    "julio":7,"agosto":8,"septiembre":9,"setiembre":9,"octubre":10,"noviembre":11,"diciembre":12
}

def _norm(s:str)->str:
    return re.sub(r"\s+"," ",str(s or "")).strip().lower()

def _roles(schema:Dict[str,Any], hoja:str)->Dict[str,str]:
    return (schema.get(hoja) or {}).get("roles") or {}

def _cols(schema:Dict[str,Any], hoja:str)->List[str]:
    return (schema.get(hoja) or {}).get("columns") or []

def _pick_date_col(schema, hoja)->str:
    roles = _roles(schema, hoja)
    for c,r in roles.items():
        if r=="date": return c
    return ""

def _pick_money_col(schema, hoja)->str:
    roles = _roles(schema, hoja)
    prefer = ["MONTO NETO","Monto Neto","Neto","TOTAL","Total","Importe","Valor","UTILIDAD","Utilidad","Ingresos","Costos"]
    for p in prefer:
        for c,r in roles.items():
            if r=="money" and _norm(c)==_norm(p): return c
    for c,r in roles.items():
        if r=="money": return c
    return ""

def _pick_id_col(schema, hoja)->str:
    roles = _roles(schema, hoja)
    prefer = ["OT","N° OT","NUMERO DE FACTURA","Número De Factura","PATENTE"]
    for p in prefer:
        for c,r in roles.items():
            if r=="id" and _norm(c)==_norm(p): return c
    for c,r in roles.items():
        if r=="id": return c
    return ""

def _sheet_from_question(q:str)->str:
    qn = _norm(q)
    if any(w in qn for w in ["monto","ingreso","ingresos","costo","costos","utilidad","margen","factura","facturación","pago","pagos"]):
        return "FINANZAS"
    return "MODELO_BOT"

def _op_from_question(q:str, value_role:str)->str:
    qn = _norm(q)
    if any(w in qn for w in ["promedio","media","promedi","avg"]): return "avg"
    if any(w in qn for w in ["máximo","maximo","mayor","peak","tope"]): return "max"
    if any(w in qn for w in ["mínimo","minimo","menor"]): return "min"
    if any(w in qn for w in ["cuántas","cuantos","número de","cantidad de","count","cuenta"]) or value_role=="id":
        return "count"
    return "sum" if value_role!="percent" else "avg"

def _group_by_from_question(q:str)->str:
    qn = _norm(q)
    if re.search(r"\bpor\s+mes(es)?\b", qn) or "mensual" in qn: return "month"
    if re.search(r"\bpor\s+año(s)?\b", qn) or "anual" in qn: return "year"
    return "none"

def _category_from_question(q:str, schema:Dict[str,Any], hoja:str)->str:
    qn = _norm(q)
    m = re.search(r"\bpor\s+([a-záéíóúñ\s]+)$", qn)
    if m:
        target = m.group(1).strip()
        cols = _cols(schema, hoja)
        tgt = _norm(target)
        for c in cols:
            if tgt in _norm(c):
                return c
    hints = ["tipo cliente","cliente","marca","modelo","proceso","estado servicio","estado presupuesto","patente","ot"]
    for h in hints:
        if h in qn:
            tokens = [t for t in h.split() if t]
            cols = _cols(schema, hoja)
            for c in cols:
                nc = _norm(c)
                if all(t in nc for t in tokens):
                    return c
    return ""

def _date_filters(q:str, schema:Dict[str,Any], hoja:str)->List[Dict[str,str]]:
    qn = _norm(q)
    col = _pick_date_col(schema, hoja)
    if not col: return []
    out = []
    # Año explícito
    m = re.search(r"\b(20\d{2})\b", qn)
    if m:
        y = int(m.group(1))
        out.append({"col":col,"op":"gte","val":f"{y}-01-01"})
        out.append({"col":col,"op":"lte","val":f"{y}-12-31"})
    # Mes explícito (con o sin año)
    for mes, num in MESES.items():
        if mes in qn:
            ym = re.search(rf"{mes}\s+de?\s*(20\d{{2}})?", qn)
            y = int(ym.group(1)) if (ym and ym.group(1)) else dt.date.today().year
            d1 = dt.date(y, num, 1)
            d2 = dt.date(y, 12, 31) if num==12 else (dt.date(y, num+1, 1) - dt.timedelta(days=1))
            out.append({"col":col,"op":"gte","val":str(d1)})
            out.append({"col":col,"op":"lte","val":str(d2)})
            break
    # Últimos N meses
    m2 = re.search(r"últim[oa]s?\s+(\d{1,2})\s+mes", qn)
    if m2:
        n = int(m2.group(1))
        today = dt.date.today()
        month_back = today.month - n
        year = today.year
        while month_back <= 0:
            month_back += 12; year -= 1
        d1 = dt.date(year, month_back, 1)
        out.append({"col":col,"op":"gte","val":str(d1)})
    return out

def plan_from_rules(pregunta:str, schema:Dict[str,Any]) -> Dict[str,Any]:
    hoja = _sheet_from_question(pregunta)
    roles = _roles(schema, hoja)
    # elegir columna de valor
    value_col = _pick_money_col(schema, hoja) if hoja=="FINANZAS" else _pick_id_col(schema, hoja)
    value_role = roles.get(value_col, "money" if hoja=="FINANZAS" else "id")
    # si pregunta por facturas/OT, favorecer id incluso en FINANZAS
    if any(w in _norm(pregunta) for w in ["factura","facturas","ot","órdenes","ordenes","orden de trabajo"]):
        cand = _pick_id_col(schema, hoja)
        if cand: value_col, value_role = cand, "id"
    op = _op_from_question(pregunta, value_role)
    category_col = _category_from_question(pregunta, schema, hoja)
    group_by = _group_by_from_question(pregunta)
    filters = _date_filters(pregunta, schema, hoja)
    return {
        "sheet": hoja,
        "value_col": value_col or "",
        "category_col": category_col or "",
        "op": op,
        "filters": filters,
        "group_by": group_by
    }
