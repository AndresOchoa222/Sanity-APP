# app_v2.py — CAGE Code Dashboard (interactivo, v2)
# Ejecuta: streamlit run app_v2.py
import io, json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="CAGE Code Dashboard v2", layout="wide")

st.title("📊 CAGE Code Dashboard — v2 (más interactivo)")

with st.sidebar:
    st.header("1) Datos")
    up = st.file_uploader("Sube tu archivo (.csv o .xlsx)", type=["csv","xlsx","xls"])

    st.markdown("---")
    st.header("Ajustes")
    heatmap_palette = st.selectbox("Paleta del heatmap", [
        "YlOrRd","Viridis","Cividis","Plasma","Inferno","Magma","Blues","Greens","Reds","Oranges"
    ], index=0)
    usar_montos = st.toggle("Graficar por monto (si existe la columna)", value=False)

st.caption("Mapea columnas, aplica filtros, revisa KPIs, explora por país y baja resúmenes.")

def _read_any(file):
    name = file.name.lower()
    if name.endswith((".xlsx",".xls")):
        return pd.read_excel(file)
    # CSV con autodetección de separador
    raw = file.read().decode("utf-8", errors="ignore")
    sample = io.StringIO(raw)
    peek = sample.read(4096); sample.seek(0)
    sep_candidates = [",",";","|","\t"]
    sep = max(sep_candidates, key=lambda s: peek.count(s))
    try:
        df = pd.read_csv(sample, sep=sep, engine="c")
    except Exception:
        sample.seek(0)
        df = pd.read_csv(sample, sep=sep, engine="python")
    return df

@st.cache_data(show_spinner=False)
def load_df(file):
    df = _read_any(file)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def coerce_numeric(s):
    if s.dtype == "object":
        cleaned = s.str.replace(r"[.\s]", "", regex=True).str.replace(",", ".", regex=False)
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(s, errors="coerce")

if up is None:
    st.info("⬅️ Sube un archivo para continuar.")
    st.stop()

df = load_df(up).copy()
st.success(f"Datos cargados: {len(df):,} filas · {df.shape[1]} columnas")

# ====== 2) Mapeo de columnas ======
with st.sidebar:
    st.header("2) Mapeo de columnas")
    colnames = ["—"] + list(df.columns)
    cage_col   = st.selectbox("Cage Code / País (numérico)", colnames, index=(colnames.index("Cage Code") if "Cage Code" in df.columns else 0))
    method_col = st.selectbox("Payment Method / Método", colnames, index=(colnames.index("Payment Method") if "Payment Method" in df.columns else 0))
    status_col = st.selectbox("Status / Estado", colnames, index=(colnames.index("Status") if "Status" in df.columns else 0))
    date_col   = st.selectbox("Fecha (Create Time Minute)", colnames, index=(colnames.index("Create Time Minute") if "Create Time Minute" in df.columns else 0))
    qty_col    = st.selectbox("Cantidad de trx", colnames, index=(colnames.index("Cantidad de trx") if "Cantidad de trx" in df.columns else 0))
    amt_col    = st.selectbox("Total Amount / Monto", colnames, index=(colnames.index("Total Amount") if "Total Amount" in df.columns else 0))

# Validaciones mínimas
req = [cage_col, method_col, status_col]
if any(c == "—" for c in req):
    st.error("Selecciona al menos: Cage Code, Método y Status.")
    st.stop()

# Tipos
if date_col != "—":
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

if qty_col != "—":
    df[qty_col] = coerce_numeric(df[qty_col])

if amt_col != "—":
    df[amt_col] = coerce_numeric(df[amt_col])

for c in [cage_col, method_col, status_col]:
    df[c] = df[c].astype(str).str.strip()

# Código de país a numérico
df["_country_code_num"] = pd.to_numeric(df[cage_col], errors="coerce").astype("Int64")

# ====== 3) Filtros ======
with st.sidebar:
    st.header("3) Filtros")
    codes_available = sorted([int(x) for x in df["_country_code_num"].dropna().unique()])
    selected_codes = st.multiselect("Códigos (CAGE/País)", options=codes_available, default=codes_available[:3])
    has_date = date_col != "—" and pd.api.types.is_datetime64_any_dtype(df[date_col])
    # Rango de fechas
    if has_date:
        min_d = pd.to_datetime(df[date_col].min()).date()
        max_d = pd.to_datetime(df[date_col].max()).date()
        dr = st.date_input("Rango de fechas", (min_d, max_d))
        if isinstance(dr, tuple) and len(dr)==2:
            df = df[(df[date_col].dt.date >= dr[0]) & (df[date_col].dt.date <= dr[1])]

    # Estado y método
    statuses = sorted(df[status_col].dropna().unique())
    methods  = sorted(df[method_col].dropna().unique())
    sel_status = st.multiselect("Estados", options=statuses, default=statuses)
    sel_methods = st.multiselect("Métodos de pago", options=methods, default=methods)

    df = df[df[status_col].isin(sel_status) & df[method_col].isin(sel_methods)]

# Agregación
agg_col = None
if usar_montos and amt_col != "—":
    agg_col = amt_col
elif qty_col != "—":
    agg_col = qty_col

def agg_series(group: pd.DataFrame):
    if agg_col and agg_col in group.columns:
        return group[agg_col].sum(min_count=1)
    return float(len(group))

def fmt_num(v, is_amount=False):
    try:
        if is_amount:
            return f"{float(v):,.0f}"
        return f"{int(v):,}"
    except Exception:
        return str(v)

COUNTRY_LABEL = {57:"Colombia", 52:"México", 51:"Perú"}

# ====== 4) KPIs globales ======
total_val = df[agg_col].sum(min_count=1) if agg_col else float(len(df))
unique_methods = df[method_col].nunique()
unique_status  = df[status_col].nunique()
date_span = ""
if has_date and not df.empty:
    date_span = f"{pd.to_datetime(df[date_col].min()).date()} → {pd.to_datetime(df[date_col].max()).date()}"

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total " + ("Monto" if agg_col==amt_col else "Transacciones"), fmt_num(total_val, agg_col==amt_col))
k2.metric("Métodos únicos", unique_methods)
k3.metric("Estados únicos", unique_status)
k4.metric("Rango de fechas", date_span if date_span else "—")

st.markdown("### Datos filtrados")
st.dataframe(df.head(300), use_container_width=True)

# Export de filtros/preset
preset = {
    "usar_montos": usar_montos,
    "heatmap_palette": heatmap_palette,
    "selected_codes": selected_codes,
    "sel_status": sel_status,
    "sel_methods": sel_methods,
    "date_enabled": has_date,
}
st.download_button("⬇️ Descargar preset (JSON)", data=json.dumps(preset).encode("utf-8"),
                   file_name="preset_dashboard.json", mime="application/json")

# ====== 5) Tabs por código + 3 gráficas ======
if not selected_codes:
    st.warning("Selecciona al menos un código.")
    st.stop()

tabs = st.tabs([f"{COUNTRY_LABEL.get(c, 'Código')} · {c}" for c in selected_codes])

for code, tab in zip(selected_codes, tabs):
    with tab:
        sub = df[df["_country_code_num"] == code].copy()
        st.subheader(f"{COUNTRY_LABEL.get(code, 'Código')} ({code})")
        if sub.empty:
            st.info("No hay datos para este código.")
            continue

        # Serie temporal por estado
        if has_date:
            tmp = sub[[date_col, status_col] + ([agg_col] if agg_col else [])].copy()
            tmp["date"] = tmp[date_col].dt.floor("D")
            grp = tmp.groupby(["date", status_col]).apply(agg_series).reset_index(name="value")
            if not grp.empty:
                linefig = px.line(grp, x="date", y="value", color=status_col, markers=True)
                linefig.update_layout(
                    title="Flujo por estado",
                    xaxis_title="Fecha",
                    yaxis_title=("Monto" if agg_col==amt_col else "Transacciones"),
                    legend_title="Estado",
                    hovermode="x unified",
                    margin=dict(l=10,r=10,t=60,b=10),
                )
                st.plotly_chart(linefig, use_container_width=True)
            else:
                st.info("Sin fechas válidas para graficar serie.")
        else:
            st.info("No se detectó una columna de fecha válida.")

        # Barras apiladas método × estado
        if agg_col:
            ct = sub.groupby([method_col, status_col])[agg_col].sum().reset_index()
        else:
            ct = sub.groupby([method_col, status_col]).size().reset_index(name="value")
            agg_col_tmp = "value"
            ct.rename(columns={"value": agg_col_tmp}, inplace=True)
        ycol = agg_col if agg_col else agg_col_tmp

        barfig = px.bar(
            ct, x=method_col, y=ycol, color=status_col, barmode="stack",
            title="Métodos por estado",
        )
        barfig.update_layout(
            xaxis_title="Método de pago",
            yaxis_title=("Monto" if (agg_col and agg_col==amt_col) else "Transacciones"),
            legend_title="Estado",
            margin=dict(l=10,r=10,t=60,b=10),
        )
        st.plotly_chart(barfig, use_container_width=True)

        # Heatmap método × estado con anotaciones
        if agg_col:
            mat = sub.groupby([method_col, status_col])[agg_col].sum().unstack(status_col).fillna(0).astype(float)
        else:
            mat = sub.groupby([method_col, status_col]).size().unstack(status_col).fillna(0).astype(float)

        z = mat.values
        x = list(mat.columns.astype(str))
        y = list(mat.index.astype(str))
        text_ann = [[fmt_num(z[i][j], usar_montos and (amt_col != "—")) for j in range(len(x))] for i in range(len(y))]

        heat = go.Figure(data=go.Heatmap(
            z=z, x=x, y=y, colorscale=heatmap_palette, text=text_ann, texttemplate="%{text}",
            hovertemplate="Método=%{y}<br>Estado=%{x}<br>Valor=%{z}<extra></extra>"
        ))
        heat.update_layout(
            title="Heatmap métodos × estado",
            xaxis_title="Estado",
            yaxis_title="Método de pago",
            margin=dict(l=10,r=10,t=60,b=10),
        )
        st.plotly_chart(heat, use_container_width=True)

        # Top-N métodos (FIX aplicado)
        st.markdown("#### Top-N métodos por total")
        n_methods = sub[method_col].nunique()
        if n_methods == 0:
            st.info("No hay métodos para este código con los filtros actuales.")
        else:
            topn = st.slider("N", min_value=1, max_value=max(1, n_methods), value=min(10, n_methods))
            if agg_col:
                totals = sub.groupby(method_col)[agg_col].sum().sort_values(ascending=False).head(topn)
            else:
                totals = sub.groupby(method_col).size().sort_values(ascending=False).head(topn)
            totals_df = totals.rename_axis(method_col).reset_index(name="Total")
            st.dataframe(totals_df, use_container_width=True)

        # Drill-down detalle
        st.markdown("#### Detalle por método y estado")
        dd_cols = st.columns(2)
        sel_m = dd_cols[0].selectbox("Método", options=y)
        sel_s = dd_cols[1].selectbox("Estado", options=x)
        det = sub[(sub[method_col]==sel_m) & (sub[status_col]==sel_s)].copy()
        st.dataframe(det.head(500), use_container_width=True)
        st.download_button(
            "⬇️ Descargar detalle (CSV)",
            data=det.to_csv(index=False).encode("utf-8"),
            file_name=f"detalle_{code}_{sel_m}_{sel_s}.csv",
            mime="text/csv"
        )
