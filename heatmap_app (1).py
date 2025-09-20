from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Column definitions
DATA_FILE = "LATAM_SF player_payments_table 2025-09-19T2158.csv"
COL_COUNTRY = "Cage Code"
COL_METHOD = "Payment Method"
COL_STATUS = "Status"
COL_DATE = "Create Time Minute"
COL_TYPE = "Type"
COL_STATUS_TIME = "Status Time Minute"

# Labels and ordering
COUNTRY_LABEL = {57: "Colombia", 52: "México", 51: "Perú"}
COUNTRY_CODES: tuple[int, ...] = tuple(COUNTRY_LABEL)
TYPE_ORDER = ("DEPOSIT", "WITHDRAWAL")
TYPE_CMAP = {"DEPOSIT": "BuGn", "WITHDRAWAL": "PuRd"}
SEPARATORS = (",", ";", "|", "\t")

# Matplotlib configuration
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 11,
    }
)


def read_table(path: str | Path) -> pd.DataFrame:
    """Load CSV/Excel files detecting separator automatically from disk path."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    # Detect separator for CSV
    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        head = handle.read(4096)
    separator = max(SEPARATORS, key=head.count)
    try:
        return pd.read_csv(file_path, sep=separator, engine="c", encoding="utf-8")
    except Exception:
        return pd.read_csv(file_path, sep=separator, engine="python", encoding="utf-8")


def read_uploaded_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Load an uploaded file provided as bytes.

    Parameters
    ----------
    file_bytes : bytes
        Raw bytes of the uploaded file.
    filename : str
        Original filename, used to infer extension.

    Returns
    -------
    pd.DataFrame
        Parsed DataFrame from the uploaded file.
    """
    ext = Path(filename).suffix.lower()
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(pd.io.common.BytesIO(file_bytes))
    # CSV: detect separator by sampling text
    text = file_bytes.decode("utf-8", errors="ignore")
    sample = text[:4096]
    sep = max(SEPARATORS, key=sample.count)
    return pd.read_csv(pd.io.common.StringIO(text), sep=sep, engine="python")


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize key columns and convert dates to datetime."""
    df = df.rename(columns=lambda column: str(column).strip()).copy()
    df[COL_DATE] = pd.to_datetime(df.get(COL_DATE), errors="coerce")
    df[COL_STATUS_TIME] = pd.to_datetime(df.get(COL_STATUS_TIME), errors="coerce")
    df["_country_code_num"] = pd.to_numeric(df[COL_COUNTRY], errors="coerce").astype("Int64")
    df[COL_TYPE] = (
        df[COL_TYPE]
        .astype(str)
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
        .replace({"DEPOSITO": "DEPOSIT", "WITHDRAW": "WITHDRAWAL"})
    )
    return df


def _oldest_in_progress(per_method: pd.DataFrame) -> Dict[str, str]:
    """Return the oldest IN_PROGRESS date/time per method, formatted without seconds."""
    in_progress = per_method.loc[per_method[COL_STATUS] == "IN_PROGRESS", [COL_METHOD, COL_STATUS_TIME]]
    if in_progress.empty:
        return {}
    formatted = (
        in_progress.groupby(COL_METHOD)[COL_STATUS_TIME]
        .min()
        .dt.strftime("%Y-%m-%d %H:%M")
        .dropna()
    )
    return formatted.to_dict()


def _build_pivot(table: pd.DataFrame) -> pd.DataFrame:
    """Construct a pivot table of counts by method and status."""
    pivot = (
        table.groupby([COL_METHOD, COL_STATUS])
        .size()
        .unstack(COL_STATUS, fill_value=0)
        .astype(float)
    )
    return pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]


def plot_heatmap(data: pd.DataFrame, country_code: int, trx_type: str, *, show: bool = True) -> Optional[plt.Figure]:
    """
    Plot a heatmap (method vs status) and optionally display it.

    If `show` is False, the function returns the created figure without calling plt.show().
    If `show` is True, the figure is shown via plt.show() and None is returned.
    """
    per_type = data.loc[data[COL_TYPE] == trx_type]
    if per_type.empty:
        return None
    pivot = _build_pivot(per_type)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    annotations = _oldest_in_progress(per_type)
    n_methods = max(1, pivot.shape[0])
    fig, ax = plt.subplots(figsize=(8, max(6, n_methods * 0.4)))
    heat = ax.imshow(pivot.values, aspect="auto", cmap=TYPE_CMAP.get(trx_type, "BuGn"))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Estado")
    ax.set_ylabel("Método de pago")
    vmax = float(np.nanmax(pivot.values)) if pivot.size else 0.0
    threshold = vmax * 0.5 if vmax else 0.0
    for row, method in enumerate(pivot.index):
        for col, status in enumerate(pivot.columns):
            value = pivot.iat[row, col]
            label = f"{int(value):,}"
            if status == "IN_PROGRESS" and method in annotations:
                label = f"{label}\n{annotations[method]}"
            color = "white" if value >= threshold else "black"
            ax.text(col, row, label, ha="center", va="center", fontsize=9, weight="bold", color=color)
    dates = per_type[COL_DATE].dropna()
    if not dates.empty:
        period = f"{dates.min():%Y-%m-%d} → {dates.max():%Y-%m-%d}"
    else:
        period = "Sin fechas"
    country_label = COUNTRY_LABEL.get(country_code, f"Código {country_code}")
    ax.set_title(f"Heatmap {trx_type} - {country_label} ({country_code})\n{period}", weight="bold")
    fig.colorbar(heat, ax=ax, label="Cantidad de trx")
    plt.tight_layout()
    if show:
        plt.show()
        return None
    return fig


def plot_all_countries(df: pd.DataFrame, codes: Iterable[int] = COUNTRY_CODES) -> None:
    """Plot heatmaps for all specified country codes and transaction types."""
    for code in codes:
        subset = df.loc[df["_country_code_num"] == code]
        for trx_type in TYPE_ORDER:
            plot_heatmap(subset, code, trx_type, show=True)


def main() -> None:
    """Run the heatmap generator as a command-line script."""
    df = normalize_frame(read_table(DATA_FILE))
    plot_all_countries(df)


def run_app() -> None:

    import streamlit as st  # type: ignore
    st.title("Heatmaps de métodos de pago")
    st.write(
        "Carga un archivo CSV o Excel con tus transacciones. La aplicación "
        "generará mapas de calor por método de pago y estado, separados por "
        "país (código CAGE) y tipo de transacción."
    )
    uploaded_file = st.file_uploader("Sube tu archivo", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        try:
            df = read_uploaded_file(uploaded_file.getvalue(), uploaded_file.name)
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
            return
        df = normalize_frame(df)
        codes = sorted(df["_country_code_num"].dropna().unique())
        if not codes:
            st.warning("No se encontraron códigos de país en los datos.")
            return
        for code_int in codes:
            sub_df = df[df["_country_code_num"] == code_int]
            if sub_df.empty:
                continue
            label = COUNTRY_LABEL.get(code_int, f"Código {code_int}")
            st.header(f"Cage Code {code_int} - {label}")
            for trx_type in TYPE_ORDER:
                st.subheader(trx_type)
                fig = plot_heatmap(sub_df, code_int, trx_type, show=False)
                if fig is not None:
                    st.pyplot(fig)
                    st.write(
                        f"Mapa de calor para el tipo **{trx_type}** en el país {label}. "
                        f"Cada celda muestra el número de transacciones por método de pago y estado. "
                        "Si el estado es **IN_PROGRESS**, se muestra debajo la fecha y hora más antigua en "
                        "que se registró ese estado para ese método."
                    )
                    st.markdown("---")


if __name__ == "__main__":
  
    run_app()
