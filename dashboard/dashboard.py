from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="E-Commerce Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


COLOR_BG = "#0F172A"
COLOR_CARD = "#1E293B"
COLOR_BORDER = "#334155"
COLOR_TEXT = "#E2E8F0"
COLOR_SUBTEXT = "#94A3B8"
COLOR_PRIMARY = "#4E79A7"
COLOR_HIGHLIGHT = "#F28E2B"
COLOR_SUCCESS = "#34D399"
COLOR_WARNING = "#FBBF24"
COLOR_DANGER = "#F87171"

DELAY_ORDER = [
    "Lebih cepat",
    "Tepat waktu",
    "Terlambat 1-3 hari",
    "Terlambat 4-7 hari",
    "Terlambat >7 hari",
]

DELAY_COLOR_MAP = {
    "Lebih cepat": COLOR_SUCCESS,
    "Tepat waktu": COLOR_PRIMARY,
    "Terlambat 1-3 hari": COLOR_WARNING,
    "Terlambat 4-7 hari": "#FB923C",
    "Terlambat >7 hari": COLOR_DANGER,
}

BRAZIL_STATE_CENTROIDS = {
    "AC": (-8.77, -70.55),
    "AL": (-9.62, -36.82),
    "AM": (-3.47, -65.10),
    "AP": (1.41, -51.77),
    "BA": (-12.96, -38.51),
    "CE": (-3.71, -38.54),
    "DF": (-15.83, -47.86),
    "ES": (-20.32, -40.34),
    "GO": (-16.64, -49.31),
    "MA": (-2.53, -44.30),
    "MG": (-19.92, -43.94),
    "MS": (-20.45, -54.61),
    "MT": (-15.60, -56.10),
    "PA": (-1.45, -48.49),
    "PB": (-7.12, -34.86),
    "PE": (-8.05, -34.90),
    "PI": (-5.09, -42.80),
    "PR": (-25.42, -49.27),
    "RJ": (-22.90, -43.20),
    "RN": (-5.79, -35.21),
    "RO": (-8.76, -63.90),
    "RR": (2.82, -60.67),
    "RS": (-30.03, -51.23),
    "SC": (-27.59, -48.55),
    "SE": (-10.90, -37.07),
    "SP": (-23.55, -46.63),
    "TO": (-10.25, -48.25),
}

PLOT_CONFIG = {
    "displayModeBar": True,
    "responsive": True,
    "scrollZoom": True,
}


def apply_style() -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: linear-gradient(180deg, #0B1220 0%, {COLOR_BG} 100%);
                color: {COLOR_TEXT};
            }}

            .block-container {{
                padding-top: 2rem;
                padding-bottom: 2rem;
            }}

            section[data-testid="stSidebar"] {{
                background: #111827;
                border-right: 1px solid {COLOR_BORDER};
            }}

            section[data-testid="stSidebar"] * {{
                color: {COLOR_TEXT} !important;
            }}

            .hero-box {{
                background: rgba(30, 41, 59, 0.96);
                border: 1px solid {COLOR_BORDER};
                border-radius: 14px;
                padding: 1rem 1.1rem;
                margin-bottom: 1rem;
                color: {COLOR_TEXT};
            }}

            .guide-box {{
                background: rgba(78, 121, 167, 0.14);
                border: 1px solid rgba(78, 121, 167, 0.7);
                border-radius: 14px;
                padding: 1rem 1.1rem;
                margin-bottom: 1rem;
                color: {COLOR_TEXT};
            }}

            .insight-box {{
                background: rgba(30, 41, 59, 0.96);
                border: 1px solid {COLOR_BORDER};
                border-left: 4px solid {COLOR_HIGHLIGHT};
                border-radius: 14px;
                padding: 1rem 1.1rem;
                margin-bottom: 1rem;
                color: {COLOR_TEXT};
            }}

            div[data-testid="stMetric"] {{
                background: rgba(30, 41, 59, 0.96);
                border: 1px solid {COLOR_BORDER};
                border-radius: 14px;
                padding: 1rem;
                box-shadow: 0 6px 18px rgba(0, 0, 0, 0.18);
                min-height: 110px;
            }}

            div[data-testid="stMetricLabel"] {{
                color: {COLOR_SUBTEXT} !important;
            }}

            div[data-testid="stMetricLabel"] label {{
                color: {COLOR_SUBTEXT} !important;
                font-size: 0.92rem !important;
            }}

            div[data-testid="stMetricValue"] {{
                color: {COLOR_TEXT} !important;
                font-size: 1.85rem !important;
                line-height: 1.1 !important;
            }}

            div[data-testid="stMetricDelta"] {{
                color: {COLOR_HIGHLIGHT} !important;
            }}

            h1, h2, h3, h4 {{
                color: {COLOR_TEXT} !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_currency(value: float) -> str:
    if pd.isna(value):
        value = 0.0
    return f"R$ {float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_number(value: float) -> str:
    if pd.isna(value):
        value = 0
    return f"{int(value):,}".replace(",", ".")


def safe_value(df: pd.DataFrame, column: str, default):
    if df.empty or column not in df.columns:
        return default
    return df.iloc[0][column]


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def resolve_data_path() -> Path:
    dashboard_dir = Path(__file__).resolve().parent
    project_root = dashboard_dir.parent

    candidate_paths = [
        dashboard_dir / "main_data.csv",
        project_root / "dashboard" / "main_data.csv",
        project_root / "data" / "main_data.csv",
        project_root / "main_data.csv",
    ]

    for path in candidate_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        "File main_data.csv tidak ditemukan. "
        "Pastikan file tersimpan di folder dashboard atau project root."
    )


def detect_csv_separator(path: Path) -> str:
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        sample = file.read(4096)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def read_csv_safely(path: Path) -> Tuple[pd.DataFrame, str]:
    detected_separator = detect_csv_separator(path)
    df = pd.read_csv(path, sep=detected_separator, encoding="utf-8-sig")

    if len(df.columns) == 1:
        for separator in [",", ";", "\t", "|"]:
            retry_df = pd.read_csv(path, sep=separator, encoding="utf-8-sig")
            if len(retry_df.columns) > 1:
                return retry_df, separator

    return df, detected_separator


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = df.copy()
    normalized_df.columns = [
        str(col).strip().lower().replace(" ", "_") for col in normalized_df.columns
    ]
    return normalized_df


def parse_datetime_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def parse_locale_number(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if text == "" or text.lower() in {"nan", "none", "null"}:
        return np.nan

    text = text.replace(" ", "")

    if "." in text and "," in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")

    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_locale_numeric_series(series: pd.Series) -> pd.Series:
    return series.apply(parse_locale_number)


def normalize_boolean_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
            }
        )
        .fillna(False)
    )


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = normalize_columns(df)

    datetime_columns = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for column in datetime_columns:
        if column in normalized_df.columns:
            normalized_df[column] = parse_datetime_series(normalized_df[column])

    numeric_columns = [
        "review_score",
        "sales",
        "total_sales",
        "geolocation_lat",
        "geolocation_lng",
    ]
    for column in numeric_columns:
        if column in normalized_df.columns:
            normalized_df[column] = parse_locale_numeric_series(normalized_df[column])

    if "sales" not in normalized_df.columns and "total_sales" in normalized_df.columns:
        normalized_df["sales"] = normalized_df["total_sales"]

    if "purchase_month" not in normalized_df.columns and "order_purchase_timestamp" in normalized_df.columns:
        normalized_df["purchase_month"] = (
            normalized_df["order_purchase_timestamp"].dt.to_period("M").astype(str)
        )

    if "is_late" in normalized_df.columns:
        normalized_df["is_late"] = normalize_boolean_series(normalized_df["is_late"])
    else:
        normalized_df["is_late"] = False

    fill_map = {
        "customer_state": "Unknown",
        "customer_city": "Unknown",
        "product_category_name_english": "unknown",
        "delay_category": "Unknown",
    }
    for column, default_value in fill_map.items():
        if column in normalized_df.columns:
            normalized_df[column] = normalized_df[column].fillna(default_value)
        else:
            normalized_df[column] = default_value

    return normalized_df


@st.cache_data
def load_data(path: Path) -> Tuple[pd.DataFrame, str]:
    raw_df, separator = read_csv_safely(path)
    normalized_df = prepare_dataframe(raw_df)
    return normalized_df, separator


def validate_data(df: pd.DataFrame) -> None:
    required_columns = [
        "order_id",
        "customer_unique_id",
        "customer_state",
        "product_category_name_english",
        "review_score",
        "sales",
        "purchase_month",
        "is_late",
        "delay_category",
        "order_purchase_timestamp",
    ]

    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        st.error("Struktur file main_data.csv tidak sesuai dengan kebutuhan dashboard.")
        st.write("Kolom yang hilang:", missing_columns)
        st.write("Kolom yang terbaca:", df.columns.tolist())
        st.stop()


def build_order_df(item_df: pd.DataFrame) -> pd.DataFrame:
    aggregation_map = {
        "customer_unique_id": "first",
        "customer_state": "first",
        "customer_city": "first",
        "review_score": "first",
        "is_late": "first",
        "delay_category": "first",
        "order_purchase_timestamp": "first",
        "sales": "sum",
    }

    optional_columns = [
        "geolocation_lat",
        "geolocation_lng",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for column in optional_columns:
        if column in item_df.columns:
            aggregation_map[column] = "first"

    order_df = item_df.groupby("order_id", as_index=False).agg(aggregation_map)
    order_df["purchase_month"] = order_df["order_purchase_timestamp"].dt.to_period("M").astype(str)
    return order_df


def get_filter_options(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str], pd.Timestamp, pd.Timestamp]:
    states = sorted(df["customer_state"].dropna().astype(str).unique().tolist())
    categories = sorted(df["product_category_name_english"].dropna().astype(str).unique().tolist())
    min_date = df["order_purchase_timestamp"].min()
    max_date = df["order_purchase_timestamp"].max()
    return states, categories, min_date, max_date


def get_preset_range(
    preset: str,
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if preset == "Semua data":
        return min_date, max_date

    if preset == "90 hari terakhir":
        return max(min_date, max_date - pd.Timedelta(days=89)), max_date

    if preset == "180 hari terakhir":
        return max(min_date, max_date - pd.Timedelta(days=179)), max_date

    return min_date, max_date


def render_sidebar(df: pd.DataFrame, data_path: Path) -> Dict:
    states, categories, min_date, max_date = get_filter_options(df)

    st.sidebar.markdown("## Filter Dashboard")
    st.sidebar.caption(f"Sumber data: {data_path.name}")

    with st.sidebar.expander("Panduan penggunaan", expanded=True):
        st.markdown(
            """
            1. Pilih **preset periode** atau gunakan **tanggal kustom**.  
            2. Gunakan filter **state** dan **kategori** untuk eksplorasi lebih spesifik.  
            3. Ubah **Top N** untuk mengatur jumlah kategori/state yang tampil.  
            4. Gunakan tab untuk berpindah antara performa bisnis, pelanggan, dan geografis.  
            5. Unduh hasil filter pada bagian **Download data**.
            """
        )

    preset = st.sidebar.radio(
        "Preset periode",
        ["Semua data", "90 hari terakhir", "180 hari terakhir", "Kustom"],
        index=0,
    )

    if preset == "Kustom":
        date_value = st.sidebar.date_input(
            "Rentang tanggal transaksi",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )

        if isinstance(date_value, tuple) and len(date_value) == 2:
            start_date, end_date = date_value
        else:
            start_date = date_value
            end_date = date_value

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
    else:
        start_date, end_date = get_preset_range(preset, min_date, max_date)

    selected_states = st.sidebar.multiselect(
        "Filter state pelanggan",
        options=states,
        default=states,
    )

    selected_categories = st.sidebar.multiselect(
        "Filter kategori produk",
        options=categories,
        default=categories,
    )

    top_n = st.sidebar.slider(
        "Jumlah kategori/state yang ditampilkan",
        min_value=5,
        max_value=15,
        value=10,
        step=1,
    )

    map_mode = st.sidebar.selectbox(
        "Mode peta",
        ["Otomatis", "Koordinat pelanggan", "Centroid per state"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Periode data penuh: {min_date.date()} s.d. {max_date.date()}")
    st.sidebar.caption(f"State terpilih: {len(selected_states)}")
    st.sidebar.caption(f"Kategori terpilih: {len(selected_categories)}")

    return {
        "start_date": pd.to_datetime(start_date),
        "end_date": pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
        "states": selected_states,
        "categories": selected_categories,
        "top_n": top_n,
        "map_mode": map_mode,
        "preset": preset,
    }


def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    filtered_df = df.copy()

    filtered_df = filtered_df[
        (filtered_df["order_purchase_timestamp"] >= filters["start_date"])
        & (filtered_df["order_purchase_timestamp"] <= filters["end_date"])
    ]

    if filters["states"]:
        filtered_df = filtered_df[filtered_df["customer_state"].isin(filters["states"])]

    if filters["categories"]:
        filtered_df = filtered_df[
            filtered_df["product_category_name_english"].isin(filters["categories"])
        ]

    return filtered_df


def calculate_kpis(item_df: pd.DataFrame, order_df: pd.DataFrame) -> Dict[str, float]:
    total_revenue = item_df["sales"].sum()
    total_orders = order_df["order_id"].nunique()
    total_customers = order_df["customer_unique_id"].nunique()
    avg_order_value = total_revenue / total_orders if total_orders else 0.0
    avg_review = order_df["review_score"].mean() if "review_score" in order_df.columns else 0.0
    late_rate = order_df["is_late"].mean() * 100 if "is_late" in order_df.columns else 0.0

    return {
        "total_revenue": total_revenue,
        "total_orders": total_orders,
        "total_customers": total_customers,
        "avg_order_value": avg_order_value,
        "avg_review": avg_review,
        "late_rate": late_rate,
    }


def monthly_summary(order_df: pd.DataFrame) -> pd.DataFrame:
    return (
        order_df.groupby("purchase_month", as_index=False)
        .agg(
            total_orders=("order_id", "nunique"),
            total_revenue=("sales", "sum"),
        )
        .sort_values("purchase_month")
    )


def top_categories(item_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    return (
        item_df.groupby("product_category_name_english", as_index=False)
        .agg(
            total_revenue=("sales", "sum"),
            total_orders=("order_id", "nunique"),
            avg_review=("review_score", "mean"),
        )
        .sort_values("total_revenue", ascending=False)
        .head(top_n)
    )


def review_by_delay(order_df: pd.DataFrame) -> pd.DataFrame:
    review_df = (
        order_df.groupby("delay_category", as_index=False)
        .agg(
            avg_review=("review_score", "mean"),
            total_orders=("order_id", "nunique"),
        )
    )
    review_df = review_df[review_df["delay_category"].isin(DELAY_ORDER)].copy()
    review_df["delay_category"] = pd.Categorical(
        review_df["delay_category"],
        categories=DELAY_ORDER,
        ordered=True,
    )
    return review_df.sort_values("delay_category")


def state_summary(order_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if order_df.empty:
        return pd.DataFrame(
            columns=["customer_state", "total_revenue", "total_orders", "avg_review", "late_rate"]
        )

    summary_df = (
        order_df.groupby("customer_state", as_index=False)
        .agg(
            total_revenue=("sales", "sum"),
            total_orders=("order_id", "nunique"),
            avg_review=("review_score", "mean"),
            late_rate=("is_late", "mean"),
        )
        .sort_values("total_revenue", ascending=False)
        .head(top_n)
    )

    summary_df["late_rate"] = summary_df["late_rate"] * 100
    return summary_df


def score_by_quartile(series: pd.Series, higher_is_better: bool) -> pd.Series:
    valid_series = series.dropna()
    scores = pd.Series(1, index=series.index, dtype=int)

    if valid_series.empty:
        return scores

    percentile = valid_series.rank(
        method="average",
        pct=True,
        ascending=higher_is_better,
    )
    scored = np.ceil(percentile * 4).clip(1, 4).astype(int)
    scores.loc[valid_series.index] = scored
    return scores


def rfm_summary(order_df: pd.DataFrame) -> pd.DataFrame:
    if order_df.empty:
        return pd.DataFrame(columns=["segment", "total_customers", "avg_monetary"])

    latest_date = order_df["order_purchase_timestamp"].max()

    rfm_df = (
        order_df.groupby("customer_unique_id")
        .agg(
            last_purchase=("order_purchase_timestamp", "max"),
            frequency=("order_id", "nunique"),
            monetary=("sales", "sum"),
        )
        .reset_index()
    )

    rfm_df["recency"] = (latest_date - rfm_df["last_purchase"]).dt.days
    rfm_df["r_score"] = score_by_quartile(rfm_df["recency"], higher_is_better=False)
    rfm_df["f_score"] = score_by_quartile(rfm_df["frequency"], higher_is_better=True)
    rfm_df["m_score"] = score_by_quartile(rfm_df["monetary"], higher_is_better=True)

    def assign_segment(row: pd.Series) -> str:
        if row["r_score"] >= 4 and row["f_score"] >= 4 and row["m_score"] >= 4:
            return "Best Customers"
        if row["r_score"] >= 3 and row["f_score"] >= 3 and row["m_score"] >= 3:
            return "Loyal Customers"
        if row["r_score"] >= 3 and row["f_score"] <= 2:
            return "Recent Customers"
        if row["r_score"] <= 2 and row["f_score"] >= 3:
            return "At Risk"
        return "Others"

    rfm_df["segment"] = rfm_df.apply(assign_segment, axis=1)

    summary_df = (
        rfm_df.groupby("segment", as_index=False)
        .agg(
            total_customers=("customer_unique_id", "count"),
            avg_monetary=("monetary", "mean"),
        )
    )

    segment_order = ["Best Customers", "Loyal Customers", "Recent Customers", "At Risk", "Others"]
    summary_df["segment"] = pd.Categorical(
        summary_df["segment"],
        categories=segment_order,
        ordered=True,
    )

    return summary_df.sort_values("segment")


def spending_group_summary(order_df: pd.DataFrame) -> pd.DataFrame:
    if order_df.empty:
        return pd.DataFrame(columns=["spending_group", "total_customers", "total_revenue"])

    spending_df = (
        order_df.groupby("customer_unique_id", as_index=False)
        .agg(total_spending=("sales", "sum"))
    )

    spending_df["spending_group"] = pd.cut(
        spending_df["total_spending"],
        bins=[-np.inf, 100, 500, np.inf],
        labels=["Low Spender", "Medium Spender", "High Spender"],
    )

    return (
        spending_df.groupby("spending_group", as_index=False, observed=True)
        .agg(
            total_customers=("customer_unique_id", "count"),
            total_revenue=("total_spending", "sum"),
        )
    )


def build_state_fallback_map(order_df: pd.DataFrame) -> pd.DataFrame:
    fallback_df = (
        order_df.groupby("customer_state", as_index=False)
        .agg(
            total_revenue=("sales", "sum"),
            total_orders=("order_id", "nunique"),
        )
    )

    fallback_df["lat"] = fallback_df["customer_state"].map(
        lambda state: BRAZIL_STATE_CENTROIDS.get(state, (np.nan, np.nan))[0]
    )
    fallback_df["lon"] = fallback_df["customer_state"].map(
        lambda state: BRAZIL_STATE_CENTROIDS.get(state, (np.nan, np.nan))[1]
    )

    return fallback_df.dropna(subset=["lat", "lon"])


def build_granular_map(order_df: pd.DataFrame) -> pd.DataFrame:
    if not {"geolocation_lat", "geolocation_lng"}.issubset(order_df.columns):
        return pd.DataFrame()

    points_df = order_df.copy()
    points_df["geolocation_lat"] = parse_locale_numeric_series(points_df["geolocation_lat"])
    points_df["geolocation_lng"] = parse_locale_numeric_series(points_df["geolocation_lng"])

    points_df = points_df.dropna(subset=["geolocation_lat", "geolocation_lng"]).copy()

    if points_df.empty:
        return points_df

    points_df = points_df[
        points_df["geolocation_lat"].between(-35, 6)
        & points_df["geolocation_lng"].between(-75, -30)
    ]

    if len(points_df) > 3000:
        points_df = points_df.sample(3000, random_state=42)

    return points_df


def get_map_data(order_df: pd.DataFrame, map_mode: str) -> Tuple[pd.DataFrame, str]:
    granular_df = build_granular_map(order_df)
    fallback_df = build_state_fallback_map(order_df)

    if map_mode == "Koordinat pelanggan":
        if not granular_df.empty:
            return granular_df, "granular"
        return fallback_df, "fallback"

    if map_mode == "Centroid per state":
        return fallback_df, "fallback"

    if not granular_df.empty:
        return granular_df, "granular"

    return fallback_df, "fallback"


def build_export_summary(kpis: Dict[str, float], filters: Dict, filtered_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": [
                "preset_periode",
                "tanggal_mulai",
                "tanggal_selesai",
                "jumlah_state_terpilih",
                "jumlah_kategori_terpilih",
                "jumlah_baris_terfilter",
                "total_revenue",
                "total_orders",
                "total_customers",
                "avg_order_value",
                "avg_review",
                "late_rate",
            ],
            "value": [
                filters["preset"],
                filters["start_date"].date(),
                filters["end_date"].date(),
                len(filters["states"]),
                len(filters["categories"]),
                len(filtered_df),
                round(kpis["total_revenue"], 2),
                kpis["total_orders"],
                kpis["total_customers"],
                round(kpis["avg_order_value"], 2),
                round(kpis["avg_review"], 2),
                round(kpis["late_rate"], 2),
            ],
        }
    )


def base_layout(fig: go.Figure, height: Optional[int] = None) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=COLOR_CARD,
        plot_bgcolor=COLOR_CARD,
        font=dict(color=COLOR_TEXT),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor=COLOR_BORDER,
        tickfont=dict(color=COLOR_SUBTEXT),
    )
    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.18)",
        linecolor=COLOR_BORDER,
        tickfont=dict(color=COLOR_SUBTEXT),
    )

    if height is not None:
        fig.update_layout(height=height)

    return fig


def bar_colors(length: int, highlight_index: int) -> List[str]:
    colors = [COLOR_PRIMARY] * length
    if 0 <= highlight_index < length:
        colors[highlight_index] = COLOR_HIGHLIGHT
    return colors


def render_header() -> None:
    st.title("Dashboard Analisis E-Commerce Public Dataset")
    st.caption(
        "Dashboard interaktif ini menggunakan data hasil cleaning untuk mengeksplorasi "
        "performa bisnis, pelanggan, dan persebaran geografis."
    )

    st.markdown(
        """
        <div class="hero-box">
            <b>Tujuan dashboard</b><br>
            Dashboard ini membantu mengeksplorasi tren revenue, performa kategori produk,
            hubungan keterlambatan pengiriman dengan review pelanggan, segmentasi pelanggan,
            dan persebaran geografis pelanggan berdasarkan filter yang dipilih.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Cara menggunakan dashboard", expanded=False):
        st.markdown(
            """
            **Langkah penggunaan:**
            1. Pilih preset periode atau tanggal kustom di sidebar.
            2. Filter data berdasarkan state pelanggan dan kategori produk.
            3. Ubah nilai **Top N** untuk mengatur banyaknya kategori/state yang ditampilkan.
            4. Buka tab **Business Performance**, **Customer Intelligence**, atau **Geographic Distribution**.
            5. Gunakan tombol **Download** untuk menyimpan data hasil filter atau ringkasan KPI.
            """
        )


def render_download_section(filtered_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    st.subheader("Download Data")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Download data hasil filter (CSV)",
            data=dataframe_to_csv_bytes(filtered_df),
            file_name="filtered_main_data.csv",
            mime="text/csv",
        )

    with col2:
        st.download_button(
            label="Download ringkasan KPI (CSV)",
            data=dataframe_to_csv_bytes(summary_df),
            file_name="dashboard_summary.csv",
            mime="text/csv",
        )


def render_kpis(kpis: Dict[str, float]) -> None:
    st.subheader("Business Performance Overview")

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    col1.metric("Total Revenue", format_currency(kpis["total_revenue"]))
    col2.metric("Total Orders", format_number(kpis["total_orders"]))
    col3.metric("Total Customers", format_number(kpis["total_customers"]))
    col4.metric("Average Order Value", format_currency(kpis["avg_order_value"]))
    col5.metric("Average Review Score", f"{kpis['avg_review']:.2f}")
    col6.metric("Late Delivery Rate", f"{kpis['late_rate']:.2f}%")


def render_business_tab(item_df: pd.DataFrame, order_df: pd.DataFrame, kpis: Dict[str, float], top_n: int) -> None:
    render_kpis(kpis)
    st.markdown("---")

    monthly_df = monthly_summary(order_df)
    category_df = top_categories(item_df, top_n=top_n)

    left_column, right_column = st.columns([2, 1])

    with left_column:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=monthly_df["purchase_month"],
                y=monthly_df["total_revenue"],
                mode="lines+markers",
                line=dict(color=COLOR_PRIMARY, width=3),
                marker=dict(color=COLOR_HIGHLIGHT, size=7),
                name="Revenue",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=monthly_df["purchase_month"],
                y=monthly_df["total_orders"],
                mode="lines+markers",
                line=dict(color=COLOR_SUCCESS, width=2, dash="dot"),
                marker=dict(color=COLOR_SUCCESS, size=6),
                name="Orders",
                yaxis="y2",
            )
        )

        fig.update_layout(
            title="Tren Revenue dan Jumlah Order per Bulan",
            xaxis_title="Bulan",
            yaxis_title="Total Revenue",
            yaxis2=dict(
                title="Jumlah Order",
                overlaying="y",
                side="right",
                showgrid=False,
                tickfont=dict(color=COLOR_SUBTEXT),
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color=COLOR_TEXT),
            ),
            height=420,
        )

        st.plotly_chart(base_layout(fig), width="stretch", config=PLOT_CONFIG)

    with right_column:
        peak_month = (
            monthly_df.loc[monthly_df["total_revenue"].idxmax(), "purchase_month"]
            if not monthly_df.empty else "-"
        )
        peak_revenue = (
            monthly_df.loc[monthly_df["total_revenue"].idxmax(), "total_revenue"]
            if not monthly_df.empty else 0
        )
        top_category_name = safe_value(category_df, "product_category_name_english", "-")
        top_category_value = safe_value(category_df, "total_revenue", 0)

        st.markdown(
            f"""
            <div class="insight-box">
                <h4 style="margin-top:0;">Insight Utama</h4>
                <p>
                    Pada filter aktif, puncak revenue terjadi pada <b>{peak_month}</b>
                    dengan total sekitar <b>{format_currency(peak_revenue)}</b>.
                </p>
                <p>
                    Kategori produk dengan kontribusi revenue terbesar adalah
                    <b>{top_category_name}</b> dengan nilai sekitar
                    <b>{format_currency(top_category_value)}</b>.
                </p>
                <p>
                    Temuan ini dapat digunakan untuk menentukan prioritas promosi,
                    alokasi stok, dan evaluasi momentum penjualan bulanan.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    left_column, right_column = st.columns(2)

    with left_column:
        revenue_plot = category_df.sort_values("total_revenue", ascending=True).reset_index(drop=True)
        colors = bar_colors(len(revenue_plot), len(revenue_plot) - 1)

        fig = go.Figure(
            go.Bar(
                x=revenue_plot["total_revenue"],
                y=revenue_plot["product_category_name_english"],
                orientation="h",
                marker_color=colors,
            )
        )
        fig.update_layout(
            title=f"Top {top_n} Kategori Produk Berdasarkan Revenue",
            xaxis_title="Total Revenue",
            yaxis_title="Kategori Produk",
            height=430,
        )
        st.plotly_chart(base_layout(fig), width="stretch", config=PLOT_CONFIG)

    with right_column:
        order_plot = category_df.sort_values("total_orders", ascending=True).reset_index(drop=True)
        highlight_idx = int(order_plot["total_orders"].idxmax()) if not order_plot.empty else -1
        colors = bar_colors(len(order_plot), highlight_idx)

        fig = go.Figure(
            go.Bar(
                x=order_plot["total_orders"],
                y=order_plot["product_category_name_english"],
                orientation="h",
                marker_color=colors,
            )
        )
        fig.update_layout(
            title=f"Top {top_n} Kategori Produk Berdasarkan Jumlah Order",
            xaxis_title="Jumlah Order",
            yaxis_title="Kategori Produk",
            height=430,
        )
        st.plotly_chart(base_layout(fig), width="stretch", config=PLOT_CONFIG)


def render_customer_tab(order_df: pd.DataFrame) -> None:
    review_df = review_by_delay(order_df)
    rfm_df = rfm_summary(order_df)
    spending_df = spending_group_summary(order_df)

    left_column, right_column = st.columns([2, 1])

    with left_column:
        fig = px.bar(
            review_df,
            x="delay_category",
            y="avg_review",
            color="delay_category",
            color_discrete_map=DELAY_COLOR_MAP,
            title="Rata-rata Review Score Berdasarkan Ketepatan Pengiriman",
        )
        fig.update_layout(
            xaxis_title="Kategori Keterlambatan",
            yaxis_title="Rata-rata Review Score",
            height=420,
            showlegend=False,
        )
        st.plotly_chart(base_layout(fig), width="stretch", config=PLOT_CONFIG)

    with right_column:
        if review_df.empty:
            best_delay_name = "-"
            best_delay_score = 0.0
            worst_delay_name = "-"
            worst_delay_score = 0.0
        else:
            best_row = review_df.sort_values("avg_review", ascending=False).iloc[0]
            worst_row = review_df.sort_values("avg_review", ascending=True).iloc[0]
            best_delay_name = best_row["delay_category"]
            best_delay_score = best_row["avg_review"]
            worst_delay_name = worst_row["delay_category"]
            worst_delay_score = worst_row["avg_review"]

        st.markdown(
            f"""
            <div class="insight-box">
                <h4 style="margin-top:0;">Insight Pengiriman</h4>
                <p>
                    Kategori pengiriman dengan review tertinggi adalah
                    <b>{best_delay_name}</b> dengan rata-rata review
                    <b>{best_delay_score:.2f}</b>.
                </p>
                <p>
                    Kategori dengan review terendah adalah
                    <b>{worst_delay_name}</b> dengan rata-rata review
                    <b>{worst_delay_score:.2f}</b>.
                </p>
                <p>
                    Ini menunjukkan bahwa kualitas pengiriman berhubungan langsung
                    dengan kepuasan pelanggan, sehingga perbaikan SLA logistik
                    dapat membantu menjaga review tetap tinggi.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    left_column, right_column = st.columns(2)

    with left_column:
        rfm_plot = rfm_df.sort_values("total_customers", ascending=True).reset_index(drop=True)
        highlight_idx = int(rfm_plot["total_customers"].idxmax()) if not rfm_plot.empty else -1
        colors = bar_colors(len(rfm_plot), highlight_idx)

        fig = go.Figure(
            go.Bar(
                x=rfm_plot["total_customers"],
                y=rfm_plot["segment"],
                orientation="h",
                marker_color=colors,
            )
        )
        fig.update_layout(
            title="Distribusi Pelanggan Berdasarkan Segmen RFM",
            xaxis_title="Jumlah Pelanggan",
            yaxis_title="Segmen",
            height=430,
        )
        st.plotly_chart(base_layout(fig), width="stretch", config=PLOT_CONFIG)

    with right_column:
        spending_plot = spending_df.sort_values("total_revenue", ascending=True).reset_index(drop=True)
        highlight_idx = int(spending_plot["total_revenue"].idxmax()) if not spending_plot.empty else -1
        colors = bar_colors(len(spending_plot), highlight_idx)

        fig = go.Figure(
            go.Bar(
                x=spending_plot["total_revenue"],
                y=spending_plot["spending_group"].astype(str),
                orientation="h",
                marker_color=colors,
            )
        )
        fig.update_layout(
            title="Kontribusi Revenue Berdasarkan Kelompok Nilai Belanja",
            xaxis_title="Total Revenue",
            yaxis_title="Kelompok Pelanggan",
            height=430,
        )
        st.plotly_chart(base_layout(fig), width="stretch", config=PLOT_CONFIG)


def render_geographic_tab(order_df: pd.DataFrame, top_n: int, map_mode: str) -> None:
    top_state_df = state_summary(order_df, top_n=top_n)

    if top_state_df.empty:
        st.warning("Data geografis tidak tersedia untuk filter yang dipilih.")
        return

    left_column, right_column = st.columns([2, 1])

    with left_column:
        plot_df = top_state_df.sort_values("total_revenue", ascending=True).reset_index(drop=True)
        colors = bar_colors(len(plot_df), len(plot_df) - 1)

        fig = go.Figure(
            go.Bar(
                x=plot_df["total_revenue"],
                y=plot_df["customer_state"],
                orientation="h",
                marker_color=colors,
            )
        )
        fig.update_layout(
            title=f"Top {top_n} State Berdasarkan Revenue",
            xaxis_title="Total Revenue",
            yaxis_title="State",
            height=420,
        )
        st.plotly_chart(base_layout(fig), width="stretch", config=PLOT_CONFIG)

    with right_column:
        top_state = safe_value(top_state_df, "customer_state", "-")
        top_revenue = safe_value(top_state_df, "total_revenue", 0)
        high_late_state = (
            top_state_df.sort_values("late_rate", ascending=False).iloc[0]["customer_state"]
            if not top_state_df.empty else "-"
        )

        st.markdown(
            f"""
            <div class="insight-box">
                <h4 style="margin-top:0;">Insight Geografis</h4>
                <p>
                    State dengan kontribusi revenue terbesar pada filter aktif adalah
                    <b>{top_state}</b> dengan total sekitar
                    <b>{format_currency(top_revenue)}</b>.
                </p>
                <p>
                    State yang perlu perhatian lebih dari sisi keterlambatan pengiriman
                    pada kelompok teratas adalah <b>{high_late_state}</b>.
                </p>
                <p>
                    Ini menunjukkan bahwa wilayah prioritas promosi dan wilayah prioritas
                    perbaikan operasional belum tentu selalu sama.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    map_df, map_kind = get_map_data(order_df, map_mode)

    if map_df.empty:
        st.warning("Data peta tidak tersedia untuk filter yang dipilih.")
        return

    if map_kind == "granular":
        fig = px.scatter_map(
            map_df,
            lat="geolocation_lat",
            lon="geolocation_lng",
            color="sales",
            hover_name="customer_state",
            hover_data={"sales": ":.2f"},
            zoom=3.2,
            center={"lat": -14.2350, "lon": -51.9253},
            color_continuous_scale=[COLOR_PRIMARY, COLOR_HIGHLIGHT],
            height=520,
            title="Peta Persebaran Titik Transaksi Pelanggan",
            map_style="carto-positron",
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor=COLOR_CARD,
            font=dict(color=COLOR_TEXT),
        )
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
        st.caption("Peta menggunakan koordinat pelanggan hasil cleaning.")
    else:
        fig = px.scatter_map(
            map_df,
            lat="lat",
            lon="lon",
            size="total_revenue",
            color="total_revenue",
            hover_name="customer_state",
            hover_data={"total_revenue": ":.2f", "total_orders": True},
            zoom=3.2,
            center={"lat": -14.2350, "lon": -51.9253},
            color_continuous_scale=[COLOR_PRIMARY, COLOR_HIGHLIGHT],
            height=520,
            title="Peta Persebaran Wilayah Pelanggan",
            map_style="carto-positron",
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor=COLOR_CARD,
            font=dict(color=COLOR_TEXT),
        )
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
        st.caption(
            "Dashboard memakai centroid per state karena mode peta dipilih fallback atau koordinat granular tidak tersedia."
        )


def render_filtered_preview(df: pd.DataFrame) -> None:
    with st.expander("Lihat data hasil filter"):
        st.dataframe(df.head(100), width="stretch", hide_index=True)


def main() -> None:
    apply_style()
    render_header()

    data_path = resolve_data_path()

    try:
        df, separator = load_data(data_path)
    except Exception as exc:
        st.error(f"Gagal membaca data: {exc}")
        st.stop()

    if df.empty:
        st.error("File main_data.csv berhasil dibaca, tetapi isinya kosong.")
        st.stop()

    validate_data(df)

    filters = render_sidebar(df, data_path)
    filtered_df = apply_filters(df, filters)

    if filtered_df.empty:
        st.warning("Tidak ada data yang sesuai dengan filter yang dipilih.")
        st.stop()

    order_df = build_order_df(filtered_df)
    kpis = calculate_kpis(filtered_df, order_df)
    summary_export_df = build_export_summary(kpis, filters, filtered_df)

    render_download_section(filtered_df, summary_export_df)

    st.markdown("---")

    business_tab, customer_tab, geographic_tab = st.tabs(
        [
            "Business Performance",
            "Customer Intelligence",
            "Geographic Distribution",
        ]
    )

    with business_tab:
        render_business_tab(filtered_df, order_df, kpis, top_n=filters["top_n"])

    with customer_tab:
        render_customer_tab(order_df)

    with geographic_tab:
        render_geographic_tab(
            order_df,
            top_n=filters["top_n"],
            map_mode=filters["map_mode"],
        )

    st.markdown("---")
    render_filtered_preview(filtered_df)
    st.caption(
        f"E-Commerce Public Dataset | Separator terdeteksi: {repr(separator)} | "
        "Dashboard interaktif berbasis data hasil cleaning"
    )


if __name__ == "__main__":
    main()