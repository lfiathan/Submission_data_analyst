from pathlib import Path
from typing import Dict, List, Tuple
import csv

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="E-Commerce Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# DESIGN SYSTEM
# =========================================================
COLOR_BG = "#0F172A"
COLOR_BG_SOFT = "#111827"
COLOR_CARD = "#1E293B"
COLOR_CARD_2 = "#172033"
COLOR_BORDER = "#334155"

COLOR_TEXT = "#E2E8F0"
COLOR_SUBTEXT = "#94A3B8"

COLOR_PRIMARY = "#38BDF8"
COLOR_SECONDARY = "#818CF8"
COLOR_SUCCESS = "#34D399"
COLOR_WARNING = "#FBBF24"
COLOR_ORANGE = "#FB923C"
COLOR_DANGER = "#F87171"

MAP_STYLE_LIGHT = "carto-positron"
MAP_CENTER_BRAZIL = {"lat": -14.2350, "lon": -51.9253}
MAP_ZOOM_DEFAULT = 3.2
MAP_MAX_POINTS = 3000

DELAY_COLOR_MAP = {
    "Lebih cepat": COLOR_SUCCESS,
    "Tepat waktu": COLOR_PRIMARY,
    "Terlambat 1-3 hari": COLOR_WARNING,
    "Terlambat 4-7 hari": COLOR_ORANGE,
    "Terlambat >7 hari": COLOR_DANGER,
}

DELAY_ORDER = [
    "Lebih cepat",
    "Tepat waktu",
    "Terlambat 1-3 hari",
    "Terlambat 4-7 hari",
    "Terlambat >7 hari",
]

REQUIRED_COLUMNS = [
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
    "geolocation_lat",
    "geolocation_lng",
]

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


# =========================================================
# UI STYLING
# =========================================================
def apply_custom_style() -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{
                background:
                    radial-gradient(circle at top left, #162033 0%, {COLOR_BG} 45%),
                    linear-gradient(180deg, #0B1220 0%, {COLOR_BG_SOFT} 100%);
                color: {COLOR_TEXT};
            }}

            .block-container {{
                padding-top: 2rem;
                padding-bottom: 2rem;
            }}

            section[data-testid="stSidebar"] {{
                background: linear-gradient(180deg, #111827 0%, #172033 100%);
                border-right: 1px solid {COLOR_BORDER};
            }}

            section[data-testid="stSidebar"] * {{
                color: {COLOR_TEXT} !important;
            }}

            .hero-box {{
                background: linear-gradient(
                    135deg,
                    rgba(56, 189, 248, 0.16) 0%,
                    rgba(129, 140, 248, 0.10) 100%
                );
                border: 1px solid rgba(56, 189, 248, 0.28);
                border-radius: 18px;
                padding: 1.15rem 1.2rem;
                margin-bottom: 1.2rem;
                color: {COLOR_TEXT};
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
            }}

            .sidebar-box {{
                background: rgba(30, 41, 59, 0.88);
                border: 1px solid {COLOR_BORDER};
                border-radius: 14px;
                padding: 0.95rem 1rem;
                margin-bottom: 1rem;
                color: {COLOR_TEXT};
                font-size: 0.93rem;
                line-height: 1.55;
            }}

            .insight-box {{
                background: linear-gradient(
                    180deg,
                    rgba(30, 41, 59, 0.95) 0%,
                    rgba(23, 32, 51, 0.95) 100%
                );
                border: 1px solid {COLOR_BORDER};
                border-radius: 16px;
                padding: 1rem 1.05rem;
                box-shadow: 0 8px 22px rgba(0, 0, 0, 0.20);
                color: {COLOR_TEXT};
            }}

            div[data-testid="stMetric"] {{
                background: linear-gradient(
                    180deg,
                    rgba(30,41,59,0.96) 0%,
                    rgba(23,32,51,0.96) 100%
                );
                border: 1px solid {COLOR_BORDER};
                border-radius: 16px;
                padding: 1rem;
                box-shadow: 0 8px 22px rgba(0,0,0,0.22);
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
                font-size: 1.95rem !important;
                line-height: 1.1 !important;
            }}

            h1, h2, h3, h4 {{
                color: {COLOR_TEXT} !important;
            }}

            .small-note {{
                color: {COLOR_SUBTEXT};
                font-size: 0.92rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# FORMAT HELPERS
# =========================================================
def format_currency(value: float) -> str:
    safe_value = 0.0 if pd.isna(value) else float(value)
    return f"R$ {safe_value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_number(value: float) -> str:
    safe_value = 0 if pd.isna(value) else int(value)
    return f"{safe_value:,}".replace(",", ".")


def safe_first_row_value(df: pd.DataFrame, column: str, default):
    if df.empty or column not in df.columns:
        return default
    return df.iloc[0][column]


# =========================================================
# FILE HELPERS
# =========================================================
def resolve_data_path() -> Path:
    dashboard_dir = Path(__file__).resolve().parent
    project_root = dashboard_dir.parent

    candidate_paths = [
        project_root / "data" / "main_data.csv",
        dashboard_dir / "main_data.csv",
    ]

    for path in candidate_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        "File main_data.csv tidak ditemukan.\n"
        f"- {project_root / 'data' / 'main_data.csv'}\n"
        f"- {dashboard_dir / 'main_data.csv'}"
    )


def detect_csv_separator(path: Path) -> str:
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        sample = file.read(4096)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample else ""
        separator_scores = {
            ",": first_line.count(","),
            ";": first_line.count(";"),
            "\t": first_line.count("\t"),
            "|": first_line.count("|"),
        }
        return max(separator_scores, key=separator_scores.get)


def read_csv_safely(path: Path) -> Tuple[pd.DataFrame, str]:
    detected_separator = detect_csv_separator(path)
    df = pd.read_csv(path, sep=detected_separator, encoding="utf-8-sig")

    if len(df.columns) == 1:
        for separator in [",", ";", "\t", "|"]:
            if separator == detected_separator:
                continue
            retry_df = pd.read_csv(path, sep=separator, encoding="utf-8-sig")
            if len(retry_df.columns) > 1:
                return retry_df, separator

    return df, detected_separator


# =========================================================
# PARSING HELPERS
# =========================================================
def parse_datetime_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    series = series.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    known_formats = [
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ]

    for fmt in known_formats:
        current = pd.to_datetime(series, format=fmt, errors="coerce")
        parsed = parsed.fillna(current)

    remaining_mask = parsed.isna()
    if remaining_mask.any():
        parsed.loc[remaining_mask] = pd.to_datetime(
            series.loc[remaining_mask],
            errors="coerce",
        )

    return parsed


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


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = df.copy()
    normalized_df.columns = [
        str(col).strip().lower().replace(" ", "_") for col in normalized_df.columns
    ]
    return normalized_df


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


# =========================================================
# DATA PREPARATION
# =========================================================
def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = normalize_columns(df)

    datetime_columns = [
        "shipping_limit_date",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for column in datetime_columns:
        if column in normalized_df.columns:
            normalized_df[column] = parse_datetime_series(normalized_df[column])

    numeric_columns = [
        "order_item_id",
        "price",
        "freight_value",
        "review_score",
        "sales",
        "delivery_time_days",
        "delivery_delay_days",
        "customer_zip_code_prefix",
        "geolocation_zip_code_prefix",
        "geolocation_lat",
        "geolocation_lng",
    ]
    for column in numeric_columns:
        if column in normalized_df.columns:
            normalized_df[column] = parse_locale_numeric_series(normalized_df[column])

    fill_map = {
        "customer_state": "Unknown",
        "customer_city": "Unknown",
        "product_category_name_english": "unknown",
        "delay_category": "Unknown",
        "order_status": "unknown",
    }
    for column, default_value in fill_map.items():
        if column in normalized_df.columns:
            normalized_df[column] = normalized_df[column].fillna(default_value)

    if "purchase_month" in normalized_df.columns:
        normalized_df["purchase_month"] = normalized_df["purchase_month"].astype(str).str.strip()
    elif "order_purchase_timestamp" in normalized_df.columns:
        normalized_df["purchase_month"] = (
            normalized_df["order_purchase_timestamp"].dt.to_period("M").astype(str)
        )
    else:
        normalized_df["purchase_month"] = "Unknown"

    if "sales" not in normalized_df.columns:
        if "price" in normalized_df.columns and "freight_value" in normalized_df.columns:
            normalized_df["sales"] = (
                normalized_df["price"].fillna(0) + normalized_df["freight_value"].fillna(0)
            )
        elif "price" in normalized_df.columns:
            normalized_df["sales"] = normalized_df["price"].fillna(0)
        else:
            normalized_df["sales"] = 0.0

    if "is_late" in normalized_df.columns:
        normalized_df["is_late"] = normalize_boolean_series(normalized_df["is_late"])
    elif "delivery_delay_days" in normalized_df.columns:
        normalized_df["is_late"] = normalized_df["delivery_delay_days"] > 0
    else:
        normalized_df["is_late"] = False

    if "delay_category" in normalized_df.columns:
        normalized_df["delay_category"] = normalized_df["delay_category"].fillna("Unknown")

    return normalized_df


def validate_required_columns(df: pd.DataFrame) -> List[str]:
    return [column for column in REQUIRED_COLUMNS if column not in df.columns]


@st.cache_data
def load_data(path: Path) -> Tuple[pd.DataFrame, str]:
    raw_df, separator = read_csv_safely(path)
    normalized_df = normalize_dataframe(raw_df)
    return normalized_df, separator


# =========================================================
# FILTER HELPERS
# =========================================================
def get_filter_options(df: pd.DataFrame) -> Dict[str, List[str]]:
    return {
        "states": sorted(df["customer_state"].dropna().astype(str).unique().tolist()),
        "categories": sorted(
            df["product_category_name_english"].dropna().astype(str).unique().tolist()
        ),
        "months": sorted(df["purchase_month"].dropna().astype(str).unique().tolist()),
    }


def render_sidebar_header(data_path: Path, df: pd.DataFrame) -> None:
    st.sidebar.markdown("## Filter Dashboard")
    st.sidebar.markdown(
        """
        <div class="sidebar-box">
            <b>Cara menggunakan dashboard</b><br>
            1. Gunakan filter state, kategori, dan bulan.<br>
            2. Aktifkan <i>Pilih semua</i> untuk melihat seluruh data.<br>
            3. Ubah filter untuk mensimulasikan perubahan KPI, grafik, dan insight.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.caption(f"Sumber data: {data_path}")
    st.sidebar.caption(f"Total baris data: {format_number(len(df))}")


def render_multiselect_section(
    section_title: str,
    label: str,
    options: List[str],
    key_prefix: str,
    default_select_all: bool = True,
) -> List[str]:
    with st.sidebar.expander(section_title, expanded=False):
        select_all = st.checkbox(
            f"Pilih semua {label.lower()}",
            value=default_select_all,
            key=f"{key_prefix}_all",
        )

        default_value = options if select_all else options[: min(5, len(options))]

        selected_values = st.multiselect(
            label,
            options=options,
            default=default_value,
            key=f"{key_prefix}_multiselect",
            help=f"Ubah filter untuk mensimulasikan subset tertentu dari {label.lower()}.",
        )

    return selected_values


def render_sidebar_filters(df: pd.DataFrame, data_path: Path) -> Dict[str, List[str]]:
    render_sidebar_header(data_path, df)
    options = get_filter_options(df)

    selected_states = render_multiselect_section(
        section_title="Filter State Pelanggan",
        label="State pelanggan",
        options=options["states"],
        key_prefix="states",
    )
    selected_categories = render_multiselect_section(
        section_title="Filter Kategori Produk",
        label="Kategori produk",
        options=options["categories"],
        key_prefix="categories",
    )
    selected_months = render_multiselect_section(
        section_title="Filter Bulan Transaksi",
        label="Bulan transaksi",
        options=options["months"],
        key_prefix="months",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Ringkasan Filter")
    st.sidebar.caption(f"State terpilih: {len(selected_states)}")
    st.sidebar.caption(f"Kategori terpilih: {len(selected_categories)}")
    st.sidebar.caption(f"Bulan terpilih: {len(selected_months)}")

    return {
        "states": selected_states,
        "categories": selected_categories,
        "months": selected_months,
    }


def apply_filters(df: pd.DataFrame, filters: Dict[str, List[str]]) -> pd.DataFrame:
    filtered_df = df.copy()

    if filters["states"]:
        filtered_df = filtered_df[filtered_df["customer_state"].isin(filters["states"])]

    if filters["categories"]:
        filtered_df = filtered_df[
            filtered_df["product_category_name_english"].isin(filters["categories"])
        ]

    if filters["months"]:
        filtered_df = filtered_df[filtered_df["purchase_month"].isin(filters["months"])]

    return filtered_df


# =========================================================
# ORDER-LEVEL DATASET
# =========================================================
def build_order_level_dataframe(item_df: pd.DataFrame) -> pd.DataFrame:
    aggregation_map = {}

    first_value_columns = [
        "customer_id",
        "customer_unique_id",
        "customer_state",
        "customer_city",
        "order_status",
        "review_score",
        "is_late",
        "delay_category",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "geolocation_lat",
        "geolocation_lng",
    ]

    for column in first_value_columns:
        if column in item_df.columns:
            aggregation_map[column] = "first"

    if "sales" in item_df.columns:
        aggregation_map["sales"] = "sum"

    order_df = item_df.groupby("order_id", as_index=False).agg(aggregation_map).copy()
    return order_df


# =========================================================
# BUSINESS CALCULATIONS
# =========================================================
def calculate_kpis(item_df: pd.DataFrame, order_df: pd.DataFrame) -> Dict[str, float]:
    total_revenue = item_df["sales"].sum()
    total_orders = order_df["order_id"].nunique()
    total_customers = order_df["customer_unique_id"].nunique()
    avg_order_value = total_revenue / total_orders if total_orders else 0.0
    avg_review_score = order_df["review_score"].mean() if "review_score" in order_df.columns else 0.0
    late_delivery_rate = order_df["is_late"].mean() * 100 if "is_late" in order_df.columns else 0.0

    return {
        "total_revenue": total_revenue,
        "total_orders": total_orders,
        "total_customers": total_customers,
        "avg_order_value": avg_order_value,
        "avg_review_score": avg_review_score,
        "late_delivery_rate": late_delivery_rate,
    }


def get_monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("purchase_month", as_index=False)["sales"]
        .sum()
        .sort_values("purchase_month")
    )


def get_top_categories_by_revenue(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    return (
        df.groupby("product_category_name_english", as_index=False)["sales"]
        .sum()
        .sort_values("sales", ascending=False)
        .head(top_n)
    )


def get_top_categories_by_orders(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    return (
        df.groupby("product_category_name_english", as_index=False)["order_id"]
        .nunique()
        .rename(columns={"order_id": "total_orders"})
        .sort_values("total_orders", ascending=False)
        .head(top_n)
    )


def get_review_by_delay(order_df: pd.DataFrame) -> pd.DataFrame:
    review_df = (
        order_df.groupby("delay_category", as_index=False)["review_score"]
        .mean()
    )
    review_df = review_df[review_df["delay_category"].isin(DELAY_ORDER)].copy()
    review_df["delay_category"] = pd.Categorical(
        review_df["delay_category"],
        categories=DELAY_ORDER,
        ordered=True,
    )
    return review_df.sort_values("delay_category")


def get_top_states_by_revenue(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    return (
        df.groupby("customer_state", as_index=False)["sales"]
        .sum()
        .sort_values("sales", ascending=False)
        .head(top_n)
    )


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


def compute_rfm_segment_summary(order_df: pd.DataFrame) -> pd.DataFrame:
    if order_df.empty:
        return pd.DataFrame(columns=["segment", "total_customers"])

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
        if row["r_score"] >= 3 and (row["f_score"] >= 2 or row["m_score"] >= 2):
            return "Recent Customers"
        if row["r_score"] <= 2 and row["f_score"] <= 2 and row["m_score"] <= 2:
            return "At Risk"
        return "Others"

    rfm_df["segment"] = rfm_df.apply(assign_segment, axis=1)

    summary_df = (
        rfm_df.groupby("segment", as_index=False)
        .size()
        .rename(columns={"size": "total_customers"})
    )

    segment_order = [
        "Best Customers",
        "Loyal Customers",
        "Recent Customers",
        "At Risk",
        "Others",
    ]
    summary_df["segment"] = pd.Categorical(
        summary_df["segment"],
        categories=segment_order,
        ordered=True,
    )

    return summary_df.sort_values("segment")


def compute_monetary_cluster_summary(order_df: pd.DataFrame) -> pd.DataFrame:
    if order_df.empty:
        return pd.DataFrame(columns=["monetary_cluster", "total_customers"])

    monetary_df = (
        order_df.groupby("customer_unique_id", as_index=False)["sales"]
        .sum()
        .rename(columns={"sales": "monetary"})
    )

    q1 = monetary_df["monetary"].quantile(0.25)
    q2 = monetary_df["monetary"].quantile(0.50)
    q3 = monetary_df["monetary"].quantile(0.75)

    def assign_cluster(value: float) -> str:
        if value <= q1:
            return "Low"
        if value <= q2:
            return "Medium"
        if value <= q3:
            return "High"
        return "Very High"

    monetary_df["monetary_cluster"] = monetary_df["monetary"].apply(assign_cluster)

    summary_df = (
        monetary_df.groupby("monetary_cluster", as_index=False)
        .size()
        .rename(columns={"size": "total_customers"})
    )

    cluster_order = ["Low", "Medium", "High", "Very High"]
    summary_df["monetary_cluster"] = pd.Categorical(
        summary_df["monetary_cluster"],
        categories=cluster_order,
        ordered=True,
    )

    return summary_df.sort_values("monetary_cluster")


# =========================================================
# MAP DATA HELPERS
# =========================================================
def get_granular_map_data(df: pd.DataFrame, max_points: int = MAP_MAX_POINTS) -> pd.DataFrame:
    required_columns = ["order_id", "geolocation_lat", "geolocation_lng", "customer_state", "sales"]
    available_columns = [column for column in required_columns if column in df.columns]

    if "geolocation_lat" not in available_columns or "geolocation_lng" not in available_columns:
        return pd.DataFrame()

    map_df = df[available_columns].copy()

    map_df["geolocation_lat"] = parse_locale_numeric_series(map_df["geolocation_lat"])
    map_df["geolocation_lng"] = parse_locale_numeric_series(map_df["geolocation_lng"])
    map_df = map_df.dropna(subset=["geolocation_lat", "geolocation_lng"])

    if "order_id" in map_df.columns:
        map_df = map_df.drop_duplicates(subset=["order_id"])

    brazil_bounds_df = map_df[
        map_df["geolocation_lat"].between(-35, 6)
        & map_df["geolocation_lng"].between(-75, -30)
    ]
    if not brazil_bounds_df.empty:
        map_df = brazil_bounds_df

    if len(map_df) > max_points:
        map_df = map_df.sample(max_points, random_state=42)

    return map_df


def get_state_centroid_map_data(df: pd.DataFrame) -> pd.DataFrame:
    if "customer_state" not in df.columns or "sales" not in df.columns:
        return pd.DataFrame()

    state_df = (
        df.groupby("customer_state", as_index=False)
        .agg(
            total_revenue=("sales", "sum"),
            total_orders=("order_id", "nunique"),
        )
        .copy()
    )

    state_df["lat"] = state_df["customer_state"].map(
        lambda state: BRAZIL_STATE_CENTROIDS.get(state, (np.nan, np.nan))[0]
    )
    state_df["lon"] = state_df["customer_state"].map(
        lambda state: BRAZIL_STATE_CENTROIDS.get(state, (np.nan, np.nan))[1]
    )

    return state_df.dropna(subset=["lat", "lon"])


# =========================================================
# CHART STYLING
# =========================================================
def apply_figure_style(fig: go.Figure, height: int | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor=COLOR_CARD,
        paper_bgcolor=COLOR_CARD,
        font=dict(color=COLOR_TEXT, size=12),
        title=dict(font=dict(size=18, color=COLOR_TEXT)),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=COLOR_TEXT),
        ),
    )

    if height is not None:
        fig.update_layout(height=height)

    fig.update_xaxes(
        showgrid=False,
        linecolor=COLOR_BORDER,
        tickfont=dict(color=COLOR_SUBTEXT),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
        linecolor=COLOR_BORDER,
        tickfont=dict(color=COLOR_SUBTEXT),
    )
    return fig


def get_plotly_config() -> Dict:
    return {
        "displayModeBar": True,
        "scrollZoom": True,
        "responsive": True,
    }


def build_customer_map_figure(map_df: pd.DataFrame) -> go.Figure:
    fig = px.scatter_map(
        map_df,
        lat="geolocation_lat",
        lon="geolocation_lng",
        color="sales",
        hover_name="customer_state",
        hover_data={
            "sales": ":.2f",
            "geolocation_lat": False,
            "geolocation_lng": False,
        },
        color_continuous_scale=["#BFDBFE", "#60A5FA", "#2563EB"],
        zoom=MAP_ZOOM_DEFAULT,
        center=MAP_CENTER_BRAZIL,
        title="Geographic Distribution of Customers in Brazil",
        height=520,
    )

    fig.update_traces(marker=dict(size=8, opacity=0.70))
    fig.update_layout(
        map_style=MAP_STYLE_LIGHT,
        coloraxis_colorbar=dict(
            title=dict(text="Sales", font=dict(color=COLOR_TEXT)),
            tickfont=dict(color=COLOR_TEXT),
        ),
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor=COLOR_CARD,
        font=dict(color=COLOR_TEXT),
    )
    return fig


def build_state_map_figure(state_df: pd.DataFrame) -> go.Figure:
    fig = px.scatter_map(
        state_df,
        lat="lat",
        lon="lon",
        size="total_revenue",
        color="total_revenue",
        hover_name="customer_state",
        hover_data={
            "total_revenue": ":.2f",
            "total_orders": True,
            "lat": False,
            "lon": False,
        },
        color_continuous_scale=["#BFDBFE", "#60A5FA", "#2563EB"],
        zoom=MAP_ZOOM_DEFAULT,
        center=MAP_CENTER_BRAZIL,
        title="Geographic Distribution by State Centroid (Fallback Mode)",
        height=520,
    )

    fig.update_traces(marker=dict(opacity=0.75))
    fig.update_layout(
        map_style=MAP_STYLE_LIGHT,
        coloraxis_colorbar=dict(
            title=dict(text="Revenue", font=dict(color=COLOR_TEXT)),
            tickfont=dict(color=COLOR_TEXT),
        ),
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor=COLOR_CARD,
        font=dict(color=COLOR_TEXT),
    )
    return fig


# =========================================================
# VALIDATION
# =========================================================
def validate_dataset(df: pd.DataFrame, path: Path, separator: str) -> None:
    missing_columns = validate_required_columns(df)
    if not missing_columns:
        return

    st.error("Struktur file `main_data.csv` tidak sesuai dengan kebutuhan dashboard.")
    st.write("File yang dibaca:", str(path))
    st.write("Separator yang terdeteksi:", repr(separator))
    st.write("Kolom yang hilang:", missing_columns)
    st.write("Kolom yang berhasil dibaca:", df.columns.tolist())

    with st.expander("Lihat preview data yang terbaca"):
        st.dataframe(df.head(10), width="stretch")

    st.stop()


# =========================================================
# RENDER HELPERS
# =========================================================
def render_header() -> None:
    st.title("Dashboard Analisis E-Commerce Public Dataset")
    st.caption(
        "Analytical dashboard based on E-Commerce Public Dataset. "
        "Dashboard ini merangkum performa bisnis, kepuasan pelanggan, "
        "analisis RFM, dan distribusi geografis pelanggan."
    )
    st.caption("Klik ikon panah di kiri atas untuk membuka atau menutup panel filter.")

    st.markdown(
        """
        <div class="hero-box">
            <b>Tujuan dashboard</b><br>
            Dashboard ini digunakan untuk menjawab dua pertanyaan bisnis utama:
            kategori produk mana yang paling berkontribusi terhadap penjualan,
            dan bagaimana performa pengiriman berhubungan dengan kepuasan pelanggan.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(kpis: Dict[str, float]) -> None:
    st.subheader("Business Performance Overview")

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    col1.metric("Total Revenue", format_currency(kpis["total_revenue"]))
    col2.metric("Total Orders", format_number(kpis["total_orders"]))
    col3.metric("Total Customers", format_number(kpis["total_customers"]))
    col4.metric("Average Order Value", format_currency(kpis["avg_order_value"]))
    col5.metric("Average Review Score", f"{kpis['avg_review_score']:.2f}")
    col6.metric("Late Delivery Rate", f"{kpis['late_delivery_rate']:.2f}%")


def render_business_tab(item_df: pd.DataFrame, order_df: pd.DataFrame, kpis: Dict[str, float]) -> None:
    render_kpis(kpis)
    st.markdown("---")

    left_column, right_column = st.columns([2, 1])

    with left_column:
        monthly_sales_df = get_monthly_sales(item_df)

        fig_monthly_sales = px.line(
            monthly_sales_df,
            x="purchase_month",
            y="sales",
            markers=True,
            title="Monthly Revenue Trend",
        )
        fig_monthly_sales.update_traces(
            line=dict(color=COLOR_PRIMARY, width=3),
            marker=dict(size=7, color=COLOR_WARNING),
        )
        fig_monthly_sales.update_layout(
            xaxis_title="Purchase Month",
            yaxis_title="Total Revenue (R$)",
            height=420,
        )
        st.plotly_chart(
            apply_figure_style(fig_monthly_sales),
            width="stretch",
            config=get_plotly_config(),
        )

    with right_column:
        top_category_df = get_top_categories_by_revenue(item_df, top_n=10)
        top_category_name = safe_first_row_value(top_category_df, "product_category_name_english", "-")
        top_category_value = safe_first_row_value(top_category_df, "sales", 0)

        st.markdown(
            f"""
            <div class="insight-box">
                <h4 style="margin-top:0;">Key Insight</h4>
                <p>
                    Pendapatan terkonsentrasi pada sejumlah kecil kategori utama.
                    Pada filter aktif, kategori dengan penjualan tertinggi adalah
                    <b>{top_category_name}</b> dengan total sekitar
                    <b>{format_currency(top_category_value)}</b>.
                </p>
                <p>
                    Kategori ini layak diprioritaskan untuk strategi promosi,
                    manajemen stok, dan pengembangan produk unggulan.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    left_column, right_column = st.columns(2)

    with left_column:
        revenue_by_category_df = get_top_categories_by_revenue(item_df, top_n=10)

        fig_revenue_category = px.bar(
            revenue_by_category_df,
            x="sales",
            y="product_category_name_english",
            orientation="h",
            title="Top Product Categories by Revenue",
            labels={
                "sales": "Total Revenue",
                "product_category_name_english": "Product Category",
            },
            color="sales",
            color_continuous_scale=["#183B5B", "#2563EB", "#38BDF8"],
        )
        fig_revenue_category.update_layout(
            height=430,
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
        )
        st.plotly_chart(
            apply_figure_style(fig_revenue_category),
            width="stretch",
            config=get_plotly_config(),
        )

    with right_column:
        orders_by_category_df = get_top_categories_by_orders(item_df, top_n=10)

        fig_orders_category = px.bar(
            orders_by_category_df,
            x="total_orders",
            y="product_category_name_english",
            orientation="h",
            title="Top Product Categories by Orders",
            labels={
                "total_orders": "Total Orders",
                "product_category_name_english": "Product Category",
            },
            color="total_orders",
            color_continuous_scale=["#1E293B", "#818CF8", "#38BDF8"],
        )
        fig_orders_category.update_layout(
            height=430,
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
        )
        st.plotly_chart(
            apply_figure_style(fig_orders_category),
            width="stretch",
            config=get_plotly_config(),
        )

    st.markdown("---")

    st.markdown(
        f"""
        <div class="insight-box">
            <h4 style="margin-top:0;">Business Interpretation</h4>
            <p>
                Nilai rata-rata transaksi pada filter aktif adalah
                <b>{format_currency(kpis["avg_order_value"])}</b>.
            </p>
            <p>
                Dengan melihat kategori unggulan dari sisi <b>pendapatan</b> dan
                <b>jumlah pesanan</b> secara bersamaan, dashboard ini lebih sinkron
                dengan pertanyaan bisnis pada notebook.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_customer_tab(order_df: pd.DataFrame) -> None:
    st.subheader("Customer Intelligence Overview")

    left_column, right_column = st.columns([2, 1])
    review_by_delay_df = get_review_by_delay(order_df)

    with left_column:
        fig_review_delay = px.bar(
            review_by_delay_df,
            x="delay_category",
            y="review_score",
            title="Average Review Score by Delivery Delay Category",
            labels={
                "delay_category": "Delay Category",
                "review_score": "Average Review Score",
            },
            color="delay_category",
            color_discrete_map=DELAY_COLOR_MAP,
        )
        fig_review_delay.update_layout(height=420, showlegend=False)
        st.plotly_chart(
            apply_figure_style(fig_review_delay),
            width="stretch",
            config=get_plotly_config(),
        )

    with right_column:
        if review_by_delay_df.empty:
            best_delay_name = "-"
            best_delay_score = 0.0
            worst_delay_name = "-"
            worst_delay_score = 0.0
        else:
            best_row = review_by_delay_df.sort_values("review_score", ascending=False).iloc[0]
            worst_row = review_by_delay_df.sort_values("review_score", ascending=True).iloc[0]
            best_delay_name = best_row["delay_category"]
            best_delay_score = best_row["review_score"]
            worst_delay_name = worst_row["delay_category"]
            worst_delay_score = worst_row["review_score"]

        st.markdown(
            f"""
            <div class="insight-box">
                <h4 style="margin-top:0;">Delivery Insight</h4>
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
                    Ini menguatkan hasil notebook bahwa performa pengiriman
                    memiliki hubungan kuat dengan kepuasan pelanggan.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    rfm_summary_df = compute_rfm_segment_summary(order_df)
    monetary_summary_df = compute_monetary_cluster_summary(order_df)

    left_column, right_column = st.columns(2)

    with left_column:
        fig_rfm = px.bar(
            rfm_summary_df,
            x="total_customers",
            y="segment",
            orientation="h",
            title="Customer Distribution by RFM Segment",
            labels={
                "total_customers": "Number of Customers",
                "segment": "Customer Segment",
            },
            color_discrete_sequence=[COLOR_SUCCESS],
        )
        fig_rfm.update_layout(height=430)
        st.plotly_chart(
            apply_figure_style(fig_rfm),
            width="stretch",
            config=get_plotly_config(),
        )

    with right_column:
        fig_cluster = px.bar(
            monetary_summary_df,
            x="total_customers",
            y="monetary_cluster",
            orientation="h",
            title="Customer Clustering Based on Monetary Value",
            labels={
                "total_customers": "Number of Customers",
                "monetary_cluster": "Customer Cluster",
            },
            color_discrete_sequence=[COLOR_WARNING],
        )
        fig_cluster.update_layout(height=430)
        st.plotly_chart(
            apply_figure_style(fig_cluster),
            width="stretch",
            config=get_plotly_config(),
        )

    st.markdown("---")

    best_count = (
        rfm_summary_df.loc[rfm_summary_df["segment"] == "Best Customers", "total_customers"].sum()
        if "Best Customers" in rfm_summary_df["segment"].astype(str).values
        else 0
    )
    high_value_count = (
        monetary_summary_df.loc[
            monetary_summary_df["monetary_cluster"] == "Very High", "total_customers"
        ].sum()
        if "Very High" in monetary_summary_df["monetary_cluster"].astype(str).values
        else 0
    )

    st.markdown(
        f"""
        <div class="insight-box">
            <h4 style="margin-top:0;">Customer Insight</h4>
            <p>
                Jumlah pelanggan pada segmen <b>Best Customers</b> adalah
                <b>{format_number(best_count)}</b>.
            </p>
            <p>
                Dari sisi monetary clustering, pelanggan pada kelompok
                <b>Very High</b> berjumlah <b>{format_number(high_value_count)}</b>.
                Kelompok ini bernilai strategis tinggi untuk retensi dan personalisasi.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_geographic_tab(item_df: pd.DataFrame) -> None:
    st.subheader("Geographic Distribution of Customers")

    left_column, right_column = st.columns([2, 1])
    top_states_df = get_top_states_by_revenue(item_df, top_n=10)

    with left_column:
        fig_geo_bar = px.bar(
            top_states_df,
            x="sales",
            y="customer_state",
            orientation="h",
            title="Top Regions by Revenue",
            labels={
                "sales": "Total Revenue",
                "customer_state": "State",
            },
            color="sales",
            color_continuous_scale=["#183B5B", "#2563EB", "#38BDF8"],
        )
        fig_geo_bar.update_layout(
            height=420,
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
        )
        st.plotly_chart(
            apply_figure_style(fig_geo_bar),
            width="stretch",
            config=get_plotly_config(),
        )

    with right_column:
        top_state_name = safe_first_row_value(top_states_df, "customer_state", "-")
        top_state_value = safe_first_row_value(top_states_df, "sales", 0)

        st.markdown(
            f"""
            <div class="insight-box">
                <h4 style="margin-top:0;">Geographic Insight</h4>
                <p>
                    State dengan kontribusi penjualan terbesar pada filter aktif adalah
                    <b>{top_state_name}</b> dengan total sekitar
                    <b>{format_currency(top_state_value)}</b>.
                </p>
                <p>
                    Informasi ini relevan untuk strategi pemasaran regional,
                    penguatan distribusi, dan optimasi logistik.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    granular_map_df = get_granular_map_data(item_df)

    if not granular_map_df.empty:
        fig_customer_map = build_customer_map_figure(granular_map_df)
        st.plotly_chart(
            fig_customer_map,
            width="stretch",
            config=get_plotly_config(),
        )
        st.caption(
            "Peta dibuat lebih terang menggunakan style `carto-positron`, "
            "serta dibatasi maksimal 3.000 titik agar dashboard tetap ringan."
        )
        return

    state_map_df = get_state_centroid_map_data(item_df)

    if state_map_df.empty:
        st.warning("Data koordinat tidak tersedia dan fallback peta per state juga tidak dapat dibentuk.")
        return

    fig_state_map = build_state_map_figure(state_map_df)
    st.plotly_chart(
        fig_state_map,
        width="stretch",
        config=get_plotly_config(),
    )
    st.caption(
        "Koordinat granular pelanggan tidak terbaca dengan baik, sehingga dashboard memakai fallback centroid per state."
    )


def render_filtered_data(df: pd.DataFrame) -> None:
    with st.expander("Lihat data hasil filter"):
        st.dataframe(df.head(100), width="stretch")


# =========================================================
# APP ORCHESTRATION
# =========================================================
def render_dashboard_tabs(filtered_df: pd.DataFrame) -> None:
    order_df = build_order_level_dataframe(filtered_df)
    kpis = calculate_kpis(filtered_df, order_df)

    st.markdown("---")

    business_tab, customer_tab, geographic_tab = st.tabs(
        [
            "Business Performance",
            "Customer Intelligence",
            "Geographic Distribution",
        ]
    )

    with business_tab:
        render_business_tab(filtered_df, order_df, kpis)

    with customer_tab:
        render_customer_tab(order_df)

    with geographic_tab:
        render_geographic_tab(filtered_df)


def main() -> None:
    apply_custom_style()
    render_header()

    data_path = resolve_data_path()

    try:
        df, separator = load_data(data_path)
    except Exception as exc:
        st.error(f"Gagal membaca data: {exc}")
        st.stop()

    if df.empty:
        st.error("File `main_data.csv` berhasil dibaca, tetapi isinya kosong.")
        st.stop()

    validate_dataset(df, data_path, separator)

    filters = render_sidebar_filters(df, data_path)
    filtered_df = apply_filters(df, filters)

    if filtered_df.empty:
        st.warning("Tidak ada data yang sesuai dengan filter yang dipilih.")
        st.stop()

    render_dashboard_tabs(filtered_df)

    st.markdown("---")
    render_filtered_data(filtered_df)
    st.caption("E-Commerce Public Dataset | Analytical Dashboard")


if __name__ == "__main__":
    main()