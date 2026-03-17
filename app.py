from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
import base64

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

# ── Dark car-themed chart style ───────────────────────
_BG = "#161b27"
_BG_AXES = "#1e2535"
_INK = "#e2e8f0"
_INK_MUTED = "#94a3b8"
_BORDER = "#2d3748"
mpl.rcParams.update({
    "figure.facecolor": _BG,
    "axes.facecolor": _BG_AXES,
    "axes.edgecolor": _BORDER,
    "axes.labelcolor": _INK_MUTED,
    "axes.titlecolor": _INK,
    "xtick.color": _INK_MUTED,
    "ytick.color": _INK_MUTED,
    "text.color": _INK,
    "grid.color": _BORDER,
    "grid.alpha": 0.6,
    "legend.facecolor": _BG,
    "legend.edgecolor": _BORDER,
    "legend.labelcolor": _INK,
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "sans-serif"],
    "axes.prop_cycle": mpl.cycler(
        color=["#60a5fa", "#f472b6", "#34d399", "#fbbf24", "#a78bfa", "#fb923c"]
    ),
})
from flask import Flask, render_template


app = Flask(__name__)
DATA_PATH = Path(__file__).resolve().parent / "vehicles.csv"


def classify_body_style(value: str) -> str:
    text = str(value)
    if "Pickup" in text:
        return "Pickup Truck"
    if "Sport Utility Vehicle" in text or "SUV" in text:
        return "SUV"
    if "Van" in text:
        return "Van"
    if "Wagon" in text:
        return "Wagon"
    if "Car" in text or "Seaters" in text:
        return "Car"
    if "Special Purpose" in text:
        return "Special Purpose"
    return "Other"


@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find dataset at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    for column in ["year", "comb08", "displ"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["body_style"] = df["VClass"].apply(classify_body_style)
    return df


def fig_to_base64(fig: plt.Figure) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def build_fuel_economy_chart(df: pd.DataFrame) -> tuple[str, dict[str, float]]:
    yearly = (
        df.dropna(subset=["year", "comb08"])
        .groupby("year", as_index=False)["comb08"]
        .mean()
        .sort_values("year")
    )

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(yearly["year"], yearly["comb08"], color="#2dd4bf", linewidth=2.4)
    ax.set_title("Average Fuel Economy Over Time")
    ax.set_xlabel("Model Year")
    ax.set_ylabel("Average Combined MPG")
    ax.grid(alpha=0.3)

    details = {
        "start_year": int(yearly["year"].iloc[0]),
        "end_year": int(yearly["year"].iloc[-1]),
        "start_value": float(yearly["comb08"].iloc[0]),
        "end_value": float(yearly["comb08"].iloc[-1]),
    }
    return fig_to_base64(fig), details


def build_engine_size_chart(df: pd.DataFrame) -> tuple[str, dict[str, float]]:
    yearly = (
        df.dropna(subset=["year", "displ"])
        .query("displ > 0")
        .groupby("year", as_index=False)["displ"]
        .mean()
        .sort_values("year")
    )

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(yearly["year"], yearly["displ"], color="#60a5fa", linewidth=2.4)
    ax.set_title("Average Engine Size Over Time")
    ax.set_xlabel("Model Year")
    ax.set_ylabel("Average Engine Displacement (L)")
    ax.grid(alpha=0.3)

    details = {
        "start_year": int(yearly["year"].iloc[0]),
        "end_year": int(yearly["year"].iloc[-1]),
        "start_value": float(yearly["displ"].iloc[0]),
        "end_value": float(yearly["displ"].iloc[-1]),
    }
    return fig_to_base64(fig), details


def build_body_style_chart(df: pd.DataFrame) -> tuple[str, dict[str, str]]:
    body_counts = (
        df.dropna(subset=["year"]) 
        .groupby(["year", "body_style"]) 
        .size() 
        .rename("count") 
        .reset_index()
    )

    top_styles = (
        body_counts.groupby("body_style")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .index
        .tolist()
    )

    filtered = body_counts[body_counts["body_style"].isin(top_styles)].copy()
    pivot = filtered.pivot(index="year", columns="body_style", values="count").fillna(0)
    shares = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(9, 5.2))
    shares.plot.area(ax=ax, linewidth=0)
    ax.set_title("Body Style Share by Year (Top 5 Styles)")
    ax.set_xlabel("Model Year")
    ax.set_ylabel("Share of Vehicles (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=8)

    latest_year = int(shares.index.max())
    leader = shares.loc[latest_year].sort_values(ascending=False).index[0]
    details = {
        "latest_year": str(latest_year),
        "leading_style": str(leader),
    }
    return fig_to_base64(fig), details


def build_body_style_displacement_chart(df: pd.DataFrame) -> tuple[str, dict[str, str]]:
    displacement_rows = (
        df.dropna(subset=["year", "displ"])
        .query("displ > 0")
        .copy()
    )

    top_styles = (
        displacement_rows["body_style"]
        .value_counts()
        .head(5)
        .index
        .tolist()
    )

    filtered = displacement_rows[displacement_rows["body_style"].isin(top_styles)]
    yearly_displ = (
        filtered.groupby(["year", "body_style"], as_index=False)["displ"]
        .mean()
        .sort_values(["body_style", "year"])
    )

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for style in top_styles:
        style_data = yearly_displ[yearly_displ["body_style"] == style]
        ax.plot(style_data["year"], style_data["displ"], linewidth=2.0, label=style)

    ax.set_title("Average Engine Displacement by Body Style Over Time")
    ax.set_xlabel("Model Year")
    ax.set_ylabel("Average Engine Displacement (L)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    latest_year = int(yearly_displ["year"].max())
    latest_slice = yearly_displ[yearly_displ["year"] == latest_year]
    highest_style = latest_slice.sort_values("displ", ascending=False).iloc[0]["body_style"]
    details = {
        "tracked_styles": ", ".join(top_styles),
        "latest_year": str(latest_year),
        "highest_style": str(highest_style),
    }
    return fig_to_base64(fig), details


def build_fuel_type_chart(df: pd.DataFrame) -> tuple[str, dict[str, str]]:
    fuel_rows = df.dropna(subset=["year", "fuelType1"]).copy()
    fuel_rows["is_diesel"] = fuel_rows["fuelType1"].astype(str).str.contains("diesel", case=False, na=False)

    yearly = (
        fuel_rows.groupby("year", as_index=False)
        .agg(total_count=("is_diesel", "size"), diesel_count=("is_diesel", "sum"))
        .sort_values("year")
    )
    yearly["diesel_share"] = (yearly["diesel_count"] / yearly["total_count"]) * 100

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(yearly["year"], yearly["diesel_share"], color="#f97316", linewidth=2.4)
    ax.set_title("Diesel Vehicle Share Over Time")
    ax.set_xlabel("Model Year")
    ax.set_ylabel("Diesel Share of Vehicles (%)")
    ax.grid(alpha=0.3)

    peak_row = yearly.sort_values("diesel_share", ascending=False).iloc[0]
    details = {
        "start_year": str(int(yearly["year"].iloc[0])),
        "end_year": str(int(yearly["year"].iloc[-1])),
        "start_share": f"{float(yearly['diesel_share'].iloc[0]):.2f}",
        "end_share": f"{float(yearly['diesel_share'].iloc[-1]):.2f}",
        "peak_year": str(int(peak_row["year"])),
        "peak_share": f"{float(peak_row['diesel_share']):.2f}",
    }
    return fig_to_base64(fig), details


def five_year_snapshot(df: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    years = sorted(int(y) for y in df["year"].dropna().unique() if int(y) % 5 == 0)
    for year in years:
        subset = df[df["year"] == year]
        avg_mpg = subset["comb08"].mean()
        avg_displ = subset.loc[subset["displ"] > 0, "displ"].mean()
        top_style = subset["body_style"].value_counts().idxmax()
        rows.append(
            {
                "year": str(year),
                "avg_mpg": f"{avg_mpg:.2f}",
                "avg_displ": f"{avg_displ:.2f}",
                "top_style": str(top_style),
                "sample": f"{len(subset):,}",
            }
        )
    return rows


@app.route("/")
def index() -> str:
    df = load_data()

    fuel_chart, fuel_details = build_fuel_economy_chart(df)
    engine_chart, engine_details = build_engine_size_chart(df)
    body_chart, body_details = build_body_style_chart(df)
    body_displ_chart, body_displ_details = build_body_style_displacement_chart(df)
    fuel_type_chart, fuel_type_details = build_fuel_type_chart(df)

    context = {
        "page_title": "Vehicles in California Analysis",
        "dataset_name": DATA_PATH.name,
        "rows": f"{len(df):,}",
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "notes": [
            "This dashboard is scoped to the full California dataset provided for this project.",
            "Results summarize all rows in that California dataset rather than a single city or county subset.",
            "Each row is a vehicle configuration entry, not registration or sales volume.",
            "Fuel economy values are combined MPG/MPGe from the source dataset.",
        ],
        "fuel_chart": fuel_chart,
        "engine_chart": engine_chart,
        "body_chart": body_chart,
        "body_displ_chart": body_displ_chart,
        "fuel_type_chart": fuel_type_chart,
        "fuel_details": fuel_details,
        "engine_details": engine_details,
        "body_details": body_details,
        "body_displ_details": body_displ_details,
        "fuel_type_details": fuel_type_details,
        "snapshots": five_year_snapshot(df),
    }
    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(debug=True)
