from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ENGINE_SIZE_THRESHOLD = 3.0


def find_csv() -> Path:
	workspace = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
	csv_files = sorted(workspace.glob("*.csv"))
	if not csv_files:
		raise FileNotFoundError("No CSV file was found in the project folder.")
	return csv_files[0]


def load_data(csv_path: Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path, low_memory=False)
	for column in ["displ", "comb08", "year"]:
		df[column] = pd.to_numeric(df[column], errors="coerce")
	return df


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


def percent_summary(series: pd.Series) -> pd.DataFrame:
	counts = series.value_counts().rename_axis("category").reset_index(name="count")
	counts["percent"] = counts["count"] / counts["count"].sum() * 100
	return counts


def ev_vs_gas_summary(df: pd.DataFrame) -> pd.DataFrame:
	ev_count = int(df["fuelType1"].eq("Electricity").sum())
	gas_count = int(df["fuelType1"].astype(str).str.contains("Gasoline", case=False, na=False).sum())
	summary = pd.DataFrame(
		{
			"category": ["Electric vehicles", "Gasoline vehicles"],
			"count": [ev_count, gas_count],
		}
	)
	summary["percent"] = summary["count"] / len(df) * 100
	return summary


def engine_size_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
	engine_df = df[df["displ"].notna() & (df["displ"] > 0)].copy()
	engine_df["engine_size"] = engine_df["displ"].apply(
		lambda value: f"Big engine (>={ENGINE_SIZE_THRESHOLD:.1f}L)" if value >= ENGINE_SIZE_THRESHOLD else f"Small engine (<{ENGINE_SIZE_THRESHOLD:.1f}L)"
	)
	return percent_summary(engine_df["engine_size"]), len(df) - len(engine_df)


def fuel_economy_summary(df: pd.DataFrame) -> pd.DataFrame:
	grouped = (
		df.groupby(df["fuelType1"].replace({"Electricity": "Electric"}))["comb08"]
		.agg(["count", "mean", "median"])
		.sort_values("mean", ascending=False)
		.reset_index()
	)
	grouped.rename(columns={"fuelType1": "fuel_type", "mean": "average_combined_mpg", "median": "median_combined_mpg"}, inplace=True)
	return grouped


def body_style_summary(df: pd.DataFrame) -> pd.DataFrame:
	body_style = df["VClass"].apply(classify_body_style)
	return percent_summary(body_style)


def print_section(title: str) -> None:
	print(f"\n{title}")
	print("-" * len(title))


def print_summary(df: pd.DataFrame, source_path: Path, missing_engine_rows: int) -> None:
	print(f"Dataset: {source_path.name}")
	print(f"Rows: {len(df):,}")
	print(f"Columns: {len(df.columns)}")
	print(f"Year range: {int(df['year'].min())} to {int(df['year'].max())}")

	print_section("EV vs Gas Vehicles")
	print(ev_vs_gas_summary(df).to_string(index=False, formatters={"percent": "{:.2f}%".format}))

	engine_summary, _ = engine_size_summary(df)
	print_section("Big Engines vs Small Engines")
	print(engine_summary.to_string(index=False, formatters={"percent": "{:.2f}%".format}))
	print(f"Rows excluded from engine-size comparison: {missing_engine_rows:,}")
	print(f"Engine-size rule: big engines are {ENGINE_SIZE_THRESHOLD:.1f}L or larger.")

	fuel_summary = fuel_economy_summary(df)
	print_section("Fuel Economy by Fuel Type")
	print(fuel_summary.to_string(index=False, formatters={"average_combined_mpg": "{:.2f}".format, "median_combined_mpg": "{:.2f}".format}))
	print("Note: electric vehicles use MPGe in this dataset, so direct MPG comparisons should be discussed carefully.")

	print_section("Body Style Distribution")
	print(body_style_summary(df).to_string(index=False, formatters={"percent": "{:.2f}%".format}))


def save_bar_chart(data: pd.DataFrame, x_col: str, y_col: str, title: str, output_path: Path, color: str) -> None:
	plt.figure(figsize=(10, 6))
	plt.bar(data[x_col], data[y_col], color=color)
	plt.title(title)
	plt.ylabel(y_col.replace("_", " ").title())
	plt.xticks(rotation=20, ha="right")
	plt.tight_layout()
	plt.savefig(output_path)
	plt.close()


def save_charts(df: pd.DataFrame, output_dir: Path) -> None:
	output_dir.mkdir(exist_ok=True)

	ev_gas = ev_vs_gas_summary(df)
	save_bar_chart(ev_gas, "category", "percent", "EV vs Gas Vehicles", output_dir / "ev_vs_gas_percent.png", "seagreen")

	engine_summary, _ = engine_size_summary(df)
	save_bar_chart(engine_summary, "category", "percent", "Big vs Small Engines", output_dir / "engine_size_percent.png", "steelblue")

	body_summary = body_style_summary(df)
	save_bar_chart(body_summary, "category", "percent", "Body Style Distribution", output_dir / "body_style_percent.png", "darkorange")

	fuel_summary = fuel_economy_summary(df).head(6)
	save_bar_chart(fuel_summary, "fuel_type", "average_combined_mpg", "Average Fuel Economy by Fuel Type", output_dir / "fuel_economy_by_fuel_type.png", "slategray")


def main() -> None:
	csv_path = find_csv()
	df = load_data(csv_path)
	_, missing_engine_rows = engine_size_summary(df)
	print_summary(df, csv_path, missing_engine_rows)

	output_dir = csv_path.parent / "analysis_outputs"
	save_charts(df, output_dir)
	print_section("Chart Files")
	for chart_path in sorted(output_dir.glob("*.png")):
		print(chart_path.name)


if __name__ == "__main__":
	main()