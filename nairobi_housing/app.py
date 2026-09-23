from pathlib import Path
import json
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
COLUMNS_PATH = BASE_DIR / "model_columns.json"
LISTINGS_PATH = BASE_DIR / "listings.json"

app = Flask(__name__)


def load_json(path):
	with path.open(encoding="utf-8") as file:
		return json.load(file)


def load_artifacts():
	with MODEL_PATH.open("rb") as file:
		model = pickle.load(file)
	columns = load_json(COLUMNS_PATH)
	listings = load_json(LISTINGS_PATH)
	return model, columns, listings


def get_options(columns):
	locations = sorted(
		column.removeprefix("Location_")
		for column in columns
		if column.startswith("Location_")
	)
	property_types = sorted(
		column.removeprefix("propertyType_")
		for column in columns
		if column.startswith("propertyType_")
	)
	return locations, property_types


def artifact_error():
	return (
		"Model artifacts are not ready. Run the training notebook to create "
		"model.pkl, model_columns.json, and listings.json."
	)


@app.route("/")
def home():
	try:
		_, columns, listings = load_artifacts()
		locations, property_types = get_options(columns)
		error = None
	except (OSError, EOFError, json.JSONDecodeError, pickle.UnpicklingError, TypeError):
		locations, property_types, listings = [], [], []
		error = artifact_error()

	return render_template(
		"index.html",
		listings=listings,
		locations=locations,
		property_types=property_types,
		error=error,
	)


@app.route("/predict", methods=["POST"])
def predict():
	try:
		model, columns, _ = load_artifacts()
		data = request.get_json(silent=True) or {}
		bedroom = int(data["bedroom"])
		bathroom = int(data["bathroom"])
		if bedroom < 1 or bathroom < 1:
			raise ValueError

		row = {column: 0 for column in columns}
		row["Bedroom"] = bedroom
		row["bathroom"] = bathroom
		for prefix, key in (("Location_", "location"), ("propertyType_", "property_type")):
			option_column = f"{prefix}{data.get(key, '')}"
			if option_column in row:
				row[option_column] = 1

		features = pd.DataFrame([row], columns=columns)
		price = float(model.predict(features)[0])
		return jsonify(price=round(price, 2), price_formatted=f"KSh {price:,.0f}")
	except (OSError, EOFError, json.JSONDecodeError, pickle.UnpicklingError, TypeError):
		return jsonify(error=artifact_error()), 503
	except (KeyError, ValueError):
		return jsonify(error="Enter valid bedroom and bathroom values."), 400


if __name__ == "__main__":
	app.run(debug=True)
