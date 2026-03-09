# PharmaLoc

PharmaLoc is a geospatial machine learning project for identifying promising pharmacy locations in Moscow. The project combines OpenStreetMap data, data.mos.ru enrichment, H3 spatial indexing, feature engineering, and ensemble models to rank candidate locations.

## Project Structure

The main application code lives in `app_pharmacy/`.

- `main.py` orchestrates the end-to-end pipeline.
- `data_loader.py` and `data_mos.py` load source data.
- `features.py` and `parallel_processing.py` generate spatial features.
- `modeling.py` trains and validates machine learning models.
- `analysis.py` scores locations and prepares recommendations.
- `visualization.py` exports plots and interactive maps.

Additional project materials are stored in `docs/`, while `iat-typst-template.typ` contains the thesis manuscript source.

## Requirements

- Python 3.10+
- Dependencies from `app_pharmacy/requirements.txt`

Install the environment with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r app_pharmacy/requirements.txt
```

## Running the Pipeline

Train the model and exclude leakage-prone features:

```bash
python3 -m app_pharmacy.main --train --no-leakage
```

Validate the trained model on the held-out district:

```bash
python3 -m app_pharmacy.main --validate
```

Reuse cached data and an existing model:

```bash
python3 -m app_pharmacy.main --load --skip-data
```

## Outputs

After a successful run, generated artifacts typically appear in `data/`:

- `potential_map.html` with the interactive recommendation map
- `top_10_locations_summary.csv` with ranked candidate locations
- diagnostic plots and model evaluation figures used in the thesis

## Thesis Context

This repository accompanies the final certification work on pharmacy location optimization using machine learning and OpenStreetMap spatial data. The manuscript source is maintained in Typst and can be compiled from `iat-typst-template.typ`.
