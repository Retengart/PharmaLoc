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

Generated datasets, model files, reports, maps, and private notes are local artifacts and are not part of the public repository.

## Requirements

- Python 3.10+
- Dependencies from `app_pharmacy/requirements.txt`

Run commands with `uv` and the checked-in requirements file:

```bash
uv run --with-requirements app_pharmacy/requirements.txt python -m app_pharmacy.main --help
```

## Running the Pipeline

Train the model and exclude leakage-prone features:

```bash
uv run --with-requirements app_pharmacy/requirements.txt python -m app_pharmacy.main --train --no-leakage
```

Validate the trained model on the held-out district:

```bash
uv run --with-requirements app_pharmacy/requirements.txt python -m app_pharmacy.main --validate
```

Reuse cached data and an existing model:

```bash
uv run --with-requirements app_pharmacy/requirements.txt python -m app_pharmacy.main --load --skip-data
```

## Outputs

After a successful run, generated artifacts typically appear in `data/`:

- `potential_map.html` with the interactive recommendation map
- `top_10_locations_summary.csv` with ranked candidate locations
- diagnostic plots and model evaluation figures

## Public Scope

The public repository is limited to the application source code and neutral setup instructions.
