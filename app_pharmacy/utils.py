import importlib

REQUIRED_PACKAGES = [
    "osmnx",
    "geopandas",
    "pandas",
    "numpy",
    "folium",
    "h3",
    "matplotlib",
    "seaborn",
    "shapely",
    "geopy",
    "plotly",
    "scikit-learn",
    "imblearn",  # imbalanced-learn
    "joblib",
    "catboost",
    "optuna"
]

def check_and_install_dependencies():
    """Проверяет наличие необходимых библиотек и предлагает их установку."""
    missing = []
    for package in REQUIRED_PACKAGES:
        try:
            import_name = package
            if package == "scikit-learn":
                import_name = "sklearn"
            elif package == "imblearn":
                import_name = "imblearn" 
            
            importlib.import_module(import_name)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"⚠️ Отсутствуют следующие библиотеки: {', '.join(missing)}")
        print("Рекомендуется установить их с помощью команды:")
        print(f"uv pip install {' '.join(missing)}")
        return False
    
    print("✅ Все необходимые библиотеки установлены.")
    return True

