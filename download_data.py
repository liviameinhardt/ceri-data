# %%
from pathlib import Path
import cdsapi
import zipfile

# ======================
# CONSTANTS
# ======================

DATASET = "projections-cmip6"

MONTHS = [
    "01","02","03","04","05","06",
    "07","08","09","10","11","12"
]

VARIABLES = [
    "daily_maximum_near_surface_air_temperature",
    "daily_minimum_near_surface_air_temperature",
    "near_surface_air_temperature",
    "near_surface_specific_humidity",
    "near_surface_wind_speed",
    "precipitation",

]

MODELS = [
    "miroc6",
    "mri_esm2_0",
    "noresm2_mm",
    "gfdl_esm4",
    "access_esm1_5",
    "ipsl_cm6a_lr"
]

AREA = [6, -75, -35, 36]

BASE_DIR = Path("cmip6")


# ======================
# CLIENT
# ======================

client = cdsapi.Client(
    url="https://cds.climate.copernicus.eu/api",
    key="b0474ce3-c4ed-4747-b666-66dcca14d8ea"
)


# ======================
# CONFIGURATION
# ======================

historical_config = [
    {
        "years": range(1995, 2015)
    }
]

future_periods = [
    {"years": range(2021, 2041)},
    {"years": range(2031, 2051)},
    {"years": range(2041, 2061)},
]

SCENARIOS = {
    "historical": historical_config,
    "ssp1_2_6": future_periods,
    "ssp2_4_5": future_periods,
    "ssp3_7_0": future_periods,
}


# ======================
# FUNCTIONS
# ======================

def unzip_and_delete(zip_file, destination):

    with zipfile.ZipFile(zip_file, 'r') as z:
        z.extractall(destination)

    zip_file.unlink()


def download_variable(experiment, variable, model, years, save_dir):

    request = {
        "temporal_resolution": "daily",
        "experiment": experiment,
        "variable": variable,
        "model": model,
        "month": MONTHS,
        "year": [str(y) for y in years],
        "area":AREA
    }

    zip_path = client.retrieve(DATASET, request).download()

    unzip_and_delete(Path(zip_path), save_dir)


def download_scenario(scenario, configs):

    scenario_dir = BASE_DIR / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        for var in VARIABLES:
            for model in MODELS:

                print(f"{scenario} | {model} | {var}")

                try:
                    download_variable(
                        experiment=scenario,
                        variable=var,
                        model=model,
                        years=cfg["years"],
                        save_dir=scenario_dir
                    )
                    
                except Exception as e:
                    with open("error_log.txt", "a") as f:
                        f.write(
                            f"{scenario} | {model} | {var} | "
                            f"years={cfg['years']} | Error: {str(e)}\n"
                        )

# ======================
# MAIN
# ======================

def main():

    for scenario, configs in SCENARIOS.items():
        download_scenario(scenario, configs)


if __name__ == "__main__":
    main()