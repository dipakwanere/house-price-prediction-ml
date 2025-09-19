import os
import sys
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


COMPETITION = "house-prices-advanced-regression-techniques"
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"


def ensure_kaggle_credentials() -> None:
    """Validate Kaggle credentials location and permissions.

    Reference: Kaggle official API docs and repo.
    """
    # Allow project-local config dir via KAGGLE_CONFIG_DIR; fall back to ~/.kaggle
    cfg_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    kaggle_json = (
        Path(cfg_dir) / "kaggle.json"
        if cfg_dir
        else Path.home() / ".kaggle" / "kaggle.json"
    )
    if not kaggle_json.exists():
        hint = (
            f"Not found: {kaggle_json}. Place kaggle.json under KAGGLE_CONFIG_DIR or ~/.kaggle. "
            "Docs: https://github.com/Kaggle/kaggle-api/"
        )
        raise FileNotFoundError(hint)

    try:
        # chmod may fail on Windows; ignore errors there
        os.chmod(kaggle_json, 0o600)
    except Exception:
        if sys.platform.startswith("win"):
            # On Windows, permissions are handled differently; continue
            pass
        else:
            raise


def download_competition_files() -> None:
    ensure_kaggle_credentials()
    api = KaggleApi()
    api.authenticate()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # Download train.csv and test.csv (and data description) to data/raw
    api.competition_download_file(COMPETITION, "train.csv", path=str(RAW_DIR))
    api.competition_download_file(COMPETITION, "test.csv", path=str(RAW_DIR))
    api.competition_download_file(COMPETITION, "sample_submission.csv", path=str(RAW_DIR))
    api.competition_download_file(COMPETITION, "data_description.txt", path=str(RAW_DIR))

    # The API downloads as .csv.zip — unzip if needed
    for zipped in RAW_DIR.glob("*.zip"):
        import zipfile

        with zipfile.ZipFile(zipped, "r") as zf:
            zf.extractall(RAW_DIR)
        zipped.unlink()


if __name__ == "__main__":
    download_competition_files()


