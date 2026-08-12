import pandas as pd

PACKAGE = "fs437_package.hdf5"

METADATA_KEY = "/fs437_wholelife_metadata"
EVENTS_KEY = "/fs437_wholelife_events"
STIMULATIONS_KEY = "/fs437_wholelife_stimulations"


def load_metadata(package_path: str = PACKAGE) -> pd.DataFrame:
    return pd.read_hdf(package_path, key=METADATA_KEY)


def load_events(package_path: str = PACKAGE) -> pd.DataFrame:
    return pd.read_hdf(package_path, key=EVENTS_KEY)


def load_stimulations(package_path: str = PACKAGE) -> pd.DataFrame:
    return pd.read_hdf(package_path, key=STIMULATIONS_KEY)
