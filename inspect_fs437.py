import pandas as pd

PACKAGE = "fs437_package.hdf5"
RAW = "fs437_raw.hdf5"

print("=== PACKAGE FILE KEYS ===")
with pd.HDFStore(PACKAGE, mode="r") as store:
    print(store.keys())

print("\n=== RAW FILE KEYS ===")
with pd.HDFStore(RAW, mode="r") as store:
    print(store.keys())
