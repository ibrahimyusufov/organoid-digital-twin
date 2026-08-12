import pandas as pd

PACKAGE = "fs437_package.hdf5"

events = pd.read_hdf(PACKAGE, key="/fs437_wholelife_events")
stims = pd.read_hdf(PACKAGE, key="/fs437_wholelife_stimulations")
metadata = pd.read_hdf(PACKAGE, key="/fs437_wholelife_metadata")

print("=== METADATA ===")
print(metadata.T)

print("\n=== EVENTS SHAPE ===")
print(events.shape)

print("\n=== EVENTS COLUMNS ===")
print(events.columns.tolist())

print("\n=== FIRST 10 SPIKE EVENTS ===")
print(events.head(10))

print("\n=== EVENT TIME RANGE ===")
print(events["time_of_event"].min())
print(events["time_of_event"].max())

print("\n=== SPIKES PER ELECTRODE ===")
print(events["electrode"].value_counts().sort_index())

print("\n=== STIMULATIONS SHAPE ===")
print(stims.shape)

print("\n=== FIRST 10 STIMULATIONS ===")
print(stims.head(10))
