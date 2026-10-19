"""u85_weather_extract.py -- cut the Shifts weather data (canonical partition) into place-and-month cells.

Reads every CSV of the canonical partition of the Shifts Weather Prediction dataset (Malinin et al.
2021, CC BY-NC-SA 4.0), assigns each row to a cell (climate type, calendar month of fact_time in UTC),
and keeps a uniform random sample of at most CAP rows per cell, with a fixed seed.  No model is fitted
and no outcome is summarized here; the only output besides the sample is the row count per cell.

    python u85_weather_extract.py <dir with the canonical CSVs> <out dir>
Writes weather_cells.csv.gz and weather_cell_counts.csv.
"""
import sys
import numpy as np, pandas as pd
from pathlib import Path

CAP, SEED, CHUNK = 45_000, 0, 500_000
src, out = Path(sys.argv[1]), Path(sys.argv[2])
files = sorted(src.rglob('*.csv'))

def cell(df):
    ym = pd.to_datetime(df.fact_time, unit='s', utc=True).dt.strftime('%Y-%m')
    return df.climate.astype(str) + '|' + ym

counts = {}
for f in files:
    for df in pd.read_csv(f, chunksize=CHUNK, usecols=['fact_time', 'climate']):
        for k, v in cell(df).value_counts().items():
            counts[k] = counts.get(k, 0) + v
print('cells', len(counts), flush=True)

rng = np.random.default_rng(SEED)
keep = []
for f in files:
    for df in pd.read_csv(f, chunksize=CHUNK):
        c = cell(df)
        p = c.map(lambda k: min(1.0, CAP / counts[k])).values
        m = rng.random(len(df)) < p
        d = df[m].copy(); d['cell'] = c[m].values; d['source'] = f.stem
        keep.append(d)
    print('read', f.name, flush=True)
d = pd.concat(keep, ignore_index=True)
d.to_csv(out / 'weather_cells.csv.gz', index=False)
pd.Series(counts, name='rows').rename_axis('cell').sort_index().to_csv(out / 'weather_cell_counts.csv')
print('EXTRACT_DONE', len(d), flush=True)
