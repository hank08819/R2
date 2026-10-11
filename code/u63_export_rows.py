"""u63_export_rows.py -- the raw rows of the two real-shift cases, for the LLM embedding (u64).

Saves data/real_rows/acs_<ST>_<YEAR>.npz (X in FEAT order, y) for California 2018 and the 52
deployments, and beijing.npz (X in u61's column order, its column names).  Row order is
exactly what u59.load and u61.load return, so an embedding row i belongs to data row i.
"""
import importlib.util, warnings; warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'data' / 'real_rows'; OUT.mkdir(parents=True, exist_ok=True)
def mod(name, fn):
    s = importlib.util.spec_from_file_location(name, HERE / fn); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
u59 = mod('u59', 'u59_acs_real_shift.py')
from folktables.load_acs import state_list
keys = [('CA', '2018')] + [(s, '2018') for s in state_list if s not in ('CA', 'PR')] + [('CA', y) for y in ('2019', '2021', '2022')]
n = 0
for st, yr in keys:
    X, y = u59.load(st, yr); u59._cache.clear()
    np.savez_compressed(OUT / f'acs_{st}_{yr}.npz', X=X, y=y); n += len(y)
print(f'ACS: {len(keys)} files, {n:,} rows')
src = open(HERE / 'u61_beijing_real_shift.py').read().split('rows, probs = [], []')[0]
ns = {'__file__': str(HERE / 'u61_beijing_real_shift.py')}; exec(compile(src, 'u61_head', 'exec'), ns)
d, X = ns['load']()
wind = sorted(c for c in d.cbwd.unique())
cols = ['month', 'hour', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir'] + [f'wd_{w}' for w in wind]
np.savez_compressed(OUT / 'beijing.npz', X=X, y=d.y.values, year=d.year.values, cols=np.array(cols))
print(f'Beijing: {len(X):,} rows, columns {cols}')
