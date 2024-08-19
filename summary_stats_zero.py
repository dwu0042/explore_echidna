"""Compute summary statistics of the various types of simulation realisations for comparison"""

import re
from pathlib import Path
from typing import Iterable
from matplotlib import pyplot as plt
import numpy as np
import polars as pl
import h5py

"""
- hitting time
- raw movement numbers
- "steady state" infection numbers
"""

class Realisation:
    __slots__ = [
        "path",
        "ts",
        "history",
        "mover_out",
        "mover_in",
    ]

    base_name_patterns = {
        'zero_delay' : re.compile(r"sim_(\d+)_(\d+)_(\d+)"),
        'delay_sim': re.compile(r"sim_(\d+)_(\d+)_(\d+)_(\d+)"),
    }

    base_name_schema = {
        'zero_delay': ('seed', 'simdate', 'simtime'),
        'delay_sim': ('seed', 'delay', 'simdate', 'simtime'),
    }

    def __init__(self, file):
        self.path = Path(file)
        fl = np.load(file)
        for name, value in fl.items():
            setattr(self, name, value)
        
        self.mover_out = self.mover_out.squeeze().T
        self.mover_in = self.mover_in.squeeze().T

    def __contains__(self, key):
        return key in self.__dict__

    def hitting_time(self, null=np.nan):
        mask = (self.history != 0)
        return np.where(mask.any(axis=1), mask.argmax(axis=1), null)

    def movement_out_total(self):
        return self.mover_out.sum(axis=1)

    def movement_in_total(self):
        return self.mover_in.sum(axis=1)

    def steady_state_infected(self):
        init = self.history[:,0].sum()
        return self.history.sum(axis=0) / init

    def emptying_time(self, null=np.nan):
        xs = self.history.sum(axis=0)
        mask = (xs == 0)
        return mask.argmax() if mask.any() else null

    def name_patterns(self, pattern_style: str):
        return self.extract_name_patterns(self.path, pattern_style)

    @staticmethod
    def extract_name_patterns(path: Path, pattern_style: str):
        pattern = Realisation.base_name_patterns[pattern_style]
        name = str(path.stem)
        matches = pattern.match(name)
        if matches:
            return [int(x) for x in matches.groups()]
        else:
            return None



class RealisationArray():

    def __init__(self, rootpath, pattern_style='zero_delay'):
        self.rootpath = Path(rootpath)
        self.name_pattern = pattern_style
        # use numpy array for smart indexing later
        self.realisations = np.array(list(self.rootpath.glob("sim_*.npz")))
        self._metadata = None

    def __iter__(self):
        for rlz in self.realisations:
            yield rlz

    def collect(self, metric: str, filter=slice(0, None), *args, **kwargs):
        collation = []
        for rlz in self.realisations[filter]:
            collation.append(getattr(Realisation(rlz), metric)(*args, **kwargs))
        return collation

    def multicollect(self, metrics: Iterable[str], filter=slice(0, None), *args, **kwargs):
        collation = {metric:[] for metric in metrics}
        for rlz in self.realisations[filter]:
            for metric in metrics:
                collation[metric].append(getattr(Realisation(rlz), metric)(*args, **kwargs))
        return collation

    @property
    def metadata(self):
        if self._metadata is None:
            raw_metadata = [Realisation.extract_name_patterns(path, self.name_pattern) 
                              for path in self.realisations]
            self._metadata = pl.from_records(
                raw_metadata, 
                schema=Realisation.base_name_schema[self.name_pattern]
            ).with_row_count()
        return self._metadata
    

    def to_hdf5(self, out):
        with h5py.File(out, 'w') as h5f:
            for rlz in self.metadata.to_dicts():
                g = h5f.create_group(f"sim_{rlz['row_nr']}")
                g.attrs['seed'] = rlz['seed']
                g.attrs['simdate'] = rlz['simdate']
                g.attrs['simtime'] = rlz['simtime']
                rfl = self.realisations[rlz['row_nr']]
                with np.load(rfl) as ifp:
                    tarr = ifp.get('ts') or ifp['t']
                    g.create_dataset('ts', data=tarr.flatten(), compression='gzip')
                    g.create_dataset('history', data=ifp['history'], compression='gzip')
                    g.create_dataset('mover_in', data=np.squeeze(ifp['mover_in']), compression='gzip')
                    g.create_dataset('mover_out', data=np.squeeze(ifp['mover_out']), compression='gzip')