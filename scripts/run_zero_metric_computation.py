from echidna import netsim_summariser as summ
from pathlib import Path
import numpy as np
import polars as pl

root = Path(__file__).resolve().parent.parent.resolve()
simulation_archives = {
    folder.stem: folder / "sim_all_30s.h5"
    for folder in (root / "simulations/zero_sims_resized").iterdir()
}

processed_results = dict()

for label, archive in simulation_archives.items():
    print(f"Analysing: {label}")
    summariser = summ.Summariser(archive)
    result_df = summariser.metrics(ncpus=6, no_move=True, drop=[]).add_extent(30)
    result_df.write_parquet(archive.with_name("metrics_30s.parquet"))
    processed_results[label] = result_df

