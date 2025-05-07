# assumes that metrics are computed
import numpy as np
from pathlib import Path
import polars as pl
from matplotlib import pyplot as plt
import seaborn as sns

root = Path(__file__).resolve().parent.parent.resolve()

metrics_files = {
    folder.stem: folder / "metrics_30s.parquet"
    for folder in (root / "simulations/zero_sims_resized").iterdir()
}

metrics = {
    label: pl.read_parquet(file)
    for label , file in metrics_files.items()
}

_hitting_time_columns = pl.selectors.starts_with("hitting_time_")
agg_metrics = dict()
hitting_time_dists = dict()

for model, df in metrics.items():
    agg_metric = (df
                  .unpivot(
                      on=_hitting_time_columns,
                      index='seed',
                      variable_name='target',
                      value_name='hitting_time',
                  )
                  .with_columns(
                      target_seed = (
                          pl.col('target')
                          .str.strip_prefix('hitting_time_')
                          .str.to_integer()
                      )
                  )
                  .drop('target')
                 ) 
    agg_metrics[model] = agg_metric
    hitting_time_dists[model] = np.sort(agg_metric.select("hitting_time").to_series().to_numpy())

