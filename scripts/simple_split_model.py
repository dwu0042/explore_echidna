"""here we test the simple model construction with an arbitrary truncated power law dist"""

import random
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt

rng = np.random.default_rng()

# generating the empirical cdf

residence_df = pl.read_parquet("./concordant_networks/readmission_time.parquet").sort('readmission_time')

mean_window = (
    residence_df
    .with_columns((pl.col('readmission_time') * pl.col('count')).alias('weighted_sum'))
    .select(pl.col('weighted_sum').sum() / pl.col('count').sum())
    .item()
)
num_readmits = residence_df.select(pl.col('count').sum()).item()


# Naive static model #

BIGN = 25_000

# draw residence times as lambda ~ 1/mean_time

naive_static_draws = [random.expovariate(1.0 / mean_window) for _ in range(BIGN)]

# Improved static model #

# compute components of the empirical cdf

threshold = 4
shorts = residence_df.filter(pl.col('readmission_time') <= threshold)
longs = residence_df.filter(pl.col('readmission_time') > threshold)

short_window = (
    shorts
    .with_columns((pl.col('readmission_time') * pl.col('count')).alias('weighted_sum'))
    .select(pl.col('weighted_sum').sum() / pl.col('count').sum())
    .item()
)
n_shorts = shorts.select(pl.col('count').sum()).item()

long_window = (
    longs
    .with_columns((pl.col('readmission_time') * pl.col('count')).alias('weighted_sum'))
    .select(pl.col('weighted_sum').sum() / pl.col('count').sum())
    .item()
)
n_longs = longs.select(pl.col('count').sum()).item()

fast_prop = n_shorts / (n_shorts + n_longs)
lambd_fast = 1 / short_window
lambd_star = lambd_fast / fast_prop
lambd_slow_a = lambd_star * (1 - fast_prop)
lambd_slow_b = 1 / (long_window - 1 / lambd_slow_a)

lambd_slow_c = 1 / (long_window - 1/ lambd_fast)

improved_static_draws = [None] * BIGN
for i in range(BIGN):
    base_time = random.expovariate(lambd_fast)
    if random.random() < fast_prop:
        improved_static_draws[i] = base_time
    else:
        improved_static_draws[i] = base_time + random.expovariate(lambd_slow_c)

# this is taken from the interpretation from the simulation code...
slowed_draws = [None] * BIGN
for i in range(BIGN):
    base_time = random.expovariate(1.0 / mean_window)
    if random.random() < fast_prop:
        slowed_draws[i] = 0
    else:
        slowed_draws[i] = 0 + random.expovariate(1/ long_window)


ecdf_x = residence_df.select('readmission_time').to_series().to_numpy()
ecdf_y = residence_df.select(1.0 - (pl.col('count').cum_sum().alias('cumcount') / num_readmits)).to_series().to_numpy()

#------------------------------------------------------------
ys = np.linspace(1, 0, BIGN)
plt.loglog(ecdf_x, ecdf_y, 'ko', label='eSF')
plt.loglog(np.sort(naive_static_draws), ys, label='naive')
plt.loglog(np.sort(improved_static_draws), ys, label='improved')
plt.loglog(np.sort(slowed_draws), ys, label='erroneous')

plt.axvline(mean_window, color='r', linestyle='dashed', label='mean')
plt.axvline(threshold, color='r', label='threshold')

print(fast_prop)

# plt.xlim(0.25, None)
plt.xscale('asinh')

plt.legend()

#--------------------------------------------------------------
plt.figure()
readmit_times = residence_df.select('readmission_time').to_series().to_numpy()
scaled_props = residence_df.select(pl.col('count') / num_readmits * BIGN).to_series().to_numpy()
plt.loglog(readmit_times, scaled_props, 'k.', label='ePDF', zorder=99)
plt.hist(naive_static_draws, bins=readmit_times, histtype='step', label='naive')
plt.hist(improved_static_draws, bins=readmit_times, histtype='step', label='improved')
plt.hist(slowed_draws, bins=readmit_times, histtype='step', label='erroneous')
plt.axvline(mean_window, color='r', linestyle='dashed', label='mean')
plt.axvline(threshold, color='r', label='threshold')

plt.xscale('asinh')
plt.legend()

#---------------------------------------------------------------
plt.figure()

SMALLN = int(BIGN / 20)
# I want to check the _timing_ of things matches up
eprops = residence_df.select(pl.col('count') / num_readmits).to_series().to_numpy()
e_random_times = np.cumsum(rng.choice(readmit_times, p=eprops, replace=True, size=SMALLN))

naive_timeline = np.cumsum(naive_static_draws[:SMALLN])
improved_timeline = np.cumsum(improved_static_draws[:SMALLN])
slow_timeline = np.cumsum(slowed_draws[:SMALLN])

cys = np.arange(SMALLN)
plt.plot(e_random_times, cys, 'ko', label='eT')
plt.plot(naive_timeline, cys, label='naive')
plt.plot(improved_timeline, cys, label='improved')
plt.plot(slow_timeline, cys, label='erroneous')

plt.legend()

#---------------------------------------------------------------
plt.show()
