import netsim_summariser as summ

sim_archive = "small_case/simulations.h5"
summariser = summ.Summariser(sim_archive)
result_df = summariser.metrics(ncpus=6, no_move=True, drop=[])
export_file = "small_case/metrics.parquet"
result_df.write_parquet(export_file)
