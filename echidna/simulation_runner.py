import numpy as np

from . import network_conversion as conv
from . import network_simulation as simm

from os import PathLike
from typing import Mapping, Self, SupportsFloat, SupportsInt

class Runner():

    CLASSES = {
        "naive_static": {
            'converter': conv.NaiveStaticConverter,
            'simulator': simm.Simulation,
        },
        "static": {
            'converter': conv.StaticConverter,
            'simulator': simm.StaticWithHome,
        },
        "snapshot": {
            'converter': conv.SnapshotWithHomeConverter,
            'simulator': simm.SnapshotWithHome,
        },
        "temporal": {
            'converter': conv.TemporalNetworkConverter,
            'simulator': simm.TemporalSim,
        },
    }

    def __init__(
        self,
        simulation_class: str,
        graph_file: PathLike,
        ordering_file: PathLike,
        prob_final_file: PathLike,
        beta: SupportsFloat,
        output_file: PathLike,
        zero_sim = False,
        sim_n_seed_events: SupportsInt = 1,
        sim_n_seed_infections: SupportsInt = 1,
        sim_dt: SupportsFloat = 1.0,
        sim_track_movement = False,
        converter_args: Mapping | None = None,
        simulation_args: Mapping | None = None,
    ):
        
        self.simulation_class = simulation_class
        self.base_classes = self.CLASSES[simulation_class]
        
        self.ordering_file = ordering_file
        self.graph_file = graph_file
        self.prob_final_file = prob_final_file
        self.output_file = output_file
        self.beta = beta

        if converter_args is None: converter_args = dict()
        self._create_parameter_converter(**converter_args)

        if zero_sim:
            self._create_zero_parameters()
        else:
            self._create_parameters()

        self.base_sim_args = dict(
            n_seed_events=sim_n_seed_events,
            n_seed_infections=sim_n_seed_infections,
            dt=sim_dt,
            track_movement=sim_track_movement,
        )
        if simulation_args is None: simulation_args = dict()
        self.simulation_args = dict(
            **self.base_sim_args
        )
        self.simulation_args.update(simulation_args)
        self._create_simulator(
            **self.simulation_args,
        )

    def _create_parameter_converter(self, **kwargs):

        self.ordering = conv.Ordering.from_file(self.ordering_file)
        self.converter = self.base_classes['converter'].from_file(self.graph_file, ordering=self.ordering, **kwargs)

    def _create_parameters(self):

        self.prob_final = conv.ColumnDict.from_prob_final_file(self.prob_final_file)
        pf_arr = self.prob_final.organise_by(self.ordering)

        self.parameters = self.converter.map_parameters({
            'beta': self.beta,
            'prob_final_stay': pf_arr,
        })

    def _create_zero_parameters(self):

        self.prob_final = conv.ColumnDict.from_prob_final_file(self.prob_final_file)
        pf_arr = self.prob_final.organise_by(self.ordering)
        pf_zero = np.zeros_like(pf_arr)

        self.parameters = self.converter.map_parameters({
            'beta': 0.0,
            'prob_final_stay': pf_zero,
        })

    def _create_simulator(self, n_seed_events=1, n_seed_infections=1, dt=1.0, track_movement=False, pseudo_capacity=None, **simulator_args):

        if pseudo_capacity is not None:
            full_sizes = pseudo_capacity * np.ones_like(self.ordering.sizes, dtype=int)
        else:
            full_sizes = self.ordering.sizes

        # here we map converter-based args
        for arg, val in simulator_args.items():
            if callable(val):
                simulator_args[arg] = val(self.converter)

        self.sim = self.base_classes['simulator'](
            full_size=full_sizes,
            parameters=self.parameters,
            dt=dt,
            track_movement=track_movement,
            **simulator_args,
        )

        self.sim.seed(
            n_seed_events=n_seed_events,
            n_seed_number=n_seed_infections,
        )

    def run(self, until=100, nostop=False):

        self.sim.simulate(until=until, nostop=nostop)


    def export(self, to=None, identity=None, with_movers=False, **kwargs):

        if to is None:
            to = self.output_file
        
        if identity is None:
            identity = self._export_identity()

        self.sim.export_history(to, identity=identity, with_movers=with_movers, **kwargs)


    def _export_identity(self) -> str:
        
        iden = ""
        if hasattr(self.sim, 'rng'):
            iden = self.sim.rng.state['state']['state']
        
        return iden