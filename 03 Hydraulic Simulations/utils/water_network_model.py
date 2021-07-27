"""
erytheis github
"""

import wntr
import pandas as pd
from utils.math import convert_flow_to_hourly


class EpanetModel:

    def __init__(self, epanet_file_path):
        self.epanet_file_path = epanet_file_path
        self.model = wntr.network.WaterNetworkModel(epanet_file_path)

    def simulate(self):
        self.sim = wntr.sim.EpanetSimulator(self.model)
        self.results = self.sim.run_sim()
        return self.results

    def get_results(self):
        return self.results

    def get_roughnesses(self):
        self.roughnessees = {}
        for pipe_name, pipe in self.model.pipes():
            self.roughnessees[pipe_name] = pipe.roughness
        self.roughnessees = pd.DataFrame(self.roughnessees)
        return self.roughnessees

    def get_node_type(self):
        self.node_types = {}
        for node_name, node in self.model.nodes():
            self.node_types[node_name] = node.node_type
        self.node_types = pd.DataFrame(self.node_types)
        return self.node_types

    def get_simulated_pressures(self):
        return self.results.node['pressure']

    def get_nominal_demands(self, convert_to_hourly = True):
        """
        Returns nominal demands from the Epanet file
        :return: pd.DataFrame with {node_name: Q}
        """
        self.nominal_demands = {}
        for node_name, node in self.model.nodes():
            if node.node_type == 'Junction':
                self.nominal_demands[node_name] = node.base_demand

        self.nominal_demands = pd.DataFrame([self.nominal_demands])
        if convert_to_hourly:
            self.nominal_demands_per_hour = convert_flow_to_hourly(self.nominal_demands)
            return self.nominal_demands_per_hour
        else:
            return self.nominal_demands


if __name__ == '__main__':
    from utils.definitions import RAW_DATA_DIR
    from src.data_loader.dataset import get_demands
    import numpy as np

    epanet_file_path = RAW_DATA_DIR / 'EPANET' / 'L-TOWN.inp'
    wn = EpanetModel(epanet_file_path)
    wn_no_leaks = EpanetModel(epanet_file_path)

    real_demands = get_demands()
    nominal_demands = wn.get_nominal_demands()

    wn_no_leaks.simulate()
    pressures = wn_no_leaks.get_simulated_pressures()

    node = wn.model.get_node('n6')
    node.add_leak(wn.model, area = 0.15, start_time = 2*3600, end_time = 1000*3600)

    wn.simulate()
    pressures_with_leakage = wn.get_simulated_pressures()

    pressure_difference = pressures - pressures_with_leakage
    a = np.argmax(pressure_difference.sum(axis = 1))
