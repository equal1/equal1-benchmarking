from equal1.benchmarking import qbraid_wrapper
from abc import ABC, abstractmethod
import qiskit
from typing import Optional


class Experiment(ABC):
    shots: int
    circuits: list[qiskit.QuantumCircuit]
    device_name: str
    runtime_options: dict

    def run(self, simulate: bool = True, runtime_options=None):
        runtime_options = runtime_options or {}

        self.start_jobs(self.device_name, simulate, runtime_options)
        self.collect_results()

    def start_jobs(
        self, device_name: Optional[str] = None, simulate=True, runtime_options=None
    ):
        device_name = device_name or self.device_name
        device, noise = qbraid_wrapper.get_device(device_name, simulation=simulate)
        print(self.runtime_options)
        combined_runtime_options = {k: v for k, v in self.runtime_options.items()}
        if runtime_options is not None:
            combined_runtime_options |= runtime_options
        print(combined_runtime_options)

        # self.jobs = []
        # for circuit in self.circuits:
        self.jobs = qbraid_wrapper.run_on_device(
            qc=self.circuits,
            device=device,
            noise_model=noise,
            shots=self.shots,
            runtime_options=combined_runtime_options,
        )
        print(self.jobs)
        # self.jobs.append(job)
        return self.jobs

    def collect_results(self):
        self.job_results = qbraid_wrapper.get_results_from_jobs(self.jobs)

        self.distribtions = qbraid_wrapper.get_state_distribution(
            self.job_results, self.shots, get_probabilities=True
        )
        return self.distribtions

    @abstractmethod
    def plot_graph(self, ax, graph_name):
        pass

    # @abstractmethod
    # def plot_all_graphs(self):
    #     pass

    @abstractmethod
    def analyse_results(self):
        pass

    def save_results(self, filename):
        raise NotImplementedError("Results saving not implemented.")

    def load_results(self, filename):
        raise NotImplementedError("Results loading not implemented.")
