from equal1.benchmarking import qbraid_wrapper
from abc import ABC, abstractmethod
import qiskit


class Experiment(ABC):
    shots: int
    circuits: list[qiskit.QuantumCircuit]
    device: str

    def run(self, runtime_options=None):
        self.start_jobs(runtime_options)
        self.collect_results()

    def start_jobs(self, n_jobs=10):
        device = qbraid_wrapper.get_device(self.backend)

        self.jobs = []
        for circuit in self.circuits:
            job = qbraid_wrapper.run_circuit(
                qc=circuit,
                device=device,
                shots=self.shots,
                simulation_platform="CPU",
                force_bypass_transpilation=False,
                optimization_level=0,
            )
            self.jobs.append(job)
        return self.jobs

    def collect_results(self):
        self.results = qbraid_wrapper.get_results_from_jobs(
            self.jobs, shots=self.shots, get_probabilities=True
        )
        return self.results

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
