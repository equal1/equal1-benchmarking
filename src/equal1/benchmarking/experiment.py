class Experiment:
    def __init__(self):
        pass

    def run(self):
        self.start_jobs()
        self.collect_results()

    def start_jobs(self, n_jobs=10):
        device = qbraid_wrapper.get_device(self.backend)

        self.jobs = []
        for circuit in self.circuits:
            job = qbraid_wrapper.run_circuit(
                qc=circuit,
                device=device,
                shots=self.shots,
                get_probabilities=False,
                simulation_platform="CPU",
                force_bypass_transpilation=False,
                optimization_level=0,
                return_transpiled_circuits=False,
            )
            self.jobs.append(job)
        return self.jobs

    def collect_results(self):
        self.results = qbraid_wrapper.get_results_from_jobs(
            self.jobs, shots=self.shots, get_probabilities=True
        )
        return self.results

    def plot_graph(self, ax, graph_name):
        pass

    def plot_all_graphs(self):
        pass

    def analyse_results(self):
        pass

    def save_results(self, filename):
        raise NotImplementedError("Results saving not implemented.")

    def load_results(self, filename):
        raise NotImplementedError("Results loading not implemented.")
