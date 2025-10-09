from equal1.benchmarking.experiment import Experiment
import numpy as np
import qiskit
from typing import Optional


class BernsteinVazirani(Experiment):
    def __init__(
        self,
        n_qubits,
        device_name: str,
        shots=1024,
        rng=None,
        token=None,
        number_of_random_strings_to_test: int = 0,
        strings_to_test: Optional[list[str]] = None,
        runtime_options=None,
    ):
        self.n_qubits = n_qubits
        self.shots = shots
        self.device_name = device_name
        self.token = token
        self.results = []
        self.rng = np.random.default_rng(rng)
        self.results_analysis = {}
        self.runtime_options = runtime_options or {}

        if not ((not number_of_random_strings_to_test) ^ (not strings_to_test)):
            raise ValueError(
                "Either number_of_random_strings_to_test or strings_to_test must be provided, but not both."
            )
        self.hidden_strings = self._generate_hidden_strings(
            number_of_random_strings_to_test, strings_to_test
        )

        self.circuits = [
            self._make_circuit(hidden_string=s) for s in self.hidden_strings
        ]

    def _generate_hidden_strings(self, number_of_strings_to_test, strings_to_test):
        if strings_to_test:
            assert [len(s) == self.n_qubits for s in strings_to_test], (
                "All provided strings must have length equal to n_qubits"
            )
            return strings_to_test

        if number_of_strings_to_test == 2**self.n_qubits:
            return [np.binary_repr(i, self.n_qubits) for i in range(2**self.n_qubits)]

        random_numbers = self.rng.choice(
            range(2**self.n_qubits), size=number_of_strings_to_test, replace=False
        )
        return [np.binary_repr(num, self.n_qubits) for num in random_numbers]

    def _make_oracle(self, circuit, hidden_string):
        for i, bit in enumerate(reversed(hidden_string)):
            if bit == "1":
                circuit.cx(i, self.n_qubits)  # ancilla is the last one
        return circuit

    def _make_circuit(self, hidden_string):
        qc = qiskit.QuantumCircuit(self.n_qubits + 1, self.n_qubits)
        qc.x(self.n_qubits)
        qc.h(range(self.n_qubits + 1))
        qc.barrier()
        qc = self._make_oracle(qc, hidden_string)
        qc.barrier()
        qc.h(range(self.n_qubits))
        qc.barrier()
        qc.measure(range(self.n_qubits), range(self.n_qubits))
        return qc

    def analyse_results(self):
        self.results_analysis = {}
        for hidden_string, result in zip(self.hidden_strings, self.results):
            percent_success = result.get(hidden_string, 0)
            self.results_analysis[hidden_string] = percent_success

    def plot_graph(self, ax, graph_name):
        if graph_name == "success":
            self._plot_success(ax)
        else:
            raise ValueError(f"Unknown graph name: {graph_name}")

    def _plot_success(self, ax):
        if not self.results_analysis:
            raise ValueError(
                "No analysis results to plot. Run analyse_results() first."
            )

        strings = list(self.results_analysis.keys())
        success_rates = [
            self.results_analysis[s] / self.shots for s in self.results_analysis
        ]

        ax.bar(strings, success_rates)
        ax.set_xlabel("Hidden String")
        ax.set_ylabel("Success Rate")
        ax.set_title("Bernstein-Vazirani Success Rates")
        ax.set_ylim(0, 1)
