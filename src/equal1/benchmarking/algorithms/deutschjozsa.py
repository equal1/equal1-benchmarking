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
        self.job_results = []
        self.distribtions = []
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
        # extra = 0
        # if self.device_name == "bell1-6":
        #     if self.n_qubits == 5:
        #         extra = 1
        qc = qiskit.QuantumCircuit(self.n_qubits + 1, self.n_qubits)  # + extra)
        qc.x(self.n_qubits)
        qc.h(range(self.n_qubits + 1))
        qc.barrier()
        qc = self._make_oracle(qc, hidden_string)
        qc.barrier()
        qc.h(range(self.n_qubits))
        qc.barrier()
        qc.measure(range(self.n_qubits), range(self.n_qubits))
        # if self.device_name == "bell1-6":
        #     if qc.num_qubits == 6:
        #         qc.measure(5, 5)  # need to measure the last ancilla for it to run
        return qc

    def analyse_results(self):
        self.results_analysis = {}
        for hidden_string, result in zip(self.hidden_strings, self.distribtions):
            result_sub = subsystem_counts(result, list(range(self.n_qubits)))
            if hidden_string in result_sub:
                self.results_analysis[hidden_string] = result
        return self.results_analysis

    def plot_graph(self, ax, graph_name):
        probabilities = self.results_analysis[graph_name]
        strings = sorted(list(probabilities.keys()))
        freq = [probabilities.get(s, 0) for s in strings]

        ax.bar(strings, freq)
        ax.set_xlabel("Measured String")
        ax.set_ylabel("Probability of Measurement")
        ax.bar([graph_name], [probabilities.get(graph_name, 0)], color="orange")
        ax.set_title("Bernstein-Vazirani Success Rates")
        ax.set_ylim(0, 1)
        return ax


def subsystem_counts(counts: dict[str, int], subsystem: list[int]):
    def selected_state(s: str):
        substring = "".join([c for i, c in enumerate(s[::-1]) if i in subsystem])
        return substring[::-1]

    subsystem_count = {}
    for k, v in counts.items():
        subsystem_state = selected_state(k)
        subsystem_count[subsystem_state] = subsystem_count.get(subsystem_state, 0) + v

    return subsystem_count
