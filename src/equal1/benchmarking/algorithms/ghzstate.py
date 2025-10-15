from equal1.benchmarking.experiment import Experiment
import numpy as np
import qiskit
import qiskit.circuit
from typing import Optional

# class GHZState(Experiment):
#     def __init__(
#         self,
#         max_n_qubits: int,
#         min_n_qubits: int,
#         step_n_qubits: int,
#         device_name: str,
#         shots=1024,
#         token=None,
#         # number_of_phases: Optional[int] = None,
#         runtime_options=None,
#     ):
#         self.experiments = []


class GHZStateIndividual(Experiment):
    def __init__(
        self,
        n_qubits: int,
        device_name: str,
        shots=1024,
        token=None,
        number_of_phases: Optional[int] = None,
        runtime_options=None,
    ):
        self.shots = shots
        self.device_name = device_name
        self.n_qubits = n_qubits
        self.token = token
        self.job_results = []
        self.distribtions = []
        self.results_analysis = {}
        self.number_of_phases_to_test = number_of_phases or (2 * n_qubits + 2)
        self.runtime_options = runtime_options or {}

        j = np.arange(self.number_of_phases_to_test)
        self.phase_sweep = np.pi * j / (self.number_of_phases_to_test - 1)

        self.circuits = self._make_circuts()

    def _make_circuts(self):
        circ_phase, parameters = circuit_to_measure_off_diagonal(self.n_qubits)

        return [ghz_with_shuttling(self.n_qubits, measure=True)] + [
            circ_phase.assign_parameters({pr: ph for pr in parameters})
            for ph in self.phase_sweep
        ]

    def analyse_results(self):
        p00 = [r["0" * self.n_qubits] for r in self.distribtions[1:]]
        complex_coefficients = np.exp(1j * self.n_qubits * np.array(self.phase_sweep))
        # https://iopscience.iop.org/article/10.1088/2399-6528/ac1df7
        i_q = np.abs(np.sum(complex_coefficients * p00)) / (len(self.phase_sweep))
        p_0 = self.distribtions[0]["0" * self.n_qubits]
        p_1 = self.distribtions[0]["1" * self.n_qubits]
        P = p_0 + p_1
        C = 2 * np.sqrt(i_q)
        return (P + C) / 2

    def plot_graph(self, ax, graph_name):
        raise NotImplementedError

    #     probabilities = self.results_analysis[graph_name]
    #     strings = sorted(list(probabilities.keys()))
    #     freq = [probabilities.get(s, 0) for s in strings]

    #     ax.bar(strings, freq)
    #     ax.set_xlabel("Measured String")
    #     ax.set_ylabel("Probability of Measurement")
    #     ax.bar([graph_name], [probabilities.get(graph_name, 0)], color="orange")
    #     ax.set_title("Bernstein-Vazirani Success Rates")
    #     ax.set_ylim(0, 1)
    #     return ax


def ghz_with_shuttling_top(start, layer):
    gates_to_add = 2**layer
    layer_gates = []
    for gate in range(start, start - gates_to_add, -1):
        layer_gates.append((gate, gate - gates_to_add))
    return layer_gates


def ghz_with_shuttling_bot(start, layer):
    gates_to_add = 2**layer
    layer_gates = []
    for gate in range(start, start + gates_to_add, +1):
        layer_gates.append((gate, gate + gates_to_add))
    return layer_gates


def ghz_with_shuttling(n_qubits, measure=False):
    if n_qubits % 2 != 0:
        raise ValueError("Number of qubits must be even")

    circ = qiskit.QuantumCircuit(n_qubits)
    layers = int(np.ceil(np.log2(n_qubits)))

    gates = []
    mid = (n_qubits // 2) - 1
    gates.append((mid, mid + 1))
    # gates.append(("barrier", None))

    for i in range(0, layers - 1):
        gates += ghz_with_shuttling_top(mid, i)
        gates += ghz_with_shuttling_bot(mid + 1, i)
        # gates.append(("barrier", None))

    circ.h(mid)
    for g_a, g_b in gates:
        # if g_a == "barrier":
        #     pass
        # circ.barrier()
        if g_a < 0 or g_b < 0 or g_a >= n_qubits or g_b >= n_qubits:
            continue
        else:
            circ.cx(g_a, g_b)
    if measure:
        circ.measure_all()
    return circ


def circuit_to_measure_off_diagonal(n_qubits):
    ghz = ghz_with_shuttling(n_qubits)
    adj_ghz = ghz.inverse()
    parameters = [qiskit.circuit.Parameter(f"th_{i}") for i in range(n_qubits)]
    for q, p in enumerate(parameters):
        ghz.rz(p, q)
    ghz = ghz.compose(adj_ghz)
    ghz.measure_all()
    return ghz, parameters
