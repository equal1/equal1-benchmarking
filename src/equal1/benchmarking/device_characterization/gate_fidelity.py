import numpy as np
import qiskit
import qiskit.circuit.library
import qiskit.quantum_info as qqi
import scipy.optimize
import matplotlib.pyplot as plt
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.quantum_info.operators.symplectic import Clifford
from qiskit.circuit import Instruction

from equal1.benchmarking.experiment import Experiment

from equal1.benchmarking.device_characterization import randomised_benchmarking

gate_name_to_instruction = get_standard_gate_name_mapping()


def make_1q_benchmarking_circuits(
    qubit_to_test: int,
    gate_to_benchmark: Instruction,
    replicates,
    depths,
    rng,
):
    clifford_gate = qqi.Clifford.from_circuit(gate_to_benchmark)
    circuit_meta = [
        {
            "rep": rep,
            "depth": depth,
            "depth_index": d,
            "gate": clifford_gate,
            "gate_name": gate_to_benchmark.name,
        }
        for rep in range(replicates)
        for d, depth in enumerate(depths)
    ]
    circuits = [
        randomised_benchmarking.qiskit_n_qubit_clifford_benchmarking(
            [qubit_to_test], m["depth"], gate_to_benchmark=clifford_gate, rng=rng
        )
        for m in circuit_meta
    ]

    for c in circuits:
        c.measure(qubit_to_test, 0)

    return circuit_meta, circuits


def make_2q_benchmarking_circuits(
    qubits_to_test: list[int],
    gate_to_benchmark: Instruction,
    replicates,
    depths,
    rng,
    start_states=None,
):
    clifford_gate = Clifford.from_circuit(gate_to_benchmark)

    circuit_samples_ref = []
    circuit_samples_int = []
    circuit_meta = []
    start_states = start_states or ["00", "01", "10", "11"]
    for d_idx, depth in enumerate(depths):
        for start_state in start_states:
            for rep in range(replicates):
                clifford_sequence_reference = randomised_benchmarking.generate_character_rb(
                    start_state,
                    qubits_to_test,
                    qqi.Clifford.from_label("II"),
                    depth,
                    rng,
                )

                clifford_sequence_interleaved = randomised_benchmarking.generate_character_rb(
                    start_state,
                    qubits_to_benchmark=qubits_to_test,
                    gate_to_benchmark=clifford_gate,
                    length=depth,
                    rng=rng,
                )

                # Measure the circuits
                clifford_sequence_reference.barrier()
                clifford_sequence_reference.measure(qubits_to_test, [0, 1])  # type: ignore
                clifford_sequence_interleaved.barrier()
                clifford_sequence_interleaved.measure(qubits_to_test, [0, 1])  # type: ignore

                circuit_samples_ref.append(clifford_sequence_reference)
                circuit_samples_int.append(clifford_sequence_interleaved)

                circuit_meta.append(
                    {
                        "rep": rep,
                        "depth": depth,
                        "depth_index": d_idx,
                        "start_state": start_state,
                        "gate": clifford_gate,
                        "gate_name": gate_to_benchmark.name,
                    }
                )

    return circuit_meta, circuit_samples_ref, circuit_samples_int


def run_benchmarking_and_collect_results(engine, circuit_meta, circuits, shots=10_000):
    n_reps = len({m["rep"] for m in circuit_meta})
    n_depths = len({m["depth"] for m in circuit_meta})
    results = np.zeros((n_reps, n_depths), dtype=float)

    for circ, meta in zip(circuits, circuit_meta):
        engine.load_input(QiskitProgram(circ), shots=shots)
        result_map = engine.execute().results_map
        results[meta["rep"], meta["depth_index"]] = result_map.get("0" * circ.num_clbits, 0.0) / shots
    return results


def run_benchmarking_and_collect_results_crb(engine, circuit_meta, circuits, shots=10_000):
    n_reps = len({m["rep"] for m in circuit_meta})
    n_depths = len({m["depth"] for m in circuit_meta})
    start_states = ["00", "01", "10", "11"]
    results = {ss: np.zeros((n_reps, n_depths), dtype=float) for ss in start_states}

    for circ, meta in zip(circuits, circuit_meta):
        engine.load_input(QiskitProgram(circ), shots=shots)
        result_map = engine.execute().results_map
        results[meta["start_state"]][meta["rep"], meta["depth_index"]] = (
            result_map.get("0" * circ.num_clbits, 0.0) / shots
        )
    return results


class Gate1qFidelity(Experiment):
    def __init__(
        self,
        device_name: str,
        qubit_idx: int,
        gate_to_benchmark: str,
        shots: int = 10_000_000,
        depths: list[int] = [1, 2, 4, 8, 14, 24, 42, 72, 123, 211, 360, 616, 1052, 1798, 3071],
        replicates: int = 5,
    ):
        self.device_name = device_name
        self.qubit_idx = qubit_idx
        self.gate_to_benchmark = gate_name_to_instruction[gate_to_benchmark.lower()]
        self.gate_to_reference = qiskit.circuit.library.IGate()
        self.runtime_options = {"optimization_level": 0}
        self.shots = shots
        self.depths = depths
        self.replicates = replicates
        self.rng = np.random.default_rng(1337)

        self.circuits_meta = []
        self.circuits = []

        self.gates_to_benchmark = [
            self.gate_to_benchmark,
            self.gate_to_reference,
        ]
        for gate in self.gates_to_benchmark:
            circuits_meta, circuits = make_1q_benchmarking_circuits(
                qubit_idx,
                gate,
                replicates=self.replicates,
                depths=self.depths,
                rng=self.rng,
            )

            self.circuits_meta.extend(circuits_meta)
            self.circuits.extend(circuits)

        self.job_results = []
        self.distribtions = []
        self.results_analysis = {}

    def analyse_results(self):
        n_reps = {}
        n_depths = {}
        rep_results = {}

        for gate in self.gates_to_benchmark:
            gate_name = gate.name
            n_reps[gate_name] = len({m["rep"] for m in self.circuits_meta if m["gate_name"] == gate_name})
            n_depths[gate_name] = len({m["depth"] for m in self.circuits_meta if m["gate_name"] == gate_name})
            rep_results[gate_name] = np.zeros((n_reps[gate_name], n_depths[gate_name]), dtype=float)

        for circ, meta, distribution in zip(self.circuits, self.circuits_meta, self.distribtions):
            gate_name = meta["gate_name"]

            rep_results[gate_name][meta["rep"], meta["depth_index"]] = (
                distribution.get("0" * circ.num_clbits, 0.0) / self.shots
            )

        self.results = {}
        for gate in self.gates_to_benchmark:
            gate_name = gate.name
            mean_results = np.mean(rep_results[gate_name], axis=0)

            fit = scipy.optimize.curve_fit(
                randomised_benchmarking.decay_curve,
                self.depths,
                mean_results,
                p0=[1.0, 0.9, 0.5],
                maxfev=3600,
            )[0]

            self.results[gate_name] = (mean_results, fit)

        self.gate_fidelity = randomised_benchmarking.interleaved_clifford_fidelity(
            self.results[self.gate_to_reference.name][1][1],
            self.results[self.gate_to_benchmark.name][1][1],
            1,
        )

        return self.gate_fidelity

    def plot_graph(self, ax, graph_name):
        plt.plot(self.depths, self.results["id"][0], color="blue", marker="o", label="mean")
        plt.plot(
            self.depths,
            self.results[self.gate_to_benchmark.name][0],
            color="black",
            marker="o",
            label="mean",
        )
        plt.plot(
            self.depths,
            randomised_benchmarking.decay_curve(self.depths, *self.results[self.gate_to_benchmark.name][1]),
            marker="x",
            color="grey",
        )
        plt.plot(
            self.depths,
            randomised_benchmarking.decay_curve(self.depths, *self.results["id"][1]),
            marker="x",
            color="cyan",
        )
        ax.set_title(
            f"Fidelity for {self.gate_to_benchmark.name} on qubit {self.qubit_idx} = {self.gate_fidelity:.6f}, expected {self.expected_fid:.6f}"
        )

        ax.set_ylim(0, 1)
        ax.set_xlim(left=0)

        return ax


class Gate2qFidelity(Experiment):
    def __init__(
        self,
        device_name: str,
        qubit_pair: list[int] | tuple[int, int],
        gate_to_benchmark: str = "CZ",
        shots: int = 10_000_000,
        depths: list[int] = [1, 2, 4, 8, 14, 24, 42, 72, 123, 211, 360, 616, 1052, 1798, 3071],
        replicates: int = 10,
        rng_seed: int = 1337,
    ):
        # Basic config
        self.device_name = device_name
        self.qubits_to_test = list(qubit_pair)
        self.gate_to_benchmark = gate_name_to_instruction[gate_to_benchmark.lower()]
        self.replicates = replicates
        self.start_states = ["00", "01", "10", "11"]
        self.runtime_options = {"optimization_level": 0}
        self.rng = np.random.default_rng(rng_seed)

        self.depths = depths

        self.shots = shots

        meta, circuits_ref, circuits_int = make_2q_benchmarking_circuits(
            qubits_to_test=self.qubits_to_test,
            gate_to_benchmark=self.gate_to_benchmark,
            replicates=self.replicates,
            depths=self.depths,
            rng=self.rng,
            start_states=self.start_states,
        )

        self.circuits_meta: list[dict] = []
        self.circuits: list[qiskit.QuantumCircuit] = []

        for m, c in zip(meta, circuits_ref):
            mm = dict(m)
            mm["kind"] = "ref"
            self.circuits_meta.append(mm)
            self.circuits.append(c)

        for m, c in zip(meta, circuits_int):
            mm = dict(m)
            mm["kind"] = "int"
            self.circuits_meta.append(mm)
            self.circuits.append(c)

        # Placeholders filled in analyse_results()
        self.results = {}
        self.gate_fidelity: float | None = None

    def analyse_results(self):
        n_reps = len({m["rep"] for m in self.circuits_meta})
        n_depths = len({m["depth"] for m in self.circuits_meta})
        start_states = self.start_states

        res_ref = {ss: np.zeros((n_reps, n_depths), dtype=float) for ss in start_states}
        res_int = {ss: np.zeros((n_reps, n_depths), dtype=float) for ss in start_states}

        for circ, meta, distribution in zip(self.circuits, self.circuits_meta, self.distribtions):
            p00 = distribution.get("0" * circ.num_clbits, 0.0)  # probabilities already normalized
            if meta["kind"] == "ref":
                res_ref[meta["start_state"]][meta["rep"], meta["depth_index"]] = p00
            else:
                res_int[meta["start_state"]][meta["rep"], meta["depth_index"]] = p00

        df_ref = {s: d.mean(axis=0) for s, d in res_ref.items()}
        df_int = {s: d.mean(axis=0) for s, d in res_int.items()}

        # Compute p1, p2, p3
        p1_ref = df_ref["00"] - df_ref["01"] + df_ref["10"] - df_ref["11"]
        p2_ref = df_ref["00"] + df_ref["01"] - df_ref["10"] - df_ref["11"]
        p3_ref = df_ref["00"] - df_ref["01"] - df_ref["10"] + df_ref["11"]

        p1_int = df_int["00"] - df_int["01"] + df_int["10"] - df_int["11"]
        p2_int = df_int["00"] + df_int["01"] - df_int["10"] - df_int["11"]
        p3_int = df_int["00"] - df_int["01"] - df_int["10"] + df_int["11"]

        # Fit decay curves
        decay_fit_ref = [
            scipy.optimize.curve_fit(
                randomised_benchmarking.decay_curve,
                self.depths,
                p,
                p0=[1.0, 0.9, 0],
                maxfev=3600,
            )[0]
            for p in [p1_ref, p2_ref, p3_ref]
        ]
        decay_fit_int = [
            scipy.optimize.curve_fit(
                randomised_benchmarking.decay_curve,
                self.depths,
                p,
                p0=[1.0, 0.9, 0],
                maxfev=3600,
            )[0]
            for p in [p1_int, p2_int, p3_int]
        ]

        # Compute alpha averages and interleaved Clifford fidelity
        a_ref = (decay_fit_ref[0][1] + decay_fit_ref[1][1] + 3 * decay_fit_ref[2][1]) * 3 / 15
        a_int = (decay_fit_int[0][1] + decay_fit_int[1][1] + 3 * decay_fit_int[2][1]) * 3 / 15

        gate_fidelity = randomised_benchmarking.interleaved_clifford_fidelity(a_ref, a_int, 2)

        # Store analysis
        self.results = {
            "ref": {
                "means": df_ref,
                "p": {"p1": p1_ref, "p2": p2_ref, "p3": p3_ref},
                "fits": decay_fit_ref,
            },
            "int": {
                "means": df_int,
                "p": {"p1": p1_int, "p2": p2_int, "p3": p3_int},
                "fits": decay_fit_int,
            },
        }
        self.gate_fidelity = gate_fidelity
        return gate_fidelity

    def plot_graph(self, ax, graph_name):
        p1_ref = self.results["ref"]["p"]["p1"]
        p1_int = self.results["int"]["p"]["p1"]
        fit_ref = self.results["ref"]["fits"][0]
        fit_int = self.results["int"]["fits"][0]

        ax.plot(self.depths, p1_ref, color="blue", marker="o", label="ref p1 mean")
        ax.plot(self.depths, p1_int, color="black", marker="o", label="int p1 mean")

        ax.plot(
            self.depths,
            randomised_benchmarking.decay_curve(self.depths, *fit_ref),
            color="cyan",
            marker="x",
            label="ref fit",
        )
        ax.plot(
            self.depths,
            randomised_benchmarking.decay_curve(self.depths, *fit_int),
            color="grey",
            marker="x",
            label="int fit",
        )

        qubits_str = f"{self.qubits_to_test[0]}-{self.qubits_to_test[1]}"
        ax.set_title(f"Fidelity for {self.gate_to_benchmark.name} on qubits {qubits_str} = {self.gate_fidelity:.6f}")
        ax.set_ylim(0, 1)
        ax.set_xlim(left=0)
        ax.legend()
        return ax
