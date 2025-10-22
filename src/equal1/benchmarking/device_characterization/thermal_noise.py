import numpy as np
import qiskit
import scipy.optimize

from equal1.benchmarking.experiment import Experiment


def plot_decay_fit(
    ax,
    times,
    probability,
    fit_params,
    scale=(1, 0),
):
    e_inv = 1 / np.e
    # for T2 we scale e_inv to 0.5 to 1 (1/2 * e_inv + 1/2)
    ax.axhline(
        scale[0] * e_inv + scale[1], color="red", linestyle="--", label="1/e line"
    )
    ax.axvline(fit_params[0], color="red", label="calculated value")

    ax.plot(times, probability, "o-", label="simulation")
    model_2 = decay_model(times, *fit_params)
    ax.plot(
        times,
        model_2,
        "o-",
        label="fit model",
    )
    ax.legend()
    ax.axhline(0.5, color="black")

    return ax


def make_relaxation(qubit_idx, time):
    qc = qiskit.QuantumCircuit(qubit_idx + 1, 1)
    qc.sx(qubit_idx)
    qc.sx(qubit_idx)
    qc.delay(time, qubit_idx, unit="s")
    qc.measure(qubit_idx, 0)
    return qc


def make_ramsey(qubit_idx, time):
    qc = qiskit.QuantumCircuit(qubit_idx + 1, 1)
    qc.sx(qubit_idx)
    qc.delay(time, qubit_idx, unit="s")
    qc.sx(qubit_idx)
    qc.measure(qubit_idx, 0)
    return qc


def decay_model(times, lamda, scale=1, offset=0):
    return (scale * np.exp(-times / lamda)) + offset


time_units = {
    "ns": 1e-9,
    "us": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
}


class IdleNoiseT1(Experiment):
    def __init__(
        self,
        device_name: str,
        qubit_idx: int,
        max_time: float = 1.0,
        shots: int = 70_000,
        runtime_options={"optimization_level": 0},
    ) -> None:
        self.device_name = device_name
        self.qubit_idx = qubit_idx
        self.shots = shots
        self.runtime_options = runtime_options

        self.max_time = max_time

        self.rabi_durations = np.concatenate(
            [
                np.linspace(0, self.max_time / 2, 20),
                np.linspace(self.max_time / 2, self.max_time, 5),
            ]
        )

        self.circuits = self._make_circuits()

        self.job_results = []
        self.distribtions = []
        self.results_analysis = {}

    def _make_circuits(self) -> list[qiskit.QuantumCircuit]:
        circuits = []
        for t in self.rabi_durations:
            qc = make_relaxation(self.qubit_idx, t)
            circuits.append(qc)
        return circuits

    def analyse_results(self):
        results = np.array(
            [distribution.get("1", 0.0) for distribution in self.distribtions],
            dtype=float,
        )
        self.probabilities = results / self.shots

        self.fit = scipy.optimize.curve_fit(
            decay_model,
            self.rabi_durations,
            self.probabilities,
            p0=(self.max_time * 0.5, 1.0, 0.0),
        )[0]

        t1 = self.fit[0]

        return t1

    def plot_graph(self, ax, graph_name):
        plot_decay_fit(
            ax,
            self.rabi_durations,
            self.probabilities,
            self.fit,
            scale=(1, 0),
        )

        return ax


class ThermalNoiseT2Ramsey(Experiment):
    def __init__(
        self,
        device_name: str,
        qubit_idx: int,
        max_time: float = 40 * time_units["us"],
        shots: int = 100_000,
        runtime_options={"optimization_level": 0},
    ) -> None:
        self.device_name = device_name
        self.qubit_idx = qubit_idx
        self.shots = shots
        self.runtime_options = runtime_options

        self.max_time = max_time

        self.ramsey_durations = np.concatenate(
            [
                np.linspace(0, self.max_time / 2, 20),
                np.linspace(max_time / 2, max_time, 5),
            ]
        )

        self.circuits = self._make_circuits()

        self.job_results = []
        self.distribtions = []
        self.results_analysis = {}

    def _make_circuits(self) -> list[qiskit.QuantumCircuit]:
        circuits = []
        for t in self.ramsey_durations:
            qc = make_ramsey(self.qubit_idx, t)
            circuits.append(qc)
        return circuits

    def analyse_results(self):
        results = np.array(
            [distribution.get("1", 0.0) for distribution in self.distribtions],
            dtype=float,
        )
        self.probabilities = results / self.shots

        self.fit = scipy.optimize.curve_fit(
            decay_model,
            self.ramsey_durations,
            self.probabilities,
            p0=(self.max_time * 0.5, 0.5, 0.5),
        )[0]

        t2 = self.fit[0]

        return t2

    def plot_graph(self, ax, graph_name):
        plot_decay_fit(
            ax,
            self.ramsey_durations,
            self.probabilities,
            self.fit,
            scale=(0.5, 0.5),
        )

        return ax
