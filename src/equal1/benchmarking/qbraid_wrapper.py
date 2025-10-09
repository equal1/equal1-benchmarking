import base64
import warnings
from typing import Optional

import qbraid
import qiskit
# import rich.progress


def convert_to_probabilities(
    counts: dict[str, int], shots: Optional[int] = None
) -> dict[str, float]:
    """Convert counts to probabilities.

    Args:
        counts (dict[str, int]): Dictionary of counts.
        shots (int): Total number of shots.

    Returns:
        dict[str, float]: Dictionary of probabilities.
    """
    if shots is None:
        shots = sum(counts.values())
    return {state: count / shots for state, count in counts.items()}


def get_device(device_name: str, simulation: bool) -> qbraid.QuantumDevice:
    provider = qbraid.QbraidProvider()
    if simulation:
        noise_model = device_name
        device_name = "equal1_simulator"
        return device, noise_model

    device = provider.get_device(device_name)
    return device, None


def run_on_device(
    qc: qiskit.QuantumCircuit | list[qiskit.QuantumCircuit],
    device: qbraid.QuantumDevice,
    shots: int = 10_000,
    force_bypass_transpilation=False,
    optimization_level=0,
):
    if force_bypass_transpilation and optimization_level != 0:
        warnings.warn(
            "Bypassing transpilation step will ignore the optimization_level parameter."
        )

    execution_options = {
        "force_bypass_transpilation": force_bypass_transpilation,
        "optimization_level": optimization_level,
    }

    jobs = device.run(
        qc,
        shots=shots,
        runtime_options={
            "execution_options": execution_options,
        },
    )
    return jobs


def run_simulation(
    qc: qiskit.QuantumCircuit | list[qiskit.QuantumCircuit],
    device: qbraid.QuantumDevice,
    shots: int = 10_000,
    noise_model=None,
    simulation_platform="CPU",
    force_bypass_transpilation=False,
    optimization_level=0,
):
    """Run a quantum circuit on a specified qbraid device with an optional noise model.

    Args:
        qc (qiskit.QuantumCircuit): The quantum circuit to run.
        device (qiskit.providers.Backend): The backend device to run the circuit on.
        shots (int): The number of shots to run. Defaults to 10,000.
        noise_model (qiskit.providers.models.NoiseModel, optional): A noise model to apply during execution.
        get_probabilities (bool): If True, convert counts to probabilities. Defaults to False.
        simulation_platform (str): The simulation platform to use, "GPU" or "CPU". Defaults to "CPU".
        force_bypass_transpilation (bool): If True, bypass the transpilation step. Defaults to False.
        optimization_level (int): The level of circuit optimization the compiler should apply. Defaults to 0.

    Returns:
        dict[str, float] : A dictionary of measurement outcomes and probabilities.
        or
        dict[str, int] : A dictionary of measurement outcomes and their counts.
        or
        dict[str, number], list[qiskit.QuantumCircuit]: If return_transpiled_circuits is True, returns a tuple of counts/probabilities and the transpiled circuits.
    """

    if force_bypass_transpilation and optimization_level != 0:
        warnings.warn(
            "Bypassing transpilation step will ignore the optimization_level parameter."
        )

    execution_options = {
        "force_bypass_transpilation": force_bypass_transpilation,
        "optimization_level": optimization_level,
    }

    jobs = device.run(
        qc,
        shots=shots,
        noise_model=noise_model,
        runtime_options={
            "simulation_platform": simulation_platform,
            "execution_options": execution_options,
        },
    )
    return jobs


def get_results_from_jobs(
    list_of_jobs,
    shots: int,
    get_probabilities=False,
):
    counts = []
    for job in list_of_jobs:
        job.wait_for_final_state()
        if job.status() == qbraid.JobStatus.FAILED:
            result_json = job.client.get_job_results(job.id)
            # TODO probably should cancel the rest of the jobs
            raise RuntimeError(
                f"Job failed with status: {job.status()}, error: {result_json['statusText']}"
            )
        # if return_transpiled_circuits:
        #     transpiled_circuits.append(get_transpiled_circuit(job))
        result = job.result()
        counts.append(result.data.get_counts())
        # progress.advance(task_id)
        # progress.refresh()

    if get_probabilities:
        counts = [convert_to_probabilities(c, shots) for c in counts]
    # if return_transpiled_circuits:
    #     return counts, transpiled_circuits
    return counts


def get_transpiled_circuit(job):
    result_json = job.client.get_job_results(job.id)
    compiled_ir = result_json["compiledOutput"]
    decodedoutput = base64.b64decode(compiled_ir.encode("utf-8")).decode("utf-8")
    decode_circ = qiskit.QuantumCircuit.from_qasm_str(decodedoutput)
    return decode_circ
