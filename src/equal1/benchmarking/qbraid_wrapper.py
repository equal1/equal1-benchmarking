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


def get_device(
    device_name: str, simulation: bool
) -> tuple[qbraid.QuantumDevice, Optional[str]]:
    provider = qbraid.QbraidProvider()
    noise_model = None
    if simulation:
        noise_model = device_name
        device_name = "equal1_simulator"
    return provider.get_device(device_name), noise_model


def run_on_device(
    qc: qiskit.QuantumCircuit | list[qiskit.QuantumCircuit],
    device: qbraid.QuantumDevice,
    noise_model=None,
    shots: int = 10_000,
    runtime_options=None,
):
    runtime_options = runtime_options or {}
    optimization_level = runtime_options.get("optimization_level", 0)
    simulation_platform = runtime_options.get("simulation_platform", "AUTO")
    # number_of_workers = runtime_options.get("number_of_workers", 1)

    force_bypass_transpilation = runtime_options.get(
        "force_bypass_transpilation", False
    )
    if force_bypass_transpilation and optimization_level != 0:
        warnings.warn(
            "Bypassing transpilation step will ignore the optimization_level parameter."
        )
    print("Running on device:", device.profile.model_dump())
    print("Noise model:", noise_model)
    runtime_options = {
        "simulation_platform": simulation_platform,
        "execution_options": {
            # "number_of_workers": number_of_workers,
            "force_bypass_transpilation": force_bypass_transpilation,
            "optimization_level": optimization_level,
        },
    }
    print(runtime_options)

    jobs = device.run(
        qc, shots=shots, noise_model=noise_model, runtime_options=runtime_options
    )
    print(jobs)
    return jobs


def get_results_from_jobs(
    list_of_jobs,
):
    results = []
    for job in list_of_jobs:
        job.wait_for_final_state()
        if job.status() == qbraid.JobStatus.FAILED:
            result_json = job.client.get_job_results(job.id)
            print(result_json)
            # TODO probably should cancel the rest of the jobs
            raise RuntimeError(
                f"Job failed with status: {job.status()}, error: {result_json['statusText']}"
            )
        results.append(job.get_result())

    return results


def get_state_distribution(list_of_results, shots: int, get_probabilities=True):
    def get_counts(result, get_probabilities=True):
        counts = result.data.get_counts()
        if get_probabilities:
            return convert_to_probabilities(counts, shots)

    return [get_counts(result, get_probabilities) for result in list_of_results]


def get_transpiled_circuit(job):
    result_json = job.client.get_job_results(job.id)
    compiled_ir = result_json["compiledOutput"]
    decodedoutput = base64.b64decode(compiled_ir.encode("utf-8")).decode("utf-8")
    decode_circ = qiskit.QuantumCircuit.from_qasm_str(decodedoutput)
    return decode_circ
