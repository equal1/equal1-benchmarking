import pathlib
from typing import Any
from qbraid import QbraidProvider
from qiskit import QuantumCircuit, qasm2
import logging as log
import base64
import sys

this_dir = pathlib.Path(__file__).parent.resolve()
benchmark_dir = (
    this_dir.parent.parent.parent / "thirdparty" / "QC-App-Oriented-Benchmarks"
)


def init_benchmark_environmet():
    sys.path.insert(1, str(benchmark_dir / "_common"))
    sys.path.insert(1, str(benchmark_dir / "_common" / "qiskit"))


init_benchmark_environmet()
import execute  # noqa: E402


def get_benchmarks_dir() -> str:
    return str(benchmark_dir)


def get_benchmark_standard_params() -> dict[str, Any]:
    return {
        "backend_id": "qasm_simulator",
        "hub": "",
        "group": "",
        "project": "",
        "provider_backend": None,
    }


class Equal1BackEnd:
    def __init__(self):
        self.name = "Equal1Backend"


class Equal1Result:
    def __init__(self, counts, exec_time, transpiled_circuit_metrics):
        self.exec_time = exec_time
        self.counts = counts
        self.transpiled_circuit_metrics = transpiled_circuit_metrics

    def get_counts(self, qc):
        return self.counts

    def get_transpiled_circuit_metrics(self):
        return self.transpiled_circuit_metrics


class Equal1Executor:
    def __init__(self, equal1_hardware: str):
        provider = QbraidProvider()
        self.device = provider.get_device("equal1_simulator")
        self.hardware = equal1_hardware

    def __call__(
        self, qc: QuantumCircuit, backend_name: str, backend, shots, **kwargs
    ) -> Equal1Result:
        # print(f"attempting to run {qc} on {backend_name}, on {backend} with {shots} and {kwargs}")

        runtime_options = {
            "simulation_platform": "GPU",
            "execution_options": {"optimization_level": 2},
        }

        backend = "TensorNet_MPS" if qc.num_qubits > 8 else "DensityMatrix"

        job = self.device.run(
            qc,
            shots=shots,
            noise_model=self.hardware,
            runtime_options=runtime_options,
            backend=backend,
        )
        job.wait_for_final_state()  # Wait for the job to complete and get final state

        if job.status().name != "COMPLETED":
            job_result = job.client.get_job_results(job.id)
            log.error(
                f"\n@@@@@@@@@@@@@@ \n JOB FAILED for {qc} \n With error message:\n {job_result['statusText']}\n @@@@@@@@@@@@@@@@@@ \n "
            )

        result = job.result()
        counts = result.data.get_counts()
        # print(counts)

        result_json = job.client.get_job_results(job.id)
        inner_exec_time = result_json["executionMetrics"]["executor"]

        transpiled_circuit = base64.b64decode(result_json["compiledOutput"]).decode(
            "utf-8"
        )
        transpiled_qc = qasm2.loads(
            transpiled_circuit, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS
        )

        metrics = execute.get_circuit_metrics(transpiled_qc)

        return Equal1Result(counts, inner_exec_time, metrics)
