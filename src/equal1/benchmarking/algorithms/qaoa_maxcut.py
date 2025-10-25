import networkx as nx
import numpy as np
import qiskit
import qiskit.circuit
import scipy

from equal1.benchmarking import qbraid_wrapper

from typing import Union

from networkx import Graph
from networkx.algorithms.approximation.maxcut import one_exchange


# import sys
# sys.path.append("../../thirdparty/qscore/utils/")
# import max_cut
# max_cut.calculate_beta_max_cut(graph, cost)
def calculate_beta_max_cut(graph: Union[Graph, int], max_cut_result: float) -> float:
    """
    Calculate beta value for a Max-Cut optimization problem.
    If a graph is provided, beta is calculated based on exact result.

    Args:
        graph: Problem instance graph or graph size.
        max_cut_result: Found objective value.

    Returns:
        beta value for specific problem instance or problem size.
    """
    if isinstance(graph, Graph):  # only suitable for small graph sizes.
        n = len(graph)
        random_score = n * (n - 1) / 8
        exact_score = one_exchange(graph)[0]
        beta = (max_cut_result - random_score) / (exact_score - random_score)
    else:
        random_score = graph**2 / 8
        beta = (max_cut_result - random_score) / (0.178 * pow(graph, 3 / 2))
    return beta


def run_on_qbraid(circ, device):
    shots = 2**14
    return qbraid_wrapper.run_circuit(circ, device, simulate=True, shots=shots)[0]


def compute_qscore(device, sizes, replicates, executor, layers=2, seed=None):
    results = []
    rng = np.random.default_rng(seed)
    for size in sizes:
        for _ in range(replicates):
            graph_seed = int(rng.integers(0, 100000, 1))  # 2**32 - 1)
            graph = nx.erdos_renyi_graph(size, 0.5, seed=graph_seed)
            try:
                score, solution, parameters = run_qaoa_on_graph(
                    graph,
                    executor,
                    layers=layers,
                )
            except Exception as e:
                # randomly we get a key error "gamma_0",
                # as if the optimizer suggested no parameters or something.
                # I am not sure what causes it, so just skip this run
                print(f"Error running QAOA on graph size {size}: {e}")
                continue
            beta = calculate_beta_max_cut(size, score)
            results.append(
                {
                    "cuts": score,
                    # "parameters": parameters,
                    "layers": layers,
                    "beta": beta,
                    "graph_size": size,
                    "device": device,
                    "seed": graph_seed,
                }
            )
            print(results[-1])
    return results


def run_qaoa_on_graph(graph, execute_circuit, layers=1, start_params=None):
    start_params = start_params or [np.pi / 4, np.pi / 2] * layers
    ansatz = make_qaoa_maxcut_circuit(graph, layers=layers)
    result = scipy.optimize.minimize(
        objective_function,
        x0=start_params,
        method="COBYLA",
        options={"maxiter": 100},
        args=(
            graph,
            ansatz,
            execute_circuit,
        ),
    )
    prob_dist = execute_circuit(ansatz.assign_parameters(parameters_to_dict(result.x)))
    solution = solution_from_probs(prob_dist)
    cut_score = solution_to_cost(graph, solution)
    return cut_score, solution, result.x


def objective_function(params, graph, circuit, execute_circuit):
    circ_param = circuit.assign_parameters(parameters_to_dict(params))
    prob_dist = execute_circuit(circ_param)
    cost = solution_cost_no_weights(graph, prob_dist)

    return -cost


def neg_pos_eig(x, i, j):
    n = len(x)  # bit string is from right to left
    if x[n - 1 - i] != x[n - 1 - j]:
        return -1
    return 1


def solution_cost(graph: nx.Graph, probability: dict[str, float]) -> float:
    total_costs = 0
    for u, v, w in graph.edges(data=True):
        solution_weights = [
            p_s * neg_pos_eig(s, u, v) * w["weight"] for s, p_s in probability.items()
        ]
        term = np.sum(solution_weights)
        total_costs += term
    return -np.sum(total_costs)


def solution_cost_no_weights(graph: nx.Graph, probability: dict[str, float]) -> float:
    total_costs = 0
    for u, v in graph.edges():
        solution_weights = [
            p_s * neg_pos_eig(s, u, v) for s, p_s in probability.items()
        ]
        term = np.sum(solution_weights)
        total_costs += term
    return -np.sum(total_costs)


def mixer_hamiltonian(circuit, parameter_name: str = "beta"):
    beta = qiskit.circuit.parameter.Parameter(parameter_name)
    circuit.rx(2 * beta, range(circuit.num_qubits))
    return circuit


def driver_hamiltonian(circuit, graph: nx.Graph, parameter_name: str = "gamma"):
    gamma = qiskit.circuit.parameter.Parameter(parameter_name)
    for edge in graph.edges():
        circuit.cx(*edge)
        circuit.rz(gamma, edge[1])
        circuit.cx(*edge)
    return circuit


def make_qaoa_maxcut_circuit(graph: nx.Graph, layers: int = 1) -> qiskit.QuantumCircuit:
    n_qubits = graph.number_of_nodes()
    circuit = qiskit.QuantumCircuit(n_qubits)
    circuit.h(range(n_qubits))
    circuit.barrier()
    for l in range(layers):
        circuit = driver_hamiltonian(circuit, graph, parameter_name=f"gamma_{l}")
        circuit.barrier()
        circuit = mixer_hamiltonian(circuit, parameter_name=f"beta_{l}")
        circuit.barrier()
    circuit.measure_all()
    return circuit


def parameters_to_dict(params):
    param_dict = {}
    for i in range(0, len(params), 2):
        param_dict[f"gamma_{i // 2}"] = params[i]
        param_dict[f"beta_{i // 2}"] = params[i + 1]
    return param_dict


def solution_to_cost(graph, solution):
    rev_solution = solution[::-1]
    cut = {node for node in graph.nodes() if rev_solution[node] == "1"}
    current_cut_size = nx.algorithms.cut_size(graph, cut)
    return current_cut_size


def solution_from_probs(counts):
    solution = max(counts, key=lambda k: counts[k])
    return solution
