import numpy as np
import qiskit
import qiskit.quantum_info as qqi
from qiskit.circuit.library import XGate, SXGate


def clifford_to_str(clifford: qqi.Clifford):
    cd = clifford.to_dict()
    return ",".join(cd["destabilizer"]) + "|" + ",".join(cd["stabilizer"])


def labels_to_clifford(label: str) -> qqi.Clifford:
    destab, stab = label.split("|")
    destab = destab.split(",")
    stab = stab.split(",")
    return qqi.Clifford.from_dict({"destabilizer": destab, "stabilizer": stab})


def gates_to_circuit(
    number_of_qubits: int,
    gate_list: list[tuple[str, float, tuple[int, ...]]],
):
    qc = qiskit.QuantumCircuit(number_of_qubits)
    for gate, param, qubits in gate_list:
        if gate == "sx":
            qc.append(SXGate(), qubits)
        elif gate == "x":
            qc.append(XGate(), qubits)
        elif gate == "rz":
            qc.rz(param * np.pi, qubits)
        elif gate == "cz":
            qc.cz(qubits[0], qubits[1])
    return qc


# label order is destabiliser, stabiliser
q1_cliffords = {
    "+X|+Y": [("rz", -1.0, (0,)), ("sx", None, (0,)), ("rz", -1.0, (0,))],
    "+X|+Z": [],
    "+X|-Y": [("sx", None, (0,))],
    "+X|-Z": [("x", None, (0,))],
    "+Y|+X": [("sx", None, (0,)), ("rz", 0.5, (0,))],
    "+Y|+Z": [("rz", 0.5, (0,))],
    "+Y|-X": [("rz", -1.0, (0,)), ("sx", None, (0,)), ("rz", -0.5, (0,))],
    "+Y|-Z": [("x", None, (0,)), ("rz", 0.5, (0,))],
    "+Z|+X": [("rz", 0.5, (0,)), ("sx", None, (0,)), ("rz", 0.5, (0,))],
    "+Z|+Y": [("rz", 0.5, (0,)), ("sx", None, (0,)), ("rz", -1.0, (0,))],
    "+Z|-X": [("rz", 0.5, (0,)), ("sx", None, (0,)), ("rz", -0.5, (0,))],
    "+Z|-Y": [("rz", 0.5, (0,)), ("sx", None, (0,))],
    "-X|+Y": [("sx", None, (0,)), ("rz", -1.0, (0,))],
    "-X|+Z": [("rz", 1.0, (0,))],
    "-X|-Y": [("rz", -1.0, (0,)), ("sx", None, (0,))],
    "-X|-Z": [("rz", -1.0, (0,)), ("x", None, (0,))],
    "-Y|+X": [("rz", -1.0, (0,)), ("sx", None, (0,)), ("rz", 0.5, (0,))],
    "-Y|+Z": [("rz", -0.5, (0,))],
    "-Y|-X": [("sx", None, (0,)), ("rz", -0.5, (0,))],
    "-Y|-Z": [("x", None, (0,)), ("rz", -0.5, (0,))],
    "-Z|+X": [("rz", -0.5, (0,)), ("sx", None, (0,)), ("rz", 0.5, (0,))],
    "-Z|+Y": [("rz", -0.5, (0,)), ("sx", None, (0,)), ("rz", -1.0, (0,))],
    "-Z|-X": [("rz", -0.5, (0,)), ("sx", None, (0,)), ("rz", -0.5, (0,))],
    "-Z|-Y": [("rz", -0.5, (0,)), ("sx", None, (0,))],
}

# The following section is just to stop the clifford "to_circuit" function
# returning a huge circuit for CX and CZ gates.
# it turns CX 1 0 into a depth 5 circuit, which qiskit will not transpile to a single CX gate.
# we make a custom circuit that gives us a standard CX gate.
q2_cliffords = {
    "+IX,+XI|+IZ,+ZI": [],  # identity_label
    "+ZX,+XZ|+IZ,+ZI": [("cz", None, (0, 1))],
    "+IX,+XX|+ZZ,+ZI": [  # cx 0, 1
        ("rz", 0.5, (1,)),
        ("sx", None, (1,)),
        ("rz", 1, (1,)),
        ("cz", None, (0, 1)),
        ("sx", None, (1,)),
        ("rz", 0.5, (1,)),
    ],
    "+XX,+XI|+IZ,+ZZ": [  # cx10
        ("rz", 0.5, (0,)),
        ("sx", None, (0,)),
        ("rz", 1, (0,)),
        ("cz", None, (0, 1)),
        ("sx", None, (0,)),
        ("rz", 0.5, (0,)),
    ],
}

cliffords = {
    k: labels_to_clifford(k)
    for k in list(q1_cliffords.keys()) + list(q2_cliffords.keys())
}

transpiled_cliffords = {
    k: gates_to_circuit(1, gates) for k, gates in q1_cliffords.items()
} | {k: gates_to_circuit(2, gates) for k, gates in q2_cliffords.items()}


def random_1q_clifford(rng):
    label = rng.choice(list(q1_cliffords.keys()))
    return labels_to_clifford(label)


def transpile_clifford(clifford: qqi.Clifford):
    cliff_str = clifford_to_str(clifford)
    try:
        return transpiled_cliffords[cliff_str]
    except KeyError:
        transpile = qiskit.transpile(
            clifford.to_circuit(), basis_gates=["cz", "sx", "rz"], optimization_level=3
        )
        return transpile


# functions to fit to benchmarking
def decay_curve(m, A, alpha, B=0):
    return A * (alpha**m) + B


def inverse_decay_curve(alpha, x, b=0):
    return np.log(-(b - np.e ** (-x)) / x) / np.log(alpha)


def average_clifford_fidelity(alpha, n):
    d = 2**n
    fav = 1 - ((1 - alpha) * ((d - 1) / d))
    return fav


def inverse_average_clifford_fidelity(fav, n):
    d = 2**n
    alpha = (fav * d - 1) / (d - 1)
    return alpha


def max_depth_for_fidelity(fid, n, x=3):
    """
    Given a target fidelity and number of qubits, return the maximum sequence length
    needed in benchmarking to reach that fidelity.

    Args:
        fid: target fidelity
        n: number of qubits
        x: point on the decay curve to return the depth for
    """

    alpha = inverse_average_clifford_fidelity(fid, n)
    max_depth = inverse_decay_curve(alpha, x)

    return int(max_depth)


def estimate_needed_precision(infid):
    """
    Given a target infidelity, estimate the number of decimal places needed to
    first non 9 fidelity digit.
    """
    pos = int(np.floor(-np.log10(infid)))
    val = -np.log10(infid)
    if not np.isclose(val, round(val), rtol=0, atol=1e-12):
        pos += 1
    return pos


def depth_list_for_fidelity(fid, n):
    max_depth = max_depth_for_fidelity(fid, n)
    depths = np.unique(np.geomspace(1, max_depth, num=16, endpoint=True, dtype=int))
    return depths.tolist()


def shots_for_infidelity(infid):
    shots = 10 ** (int(np.ceil(-np.log10(infid))) * 2 + 1)

    return shots


def interleaved_clifford_fidelity(alpha_standard, alpha_interleaved, n):
    d = 2**n
    interleaved_gate_fidelity = ((d - 1) / d) * (
        1 - (alpha_interleaved / alpha_standard)
    )
    return 1 - interleaved_gate_fidelity


def add_1_qubit_cliffords_on_n_qubits(
    circuit: qiskit.QuantumCircuit,
    clifford_state: qqi.Clifford,
    qubits: list[int],
    rng: np.random.Generator,
) -> tuple[qiskit.QuantumCircuit, qqi.Clifford]:
    chosen_cliffords = []
    for i, q in enumerate(qubits):
        chosen_clifford = random_1q_clifford(rng)
        circuit = circuit.compose(transpile_clifford(chosen_clifford), qubits=[q])
        # circuit.barrier()
        chosen_cliffords.append(chosen_clifford.copy())

    # add the chosen cliffords to the clifford state.
    chosen_cliffords = chosen_cliffords[::-1]
    # tensor product all the cliffords one at a time.
    cliffords = chosen_cliffords[0]
    for clifford in chosen_cliffords[1:]:
        cliffords = cliffords.tensor(clifford)
    clifford_state = clifford_state.compose(cliffords)
    return circuit, clifford_state


def qiskit_n_qubit_clifford_benchmarking(
    qubits_to_benchmark,
    length,
    gate_to_benchmark: qqi.Clifford,
    rng: np.random.Generator,
):
    n_qubits = len(qubits_to_benchmark)
    circ = qiskit.QuantumCircuit(max(qubits_to_benchmark) + 1, len(qubits_to_benchmark))

    identity = qqi.Operator(np.eye(2**n_qubits))
    combined_clifford = qqi.Clifford.from_operator(identity)

    for _ in range(length):
        circ, combined_clifford = add_1_qubit_cliffords_on_n_qubits(
            circ, combined_clifford, qubits_to_benchmark, rng
        )

        # apply our gate to interleave if we have one
        chosen_clifford = gate_to_benchmark.copy()
        circ = circ.compose(
            transpile_clifford(chosen_clifford), qubits=qubits_to_benchmark
        )
        # circ.barrier()
        combined_clifford = combined_clifford.compose(chosen_clifford)

    inverse: qqi.Clifford = combined_clifford.adjoint()
    circ = circ.compose(transpile_clifford(inverse), qubits=qubits_to_benchmark)
    # debug check that the circuit is correct
    # assert qqi.Operator(circ).equiv(qqi.Operator.from_label("IIIIII"))
    return circ


def generate_character_rb(
    start_state,
    qubits_to_benchmark,
    gate_to_benchmark: qqi.Clifford,
    length: int,
    rng=None,
):
    rng = rng or np.random.default_rng()
    qc_start = qiskit.QuantumCircuit(6, 2)
    for i, st in zip(qubits_to_benchmark, start_state[::-1]):
        if st == "1":  # 1 can be made by X gate
            if rng.random(1) < 0.5:  # or half the time, Y gate (z and x)
                qc_start.rz(np.pi, [i])
            qc_start.append(XGate(), [i])

    interleaved_sequence = qiskit_n_qubit_clifford_benchmarking(
        qubits_to_benchmark,
        length,
        gate_to_benchmark=gate_to_benchmark,
        rng=rng,
    )
    return qc_start.compose(interleaved_sequence)
