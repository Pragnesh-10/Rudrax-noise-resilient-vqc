"""Transpilation helpers."""
from qiskit import transpile
from .processors import coupling_map


def compile_for_processor(circuit, config):
    return transpile(
        circuit,
        basis_gates=config["basis_gates"],
        coupling_map=coupling_map(config),
        optimization_level=config["transpilation"]["optimization_level"],
        seed_transpiler=config["transpilation"]["seed_transpiler"],
    )


def circuit_metrics(circuit):
    ops = circuit.count_ops()
    return {
        "depth": int(circuit.depth()),
        "cx_count": int(ops.get("cx", 0)),
        "swap_count": int(ops.get("swap", 0)),
        "operation_count": int(sum(ops.values())),
        "num_qubits": int(circuit.num_qubits),
    }
