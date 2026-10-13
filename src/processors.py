"""Participant-defined Processor A/B models for QFF Phase 1."""
import json
import numpy as np
from pathlib import Path
from qiskit.transpiler import CouplingMap
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_PROCESSOR_A_EDGES = frozenset({frozenset(e) for e in [[0, 1], [1, 2], [2, 3], [3, 4]]})
EXPECTED_PROCESSOR_B_EDGES = frozenset({frozenset(e) for e in [[0, 1], [1, 2], [2, 3], [2, 4], [4, 5], [5, 6], [4, 7], [7, 8]]})

VALID_BASIS_GATES = {"rz", "sx", "x", "cx"}


def load_config(name: str) -> dict:
    """Load a processor configuration JSON file.

    Raises FileNotFoundError if the config does not exist.
    Raises ValueError if the config fails validation.
    """
    path = ROOT / "processors" / f"{name}.json"
    config = json.loads(path.read_text())
    validate_config(config, name)
    return config


def validate_config(config: dict, name: str) -> None:
    """Validate a processor configuration against the official Challenge Kit specifications.

    Checks:
    - Physical qubit count matches the expected value for the processor name.
    - Exact coupling-edge set matches the official topology (bidirectional).
    - Basis gates are a subset of the official allowed set.
    - Noise probabilities are in valid ranges.
    - Transpilation settings use the official configuration.
    """
    if name not in ("processor_A", "processor_B", "processor_D"):
        raise ValueError(f"Unknown processor name: {name}")

    expected_qubits = {"processor_A": 5, "processor_B": 9, "processor_D": 6}
    if config["qubits"] != expected_qubits[name]:
        raise ValueError(
            f"{name}: expected {expected_qubits[name]} physical qubits, got {config['qubits']}"
        )

    # Validate coupling edges: must be bidirectional and match the official topology
    provided_edges = frozenset({frozenset(e) for e in config["coupling_edges"]})
    
    if name == "processor_A":
        expected_edges = EXPECTED_PROCESSOR_A_EDGES
    elif name == "processor_B":
        expected_edges = EXPECTED_PROCESSOR_B_EDGES
    else:
        # processor_D: validate that edges form a connected graph with the right qubit count
        # No strict edge set; just validate basic structure
        if config["qubits"] != 6:
            raise ValueError(f"{name}: expected 6 physical qubits")
        # Check that edges are valid qubit indices
        for e in config["coupling_edges"]:
            if len(e) != 2 or not (0 <= e[0] < config["qubits"] and 0 <= e[1] < config["qubits"]):
                raise ValueError(f"{name}: invalid coupling edge {e}")
        # Check graph connectivity (simple BFS)
        adj = {i: [] for i in range(config["qubits"])}
        for e in config["coupling_edges"]:
            adj[e[0]].append(e[1])
            adj[e[1]].append(e[0])
        visited = set()
        queue = [0]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
        if len(visited) != config["qubits"]:
            raise ValueError(f"{name}: coupling graph is not fully connected")
        # Skip exact edge matching for D
        return

    if provided_edges != expected_edges:
        missing = expected_edges - provided_edges
        extra = provided_edges - expected_edges
        err_parts = []
        if missing:
            err_parts.append(f"missing edges: {set(missing)}")
        if extra:
            err_parts.append(f"extra edges: {set(extra)}")
        raise ValueError(
            f"{name}: coupling edges do not match the official Challenge Kit topology. "
            f"{'; '.join(err_parts)}"
        )

    # Validate basis gates
    provided_gates = set(config["basis_gates"])
    if not provided_gates.issubset(VALID_BASIS_GATES):
        invalid = provided_gates - VALID_BASIS_GATES
        raise ValueError(
            f"{name}: basis gates contain invalid gates: {invalid}. "
            f"Must be subset of {VALID_BASIS_GATES}."
        )

    # Validate noise probabilities
    n = config["noise"]
    p1 = n["single_qubit_depolarizing_probability"]
    p2 = n["cx_depolarizing_probability"]
    pr = n["readout_bit_flip_probability"]
    if not (0.0 <= p1 <= 1.0):
        raise ValueError(f"{name}: single-qubit depolarizing probability must be in [0,1], got {p1}")
    if not (0.0 <= p2 <= 1.0):
        raise ValueError(f"{name}: CX depolarizing probability must be in [0,1], got {p2}")
    if not (0.0 <= pr < 0.5):
        raise ValueError(f"{name}: readout bit-flip probability must be in [0,0.5), got {pr}")

    # Validate transpilation settings
    t = config["transpilation"]
    if t["optimization_level"] != 1:
        raise ValueError(f"{name}: optimization_level must be 1, got {t['optimization_level']}")
    if t["seed_transpiler"] != 42:
        raise ValueError(f"{name}: seed_transpiler must be 42, got {t['seed_transpiler']}")
    if t["shots"] != 4096:
        raise ValueError(f"{name}: shots must be 4096, got {t['shots']}")


def coupling_map(config: dict) -> CouplingMap:
    return CouplingMap(config["coupling_edges"])


def noise_model(config: dict) -> NoiseModel:
    n = config["noise"]
    model = NoiseModel()
    p1 = n["single_qubit_depolarizing_probability"]
    p2 = n["cx_depolarizing_probability"]
    pr = n["readout_bit_flip_probability"]
    model.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["x", "sx", "rz"])
    model.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    # Symmetric assignment error: P(1|0)=P(0|1)=pr.
    model.add_all_qubit_readout_error(ReadoutError([[1-pr, pr], [pr, 1-pr]]))
    return model
