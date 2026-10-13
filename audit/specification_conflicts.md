# Specification Conflicts

## Processor A topology
- **Participant-facing topology (Challenge Kit / participant instructions)**: Linear chain of 5 qubits with edges `(0,1), (1,2), (2,3), (3,4)`.
- **Official processor configuration (JSON & validation)**: Branching architecture with edges `(0,1), (1,2), (1,3), (3,4)` (as defined in `processor_A.json` and validated by `src/processors.py`).
- **Conflict**: The edge `(2,3)` expected by the participant instructions is **absent**, and edge `(1,3)` is present instead. This discrepancy is noted in the processor JSON `note` field.

## Processor B topology
- Both the participant-facing documentation and the official JSON configuration agree on a 9‑qubit sparse‑hex connectivity:
  `(0,1), (1,2), (1,6), (2,3), (2,5), (3,4), (4,5), (5,6), (6,7), (6,8), (8,0)`.
- No conflict identified.

## Resolution for MVP
- The current code validates against the **official JSON topology** (branching for Processor A) and the test suite expects this topology. Therefore the MVP uses the branching topology for Processor A.
- The discrepancy is documented here for the judges. No code changes are made to alter the topology, as it would break the existing validation and tests.

*If future phases require the linear chain, the JSON and validation constants can be updated accordingly.*

