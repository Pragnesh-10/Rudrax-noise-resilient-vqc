
Processor Topology Decision and Specification Notes

Governing Phase 1 guidance

The QFF Phase 1 Master Guide, page 2, states that the coupling maps in the
Processor A/B Benchmark Configuration are reference architectures, not
mandatory maps. Participants may design and justify any valid graph with the
required physical qubit budget.

Accordingly, this project uses its documented participant-defined coupling
maps. The difference between these maps and reference maps in other supplied
documents is not, by itself, a specification violation.

Current project topologies

Processor A: 5-qubit linear chain with edges
(0,1), (1,2), (2,3), (3,4).

Processor B: 9-qubit graph with edges
(0,1), (1,2), (2,3), (2,4), (4,5), (5,6), (4,7), (7,8).

The JSON configurations and validation logic in processors/ and src/
define the topologies used in the current benchmark. The benchmark results
must be interpreted as results for these specific graphs and settings.

Benchmark interpretation and limitations

The latest saved benchmark reports Processor A with mean compiled depth 23,
8 CX gates, and 64 operations; Processor B has depth 24, 7 CX gates, and
57 operations.

These resource counts are configuration-specific and do not establish that
either topology is universally superior.

The readout correction uses the configured error rate rather than a measured
calibration matrix.

ZNE scales simulator noise parameters; it is not gate-folding-based noise
amplification.

COBYLA reached its 100-evaluation limit and did not confirm convergence.

The benchmark evaluates 40 held-out examples and uses three repeated runs.
Treat the results as preliminary.

Final interpretation

The Phase 1 Master Guide permits participant-designed coupling graphs, so the
current topology choice is defensible under that guidance. Document the design
rationale and limitations clearly. Do not claim that these graphs outperform
the reference graphs unless those alternatives have also been benchmarked.
