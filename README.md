# QFF 2026 Phase 1 — PS-Q3: Noise-Resilient Classification

## Status
Starter implementation scaffold aligned to the uploaded **QFF Phase 1 Master Guide**. Run and validate before submission; do not report generated metrics until the run completes successfully.

## Scope
- 4-qubit variational quantum classifier (VQC)
- Synthetic binary classification data generated with scikit-learn (`n_samples=200`, 4 features, `random_state=42`, 80/20 split)
- Participant-defined 5-qubit Processor A and 9-qubit Processor B coupling maps
- Identical synthetic noise probabilities on A and B: single-qubit depolarizing 0.001, CX depolarizing 0.005, readout bit-flip 0.01
- Ideal reference, noisy A/B, and readout-corrected A/B comparisons
- Transpilation optimization level 1 and seed 42; noisy evaluations use 4096 shots/sample

## Setup
Python 3.10+ is recommended.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

## Important methodological notes
1. The Master Guide says coupling maps are participant-designed. The included graphs are starting choices based on its reference graphs, not a claim that they are the only valid maps.
2. This scaffold trains with an ideal statevector and a small COBYLA run. Training quality may be limited; increase iterations only after the end-to-end workflow runs.
3. Readout mitigation here uses the stated symmetric bit-flip probability analytically. It is a lightweight mitigation demonstration, **not** a calibration matrix estimated from calibration circuits. Clearly disclose this distinction.
4. The current code reports depth/CX/SWAP from one representative compiled test circuit. For a stronger submission, compile and aggregate metrics across all test samples or report why the representative circuit is used.
5. Verify whether the transpiler's measurement mapping and the noise model's instruction names behave as expected in your installed Qiskit version. Do not submit without checking the run logs and saved outputs.
6. This version implements ZNE error mitigation, repeated-run variance analysis, and a comparative architecture study across Processor A, B, and D. Update with final results before submission.
7. If the organizer supplies newer topology/noise instructions, follow those and update `processors/*.json` and this README.

## Repository map
- `main.py`: end-to-end run
- `main.ipynb`: create a notebook that imports/runs the same workflow (required by the guide)
- `src/problem.py`: data generation and VQC
- `src/processors.py`: coupling maps and noise models
- `src/routing.py`: transpilation and circuit metrics
- `src/metrics.py`: noisy evaluation and readout correction
- `src/visualization.py`: plot
- `processors/`: processor configuration
- `results/`: generated outputs (do not commit fake or hand-entered metrics)

## Submission checklist
- [ ] Run from a clean environment
- [ ] Verify all output files and metric definitions
- [ ] Add `main.ipynb`
- [ ] Include exact Python/package versions used
- [ ] Add architecture rationale and limitations to `report.md`
- [ ] Ensure repository is public and submit its URL before the deadline stated in the latest official guide

- ZNE cost overhead now correctly reports total_shots as shots per sample multiplied by the number of noise factors.
