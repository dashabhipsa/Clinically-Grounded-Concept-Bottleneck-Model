"""Phase 3 reporting pipeline.

A single entry point that collates, for every available trained checkpoint:

* diagnostic / concept / spatial metrics (Phase 1 + 2 evaluation),
* intervention experiments (oracle, robustness, recovery, completeness,
  dependency) for CBM-style models,
* failure analysis (A-E categories),
* qualitative evidence figures (Grad-CAM / concept-evidence / intervention),
* structured, fact-only explanations.

**Nothing here trains a model and no result is fabricated.** Every number in
the produced report is computed from actual ``predict()`` outputs; any metric
whose checkpoint is missing is reported as ``N/A`` (pending), never invented.

Modules
-------
``discovery``  -- locate ``best_model.pth`` checkpoints and read their configs.
``analysis``   -- run the Phase 3 analyses for one model and persist artifacts.
``report``     -- assemble the final comparison + master markdown report.
"""
