# `datadriven` Boundary

The legacy `datadriven` surface was inspected during the ML migration pass, but
it remains out of scope for now.

Reason:

- It is a symbolic-regression / model-discovery surface rather than a clear
  library-wrapper provider surface.
- The source artifacts do not establish a single ownership boundary cleanly
  enough to move it into `sciona.atoms.ml` without a broader classification
  decision.

If the repo later adopts a dedicated model-discovery family, this module is the
place to revisit the split.
