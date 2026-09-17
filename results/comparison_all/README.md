# comparison_all — measured results, T1 vs T2, raw vs stabilised

Four `evaluate_run.py` reports (episodes 002, 009, 047, 048), copied verbatim from
`~/egoforce_runs/<episode>/comparison_all/report.md` on the run machine. They are kept here
because the `.npz` runs they were computed from are large local run outputs that are not in
git and did not survive that machine — so while `evaluate_run.py` is still in the repo, these
particular numbers can no longer be regenerated.

Generated 2026-09-11 (047, 048) and 2026-09-13 (002, 009).

## Two caveats to read before quoting M3

**M3 is circular for T2.** T2 is produced by `fuse_egoforce_rtmpose.py`, which refits EgoForce
to the RTMPose 2D detections and labels the accepted rows `fused` (see that file, the row
written at `'fused', 'egoforce+rtmpose'`). M3 then scores the result against *that same*
RTMPose reference. So T2's very low fingertip medians — 4.6, 3.2, 1.4 and 1.5 px across the
four episodes — largely measure how well the refit hit its own input, not accuracy. The
`wilor_pnpfail` rows in the source mix are the ones the refit rejected, and those are not
independent evidence either. The report's own header warns that agreement is not accuracy;
this is the stronger version of that warning.

**`reproj p95` is the signal in M2, not `reproj median`.** Every run reports a median of
0.0000 px because reprojection is self-consistent by construction. The p95 for the
unstabilised runs is 5.6e7 to 7.7e7 px, which is degenerate frames rather than a large error,
and it lands next to `depth implausible frac` ~0.107-0.123 and 29-67 `degenerate span count`.
Stabilisation drives all three to zero, which is the result these four reports were made to
show.
