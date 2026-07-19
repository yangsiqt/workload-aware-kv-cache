# Reports

Tracked report summaries and figures are small, reproducible artifacts. Files
whose names begin with `simulated-` come from mock backends and must not be
presented as GPU measurements. Per-request traces and raw run directories live
under `/root/workload-aware-kv-cache-data/runs` and are not committed.

`dual-h20/` contains real GPU summaries from one fixed arrival trace/profile.
These reports use p50/p90 and do not treat the automatically computed p99 as a
headline or claim statistical confidence from a single repetition.
