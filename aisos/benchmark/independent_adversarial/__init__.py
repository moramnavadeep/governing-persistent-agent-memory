"""
Independent Adversarial Benchmark
(AISOS_Independent_Adversarial_Benchmark_Problem.md).

A benchmark package deliberately built as a SEPARATE package from
aisos/benchmark/semantic/ and aisos/benchmark/systems.py, with its own
corpus, attack taxonomy, evaluator, and query set -- none copied, renamed,
or derived from aisos/benchmark/semantic/corpus.py,
aisos/benchmark/semantic/workloads.py, aisos/benchmark/semantic/adversarial.py,
aisos/benchmark/semantic/manual_corpus.py, or the multi-agent MA1-MA12
workload suite. See docs/independent_adversarial_benchmark_protocol.md
Section "Independence discipline" for the exact, honest scope of what
independence means and does not mean when the benchmark designer and the
system's own author are the same party -- a limitation stated plainly,
not concealed.

SYSTEMS UNDER TEST are reused, deliberately -- aisos/benchmark/semantic/systems.py's
S1-S7/S6b/S8(SMSR-style) implementations ARE the systems this benchmark
evaluates, unmodified. Reusing an already-built, already-tested SYSTEM
implementation is not an independence violation; only reusing the
BENCHMARK DESIGN (corpus/attacks/evaluator/queries) would be.
"""
