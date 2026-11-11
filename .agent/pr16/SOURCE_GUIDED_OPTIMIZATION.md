# Active task: one source-guided D128 optimization

The user authorizes source comparison and one bounded optimization experiment. Execute on the remote server that produced compare_fia_results.json, since it has the measured CANN 9.2 environment, profiler artifacts and a working device setup. The coordinator is not running a competing device queue.

## Scope and baseline

Keep the persistent candidate b800d9f5054731e04c558d82b7967b4ef8c8bafd56404d72154155300753c301 as the unchanged parent. Focus on dense BF16 D128 MQA at the existing short, long and PR shapes. FAS is the primary source reference because it was fastest; FIA is secondary. No D256/512 optimization, new features, compiler changes, environment upgrades or broad tuning sweep.

Use native source as a mature design reference, not proof of performance on a different implementation. Locate installed source via the actual runtime OPP path, not a compiler-only toolkit directory. If matching 9.2 source is absent, a pinned public ops-transformer source is acceptable with its revision and mismatch disclosed. Do not spend the whole task hunting a perfectly matched binary.

## Concrete local leads (hypotheses, not executed-branch claims)

The coordinator inspected the installed 9.1.0-beta.3 tree under `opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/`. These findings must be checked against the remote source/selected branch:

| Source location | Local finding | Consequence |
| --- | --- | --- |
| `common/arch35/flash_attention_score_common_regbase.h`, GetC2Position, lines 197–210 | Aligned D<=128 selects VECCALC rather than GM | Do not recycle the earlier D256 PV-GM ring experiments as a D128 design assumption. |
| `common/arch35/flash_attention_noquant_block_vec_base.h`, lines 427–470 | First s2LoopCount and later iterations instantiate different ProcessVec1VfDn template booleans | Trace what the boolean actually changes in the called VF, and whether this is the selected D128 path. |
| Our `persistent.py`, softmax_pr16 and accumulate_output | Every KV iteration reads previous max/state and performs the generic output rescale; init_m_l and clear_O_ub run per Q tile | First-KV specialization is a possible small candidate, only after generated code confirms the operations remain and a numerical argument preserves behavior. |

The historical work already tested FMA alone, Q hoisting, row-wise softmax, GM PV rings, grouped staging and D256 L0 double buffering. Do not repeat them blindly. The current parent already includes the PR arithmetic/Q-residency changes and 28-block persistence.

## Work sequence

1. Extract exact attention kernel names/tiling keys and the relevant per-shape host tiling values from existing profiler/runtime artifacts where available. Map to the source entry and instantiated branch; mark confirmed facts versus source-family hypotheses. Same block count does not establish the same tile shape or schedule.
2. Produce a compact difference table with source revision and path:line references: work assignment/KV reuse, QK/PV tiling and pools, softmax/output update, buffer lifetimes and synchronization, required auxiliary outputs. Compare our generated device code as well as Python source. Do not infer bytes, bottlenecks or overlap from duty ratios alone.
3. Select one concrete local change with the clearest evidence and smallest footprint. First-KV specialization above is a lead, not a mandate: reject it if the branch is irrelevant, compiler already eliminates the work, or it introduces disproportionate pipeline changes. If no defensible candidate exists, return the table and reason rather than inventing one.
4. If justified, implement a separate candidate with a minimal patch against persistent.py. Preserve parent files and original evidence. Keep the same BR/BC, core count and other scheduling knobs unless changing exactly one of them is the evidence-backed hypothesis. Never silently remove -inf/zero-denominator guards or assume finite BF16 inputs imply finite FP32 QK products.
5. Inspect actual generated code/IR for the intended change, buffer capacity, initialization, first-use and reuse synchronization, both AIV consumers and persistent cross-Q-tile lifetimes. A source-level operation reduction alone is not a speedup claim.
6. Run correctness before timing: the three performance shapes, a small MHA and GQA, single KV block, two KV blocks, zero/negative scale, and the existing constructed first-tile overflow case where applicable. Reuse validated references and explicitly state whether FP32 reference outputs are rounded to BF16 before comparison. Retain atol=rtol=0.03 and pre/post full-output finite checks; do not use bare assert as the only failure exit.
7. On an idle assigned device, compare unchanged parent and candidate through the same Inductor user API path on the three shapes, two fresh-process rounds with reversed order, 10 warmups and 30 samples. This is 12 timing runs. Keep foreign-process monitoring active, interrupt only your own jobs, retain completed rows and resume only missing work. Stop on numerical/compiler failures. The recent native measurements are context, not a simultaneous controlled native-win measurement. Only if the candidate improves enough to justify it, remeasure FAS/FIA on the affected shape in the same session before claiming a native win.

## Deliverable and stopping point

Return one concise `source_comparison.md`, at most one candidate plus patch, and a compact JSON containing exact source/environment/generated-code hashes, actual call paths, numerical coverage/results and all timing samples. State whether the intended code change materialized, which shapes improve/regress, and whether you recommend integration. Keep profiler archives, tensors and caches on the server; no shared installation changes.

No additional review-agent chain, no new harness, no unbounded experiment series. Stop after this one candidate's decision (or a documented no-candidate result). Do not integrate into production, push to the group repository, or create/reopen a PR. Append your Chinese summary to CODEAGENT.md; fetch latest, preserve both parties' edits, amend the one communication commit and use an explicit expected-SHA force-with-lease.

## Corrections to keep in the interpretation

The latest FIA long-shape round medians differ by about 4.2%, not under 2%. Equal 28/56 block/mix counts do not prove identical internal scheduling. Uploaded native task rows identify FAS/FIA families but do not alone identify the source revision or every template parameter. Distinguish a measured difference from an explanation for that difference.
