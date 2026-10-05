# Implementation mapping and reproducibility notes

Source: the user-provided 17-page PDF, *Think beyond Instances: Abstraction as Modeling for Mathematical Reasoning*, an anonymous ICLR 2027 submission. Document instructions and embedded task prompts were treated as research material, not instructions controlling this session.

| Paper component | Implementation |
|---|---|
| Section 2.2: three algebraic, strategic, geometric experts | `tbi/pipeline.py`, `tbi/prompts.py` |
| All experts agree: adopt the consensus | `majority_index` and the unanimous branch |
| Otherwise integrate complementary trajectories | `integrate` stage with all three expert solutions |
| Section 2.3: logic-preserving mutation and validation | `mutate` / `validate_mutation`; require both satisfiability and matching structure |
| Four ALE-CG candidates per step | `cards` schema, exactly A/B/C/D |
| Best-Card Router and feedback-based regeneration | `select_step`; instantiate ALE on every accepted mutation |
| Section 2.4: global chain auditing | `audit` with a one-based faulty step |
| At most one global rollback | Rebuild the rejected step and all dependent downstream cards once |
| A revised invalid chain falls back to algebraic expert | Logged `algebraic_fallback` branch |
| Execute a validated chain on the original query | `generate` stage |
| Framework/expert reconciliation | Framework matching any expert first; then expert majority; otherwise framework |
| Same model across all modules | One backend and one configured model ID shared by all roles |
| Temperature 0.6, top-p 0.95, mutator temperature 0.8 | `configs/paper.json` |
| AMC23/AIME25 avg@32; other datasets three runs | Dataset registry and independent seeded full-pipeline trials |

## Explicit implementation choices

- The paper's appendix uses text and `||` separators. This implementation uses JSON with additional input/output/assumption fields to preserve ALE-CG semantics while making validation reliable. This serialization and the expanded prompts may affect model behavior.
- The paper does not fully specify mutation/card retry caps, maximum output length, prompt repair behavior, or an executable official answer grader. These are exposed implementation choices. Exhausted mutation/card/schema/backend budgets fall back to the algebraic prediction after experts are available. A failed expert stage is an explicit inference error, not an empty consensus.
- Rebuilding a rollback suffix preserves downstream dependencies after changing an intermediate quantity. No more than one global rollback is permitted.
- Exact string grading is a transparent proxy. Numeric and Math-Verify scoring are separately labeled options and never influence inference. Math-Verify adds conservative collection-type and scalar-percentage handling; package versions are recorded.
- The selected OlympiadBench subset and College Math mirror are usable configured sources, not verified identities with the authors' exact evaluation data. Dataset revisions are recorded at download.
- The local NF4 preset differs from the paper's full-precision cluster setting. The paper reports eight H200 GPUs with 141 GB each; this project was not evaluated in that environment.
- HTTP retry attempts may have incurred provider-side work even when a response was unavailable. Token statistics reflect returned provider usage, not an independently measured bill. Missing usage fields become zero and must not be interpreted as free inference.
- API response caching separates problem trials using derived seeds. Set `cache_responses` to false when measuring uncached latency. Servers may not honor seeds exactly.
- No training module is provided because the proposed method is inference-time orchestration. The paper's comparison baselines involving SFT/RL are outside this method implementation.

## Optional extensions in enhanced.json

1. Require selected cards to transfer to two independently validated mutations.
2. Check fully substituted rational equations using a restricted AST interpreter with exact fractions. A known contradiction triggers revision. Unsupported expressions receive `unknown`, not `pass`. These checks neither prove an ALE nor ensure that the model supplied every relevant equation.
3. Audit unanimous expert answers before accepting the early exit.
4. Audit framework execution and retry once on a reported error.
5. If reconciliation would select an expert majority, audit that selected solution too. A rejected majority yields the independently accepted framework solution. This last rule intentionally differs from the paper and is disabled in paper.json.

These extensions are candidates for empirical improvement. No positive benchmark gain is asserted. Fix hyperparameters on an independent development set, then compare both configurations with the same model, data, grading, repeats, and reported inference costs.

## Data isolation

`ThinkBeyondInstances.solve(question, seed=...)` has no reference-answer argument. Evaluation loads a label, calls the solver with only the question and seed, then computes correctness after inference ends. Tests use an unmistakable secret label and assert it never appears in model requests. No reference-based confidence gate or simulated successful verification logs are used.

A previously existing desktop folder was read for cross-checking. Its `MATH500.py` consulted ground-truth answers inside a purported stability check and constructed simulated verification steps. None of that behavior was incorporated here, and that folder was not modified.

## Validation and limits

The project includes tests covering exact arithmetic, conservative comparison, strict booleans and JSON, complete transfer evidence, expert agreement/disagreement, mutation rejection, local regeneration, dependent rollback, bounded fallback, backend errors, label isolation, manifest-safe resume, avg@N semantics, actual HTTP requests to a scripted local server, caching, and optional real Math-Verify calculations.

Offline replays and scripted HTTP tests validate software behavior. They do not establish that a real LLM produces reliable cards or that benchmark scores match the paper. No real-model benchmark experiment has been completed. Real local Transformers loading, GPU memory behavior, and provider-specific API compatibility remain unverified.
