# Yearning-Informed Lower Bounds

This note formalises the "desire minus saturation" control signal from the salience-yearning
invariant and anchors it in proof-complexity lower bounds. It provides both an immediately usable
framework for `SPrimeController` and a path toward stronger systems without relying on heuristic
counting or exponential search assumptions.

## Setup and Notation
- Work with a CNF formula `F` on `n` Boolean variables.
- After step `t`, the solver (or proof system) has derived consequences `C_t`.
- Let the surviving assignment space be `R(t) = { x ∈ {0,1}^n : x ⊨ C_t }`.

We interpret `t` as the controller iteration index and treat derived clauses/actions as
`ControllerAction` instances in `SalienceControllerPolicy`.

## Saturation (distribution-free)
```
S*(t) = 1 - (1 / n) * log₂ |R(t)|
```
- `S*(t) = 0` when nothing has been ruled out.
- `S*(t) → 1` as the search space shrinks to a unique satisfying assignment (SAT) or to the empty set (UNSAT).
- No probabilistic assumptions are required.

## Desire via a Structural Gap (Resolution)
Let `w(t)` be the maximum clause width derived by time `t` in Resolution.
Let `w*(F)` be the minimum clause width required to refute `F` (if UNSAT) or to force a satisfying assignment by unit propagation (if SAT).
```
D*(t) = clamp( (w*(F) - w(t)) / n , 0 , 1 )
```
- Width has a provable size–width trade-off: large required width implies exponential proof size.
- `D*(t)` is the residual structural effort still needed.

Other proof systems admit analogous measures:
- Cutting Planes (CP): use proof rank as the structural gap.
- Polynomial Calculus (PC/PCR): use polynomial degree.
- Communication-lifted systems: use lifted query/communication complexity parameters.

## Yearning (Actionable Gap)
```
Y*(t) = max( 0 , D*(t) - S*(t) )
```
- Zero when the instance is decided (`D* → 0`).
- Positive when structural work remains despite current saturation.
- No additive `ε` is required; the clamp enforces non-negativity.

## Integration with `SPrimeController`
In `core/controller/s_prime.py`, yearning dynamics are encoded via `YearningState`
(`desire`, `saturation`). The instantaneous gain factor is:
```
Gain_i(t) = ε + max(0, D_i(t) - S_i(t))
```
with decay/accumulation governed by `ρ`, `σ`, `α`, `β`. To align with the theoretical definition:
- Interpret `presence_i(t)` as whether action `i` (e.g., deriving a clause of width `w`) was executed.
- Update rules (already wired into `SPrimeController.update_yearnings`) implement
  the discrete dynamics:
  - `S_{i}(t+1) = (1 - ρ) S_{i}(t) + σ · presence_i(t)`
  - `D_{i}(t+1) = D_{i}(t) + α (1 - D_{i}(t)) (1 - presence_i(t)) - β D_{i}(t) presence_i(t)`
- The yearning gain multiplies the canonical S′ score:
```
S'_i(t) = [ w₁ f(ΔA_i) + w₂ R_i + w₃ M_i ] · C_i · g_i(t) · (1 - k φ_i)ᵞ · Gain_i(t)
```
This matches the mathematical `Y*(t)` when `Gain_i(t)` approximates `D*(t) - S*(t)`.

## Immediate Consequences
- **Verification alignment**: Along any verifying path, width requirements collapse and
  `Gain_i(t) → ε`; the controller naturally throttles failed branches without unit-mismatch bugs.
- **Exponential lower bounds**: Families such as Tseitin contradictions have
  `w*(F_n) ≥ c n`. The size–width theorem implies any Resolution refutation must be exponential,
  so yearning cannot be quenched quickly. The controller therefore keeps prioritising high-yield
  actions, matching known lower bounds without assuming a brute-force search.

## Extending Beyond Resolution
| System              | Structural Gap (Desire) | Reference Insight                                  |
|---------------------|-------------------------|----------------------------------------------------|
| Cutting Planes      | Rank deficit            | Rank/size trade-offs via communication lifting     |
| Polynomial Calculus | Degree deficit          | Degree lower bounds on Tseitin / mod-p principles  |
| Lifted Systems      | Lifted query complexity | Query → communication → proof/circuit lifting     |

Each measure can feed the same saturation/desire update loop, yielding yearning-driven
scheduling in richer proof or solver architectures.

## Practical Guidance
- **Instrumentation**: Export `YearningState` diagnostics through `IntrospectionInterface`
  to visualise saturation/desire per action.
- **Parameter tuning**: Empirically calibrate `ρ, σ, α, β, ε` using benchmarks so that the
  running gain mirrors empirical `|R(t)|` shrinkage or width increases.
- **Curriculum linkage**: Tag curriculum lessons (e.g., `training/cot_curriculum/02_verification_driven`)
  with width/rank cues so the runtime learns to associate high yearning with structural progress.

## Future Work
1. Incorporate width estimators from clause traces to approximate `w(t)` in practice.
2. Add rank/degree estimators to extend yearning to CP/PC controllers.
3. Explore lifting-based curricula to train yearning-aware heuristics for higher proof systems.
4. Formalise convergence guarantees when yearning gain drives the existing salience invariant.
