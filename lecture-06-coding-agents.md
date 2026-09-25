# Lecture 6 — Agents for coding and software development

*Independent study chapter · Based on a CMU 11-768 lecture by Graham Neubig, 10 September 2026 · [Course and attribution](index.html#sources)*

## 0. Code generation is only one part of software repair [00:00](https://www.youtube.com/watch?v=1BWeH1oOM7k&t=0s)

A repository has a defect: the expression `value or default` replaces the valid value zero with a default. A useful coding agent must locate the behavior, infer the intended semantics, edit the correct file, run focused and regression tests, and explain the evidence for completion. Producing syntactically valid code addresses only one step.

A **coding model** is a language model trained to predict code and related text. A **coding agent** places such a model inside a harness that can observe a repository, invoke development tools, edit files, and evaluate results. Let $R_t$ be the repository state at step $t$, including tracked files, dependencies, and relevant environment configuration. Let $h_t$ be the recorded action-observation history and $g$ the issue specification. A coding policy chooses

$$a_t\sim\pi_\theta(\cdot\mid g,R_t,h_t),$$

where $a_t$ may inspect, search, edit, test, or terminate. The model parameters are $\theta$, and $\sim$ means “is sampled from.” The repository transition is $R_{t+1}=T(R_t,a_t,\xi_t)$, where $\xi_t$ represents tool and environment effects such as compiler output or a network failure.

The code model needs a vocabulary and training distribution that represent programming languages, identifiers, whitespace, documentation, and repository structure. **Tokenization** maps source text to discrete model units. Byte-level subword tokenization can represent arbitrary source bytes, but its segmentation affects sequence length and how easily the model recognizes identifiers or indentation. Code corpora also require deduplication: forks, vendored libraries, generated files, and copied solutions can dominate raw counts and contaminate evaluations.

The [StarCoder](https://arxiv.org/abs/2305.06161) work documents a large code pretraining corpus and its governance choices. [Code Llama](https://arxiv.org/abs/2308.12950) and [Qwen2.5-Coder](https://arxiv.org/abs/2409.12186) illustrate continued training on mixtures that include code. A code-only mixture can improve narrow coding metrics while degrading language or mathematical capabilities. Training data composition therefore defines a vector of capabilities rather than a single “coding ability.”

## 1. Infilling and repository-conditioned generation [18:00](https://www.youtube.com/watch?v=1BWeH1oOM7k&t=1080s)

Ordinary left-to-right generation predicts a suffix from a prefix. Many edits require information on both sides of a gap. **Fill-in-the-middle (FIM)** training transforms a sequence into a serialization containing the prefix, suffix, and missing middle. Let a code sequence be partitioned as $x=(p,m,s)$, where $p$ is the prefix, $m$ the missing middle, and $s$ the suffix. A FIM transformation can train the model on

$$\operatorname{FIM}(x)=(\langle P\rangle,p,\langle S\rangle,s,\langle M\rangle,m),$$

where the bracketed items are reserved control tokens. The training objective remains next-token prediction, but the transformed order lets the model condition the generated middle on both surrounding regions. [InCoder](https://arxiv.org/abs/2204.05999) and the [FIM scaling study](https://arxiv.org/abs/2207.14255) analyze infilling objectives.

Repository work adds context beyond the containing file. Let $D(R,g)$ be a retrieval procedure that selects repository evidence for issue $g$. The model sees context $c=D(R,g)$ and proposes patch $\Delta$. Applying the patch is a partial function

$$R'=\operatorname{apply}(R,\Delta),$$

because malformed context, conflicting changes, or invalid paths can make application fail. A patch that applies cleanly has passed a syntactic interface check. It has not yet satisfied the issue.

Useful additional training signals include commit diffs, review comments, execution traces, test failures, and bug-fix pairs. Each signal has a distinct semantics. A diff teaches transformations. A trace teaches dynamic behavior. A test result supplies outcome evidence. Combining them without source and role labels can teach the model to imitate tool output as if it were an action.

## 2. Evaluation requires behavioral evidence [31:00](https://www.youtube.com/watch?v=1BWeH1oOM7k&t=1860s)

Exact string match is unsuitable when many programs implement the same function. Token-overlap and representation-based metrics can measure similarity to a reference, but similarity neither proves functional correctness nor rules it out. **Execution-based evaluation** runs a candidate against tests or another executable specification.

Let $T^+$ be tests that failed before the patch and should pass afterward. Let $T^0$ be tests that passed before the patch and should remain passing. A basic repair predicate is

$$\operatorname{repair}(\Delta)=
\left(\bigwedge_{t\in T^+}t(R\oplus\Delta)=\text{pass}\right)
\land
\left(\bigwedge_{t\in T^0}t(R\oplus\Delta)=\text{pass}\right),$$

where $R\oplus\Delta$ denotes the patched repository and $\bigwedge$ means logical “and” over all listed tests. The first conjunction checks requested behavior. The second checks known regressions.

Passing tests are finite evidence, not a proof for untested inputs. A test suite can yield a **false positive** by accepting an incorrect patch or a **false negative** by rejecting behavior that the issue permitted. [EvalPlus](https://arxiv.org/abs/2305.01210) expands tests for code-generation benchmarks to expose under-specified evaluation. Real repository tasks also depend on build tools, data, services, and versioned environments.

### The meaning of pass@k

Suppose a model produces $n$ distinct sampled programs and $c$ pass the tests. For $1\le k\le n$, the standard finite-sample estimator is

$$\widehat{\operatorname{pass@}k}
=1-\frac{\binom{n-c}{k}}{\binom nk},$$

where $\binom nk$ is the number of size-$k$ subsets of an $n$-element set. The fraction is the probability that a uniformly chosen size-$k$ subset contains only failures. Its complement is the probability that the subset contains at least one passing program.

If $n=20$, $c=3$, and $k=5$, the estimate is $1-\binom{17}{5}/\binom{20}{5}\approx0.601$. This metric assumes a sampling procedure and test oracle. It does not measure the probability that one deterministic agent run repairs an arbitrary repository. The [Codex evaluation paper](https://arxiv.org/abs/2107.03374) introduced HumanEval and this estimator in its reported setting.

The companion function implements the combinatorial expression exactly for integer counts:

```python
from agent_lab.evaluation import pass_at_k

estimated_success = pass_at_k(
    sample_count=20,
    correct_count=3,
    k=5,
)
```

## 3. The localize–edit–verify loop [46:30](https://www.youtube.com/watch?v=1BWeH1oOM7k&t=2790s)

Repository repair can be factored into three recurring decisions.

1. **Localize:** identify files, symbols, and execution paths that can explain the symptom.
2. **Edit:** produce a focused state change with explicit assumptions.
3. **Verify:** run checks that can distinguish the intended repair from plausible mistakes.

Localization is an information-retrieval problem over structured artifacts. Let $F$ be repository files and $r(f\mid g,h_t)$ a relevance score. Selecting the top-scoring files can miss a low-ranked but necessary dependency. A dependency graph, symbol index, or targeted search can improve recall. Loading the entire repository may exceed the context budget and bury the evidence.

An edit should carry a **patch contract** $C_\Delta=(S,E,P,V)$. The set $S$ names intended files and symbols. The set $E$ names permitted effects. Predicate $P(R)$ states repository preconditions. Procedure $V(R,\Delta)$ states verification. The harness checks that the actual diff remains within $S$ and that forbidden files, generated artifacts, secrets, and unrelated formatting changes are absent.

For the falsy-zero defect, localization should find the defaulting expression and its callers. The edit changes `value or default` to an explicit `None` check. Focused tests cover zero, `None`, and another ordinary value. The regression suite then checks unrelated behavior. A report that says “tests passed” without naming the test command and scope provides weak evidence.

[SWE-agent](https://arxiv.org/abs/2405.15793) argues that the agent-computer interface materially affects repository navigation and editing. [Agentless](https://arxiv.org/abs/2407.01489) demonstrates a more fixed localization-and-repair workflow. Their difference is an engineering choice about control, not a universal ranking. A fixed workflow can reduce action errors on tasks that fit its decomposition. An adaptive loop can respond to unexpected test evidence.

## 4. Tool interfaces shape coding behavior [53:00](https://www.youtube.com/watch?v=1BWeH1oOM7k&t=3180s)

A shell-only interface is expressive: search, inspect, edit, and test can all be encoded as commands. Expressiveness also increases the action space and the consequences of malformed commands. A structured editor narrows the interface but may fail when its exact matching context is stale.

Common edit representations include whole-file replacement, search-and-replace blocks, unified diffs, and syntax-tree operations. Compare them by at least four dimensions:

| Representation | Context sensitivity | Application check | Typical failure |
|---|---:|---|---|
| Whole file | low | parse or compile | overwrites concurrent or unseen changes |
| Exact search/replace | high | unique match | whitespace or duplicate-match failure |
| Unified diff | medium | hunk context | wrong line context or rejected hunk |
| Syntax-tree edit | structural | node and type checks | parser/language coverage gap |

No representation proves semantic correctness. The adapter must preserve a distinction between proposal, application, and verification. A rejected patch should become an observation. It should not trigger an unbounded retry loop.

The editing mechanism also affects security. Shell text can interpolate secrets or execute unintended subprocesses. A production harness should isolate the repository, restrict network and credentials, impose process and disk limits, record commands, and require additional authorization before publication or deployment.

## 5. Repository benchmarks measure a joint system [59:00](https://www.youtube.com/watch?v=1BWeH1oOM7k&t=3540s)

[SWE-bench](https://arxiv.org/abs/2310.06770) constructs issue-resolution tasks from real repositories and corresponding changes. Each instance includes a repository state, issue description, and tests. A measured result depends jointly on the model, harness, tool interface, context policy, execution environment, and evaluation patch.

Let $Y_i(M,H,E)$ be one when system with model $M$, harness $H$, and environment $E$ resolves instance $i$. The mean success over evaluation set $I$ is

$$\widehat S(M,H,E)=\frac{1}{|I|}\sum_{i\in I}Y_i(M,H,E).$$

Changing the harness changes the measured system. It is therefore incorrect to attribute a system score solely to the model unless the comparison holds the other components fixed. Repeated runs are needed when sampling, tool timing, or external services are stochastic.

A broader outcome record should keep dimensions separate. The companion type stores task success, regression preservation, forbidden effects, cost, and latency. It defines an admissible success only when the requested behavior passes, regressions pass, and no forbidden effect occurs:

```python
from agent_lab.evaluation import OutcomeVector

result = OutcomeVector(
    task_success=True,
    regressions_preserved=True,
    forbidden_effects=0,
    cost=0.42,
    latency=91.0,
)
assert result.admissible_success
```

Do not collapse this vector into a leaderboard rank before specifying an application-specific trade-off. A slower agent with fewer unauthorized changes may be preferable in a release repository. A rapid agent may be appropriate in a disposable sandbox.

Training tasks require leakage controls. Split by repository, project lineage, or time according to the intended claim. Near-duplicate issues, copied tests, and later commits can reveal the solution. Synthetic bug injection creates scale but may teach artifacts of the mutation process rather than naturally occurring maintenance work.

## 6. Software development extends beyond issue repair [69:00](https://www.youtube.com/watch?v=1BWeH1oOM7k&t=4140s)

Requirements analysis, application construction, test generation, continuous-integration repair, dependency migration, review, and maintenance have different oracles. A unit-test oracle is strong for a pure function. It is incomplete for usability, visual fidelity, performance, accessibility, or ambiguous product intent.

For a visual repair, combine state checks with rendered evidence. Let $S$ measure behavioral success, $V$ visual agreement under a stated comparison, $A$ accessibility conformance, $C$ cost, and $D$ delay. Report $(S,V,A,C,D)$ rather than hiding the components in one score. A vision-language judge can assess open-ended artifacts, but its sensitivity to prompts and visual evidence must be measured against human adjudication.

For long-lived code, verification must address maintainability. A patch can pass current tests while increasing coupling or depending on an unsupported API. Static checks, type checks, documentation, dependency policies, and human review provide different evidence. None substitutes automatically for the others.

## 7. Predicting program behavior can reduce expensive execution [74:00](https://www.youtube.com/watch?v=1BWeH1oOM7k&t=4440s)

A **code world model** predicts an execution outcome or next program state from code, inputs, and history. Let $z$ be an execution outcome and $q_\phi(z\mid R,a)$ the world model's predictive distribution for action $a$ in repository state $R$. An agent can rank proposed edits using expected predicted utility

$$\widehat U(a)=\mathbb E_{z\sim q_\phi(\cdot\mid R,a)}[u(z)]-\lambda c(a),$$

where $u(z)$ values the predicted outcome and $c(a)$ estimates action cost. This can reduce real executions only when prediction is accurate in the decisions where rankings differ.

The world model does not replace tests. Distribution shift, undefined behavior, environment dependencies, and adversarial code can make simulation wrong. Evaluate both prediction calibration and downstream repair success. A model that predicts common traces accurately may still fail on the rare branch containing the defect.

### Verification is a sequence of increasingly expensive claims

A proposed patch $\Delta$ transforms repository state $x$ into $x'=\operatorname{apply}(x,\Delta)$. Verification asks several different questions. They should not be represented by one Boolean until the release policy combines them.

1. **Syntactic validity:** does the changed program parse and compile?
2. **Local behavioral validity:** do focused tests for the modified component pass?
3. **Regression preservation:** do tests for previously supported behavior still pass?
4. **Static validity:** do type, lint, security, and dependency checks accept the change?
5. **System validity:** does the built artifact behave correctly in its deployment environment?

Let $z_i(\Delta)\in\{0,1\}$ be the result of verification stage $i$, where $1$ means that the stage passes. A strict acceptance policy is the conjunction

$$\operatorname{accept}(\Delta)=\bigwedge_{i=1}^{m}z_i(\Delta).$$

The symbol $\bigwedge$ means logical AND over all $m$ stages. The stages are not interchangeable. A compiling patch can be behaviorally wrong; a passing focused test can coexist with a distant regression; a passing suite can miss an insecure new dependency.

The order should usually minimize expected wasted cost. Let stage $i$ cost $c_i$ and reject a bad candidate with conditional probability $q_i$. Running a cheap, discriminating check early avoids paying for expensive builds on obvious failures. The optimal order depends on correlations among failures, so the simple ratio $q_i/c_i$ is only a heuristic. It is useful when historical data show that one check cheaply removes many candidates.

### Localization is Bayesian evidence accumulation

A coding agent must decide where to inspect before it can edit. Let $F$ be the set of repository files, and let $D$ be the observed issue description, failing trace, and search evidence. For file $f\in F$, the posterior probability that $f$ contains a relevant defect is

$$P(f\text{ relevant}\mid D)\propto
P(D\mid f\text{ relevant})P(f\text{ relevant}).$$

The prior $P(f\text{ relevant})$ may reflect change history, ownership, or architectural centrality. The likelihood $P(D\mid f\text{ relevant})$ reflects symbol matches, stack frames, dependency paths, and test coverage. No implementation needs to assign calibrated probabilities to benefit from the decomposition. It prevents a common error: treating lexical similarity as proof of causal relevance.

A search result is evidence about location, not authority to edit. Before mutation, the harness should confirm that the candidate path lies inside the allowed repository root, is not generated or vendored code unless explicitly intended, and is consistent with local instructions. After mutation, the diff becomes the primary review object. Commands should run in a sandbox with bounded time, output, network access, and filesystem scope because repository tests can execute arbitrary code.

[SWE-agent](https://arxiv.org/abs/2405.15793) shows that an agent-computer interface designed around repository navigation and editing can materially change repair performance. [SWE-bench](https://arxiv.org/abs/2310.06770) shows why evaluation must execute repository tests against real issue instances. Neither result implies that a passing benchmark patch is production-ready. Hidden tests estimate task behavior under one environment; production acceptance also requires dependency, security, operational, and maintainability review.

## 8. What the abstraction captures and misses

The repository-state model captures the central loop: localize evidence, propose a patch, apply it under constraints, and verify behavior. It omits organizational context, incomplete requirements, hidden services, flaky tests, licensing, and human ownership. A deployed coding agent also needs checkpointing, rollback, secret isolation, and reviewable artifacts.

The strongest completion claim is proportional to the oracle. Passing a focused test supports the tested behavior. Passing the regression suite supports observed compatibility within that suite. Neither establishes full correctness. Report the changed files, executed checks, remaining uncertainty, and any environment deviation.

### Exercises

1. Compute pass@5 for $n=10$ samples with $c=2$ correct samples.
2. Write a patch contract for upgrading a dependency while forbidding changes to public APIs.
3. Give one false-positive and one false-negative failure of a repository test suite.
4. Design an evaluation vector for a frontend accessibility repair. Explain why a single success bit is inadequate.

### Solutions and discussion

1. The estimate is $1-\binom85/\binom{10}5=1-56/252\approx0.778$.
2. The intended scope includes the dependency manifest, lockfile, and compatibility adaptations. Preconditions include a clean baseline and supported runtime. Verification includes install, focused compatibility tests, the full suite, and an API-diff check. The contract rejects edits that change exported names or signatures.
3. A weak test may accept a hard-coded value for its only input, producing a false positive. An over-specific assertion may reject a different but permitted error message, producing a false negative.
4. Report functional success, keyboard navigation, accessible names, contrast checks, regressions, cost, and latency separately. A success bit cannot distinguish a visually corrected control that remains unusable by keyboard.

### Further reading

- [SWE-bench](https://arxiv.org/abs/2310.06770): executable real-repository issue resolution.
- [SWE-agent](https://arxiv.org/abs/2405.15793): agent-computer interfaces for software engineering.
- [Agentless](https://arxiv.org/abs/2407.01489): localization and repair through a fixed workflow.
- [EvalPlus](https://arxiv.org/abs/2305.01210): stronger test suites for code-generation evaluation.
- [InCoder](https://arxiv.org/abs/2204.05999): bidirectional code infilling.
- [Code World Models](https://arxiv.org/abs/2405.15383): predicting execution behavior for code reasoning.
