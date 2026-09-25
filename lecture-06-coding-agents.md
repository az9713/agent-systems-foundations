# Lecture 6 — Agents for coding and software development

*Independent study chapter · Based on a CMU 11-768: AI Agents lecture by Graham Neubig, 10 September 2026 · [Course and attribution](index.html#sources)*

## 0. Code generation is only one part of software repair [00:00](https://www.youtube.com/watch?v=1BWeH1oOM7k&t=0s)

A repository has a defect: the expression `value or default` replaces the valid value zero with a default. A useful coding agent must locate the behavior, infer the intended semantics, edit the correct file, run focused and regression tests, and explain the evidence for completion. Producing syntactically valid code addresses only one step.

A **coding model** is a language model trained to predict code and related text. A **coding agent** places such a model inside a harness that can observe a repository, invoke development tools, edit files, and evaluate results. Let $R_t$ be the repository state at step $t$, including tracked files, dependencies, and relevant environment configuration. Let $h_t$ be the recorded action-observation history and $g$ the issue specification. A coding policy chooses

$$a_t\sim\pi_\theta(\cdot\mid g,R_t,h_t),$$

where $a_t$ may inspect, search, edit, test, or terminate. The model parameters are $\theta$, and $\sim$ means “is sampled from.” The repository transition is $R_{t+1}=T(R_t,a_t,\xi_t)$, where $\xi_t$ represents tool and environment effects such as compiler output or a network failure.

The code model needs a vocabulary and training distribution that represent programming languages, identifiers, whitespace, documentation, and repository structure. **Tokenization** maps source text to discrete model units. Byte-level subword tokenization can represent arbitrary source bytes, but its segmentation affects sequence length and how easily the model recognizes identifiers or indentation. Code corpora also require deduplication: forks, vendored libraries, generated files, and copied solutions can dominate raw counts and contaminate evaluations.

The [StarCoder](https://arxiv.org/abs/2305.06161) work documents a large code pretraining corpus and its governance choices. [Code Llama](https://arxiv.org/abs/2308.12950) and [Qwen2.5-Coder](https://arxiv.org/abs/2409.12186) illustrate continued training on mixtures that include code. A code-only mixture can improve narrow coding metrics while degrading language or mathematical capabilities. Training data composition therefore defines a vector of capabilities rather than a single “coding ability.”

### The unit of behavior is a repository transition

The falsy-zero defect can be stated as a behavioral relation before any edit is proposed. Let function $f(v,d)$ return $d$ only when value $v$ is absent and otherwise return $v$. If absence is represented by Python value `None`, the requirement is

$$f(v,d)=\begin{cases}
d,&v=\texttt{None},\\
v,&v\ne\texttt{None}.
\end{cases}$$

The existing expression `v or d` implements a different function because it replaces every falsy value, including zero, an empty string, and an empty container. The defect is semantic rather than syntactic. A model can emit valid Python and still implement the wrong partition of the input space.

The agent's target is a repository transition $R\rightarrow R'$ satisfying a specification $\Phi$. The specification can be decomposed into requested behavior $B$, preservation requirements $I$, and effect restrictions $A$:

$$\Phi(R,R')=B(R')\land I(R,R')\land A(R,R').$$

For this repair, $B$ requires zero to be preserved and `None` to select the default. The invariant $I$ requires previously supported ordinary values to remain unchanged. The restriction $A$ can limit edits to the implementation and its tests. This formulation makes clear why a patch diff, a passing focused test, and a clean regression suite provide different evidence.

The model is only one component of the transition. Repository checkout, dependency installation, search commands, patch application, process execution, timeouts, and result capture belong to the harness. A capable model with a broken test environment cannot demonstrate correctness. A weak model can appear strong if the evaluation leaks the reference patch. Every claim about a coding agent therefore names a complete system and a controlled initial repository state.

Code training data also carries legal and statistical structure. Deduplication should operate across repository forks and copied problem solutions, not only within files. Time-based splits must respect commit ancestry. License and provenance records should survive preprocessing. The [StarCoder](https://arxiv.org/abs/2305.06161) data pipeline and the open [OLMo](https://arxiv.org/abs/2402.00838) training report are useful primary examples because they expose choices that are often hidden behind a model score.

## 1. Infilling and repository-conditioned generation [18:00](https://www.youtube.com/watch?v=1BWeH1oOM7k&t=1080s)

Ordinary left-to-right generation predicts a suffix from a prefix. Many edits require information on both sides of a gap. **Fill-in-the-middle (FIM)** training transforms a sequence into a serialization containing the prefix, suffix, and missing middle. Let a code sequence be partitioned as $x=(p,m,s)$, where $p$ is the prefix, $m$ the missing middle, and $s$ the suffix. A FIM transformation can train the model on

$$\operatorname{FIM}(x)=(\langle P\rangle,p,\langle S\rangle,s,\langle M\rangle,m),$$

where the bracketed items are reserved control tokens. The training objective remains next-token prediction, but the transformed order lets the model condition the generated middle on both surrounding regions. [InCoder](https://arxiv.org/abs/2204.05999) and the [FIM scaling study](https://arxiv.org/abs/2207.14255) analyze infilling objectives.

Repository work adds context beyond the containing file. Let $D(R,g)$ be a retrieval procedure that selects repository evidence for issue $g$. The model sees context $c=D(R,g)$ and proposes patch $\Delta$. Applying the patch is a partial function

$$R'=\operatorname{apply}(R,\Delta),$$

because malformed context, conflicting changes, or invalid paths can make application fail. A patch that applies cleanly has passed a syntactic interface check. It has not yet satisfied the issue.

Useful additional training signals include commit diffs, review comments, execution traces, test failures, and bug-fix pairs. Each signal has a distinct semantics. A diff teaches transformations. A trace teaches dynamic behavior. A test result supplies outcome evidence. Combining them without source and role labels can teach the model to imitate tool output as if it were an action.

### A worked infilling transformation

Consider the incomplete function

```python
def choose(value, default):
    return <HOLE>
```

The prefix ends after `return ` and the suffix contains the newline and any later definitions. Under a prefix-suffix-middle serialization, the model receives the prefix, a control token, the suffix, and a control token requesting the missing span. The target is `default if value is None else value`. Because the suffix may contain callers or type annotations, it can disambiguate the intended return type.

FIM is an objective transformation, not a patching protocol. It does not identify the correct file, decide whether the change is authorized, or prove that the inserted span compiles. It improves the conditional information available during generation. The [InCoder](https://arxiv.org/abs/2204.05999) experiments separated causal masking from ordinary left-to-right generation and measured the value of suffix context. The [FIM-for-free study](https://arxiv.org/abs/2207.14255) examined how infilling can be mixed into autoregressive training without materially sacrificing ordinary left-to-right capability under its settings.

At repository scale, retrieval determines the effective suffix and surrounding context. Let $Z$ be the unknown set of artifacts required for the repair and $C=D(R,g)$ the retrieved context. Retrieval recall is $|C\cap Z|/|Z|$. High lexical precision with low recall can omit the caller that defines the real contract. Loading more files raises recall but consumes context and increases distraction. Search should therefore be iterative: form a hypothesis, retrieve evidence that can confirm or refute it, and update the localization state.

[RepoCoder](https://arxiv.org/abs/2303.12570) implements an iterative retrieval-and-generation loop for repository-level completion. Its experiments concern code completion rather than full issue repair, but the control pattern transfers. A generated candidate can become a new retrieval query. Newly retrieved context can then revise the candidate. A repair agent must add mutation boundaries and behavioral verification around that loop.

The [StarCoder2](https://arxiv.org/abs/2402.19173) report treats repository structure, data governance, and infilling as parts of one training system. That combination matters. A model trained on isolated files may know language syntax while lacking the distribution of cross-file dependencies that a repository agent must navigate.

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

The estimator answers a precise sampling question. It estimates the probability that at least one candidate in a uniformly selected subset of size $k$ belongs to the $c$ test-passing candidates among $n$ generated samples. It does not estimate semantic correctness beyond the tests. It also does not measure the cost of producing, executing, and selecting $k$ candidates.

Suppose two systems both report pass@5 of $0.60$. System A produces five candidates in one second and runs a strong hidden suite. System B produces five candidates in one minute and runs three weak examples. The numerical metric is the same, but the operational and evidential claims differ. At minimum, report the sampling temperature, number of generated candidates, selection procedure, execution budget, and oracle.

The estimate also has sampling uncertainty because $c$ is observed on a finite set of generated programs and benchmark tasks. Aggregate pass@k across tasks with confidence intervals or repeated seeds. Do not treat the combinatorial formula as eliminating empirical variance. It removes a particular bias associated with estimating the chance of at least one success from a finite generated sample.

[EvalPlus](https://arxiv.org/abs/2305.01210) demonstrates that augmenting tests can substantially change which generated programs count as correct. This is direct evidence that oracle strength is part of the measurement. A benchmark result should therefore state the exact test version rather than only the task name.

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

The [Software Engineering Agent (SWE-agent)](https://arxiv.org/abs/2405.15793) study argues that the agent-computer interface materially affects repository navigation and editing. [Agentless](https://arxiv.org/abs/2407.01489) demonstrates a more fixed localization-and-repair workflow. Their difference is an engineering choice about control, not a universal ranking. A fixed workflow can reduce action errors on tasks that fit its decomposition. An adaptive loop can respond to unexpected test evidence.

### Localization as sequential hypothesis testing

Let hypotheses $H_f$ state that file $f$ contains a necessary edit. The agent begins with prior probabilities $P(H_f)$ and observes evidence such as a stack frame, symbol match, import edge, or failing assertion. For evidence item $e$, Bayes' rule gives

$$P(H_f\mid e)=\frac{P(e\mid H_f)P(H_f)}{P(e)}.$$

The equation does not require numerically calibrated probabilities to be useful. It separates the prior from the evidential likelihood. A filename match may have high likelihood when the issue names a symbol. A frequently edited central module may have a high prior. Neither alone proves relevance.

For the falsy-zero defect, a failing assertion mentioning `choose(0, 7)` points to the test and stack trace. A text search for `or default` identifies candidate implementations. Call-site inspection determines whether `None` is the only absence marker. A focused experiment reproduces the defect before editing. Each step should change the localization hypothesis. Repeating broad searches without updating a hypothesis consumes budget without accumulating structured evidence.

The patch contract then turns the hypothesis into a bounded mutation. Let the allowed path set be $S=\{\texttt{defaults.py},\texttt{test_defaults.py}\}$. Let $d(R,R')$ be the set of changed paths. The scope condition is $d(R,R')\subseteq S$. Let predicate $Q(R')$ check that no public signature changed. The harness admits the patch only when both the scope condition and $Q(R')$ hold before it runs behavioral tests.

A correct local repair may still be incomplete if the same idiom appears elsewhere. The agent should distinguish **defect localization**, which finds a causal site for the observed failure, from **pattern search**, which finds other potentially affected sites. The issue specification determines whether the task is one repair or a repository-wide semantic migration.

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

### Edits should be transactions with explicit preconditions

An edit proposal can carry an expected file digest $h$, an intended transformation $\delta$, and a postcondition $Q$. The adapter applies the edit only when the current file digest equals $h$. It then checks that $Q$ holds. This compare-and-swap pattern prevents an agent from applying a patch to content that changed after inspection.

Let $x$ be the inspected file, $H(x)$ its digest, and $x'=\delta(x)$. The transactional rule is

$$H(x)=h\;\Rightarrow\;\operatorname{commit}(x')\text{ only if }Q(x').$$

If the digest differs, the adapter returns a conflict observation. The model must re-read and reconsider the edit. Automatic fuzzy matching can silently modify the wrong repeated block. Exact matching fails more often, but failure is visible and recoverable.

The same principle applies to commands. Every command should record its working directory, environment, timeout, exit status, and bounded output. Tests can execute arbitrary repository code. Sandboxing, secret isolation, and network restrictions are part of the coding-agent threat model even when the user requested only a source edit.

Tool design changes behavior by changing the action space. [SWE-agent](https://arxiv.org/abs/2405.15793) reports gains from an interface designed for language-model repository work. [Agentless](https://arxiv.org/abs/2407.01489) shows that a deliberately constrained localization-and-repair pipeline can also be competitive. These results motivate ablations of the interface and workflow. They do not establish one universal tool set.

The editing mechanism also affects security. Shell text can interpolate secrets or execute unintended subprocesses. A production harness should isolate the repository, restrict network and credentials, impose process and disk limits, record commands, and require additional authorization before publication or deployment.

## 5. Repository benchmarks measure a joint system [59:00](https://www.youtube.com/watch?v=1BWeH1oOM7k&t=3540s)

[SWE-bench](https://arxiv.org/abs/2310.06770), a benchmark for software-engineering issue resolution, constructs issue-resolution tasks from real repositories and corresponding changes. Each instance includes a repository state, issue description, and tests. A measured result depends jointly on the model, harness, tool interface, context policy, execution environment, and evaluation patch.

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

### The benchmark estimand must be stated

An **estimand** is the quantity an evaluation intends to estimate. For a fixed benchmark distribution $D$, harness $H$, environment image $E$, and model configuration $M$, a natural estimand is

$$S(M,H,E;D)=\Pr_{i\sim D,\,\omega}\left[Y_i(M,H,E,\omega)=1\right],$$

where $\omega$ collects sampling and runtime randomness. A finite benchmark estimates this probability under its task distribution. It does not directly estimate success on a private repository, a future dependency stack, or a different harness.

[SWE-bench](https://arxiv.org/abs/2310.06770) is valuable because issue text, repository state, and executable tests jointly define realistic repair tasks. Its realism also introduces environment reconstruction and oracle challenges. Report which benchmark subset, container images, test patches, and task exclusions were used. A result that silently drops setup failures estimates performance on a selected subset rather than on the named benchmark.

Contamination changes the estimand further. If a model has seen the issue, patch, or near-duplicate repository history during training, the evaluation mixes repair with retrieval from parameters. Time cutoffs are insufficient when forks or mirrors carry later code under earlier timestamps. Decontamination should search code, issue text, and tests at repository-family level. Residual uncertainty belongs in the report.

An outcome vector should accompany mean success. At minimum, retain requested behavior, regression preservation, forbidden changes, tool cost, elapsed time, and human intervention. These quantities answer different questions. A decision rule may combine them for one deployment, but the underlying vector permits later readers to apply a different trade-off.

## 6. Software development extends beyond issue repair [69:00](https://www.youtube.com/watch?v=1BWeH1oOM7k&t=4140s)

Requirements analysis, application construction, test generation, continuous-integration repair, dependency migration, review, and maintenance have different oracles. A unit-test oracle is strong for a pure function. It is incomplete for usability, visual fidelity, performance, accessibility, or ambiguous product intent.

For a visual repair, combine state checks with rendered evidence. Let $S$ measure behavioral success, $V$ visual agreement under a stated comparison, $A$ accessibility conformance, $C$ cost, and $D$ delay. Report $(S,V,A,C,D)$ rather than hiding the components in one score. A vision-language judge can assess open-ended artifacts, but its sensitivity to prompts and visual evidence must be measured against human adjudication.

For long-lived code, verification must address maintainability. A patch can pass current tests while increasing coupling or depending on an unsupported application programming interface (API). Static checks, type checks, documentation, dependency policies, and human review provide different evidence. None substitutes automatically for the others.

A lifecycle task also changes the unit of completion. Issue repair may end with a patch and tests. A dependency migration may require updated lockfiles, release notes, compatibility checks, and staged rollout. A feature request may require clarification because no executable oracle captures product intent. A code review may produce findings rather than mutations.

Let $q$ denote task type and let $V_q$ be its verification protocol. The harness should select $V_q$ from the declared task contract rather than applying one universal “run tests” step. For example, a frontend change can require document object model (DOM) assertions, keyboard navigation, screenshots at specified viewports, and performance budgets. A database migration can require forward migration, backward compatibility, rollback rehearsal, and data-integrity checks.

The strongest automation targets have cheap, discriminating oracles and reversible changes. As oracle ambiguity or consequence grows, human review becomes part of the intended control system. The agent should present the diff, evidence, and unresolved assumptions in a form that supports that review.

## 7. Predicting program behavior can reduce expensive execution [74:00](https://www.youtube.com/watch?v=1BWeH1oOM7k&t=4440s)

A **code world model** predicts an execution outcome or next program state from code, inputs, and history. Let $z$ be an execution outcome and $q_\phi(z\mid R,a)$ the world model's predictive distribution for action $a$ in repository state $R$. An agent can rank proposed edits using expected predicted utility

$$\widehat U(a)=\mathbb E_{z\sim q_\phi(\cdot\mid R,a)}[u(z)]-\lambda c(a),$$

where $u(z)$ values the predicted outcome and $c(a)$ estimates action cost. This can reduce real executions only when prediction is accurate in the decisions where rankings differ.

The world model does not replace tests. Distribution shift, undefined behavior, environment dependencies, and adversarial code can make simulation wrong. Evaluate both prediction calibration and downstream repair success. A model that predicts common traces accurately may still fail on the rare branch containing the defect.

Prediction quality should be evaluated where it changes action selection. Let $q_\phi(y=1\mid R,\Delta)$ be the world model's predicted probability that patch $\Delta$ passes a specified verification stage, and let $y\in\{0,1\}$ be the observed result. For a set of predictions, the Brier score is

$$\operatorname{BS}=\frac1N\sum_{i=1}^{N}(q_i-y_i)^2.$$

The score is zero for perfect probabilistic predictions and grows with squared error. Calibration can also be inspected by grouping predictions into probability bins and comparing mean confidence with empirical success. High top-one accuracy with poor calibration can make cost-sensitive scheduling unreliable.

Suppose a full integration test costs thirty minutes and a focused test costs ten seconds. A calibrated world model may help order checks or choose among several patches. It should not suppress the integration test for a release merely because it predicts success. The acceptable use depends on the cost of a false positive. Simulation is most defensible for prioritization and least defensible as the sole evidence for a high-consequence mutation.

[Code World Models](https://arxiv.org/abs/2405.15383) studies executable world models synthesized as Python programs for model-based planning. The research direction is complementary to execution. Real execution supplies ground truth for the environment that actually exists. A learned model supplies cheaper but fallible estimates that can allocate that execution budget.

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

### Scope and limits of the repository-state abstraction

The repository-state model captures the central loop: localize evidence, propose a patch, apply it under constraints, and verify behavior. It omits organizational context, incomplete requirements, hidden services, flaky tests, licensing, and human ownership. A deployed coding agent also needs checkpointing, rollback, secret isolation, and reviewable artifacts.

The strongest completion claim is proportional to the oracle. Passing a focused test supports the tested behavior. Passing the regression suite supports observed compatibility within that suite. Neither establishes full correctness. Report the changed files, executed checks, remaining uncertainty, and any environment deviation.

The falsy-zero repair illustrates the evidence ladder. Parsing rules out syntax failure. Focused tests for zero, `None`, and an ordinary value distinguish the intended input partition. The existing suite checks observed regressions. A diff review confirms that only the intended expression and tests changed. None of these checks proves behavior for every possible object with unusual truth-value semantics. The completion statement should name that residual boundary.

A concise completion record can contain five fields: repository revision, changed paths, behavioral claim, executed checks with outcomes, and unresolved assumptions. This record is an auditable observation for the surrounding agent harness. It is stronger than a confident natural-language declaration because each field can be compared with tool evidence.

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

The fourth exercise can retain a vector containing keyboard completion, screen-reader label correctness, focus visibility, visual regression, changed-file scope, latency, and human review count. The components should remain separate because deployment policies differ. Keyboard completion may be a hard acceptance condition, while a small visual difference may be reviewable.

The patch-contract exercise also requires a baseline. Before upgrading the dependency, record a clean installation and test result. The allowed effects include the manifest, lockfile, and necessary compatibility adaptations. An API-diff check evaluates the public-surface prohibition independently of functional tests. If the baseline already fails, the contract must label those failures as pre-existing rather than attributing them to the patch.

The first computation assumes that the ten generated candidates are exchangeable and that exactly two pass the stated oracle. It estimates the chance that a size-five subset contains at least one of those two passing candidates. It does not imply that five new samples on another task will succeed with probability $0.778$.

The third answer distinguishes two kinds of oracle error. A false positive accepts behavior outside the intended specification. A false negative rejects a behavior the specification permits. Adding more tests can reduce both only when the added assertions represent the specification correctly. An over-specific hidden test can increase false negatives.

### Further reading

- [SWE-bench](https://arxiv.org/abs/2310.06770): executable real-repository issue resolution.
- [SWE-agent](https://arxiv.org/abs/2405.15793): agent-computer interfaces for software engineering.
- [Agentless](https://arxiv.org/abs/2407.01489): localization and repair through a fixed workflow.
- [EvalPlus](https://arxiv.org/abs/2305.01210): stronger test suites for code-generation evaluation.
- [InCoder](https://arxiv.org/abs/2204.05999): bidirectional code infilling.
- [Code World Models](https://arxiv.org/abs/2405.15383): predicting execution behavior for code reasoning.
- [OLMo](https://arxiv.org/abs/2402.00838): an open training report with data and evaluation details.
- [StarCoder2](https://arxiv.org/abs/2402.19173): code-data governance, repository context, and infilling at scale.
