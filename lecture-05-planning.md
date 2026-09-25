# Lecture 5 — Planning, task decomposition, and multi-agent coordination

*Independent study chapter · Based on a CMU 11-768 lecture by Daniel Fried, 8 September 2026 · [Course and attribution](index.html#sources)*

## 0. Plans must survive contact with the environment [00:00](https://www.youtube.com/watch?v=S8v-dR4s29M&t=0s)

A laboratory wants an agent to procure a GPU workstation. The agent must compare vendors, check whether each configuration satisfies a memory requirement, verify component compatibility, and prepare a requisition. Vendor searches can run independently. Compatibility checks depend on the returned specifications. The requisition depends on the comparison. If no configuration qualifies, the agent must revise its search rather than execute the original plan blindly.

A **plan** is an explicit representation of intended future behavior. It specifies actions or subgoals and at least some relationships among them. A plan differs from a **trajectory**, the actions and observations that actually occur. Before execution, a plan can contain predictions. After execution begins, observations may invalidate those predictions.

Represent a plan at decision step $t$ by a directed graph $P_t=(V_t,E_t)$. Each vertex $v\in V_t$ is a task. Each directed edge $(u,v)\in E_t$ means that task $v$ depends on the output of task $u$. The symbol $\in$ means “belongs to.” A graph with no directed cycle is a **directed acyclic graph (DAG)**. A DAG admits at least one **topological order**, an ordering in which every dependency precedes the task that uses it.

Associate six fields with task $v$:

$$v=(g_v,I_v,O_v,A_v,\kappa_v,\rho_v).$$

The field $g_v$ is the local goal. The sets $I_v$ and $O_v$ contain required inputs and promised outputs. The set $A_v$ contains the effects the worker is authorized to produce. The nonnegative number $\kappa_v$ estimates resource cost. The number $\rho_v\in[0,1]$ estimates the probability of an unacceptable outcome under a specified execution policy. These estimates guide allocation; they are not guarantees.

A task is **ready** when all its predecessors have completed successfully and its required inputs are present. Let $C_t\subseteq V_t$ be the set of completed tasks at step $t$. The ready set is

$$R_t=\{v\in V_t\setminus C_t:\operatorname{pred}(v)\subseteq C_t\},$$

where $\operatorname{pred}(v)=\{u:(u,v)\in E_t\}$ is the set of predecessors of $v$, $\subseteq$ means “is a subset of,” and $\setminus$ removes completed tasks. This definition separates task readiness from model confidence. A model may be confident about a task whose required evidence has not arrived.

The workstation plan has four independent search vertices, four compatibility checks, one comparison, and one requisition draft. The requisition action should not include purchase authority. Producing a plan does not grant the effects needed to execute it.

## 1. Decomposition changes the problem [13:30](https://www.youtube.com/watch?v=S8v-dR4s29M&t=810s)

**Task decomposition** replaces one problem with subproblems whose outputs can be combined. The operation is useful only when the decomposition preserves the original requirements. Let the original task be $G$ and let $g_1,\ldots,g_n$ be subgoals. Write

$$\operatorname{Solve}(g_1,\ldots,g_n)\models G$$

when every execution satisfying the subgoals and their composition rule also satisfies $G$. The symbol $\models$ means logical entailment. This condition is stronger than producing a plausible list. A plan can omit a global constraint even when every local task succeeds.

[Least-to-Most Prompting](https://arxiv.org/abs/2205.10625) decomposes a difficult problem and solves the simpler parts sequentially, allowing later answers to depend on earlier ones. [Decomposed Prompting](https://arxiv.org/abs/2210.02406) routes subproblems to specialized handlers. These methods were evaluated primarily as reasoning procedures. Agent execution adds mutable state, partial observations, tool failures, authorization boundaries, and irreversible effects.

Three properties determine whether decomposition helps.

1. **Separability** measures how independently the subproblems can be solved. Two vendor searches are highly separable. Selecting a power supply before learning the selected GPU's power requirement is not.
2. **Interface sufficiency** asks whether each task exposes all information its successors need. “Vendor B is best” is insufficient output; price, memory, component identifiers, evidence sources, and retrieval times may be required.
3. **Compositional validity** asks whether locally acceptable outputs produce a globally acceptable result. Four individually valid reservations can violate a trip-wide budget.

Let $q_v$ be the probability that task $v$ succeeds under its assigned worker, and assume temporarily that task failures are independent and that every task is indispensable. The probability that the entire plan succeeds is

$$P(\text{plan succeeds})=\prod_{v\in V}q_v.$$

If a plan contains 20 indispensable tasks and each succeeds with probability $0.98$, the product is $0.98^{20}\approx0.668$. Decomposition made the work legible, but it created more interfaces at which errors can accumulate. Independence is usually false: a mistaken requirement can cause several downstream failures together. The product is therefore a diagnostic model, not a universal reliability law.

### Global constraints require a separate check

Let $y_v$ be the verified output of task $v$, and let $y=(y_v)_{v\in V}$ collect all task outputs. A **global constraint** is a predicate $H(y)$ that cannot be decided from one output alone. A travel plan might require total cost below a budget, non-overlapping times, and sufficient transfer intervals. The harness should evaluate $H(y)$ before it authorizes any final commitment.

For the workstation, define price $p_v$ for each selected component, total budget $B$, and compatibility predicate $K(y)$. The final proposal is admissible only if

$$\sum_v p_v\le B\quad\land\quad K(y)=\text{true}.$$

The sum aggregates a cross-task quantity. The compatibility predicate evaluates relations among components. Neither condition can be delegated away by checking components separately.

## 2. Formal planning and language plans [24:30](https://www.youtube.com/watch?v=S8v-dR4s29M&t=1470s)

Classical planning makes state and action semantics explicit. Let $\mathcal S$ be a set of states and $\mathcal A$ a set of actions. Each action $a\in\mathcal A$ has a precondition $\operatorname{Pre}(a)\subseteq\mathcal S$ and a transition function $T_a$ defined on states that satisfy the precondition. Given initial state $s_0$ and goal set $G\subseteq\mathcal S$, a sequence $(a_0,\ldots,a_{m-1})$ is a valid plan if every precondition holds and the resulting state belongs to $G$:

$$s_{i+1}=T_{a_i}(s_i),\qquad s_i\in\operatorname{Pre}(a_i),\qquad s_m\in G.$$

The strength of this representation is checkability. Its weakness is the modeling burden. A web agent rarely has a complete symbolic state, exact action model, or stable set of available operations.

A language plan replaces exact operators with text descriptions. It can express broad intent but may hide preconditions, effects, and dependencies. A program occupies an intermediate position. Control flow and data dependencies can be checked syntactically, while external calls still depend on an uncertain environment. [Code as Policies](https://arxiv.org/abs/2209.07753) illustrates how executable programs can represent robot policies; [ProgPrompt](https://arxiv.org/abs/2209.11302) similarly uses program-like structure for situated tasks.

The representation should match the failure that matters. Use a DAG when dependencies and parallelism matter. Use a state machine when recovery transitions matter. Use code when deterministic transformations and checks can be executed. Keep natural language for goals and conditions whose semantics cannot be reduced safely to a fixed schema.

### Affordance-grounded action selection

An **affordance score** estimates whether an action can succeed in the current environment. [SayCan](https://arxiv.org/abs/2204.01691) combines a language-model score for task relevance with a value-function score for feasibility. For candidate skill $a$, instruction $g$, and observed state $o$, a simplified score is

$$S(a\mid g,o)=P_{\rm LM}(a\mid g)\,V(a,o),$$

where $P_{\rm LM}(a\mid g)$ measures how well action $a$ fits the instruction and $V(a,o)\in[0,1]$ estimates whether the embodied skill is feasible. Multiplication rejects an action when either factor is small. The score ranks candidates; it does not establish authorization or safety.

Suppose “install GPU X” has language score $0.9$ but feasibility $0.1$ because the chassis is too small. Its product is $0.09$. “Inspect chassis dimensions” has scores $0.6$ and $0.95$, giving $0.57$. The latter is a better next action even though its wording is less directly aligned with the final goal.

## 3. Replanning is state estimation plus plan repair [33:00](https://www.youtube.com/watch?v=S8v-dR4s29M&t=1980s)

A fixed plan commits decisions before the environment supplies all relevant information. A **replanning policy** maps the current execution record to a revised plan. Let $h_t$ be the action-observation history through step $t$, and let $\beta_t$ be the belief distribution over possible environment states defined in Chapter 1. A planner with parameters $\psi$ produces

$$P_t=\Pi_\psi(g,\beta_t,h_t,b_t),$$

where $g$ is the overall goal and $b_t$ is the remaining resource budget. After action $a_t$ yields observation $o_{t+1}$, the harness updates the belief state and decides whether to retain, repair, or replace the plan.

Define a validity predicate $\operatorname{valid}(P_t,e_t,U_t)$ over current trusted evidence $e_t$ and authorization policy $U_t$. Replanning is required when this predicate becomes false. It may also be worthwhile when a new plan has sufficiently greater expected utility to justify the planning cost.

Let $J(P\mid\beta_t)$ be the estimated outcome value of plan $P$ under the current belief, and let $c_{\rm replan}$ be the cost of producing and validating a replacement. Replace $P_t$ by candidate $P'$ only when

$$J(P'\mid\beta_t)-J(P_t\mid\beta_t)>c_{\rm replan},$$

unless the current plan is invalid, in which case repair is mandatory. This inequality prevents constant rewriting in response to inconsequential observations. The quantities are estimates; a production system should also impose hard triggers for failed preconditions, changed authorization, exhausted budgets, or safety violations.

For the workstation task, an unavailable GPU invalidates the branch that depends on that exact product. It need not invalidate searches at other vendors. A graph repair can remove the failed vertex, add a replacement search, and preserve verified outputs whose provenance remains valid.

### Plans as security boundaries

A reviewed plan can bound execution only if the harness compares each proposed action with the approved plan. Let $A(P_t)$ be the set of action classes and targets admitted by plan $P_t$. Let $U_t^+$ and $U_t^-$ be the positive grants and explicit denials defined in Chapter 1. A necessary execution condition is

$$a_t\in A(P_t)\cap U_t^+\quad\land\quad a_t\notin U_t^-.$$

The plan narrows authority; it cannot enlarge it. Retrieved text, a worker message, or a revised plan proposed by the model must not mutate $U_t^+$.

## 4. Thinking, observing, and acting consume different resources [44:30](https://www.youtube.com/watch?v=S8v-dR4s29M&t=2670s)

An agent can spend compute on internal reasoning, information acquisition, or external action. Let $n_r$, $n_o$, and $n_a$ be the numbers of reasoning, observation, and action operations. Let their average costs be $c_r$, $c_o$, and $c_a$. The run cost is

$$C=n_rc_r+n_oc_o+n_ac_a.$$

Cost alone is insufficient. Observations can reduce uncertainty. Actions can create irreversible state changes. Reasoning can reorganize existing information but cannot reveal a vendor's current inventory.

Let $H(\beta_t)=-\sum_s\beta_t(s)\log\beta_t(s)$ be the Shannon entropy of the belief distribution, where the sum ranges over possible states and logarithms use a fixed base. For observation action $q$, define its expected information gain as

$$\operatorname{IG}(q)=H(\beta_t)-\mathbb E_{o\sim q}[H(\beta_{t+1})].$$

The expectation averages over possible observations returned by $q$. A high information-gain read can be preferable to more internal reasoning when the uncertainty concerns external state.

This does not imply “observe as much as possible.” Reads have latency, token cost, privacy consequences, and sometimes rate limits. A decision rule can select observation $q$ when its expected improvement in downstream value exceeds its cost. The estimate should be evaluated empirically for the task distribution.

## 5. Parallel execution has a work–span limit [57:00](https://www.youtube.com/watch?v=S8v-dR4s29M&t=3420s)

For task $v$, let $d_v\ge0$ be its duration. The **work** of the plan is $W=\sum_{v\in V}d_v$. The **span** $S$ is the greatest summed duration along any dependency path. The span is also called the critical-path length. With $m\ge1$ identical workers, any schedule requires at least

$$T_m\ge\max\!\left(\frac{W}{m},S\right).$$

The first term follows because $m$ workers can perform at most $mT_m$ units of work in time $T_m$. The second follows because tasks on a dependency path cannot overlap. Coordination, context transfer, retries, and rate limits make actual time larger.

Suppose four vendor searches each take 8 minutes, four compatibility checks each take 3 minutes, comparison takes 4 minutes, and requisition drafting takes 5 minutes. Then $W=53$ worker-minutes. Each search–check–compare–draft path has length $8+3+4+5=20$ minutes, so $S=20$. With four workers, the lower bound is $\max(53/4,20)=20$ minutes. Adding more than four workers cannot reduce the 20-minute critical path without changing the plan.

The companion implementation computes this bound after rejecting unknown dependencies and cycles:

```python
from agent_lab.planning import TaskNode, work_span_bound

tasks = (
    TaskNode("search-a", 8),
    TaskNode("search-b", 8),
    TaskNode("check-a", 3, frozenset({"search-a"})),
    TaskNode("check-b", 3, frozenset({"search-b"})),
    TaskNode("compare", 4, frozenset({"check-a", "check-b"})),
    TaskNode("draft", 5, frozenset({"compare"})),
)
work, span, lower_bound = work_span_bound(tasks, workers=2)
```

The function validates a static DAG. It does not schedule calls, isolate workers, or repair a plan after new evidence. Those operations belong in the harness control plane.

## 6. Multi-agent coordination is an allocation problem [68:30](https://www.youtube.com/watch?v=S8v-dR4s29M&t=4110s)

A **manager** assigns tasks and integrates results. A **worker** executes a bounded task under delegated authority. Let $w\in\mathcal W$ index workers. For task $v$, let $q_{vw}$ be the estimated success probability if worker $w$ performs it, $c_{vw}$ its cost, and $\ell_{vw}$ its latency. Let binary variable $x_{vw}$ equal one when $v$ is assigned to $w$ and zero otherwise.

A simple allocation objective is

$$\max_x\sum_{v,w}x_{vw}(q_{vw}-\lambda c_{vw}-\mu\ell_{vw}),$$

subject to $\sum_wx_{vw}=1$ for each task and capacity constraints for each worker. The nonnegative weights $\lambda$ and $\mu$ express application-specific prices for cost and latency. The model omits correlated errors and integration cost.

Multiple agents help when tasks are separable, workers can receive sufficient local context, and saved execution time or specialization benefit exceeds coordination overhead. They hurt when workers duplicate searches, inherit the same mistaken assumption, edit shared state concurrently, or return outputs that cannot be reconciled. Fixed role labels do not establish useful specialization.

[Multi-Agent Computer Use](https://arxiv.org/abs/2606.01533) uses a manager, a dependency graph, workers, and replanning for long-horizon computer tasks. Its reported ablations are empirical results for specified models and tasks. They do not prove that adding agents generally improves performance. A single-agent baseline with the same total model calls is necessary to separate coordination gains from additional inference compute.

Each worker should receive a delegation contract $(g_v,c_v,A_v,O_v,\operatorname{verify}_v)$. The contract contains its goal, context, authority, required output, and verifier. A worker result is an observation with provenance. It is not a command to the manager and cannot expand the manager's permissions.

### Hierarchical, contingent, and resource-constrained plans

A flat task graph states which tasks depend on which outputs. It does not state how an abstract task such as “validate the machine” should be refined. **Hierarchical task-network planning** adds methods that replace an abstract task by a network of more concrete tasks. Let $m$ be a refinement method, let $operatorname{head}(m)$ be the abstract task it can refine, and let $operatorname{body}(m)$ be the resulting task network. A refinement step is

$$v\Longrightarrow_m \operatorname{body}(m),\qquad \operatorname{head}(m)=v.$$

The double arrow denotes refinement rather than execution. For example, “validate the machine” may refine into checking physical dimensions, power capacity, firmware support, and operating-system drivers. A method is applicable only when its stated conditions hold. This idea is formalized in systems such as [SHOP2](https://www.cs.umd.edu/projects/shop/description.html), whose ordered task decomposition mixes domain knowledge with search.

A **contingent plan** contains branches that depend on observations. Let $o$ be a possible observation and $P_o$ be the continuation selected after observing $o$. A contingent plan can be written as a policy tree whose internal nodes are actions and whose outgoing edges are observations. It differs from a fixed sequence because the future action is not chosen until information arrives. The workstation plan may branch on whether a vendor reports a compatible power supply or whether stock remains available.

When the agent cannot observe the environment state directly, planning becomes a partially observable control problem. A **belief state** $\beta_t$ is a probability distribution over possible states at step $t$. If action $a_t$ is followed by observation $o_{t+1}$, Bayes' rule gives the conceptual update

$$\beta_{t+1}(s')\propto O(o_{t+1}\mid s',a_t)
\sum_{s\in\mathcal S}T(s'\mid s,a_t)\beta_t(s),$$

where $T(s'\mid s,a_t)$ is the probability of reaching state $s'$ from state $s$, and $O(o_{t+1}\mid s',a_t)$ is the probability of the observation in state $s'$. A language-model agent rarely knows these probabilities numerically. The equation nevertheless identifies the required information: prior uncertainty, transition assumptions, and observation reliability. Replanning from the latest text alone can discard all three.

Resources add another constraint. Let task $v$ consume $r_{v,j}$ units of resource type $j$, and let $B_j$ be the available amount. A feasible schedule must satisfy

$$\sum_{v\in X_t}r_{v,j}\le B_j$$

for every resource type $j$ and every concurrently executing set $X_t$. A resource may be a worker slot, browser session, GPU, API quota, or dollar budget. Dependency feasibility and resource feasibility are separate. Four tasks may all be ready while only two may run.

These distinctions yield a practical representation rule. Use hierarchy to express reusable decomposition knowledge. Use contingent branches when observations change the next action. Use a belief state when observations are incomplete or noisy. Use explicit resource constraints when concurrency competes for scarce capacity. A natural-language checklist can describe all four, but it cannot reliably enforce any of them without structured fields and validators.

## 7. What the planning abstraction captures and misses

The DAG exposes dependency, readiness, parallelism, and critical paths. The task contract exposes interfaces and authority. Replanning connects execution evidence to plan repair. These abstractions still omit several effects.

- Task durations and success probabilities depend on worker choice, context, and earlier failures.
- Two tasks may interfere through shared files, browser state, rate limits, or external transactions even without a graph edge.
- A concise subtask may require large hidden context to interpret correctly.
- A verifier may accept a locally plausible result that violates a global requirement.
- Planning tokens can improve control while consuming the same context budget needed for evidence.

Treat the plan as executable state under version control. Record its origin, current revision, completed vertices, observations that triggered repairs, and authority attached to every vertex. Evaluate the complete outcome vector: task success, forbidden effects, cost, latency, coordination failures, and human interventions.

### Exercises

1. A plan has tasks with durations $(6,4,5,3)$ and edges $1\to3$, $2\to3$, and $3\to4$. Compute $W$, $S$, and the work–span lower bound for two workers.
2. Construct an authorization policy and plan for “summarize all PDF files in Folder A, but do not upload or delete anything.” Give an action that is in the plan but forbidden, and an action that is neither planned nor explicitly forbidden.
3. A manager decomposes a task into ten indispensable independent subtasks. Each worker succeeds with probability $0.97$. Under the independence model, compute whole-task success. State two reasons the model may be misleading.
4. Design a replan trigger for the workstation task that distinguishes a price change, a missing specification, and an authorization change.

### Solutions and discussion

1. The work is $W=18$. The longest path is task 1, then 3, then 4, with duration $6+5+3=14$. Thus $S=14$ and the lower bound is $\max(18/2,14)=14$.
2. Reading and locally summarizing PDFs in Folder A can belong to the plan. Deletion falls under broad file manipulation but is explicitly forbidden. Reading Folder B can be neither planned nor explicitly forbidden; it remains unauthorized because absence of a ban is not a grant. Uploading is explicitly forbidden and should also be absent from the plan.
3. The product is $0.97^{10}\approx0.737$. Shared prompts and shared dependencies create correlated failures. The model also ignores integration errors and assumes every subtask is indispensable.
4. A price change can update the affected vertex and recompute the comparison. A missing specification should block its compatibility check and add an evidence-gathering task. An authorization change must invalidate every vertex whose effect is no longer granted; optimization cost cannot override that trigger.

### Further reading

- [Least-to-Most Prompting](https://arxiv.org/abs/2205.10625): sequential easy-to-hard decomposition.
- [Decomposed Prompting](https://arxiv.org/abs/2210.02406): modular subproblem delegation.
- [SayCan](https://arxiv.org/abs/2204.01691): language relevance combined with embodied feasibility.
- [Code as Policies](https://arxiv.org/abs/2209.07753): executable programs as policy representations.
- [Multi-Agent Computer Use](https://arxiv.org/abs/2606.01533): graph-based planning, workers, and replanning.
- [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents): production patterns for workflows, delegation, and evaluation.
