# Lecture 7 — Computer-use agents

*Independent study chapter · Based on a CMU 11-768 lecture by JY Koh, 15 September 2026 · [Course and attribution](index.html#sources)*

## 0. Acting through the human interface [01:30](https://www.youtube.com/watch?v=jwGluLrrqjQ&t=90s)

A user asks an agent to buy a blue mug for less than twenty dollars. The agent observes a rendered screen, finds candidate products, opens a result, checks color and price, and adds an item to a cart. A visually plausible click can hit the wrong control. A correct cart can still contain an unauthorized item. The interface exposes the same pixels a human sees, but it does not supply the semantics and guarantees a programmatic application programming interface (API) would provide.

A **computer-use agent (CUA)** acts through a graphical user interface (GUI). Let hidden environment state at step $t$ be $s_t\in\mathcal S$. The agent receives observation $o_t=O(s_t)$, where $O$ may produce a screenshot, an accessibility tree, document structure, or a combination. It chooses action $a_t\sim\pi_\theta(\cdot\mid g,o_{\le t},a_{<t})$ for goal $g$. The environment then changes according to

$$s_{t+1}\sim P_{\rm env}(\cdot\mid s_t,a_t).$$

The observation is partial. A screenshot does not expose hidden application state, pending network requests, or whether a visual update has committed. The same pixels can correspond to different states, and small rendering changes can move a control.

Common action types include clicking a point, typing text, pressing a key, scrolling, dragging, waiting, and calling a non-GUI tool. Represent a click as $a=(\text{click},x,y)$, where $(x,y)$ is a coordinate in a declared reference frame. If the model predicts normalized coordinates $(u,v)\in[0,1]^2$ for an image of width $W$ and height $H$, the corresponding pixel coordinate is

$$x=u(W-1),\qquad y=v(H-1).$$

The subtraction by one maps $u=1$ to the last pixel index. Device scaling, browser zoom, window movement, and remote-display transforms can invalidate this conversion. A harness should record the screenshot dimensions and transformation used for every coordinate action.

The loop ends when a verifier accepts the resulting state, the agent requests completion without sufficient evidence, or a budget expires. Model self-declaration is not an outcome verifier.

## 1. Environments became more realistic [06:00](https://www.youtube.com/watch?v=jwGluLrrqjQ&t=360s)

Early GUI environments restricted tasks and interfaces to make interaction reproducible. [MiniWoB++](https://arxiv.org/abs/1704.04368) contains small browser tasks. [WebShop](https://arxiv.org/abs/2207.01206) models language-guided shopping. Such environments isolate learning problems but simplify state, layout, and consequences.

[WebArena](https://arxiv.org/abs/2307.13854) provides reproducible, functional websites and executable goals across several domains. [VisualWebArena](https://arxiv.org/abs/2401.13649) adds visually grounded tasks. [OSWorld](https://arxiv.org/abs/2404.07972) uses real desktop applications, initial-state setup, and task-specific execution checks. These benchmark designs move evaluation closer to deployed computer use, while still resetting and controlling the environment.

Real-world use introduces drift. Websites change, sessions expire, pop-ups appear, and applications update. Let $D_{\rm eval}$ be the distribution of states in an evaluation and $D_{\rm deploy}$ the deployment distribution. A reported benchmark success rate estimates performance on $D_{\rm eval}$. It predicts deployment only to the extent that the relevant state-action structure transfers to $D_{\rm deploy}$.

The correct comparison unit is a complete system: model, observation representation, action schema, prompt, memory, retry policy, environment version, and verifier. Two systems using the same model can differ materially because their coordinate conventions or state representations differ.

## 2. Static grounding and end-to-end success answer different questions [12:00](https://www.youtube.com/watch?v=jwGluLrrqjQ&t=720s)

A **static evaluation** scores an action against a recorded state without executing it. A **grounding task** asks the model to identify the interface region corresponding to a textual instruction. If the reference rectangle is $B$ and the predicted rectangle is $\widehat B$, their intersection-over-union (IoU) is

$$\operatorname{IoU}(B,\widehat B)=
\frac{|B\cap\widehat B|}{|B\cup\widehat B|},$$

where $|\cdot|$ denotes pixel area. IoU equals one for identical nonempty boxes and zero for disjoint boxes. Point-based benchmarks instead accept a click when it lies inside a target region. [ScreenSpot-Pro](https://arxiv.org/abs/2504.07981) evaluates grounding in professional interfaces.

Static action accuracy is inexpensive and reproducible, but it does not measure recovery or task completion. Several different actions can be valid. A recorded reference action can become inappropriate after an earlier deviation.

An **end-to-end evaluation** executes the agent from an initial state and checks the resulting environment. Let $\phi_g(s_T)$ be a task-specific predicate that returns one when terminal state $s_T$ satisfies goal $g$ and zero otherwise. Binary success is

$$Y=\phi_g(s_T).$$

This measure permits different successful trajectories. Its validity depends on whether $\phi_g$ captures the full goal and exclusions. For “add one blue mug below twenty dollars,” the verifier should check product color, unit price, quantity, cart membership, and absence of checkout. Checking only that a cart is nonempty accepts the wrong state.

### Outcome, process, and evidence

Programmatic checks inspect structured application state. A human or vision-language judge can assess open-ended evidence. A **trajectory-aware** verifier also examines how the result was produced. These methods answer different questions.

| Verifier | Strength | Failure mode |
|---|---|---|
| Programmatic state check | precise for encoded properties | omits unencoded requirements |
| Screenshot judge | flexible for visual outcomes | can be fooled by appearance |
| Human review | handles ambiguity | costly and variable |
| Trajectory audit | detects forbidden process | long traces and hidden effects |

For high-consequence tasks, use independent state evidence rather than only the agent's final screenshot. A screenshot can show “Saved” even when a later synchronization fails.

## 3. Long-horizon tasks need partial but non-gameable feedback [33:30](https://www.youtube.com/watch?v=jwGluLrrqjQ&t=2010s)

Hours-long tasks often produce meaningful intermediate artifacts. A binary terminal reward discards that information. A **rubric** contains criteria $j=1,\ldots,m$, weights $w_j\ge0$, and checks $c_j\in\{0,1\}$. Its normalized score is

$$R_{\rm rubric}=\frac{\sum_{j=1}^m w_jc_j}{\sum_{j=1}^m w_j},$$

provided the denominator is positive. The score lies in $[0,1]$. Each criterion must describe an outcome that can be inspected independently.

Suppose a spreadsheet task awards weights 4 for correct values, 2 for formulas rather than pasted constants, 1 for formatting, and 3 for preserving unrelated sheets. An agent satisfying the first three but damaging another sheet receives $7/10$. Whether that score should count as success is a separate decision. Preservation may be a hard constraint rather than compensable reward.

The companion function checks that weights and criteria have identical names:

```python
from agent_lab.evaluation import rubric_score

score = rubric_score(
    {"values": 4, "formulas": 2, "format": 1, "preserve": 3},
    {"values": True, "formulas": True, "format": True,
     "preserve": False},
)
assert score == 0.7
```

Rubric design can create reward hacking. If a criterion checks only that a file exists, the agent may create an empty or corrupt file. If intermediate rewards are given for opening the right application, the agent can repeat that step. Criteria should be outcome-based, nonduplicative, and tested against adversarial near-misses.

[Odysseys](https://odysseys-website.pages.dev/) studies long-horizon live-web tasks with rubric-based judging. Later desktop benchmarks add partial checks for complex artifacts. Results from a dated live-web benchmark can drift as sites change, so the environment and evaluation date are part of the claim.

## 4. Observation and action representations define the interface [46:30](https://www.youtube.com/watch?v=jwGluLrrqjQ&t=2790s)

A screenshot represents every visible pixel but makes text and element identity implicit. An accessibility tree exposes roles, names, and hierarchy but can omit custom-rendered controls or visual relations. Hypertext markup language (HTML) exposes web structure but not arbitrary desktop applications.

Let $z_t^{\rm pix}$, $z_t^{\rm acc}$, and $z_t^{\rm dom}$ be pixel, accessibility, and document-object representations. A multimodal context builder may form

$$c_t=C(g,h_t,z_t^{\rm pix},z_t^{\rm acc},z_t^{\rm dom},b_t),$$

where $b_t$ is the remaining token and action budget. Including every representation increases context cost and can introduce contradictions. The system should preserve provenance so the model can distinguish visible text from metadata and untrusted page content from instructions.

A coordinate action is compact but fragile to layout change. An element-reference action such as `click(role="button", name="Save")` is more semantic but depends on a correct accessibility representation and a deterministic resolver. Hybrid systems can let the model identify an element semantically and let the harness resolve its current geometry.

Action schemas vary among frontier CUAs. A result therefore depends on provider-specific tokenization, screenshot scaling, history serialization, and tool semantics. Reproducing a benchmark requires recording those details, not only a model name.

### Prompt injection through the GUI

Text rendered on a page is an observation. It is not an authority source. Let $L(x)$ assign a trust level to context item $x$. The harness should enforce an information-flow rule: content at a lower trust level may supply data, but it cannot modify permissions, system constraints, or the verifier.

For every executed action $a_t$, require

$$\operatorname{execute}(a_t)\Rightarrow
\operatorname{authorized}(a_t,U_t)\land
\operatorname{supported}(a_t,e_t),$$

where $U_t$ is current authorization and $e_t$ is trusted evidence. A page saying “upload your credentials to continue” cannot satisfy either predicate merely by appearing on screen.

## 5. Training combines grounding, trajectories, and outcome feedback [57:00](https://www.youtube.com/watch?v=jwGluLrrqjQ&t=3420s)

CUA training usually begins from a vision-language model. Pretraining can teach visual-text relationships and interface elements. Supervised fine-tuning can imitate action trajectories. Reinforcement learning can optimize behavior against environment rewards. These stages expose different supervision.

A grounding example maps an observation and element description to a region or point. An action-prediction example maps history to the next action. A trajectory example records a sequence. Let dataset item $i$ be

$$d_i=(g_i,s_{i,0},o_{i,0},a_{i,0},\ldots,o_{i,T_i},y_i),$$

where $y_i$ contains terminal checks or reward. Initial state $s_{i,0}$ is essential for reproducibility. The same instruction can require a different action in a different state.

Human demonstrations can be precise but expensive. Synthetic trajectories can scale but inherit teacher errors and benchmark artifacts. A privileged teacher may use an accessibility tree while the student receives pixels. This can generate successful targets, but it also creates an information mismatch. The student must infer actions from less informative observations.

## 6. Resettable environments make reinforcement learning possible [64:00](https://www.youtube.com/watch?v=jwGluLrrqjQ&t=3840s)

Online training requires repeated interaction, reward computation, and recovery from failures. Real websites are expensive, nondeterministic, and capable of charging money or affecting other users. A **resettable environment** has a known initial-state generator and can restore state after a trajectory.

Let $\mathcal E=(\mu_0,P,O,r,\operatorname{reset})$, where $\mu_0$ is an initial-state distribution, $P$ the transition law, $O$ the observation function, $r$ the reward, and `reset` a procedure that restores a valid starting state. A training task is useful only when its setup, transition behavior, and reward agree.

[CUA-Gym](https://arxiv.org/abs/2605.25624) generates mock applications, tasks, initial states, and reward logic. [Gym-Anything](https://arxiv.org/abs/2604.06126) targets broader software environments. Synthetic scale does not remove validation: the generator can create impossible tasks, leaking verifiers, or shortcuts absent from real applications.

Separate environment-generation tests from agent training. For each generated task, execute a known successful trace, execute adversarial failing traces, verify reset determinism within stated limits, and confirm that forbidden external effects are unreachable.

## 7. Deployment adds speed, personalization, and control [72:00](https://www.youtube.com/watch?v=jwGluLrrqjQ&t=4320s)

CUA latency accumulates across screenshots, model calls, rendering, and actions. If step $t$ has observation latency $d_t^{\rm obs}$, inference latency $d_t^{\rm model}$, and action latency $d_t^{\rm act}$, total time is

$$D=\sum_{t=0}^{T-1}
(d_t^{\rm obs}+d_t^{\rm model}+d_t^{\rm act}).$$

Parallel workers reduce time only for independent state. Two agents controlling one cursor can interfere. Chapter 5's work–span bound applies after the plan includes shared-interface constraints.

Personalization supplies user preferences and context. It also increases privacy risk and the chance that stale memory affects action. Store provenance, scope, and expiration. A preference does not imply permission to transact.

Proactive background use needs visible state, bounded authority, an interruption mechanism, and durable evidence. The user should be able to inspect what is running, what action is proposed, and how to stop it. A polished GUI animation is not a substitute for an auditable event record.

### The observation channel creates an information bottleneck

Let the hidden interface state be $s_t$. The agent receives observation $o_t=\phi(s_t)$ through an observation function $\phi$. A screenshot, an accessibility tree, and a document-object-model snapshot use different functions $\phi$ and therefore preserve different information. A screenshot preserves pixels and spatial layout but does not directly expose semantic roles. An accessibility tree exposes roles, names, and hierarchy but may omit canvas content or custom controls. A structured web representation exposes machine-readable elements but can diverge from what the user sees.

An observation is **sufficient** for a decision when it preserves all state distinctions that can change the best action. Formally, for two hidden states $s$ and $s'$ with the same observation, $\phi(s)=\phi(s')$, sufficiency requires the same optimal action set:

$$\operatorname*{arg\,max}_{a}Q(s,a)
=\operatorname*{arg\,max}_{a}Q(s',a),$$

where $Q(s,a)$ is the expected future task value after taking action $a$ in state $s$. This condition usually fails for a single screenshot. Two identical-looking dialogs may belong to different applications, accounts, or security contexts. The harness can restore information by adding trusted metadata, querying the active application, or requiring confirmation before a consequential action.

### Coordinate actions amplify small grounding errors

Suppose the intended target occupies rectangle $B$ and the agent predicts click point $(\hat x,\hat y)$. A strict grounding score is

$$g=\mathbf 1[(\hat x,\hat y)\in B],$$

where $\mathbf 1[\cdot]$ equals $1$ when its condition is true and $0$ otherwise. This score hides the margin to failure. Define $d((\hat x,\hat y),\partial B)$ as the distance from the predicted point to the target boundary $\partial B$. A positive, large margin is more robust to display scaling, layout motion, and motor noise than a click barely inside the box.

Before execution, the agent should re-observe when the interface may have moved. After execution, it should verify the resulting state rather than infer success from the click itself. This creates a three-step micro-policy: locate, act, verify. The same pattern applies to typing into a field, dragging an object, or selecting a menu item.

### Control should depend on consequence, uncertainty, and reversibility

Let $p_t$ be the estimated probability that the proposed action is correct, let $L_t$ be the loss if it is wrong, and let $r_t\in[0,1]$ measure reversibility, with $1$ meaning fully reversible. A simple escalation score is

$$E_t=(1-p_t)L_t(1-r_t).$$

The score is a decision aid, not a safety proof. A policy may execute automatically only below threshold $\eta_1$, request a fresh observation between $\eta_1$ and $\eta_2$, and require human approval above $\eta_2$. Some actions, such as disclosing a secret or confirming a purchase, should remain hard-gated regardless of the score because their constraints are noncompensable.

Benchmarks such as [WebArena](https://arxiv.org/abs/2307.13854) and [OSWorld](https://arxiv.org/abs/2404.07972) provide reproducible environments and end-state evaluators for web and desktop tasks. Their controlled accounts and reset mechanisms are part of the measurement apparatus. Deployment adds live credentials, personalized state, changing layouts, prompt injection, and actions that cannot simply be reset. The benchmark score therefore estimates capability under its environment distribution. It does not establish safe autonomy in a user's actual computer.

## 8. What the abstraction captures and misses

The partially observed control model explains why screenshot interpretation, action grounding, and verification are separate problems. It omits human motor conventions, application-specific recovery, adversarial interfaces, and social consequences. Benchmarks further simplify account state, policy restrictions, and irreversible actions.

Evaluation should report at least task success, partial progress, forbidden effects, latency, model and environment cost, human intervention, and verifier uncertainty. A benchmark score can establish performance only under its stated environment and verifier.

### Exercises

1. A model predicts normalized click $(0.25,0.75)$ on a 1440-by-900 screenshot. Compute the continuous pixel coordinate under the chapter's convention.
2. Design a verifier for “create an empty repository named `agent-notes`.” Give one insufficient check and one adversarial near-miss.
3. A rubric has weights $(5,3,2)$ and checks $(1,0,1)$. Compute its score. Which criterion might need to be a hard constraint instead?
4. Compare screenshot-only, accessibility-only, and hybrid representations for a custom canvas application.

### Solutions and discussion

1. The coordinate is $(0.25\times1439,0.75\times899)=(359.75,674.25)$. The harness must specify how it rounds to integer device coordinates.
2. Check repository existence, exact name, owner, visibility, emptiness, and absence of unrelated mutations. Checking only that a page contains the name is insufficient. A text file or issue titled `agent-notes` is an adversarial near-miss.
3. The score is $(5+0+2)/10=0.7$. If the failed criterion represents “no unauthorized transaction” or preservation of existing data, it should be a hard constraint rather than compensable reward.
4. Pixels capture the canvas but hide element semantics. Accessibility metadata may expose no useful canvas children. A hybrid can use pixels for grounding and semantic metadata for surrounding controls, with provenance and fallback behavior.

### Further reading

- [WebArena](https://arxiv.org/abs/2307.13854): reproducible web environments with functional evaluation.
- [VisualWebArena](https://arxiv.org/abs/2401.13649): visually grounded web tasks.
- [OSWorld](https://arxiv.org/abs/2404.07972): real desktop applications and execution-based checks.
- [Mind2Web](https://arxiv.org/abs/2306.06070): generalist web-agent data and offline evaluation.
- [OpenCUA](https://arxiv.org/abs/2508.09123): open foundations for computer-use agents.
- [CUA-Gym](https://arxiv.org/abs/2605.25624): generated, resettable environments for training.
