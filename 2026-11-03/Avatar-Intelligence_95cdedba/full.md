# Avatar Intelligence:

# Amplifying Human Agency Through Persistent Personal Agents

Zheng Wu<sup>1,\*</sup>, Xinbei Ma<sup>2,\*</sup>, Yao Yao<sup>1</sup>, Zhaoyuan Huang<sup>1</sup> Haowen Hu<sup>1</sup>, Yijie Lu<sup>1</sup>, Taolei Wang<sup>1</sup>, Yifei Yang<sup>1</sup> Wuyue Xiong<sup>1</sup>, Pengzhou Cheng<sup>3</sup>, Zhuosheng Zhang<sup>1,†</sup>

<sup>1</sup>Shanghai Jiao Tong University

<sup>2</sup>National University of Singapore

<sup>3</sup>Shanghai University

<sup>\*</sup>Equal contribution. <sup>†</sup>Corresponding author.

Abstract: Personal AI agents are increasingly capable of handling long-horizon tasks across applications, yet users often need to invest substantial efort in specifying goals, resolving ambiguity, and verifying outcomes. We argue that progress should be measured by how efectively agents amplify human agency: enabling people to pursue their goals with less routine supervision while retaining control over consequential decisions. We advocate Avatar Intelligence, a paradigm of persistent personal agents that represent users’ goals, values, and constraints through sustained decision delegation across contexts and time. We describe a maturity roadmap from human-directed execution to autonomous operation within delegated domains, and introduce the Agentic Oversight Index (AOI) as a framework for evaluating delivered value against agentside operational cost and human oversight burden. We identify three core research challenges: identity-continuous personalization that carries user preferences across tasks, user-situated evolution through Rehearsal Engineering, and safety mechanisms that combine technical control with accountability and recourse. These directions define a research agenda for reliable decision delegation to personal agents across tasks and over time.

Keywords: avatar intelligence; AI agents; decision delegation; human oversight; personalization; agent safety

Version: Version 1.0 Date: November 3, 2026

## Contents

1 Introduction 1   
2 Avatar Intelligence 2   
2.1 Avatar as a User-Centric System 2   
2.2 Roadmap 4   
2.3 Agentic Oversight Index (AOI) 4   
3 Applications 5   
3.1 Current Industry Products 5   
3.2 Application Scenarios 6   
4 Open Challenges 7   
4.1 Personalization: From Behavioral Alignment to Identity-Continuous Avatars . . . 7   
4.2 Evolving with Specialization . 9   
4.3 Safety 10   
5 Conclusion 11   
A Formalization of Avatar Personalization 14   
B One Instantiation of AOI 14

![](images/f6689c707769b07f220fa3a6bbf73d2db7886c5ad250bf2f68268d8a868b91af.jpg)  
Position Avatar Intelligence amplifies human agency through persistent, identity-aligned decision delegation.  
Figure 1: Conceptual overview of Avatar Intelligence. (A) The maturity roadmap traces the transfer of execution and decision responsibilities from users to avatars within delegated domains. (B) A shared avatar core supports a perception–planning–decision–execution loop and taskspecialized clones. (C) Evaluation accounts for both delivered value and the human efort required to obtain it: the Agentic Oversight Index, AOI = U − λP, combines normalized net value U with an explicit penalty for human oversight burden P, weighted by λ.

## 1 Introduction

Imagine an AI agent system that acts on your behalf: it makes decisions, executes actions, and communicates with humans or other agents with minimal human intervention. Such a system is more than a tool that waits for explicit commands; it resembles a user-centric operating system—a persistent digital representative that embodies your preferences, values, and goals and operates continuously across contexts and time.

Recent years have witnessed the rapid development of foundation models (e.g., GPT-5.5 [1], Gemini-3 [2], and DeepSeek-V4 [3]), which establish a general-purpose backbone for autonomous agents such as MUSE <sup>1</sup> and dots <sup>2</sup>. Meanwhile, agent frameworks (e.g., those equipped with prompt/context/harness engineering and memory systems) have enabled more robust long-horizon task capabilities [4]. Beyond the exchange of information in digital environments, agents can increasingly manipulate end-user devices such as computers and smartphones [5].

These developments make persistent personal agents increasingly plausible, but leave a central question unresolved: how much work must users still do to make delegation efective? When delegating a research task, for example, an agent may retrieve and summarize documents while the user continues to resolve ambiguous objectives, correct interpretations, and judge whether the result is suficient. Execution efort decreases, while the burden of specification, coordination, and verification remains substantial.

Our position is that personal agents should amplify human agency by allowing users to pursue more of their goals without supervising every intermediate decision. Human agency includes the ability to set and revise goals, choose which decisions to delegate, and redirect action when circumstances change. Progress in personal agents therefore depends on reducing the efort required to obtain reliable, user-aligned outcomes while preserving these forms of control.

We use Avatar Intelligence to describe persistent personal agents that represent a user’s goals, values, and constraints across tasks and over time. Concretely, an avatar should maintain commitments while the user is unavailable, adapt to changing circumstances, and know when to seek clarification or additional authorization. This requires learning from ongoing interactions and evaluating how reliably the agent represents the user over time.

We develop this position through four contributions:

• We characterize Avatar Intelligence through persistent user representation and sustained decision delegation, and describe a user-centric system architecture that supports these functions.

• We outline a maturity roadmap (L0–L3) for transferring operational and decision responsibilities within defined delegation scopes.

• We introduce the Agentic Oversight Index (AOI), an evaluation framework that accounts for delivered value, agent-side operational cost, and human oversight burden.

• We identify three core research directions: identity-continuous personalization, user-situated evolution through rehearsal, and safety mechanisms for controlled and accountable delegation.

## 2 Avatar Intelligence

The essence of Avatar Intelligence is to align AI agents with human needs and expectations. At its core, Avatar Intelligence reshapes the AI–human relationship by transforming agents from passive instruments into autonomous digital representatives that embody user identity, preferences, and values. Rather than merely automating isolated tasks, avatars extend human agency beyond biological constraints by enabling persistent, autonomous operation across contexts and time.

## 2.1 Avatar as a User-Centric System

We conceptualize Avatar Intelligence as a user-centric operating system that externalizes a user's cognition into a persistent and executable digital form (Figure 2). The kernel of such a system is not a thin memory cache, but a faithful and continuously updated model of the user's cognitive identity. More specifically, at the OS level, the Avatar continuously observes the user’s behavioral patterns and decision-making logic, implicitly modeling and iteratively refining its understanding of the user’s profile and behavioral patterns. Based on this evolving user model, it can automatically align with the user’s implicit intentions when carrying out tasks on the user’s behalf, thereby enabling more personalized and cognitively faithful task execution.

Existing personalization methods largely optimize behavioral outputs using both training-time methods [6–10] and decoding-time alignment [11–13], while current memory systems mainly extend context and accumulate experience [14–16].

However, current methods are still insuficient for an Avatar that must preserve identity across contexts and over time. Operationally, the Avatar is organized around four tightly coupled components: perception, planning, decision, and execution, grounded by a persistent memory substrate. Together, they form a closed loop that transforms environmental signals into useraligned action, enabling cognition to operate beyond explicit interaction.

![](images/a4964ab85231f2426f10f8f488c251c8ae985d03dc01e91023e5596bcaf24caf.jpg)  
Figure 2: Avatar Intelligence as a user-centric operating system. A persistent user model and memory support the inner loop of perception, planning, decision, and execution. The outer ring identifies three system capabilities: work continuity, proactive service, and capability expansion. Work, assistant, social, and custom clones carry the shared user identity into distinct task contexts, connecting the avatar to tools, applications, and other agents.

In practical deployment, the Avatar instantiates multiple task-specialized clones as contextual projections of a single underlying identity, including work, assistant, social, and custom clones. These clones inherit the same user model while interfacing with heterogeneous execution substrates such as tools, applications, sandboxed environments, and external agents. This clone layer gives the Avatar three system-level capabilities essential to Avatar Intelligence:

Work Continuity allows task state to persist across time, applications, and contexts, making long-horizon execution possible without repeated user specification.

Proactive Service enables the system to act on inferred intent rather than waiting for explicit invocation, shifting the paradigm from reactive response to continuous assistance.

Capability Expansion allows clones to dynamically incorporate new tools, services, and agents, turning the Avatar into an open-ended system whose functional boundary evolves with the user’s needs.

As these clones persist and scale, they will increasingly interface with other users’ agents, leading to a multi-avatar ecosystem in which many routine digital interactions are mediated by delegated representatives rather than direct human operation. In such settings, the core unit of interaction shifts from human–system to avatar–avatar coordination: scheduling, negotiation, information exchange, and joint task execution can occur through policies that represent distinct users’ identities and constraints. This shift reframes the technical problem from single-agent task completion to the design of protocols, incentives, and safety boundaries that enable cooperation under conflict, accountability under delegation, and trust under partial autonomy.

## 2.2 Roadmap

We view the path from today's agents to full Avatar Intelligence as a staged decoupling of human involvement from agentic capability (Figure 3). The stages difer not only in autonomy, but also in the required system guarantees (identity continuity, controllability, and accountability):

• L0: Human-for-the-loop. The agent behaves as an augmented application, automating fragmented subprocesses while the user remains embedded in both execution and decisions.

• L1: Human-in-the-loop. The agent executes longer workflows, but users still steer key decisions (goal refinement, routing, validation) and absorb most accountability.

• L2: Human-on-the-loop. An avatar core (persistent identity + memory + decision policy) takes over routine decisions and orchestration; humans intervene at a small number of calibrated checkpoints, with robust runtime guardrails and audit trails.

• L3: Human-out-of-the-loop. The avatar assumes end-to-end autonomy over both operations and decisions for delegated domains; humans primarily perform post-hoc auditing, dispute resolution, and policy updates.

These levels describe autonomy within a delegated domain, and the same avatar can operate at diferent levels across domains. Users retain the authority to revise goals and revoke delegation at every level.

This roadmap highlights a key research transition: moving from improving execution quality (L0/L1) to engineering decision delegation with bounded risk (L2), and ultimately institutionalgrade assurance for high-stakes autonomy (L3).

## 2.3 Agentic Oversight Index (AOI)

To evaluate progress toward Avatar Intelligence, we argue that capability improvements should be assessed jointly with the oversight burden they impose on users. Motivated by the roadmap in Figure 3, we summarize an agent's maturity by two architectural axes: Agentic Intelligence (how well it executes) and Human Involvement (how much a user must steer, validate, and audit).

Definition. We introduce the Agentic Oversight Index (AOI) as an oversight-adjusted net-value score, rather than a ratio: human involvement should penalize a weak agent as well as a strong one. For an avatar A on a task distribution τ , let $U ( A , \tau ) \in ( - 1 , 1 )$ denote normalized net value and let $P ( A , \tau ) \in [ 0 , \infty )$ denote normalized human oversight burden relative to a human-only baseline. We define

$$
U ( A , \tau ) = \operatorname { t a n h } \left( \frac { V ( A , \tau ) - C _ { \mathrm { s y s } } ( A , \tau ) } { s _ { \tau } } \right) ,\tag{1}
$$

![](images/ba5e40f7183b61104c4a1eba922690a0cb7e843923d6f07a1638e630dd81048f.jpg)  
Figure 3: Maturity roadmap toward Avatar Intelligence. The blue solid and orange dashed curves schematically illustrate increasing agentic intelligence and decreasing human involvement across L0–L3. The central transition is from execution-centric assistance to checkpointed decision delegation and ultimately end-to-end autonomy within delegated domains. The same avatar can operate at diferent levels across domains; users retain authority to revise goals and revoke delegation at every level.

$$
A O I ( A , \tau ) = U ( A , \tau ) - \lambda P ( A , \tau ) , \qquad \lambda > 0 .\tag{2}
$$

where V denotes delivered user value $( \mathrm { e . g . }$ , task success, quality, and user-perceived utility); $C _ { \mathrm { s y s } }$ denotes agent-side operational cost $( \mathrm { e . g . }$ , compute and tool usage); and $s _ { \tau } > 0$ is a task-specific reference scale used to normalize net value. The delivered value V reflects the quality and utility of the outcome, while $C _ { \mathrm { s y s } }$ accounts for agent-side resource consumption. Crucially, $C _ { \mathrm { s y s } }$ excludes human oversight, which is represented exclusively by P (e.g., intervention frequency, supervision time, decision complexity, and accountability load). The weight λ specifies the value penalty assigned to one unit of normalized oversight burden.

The score is upper-bounded by 1 but deliberately has no finite lower bound: an agent that creates arbitrarily large human workload should not be made to look merely equally bad. Holding delivered net value fixed, $\partial A O I / \partial P = - \lambda < 0 \colon$ additional human oversight always lowers the score, including when the avatar has negative net value. When $P  0$ , AOI converges to U rather than diverging.

Appendix B provides one possible instantiation of U and $P ;$ other operationalizations are compatible with the same principle.

## 3 Applications

## 3.1 Current Industry Products

A new generation of personal agents is beginning to turn persistent assistance into sustained action. Grok Bot keeps named bots and their context across tasks, allowing them to work through applications and return to the user when approval is needed [17]. Muse pursues ongoing goals through connected applications and a persistent browser, while seeking permission for actions such as sending messages or making purchases [18]. Cue envisions agents with their own email, phone, computer, and wallet, subject to boundaries set by the user [19]. OpenAI’s dots likewise continue work between conversations, draw on connected applications and memory, and bring decisions requiring human judgment back to the user [20].

These systems move beyond one-of responses by carrying work forward over time. Yet the user remains central to setting goals, defining authority, and deciding when consequential actions may proceed. In the roadmap of Avatar Intelligence, they mark a transition from human-in-the-loop execution toward human-on-the-loop oversight within delegated tasks.

## 3.2 Application Scenarios

![](images/97560074830503a47d8dc06874df5b15a9865c8ff89b159614af2edb0eafb973.jpg)  
Figure 4: Four application settings for Avatar Intelligence. (1) personal life, (2) organizational work, (3) financial decisions, and (4) coordination among multiple avatars.

Personal Avatars for Life Administration and Continuity. A personal avatar must carry commitments forward even when they are no longer visible in the immediate request. When a work trip conflicts with a parent–teacher meeting the user promised to attend weeks earlier, the avatar might first look for a later departure rather than simply clearing the calendar. If neither obligation can move, it should bring the conflict back to the user. The value of continuity lies in protecting obligations that a task-by-task assistant would easily overlook.

Enterprise Avatars for Organizational Roles and Operations. Suppose engineering flags a release-blocking defect, while the launch calendar and sales brief still promise a release today. Acting for the project manager, an avatar could decide to postpone the internal launch, update the shared schedule, and notify the teams relying on the old date. It cannot waive the engineering blocker or revise a customer commitment on the sales team’s behalf; those decisions must return to the people who hold that authority. Such coordination depends on a shared view of the project that changes with new evidence, and on the distinct authority of the roles involved [21, 22].

Financial Avatars for Economic Decision Making. The same market move can call for diferent actions depending on whose money is at stake. A planned purchase that remains acceptable for one user may push another user’s portfolio beyond a chosen exposure limit. In the latter case, the avatar should pause the order and seek a decision, even if the original instruction was to buy. Financial delegation is credible only when the user’s risk preferences shape the action itself and changes to those preferences remain under the user’s control.

Multi Avatar Interaction and Coordination. Once avatars act for diferent people, a locally sensible decision may be unacceptable to the other party. In arranging a meeting, one avatar may seek the earliest slot while another protects its user’s preparation time. They can exchange acceptable windows and settle on a time that respects both constraints, or return to their users if no such time exists. The question is how agents can reach agreements without exposing more of either user’s preferences than the negotiation requires.

## 4 Open Challenges

Avatar Intelligence sits at the interface between the user side and the rapidly advancing frontier of agent technology. On the one hand, an avatar must learn a user’s cognitive and decision patterns—capturing not only preferences, but also reasoning habits, methodologies, and value commitments. As a result, it can reliably understand the user and act as a faithful proxy. On the other hand, it must continuously grow and evolve in real-world environments: connecting the user to appropriate agent capabilities, simulating the likely consequences of alternative actions, and selecting options that are best aligned with the user’s objectives, thereby building calibrated trust. Finally, as agents become embedded in everyday life, a central question is how to ensure that such autonomy remains safe and controllable, preventing harm to the user and their environment while preserving the benefits of proactive, long-horizon assistance.

## 4.1 Personalization: From Behavioral Alignment to Identity-Continuous Avatars

Personalization is central to Avatar Intelligence: an avatar must make decisions as the user would, not merely respond in the user's preferred style. This requires learning from diverse, authorized signals (instructions, interaction traces, and environmental context) to infer stable decision structure, including values, trade-ofs, and reasoning habits.

![](images/7c5b74e2bb96c06ddb1bbcb95d2ffa0fa0218dfefe0fa23ce4704a775f8d63db.jpg)  
Figure 5: Avatar Personalization through identity modeling and interaction feedback. Thought Computability and Intent Predictability connect implicit user cognition to a computable identity representation, $\mathcal { T } _ { u } = ( z _ { u } , \mathcal { K } _ { u } )$ , where $z _ { u }$ captures personality structure and $\kappa _ { u }$ stores user-specific knowledge and experience. Instructions, interaction traces, and environmental context inform inverse modeling of this representation, which guides actions across tasks and contexts. Feedback supports continued refinement of the user model as the user and avatar co-evolve.

Most existing approaches [23, 24], however, optimize behavioral outputs (what action to take in a particular moment) rather than the latent cognitive model that generates behavior across contexts and over time. In practice, such systems often rely on memory to store user-specific facts or preferences and retrieve them at inference time. While efective for conversational personalization, memory-heavy heuristics can be brittle for agentic tasks: coverage is incomplete, preferences are context-dependent, and user behavior may be internally inconsistent. A core open problem is therefore to model an identity-continuous user representation that generalizes across tasks while remaining stable yet updateable.

Complementary to learning-based personalization, many deployed systems also use explicit configuration of identity (values and constraints), memory policies, and skill/tool scopes. A typical example is OpenClaw. <sup>3</sup> The challenge is to unify these layers into a coherent user model that remains consistent across clones, environments, and time.

We argue that Avatar Personalization should be formalized as learning an identity-continuous user model that enables transfer and generalization across tasks, while remaining stable yet evolvable.

Figure 5 illustrates how Thought Computability and Intent Predictability support identitycontinuous personalization.

<table><tr><td>Aspect</td><td>Prompt Engineering</td><td>Context Engineering</td><td>Harness Engineering</td><td>Rehearsal Engineering</td></tr><tr><td>Primary signal Intent and</td><td>constraints</td><td>Situational state and Runtime control memory</td><td>policies and rules</td><td>User-situated feedback and outcomes</td></tr><tr><td>When applied</td><td>1 Per query</td><td>Per query / per step Throughout the</td><td>action loop</td><td>Cross-episode/ continual</td></tr><tr><td>Mechanism</td><td>Instruction conditioning</td><td>Retrieval and state injection</td><td>Orchestration, gating, verification</td><td>Scenario staging + evaluation + updates</td></tr><tr><td>Optimizes for</td><td>Specification and style control</td><td>Grounding and disambiguation</td><td>Reliability and controllability</td><td>Specialization and trust calibration</td></tr><tr><td>Typical risks</td><td>Mis-specification</td><td>Missing/poisoned context</td><td>Brittleness/ over-constraint</td><td>Drift, privacy, overfitting</td></tr></table>

Table 1: Comparison of four complementary agent engineering paradigms by signal, timing, mechanism, objective, and risk. Prompt engineering specifies intent and constraints; context engineering supplies task-relevant state; harness engineering governs execution; and Rehearsal Engineering uses staged scenarios and user-situated feedback to improve behavior across episodes. Rehearsal provides an adaptation process that draws on, and updates, the other three forms of engineering.

From Behavioral Alignment to Computable Identity A central requirement for Avatar Intelligence is that the user's cognition becomes computable and predictable to the system. Concretely, we characterize Avatar Personalization with two properties:

• Thought Computability. The user's internal decision process admits a compact representation, enabling the avatar to simulate how the user would reason under diverse contexts.

• Intent Predictability. Given a context, the avatar can reliably anticipate the user's intended action without requiring explicit instruction at every step.

Together, these properties shift personalization from surface-level preference matching to learning an identity-continuous model that (i) generalizes across tasks, (ii) remains consistent across contexts, and (iii) evolves over time without drifting away from the user's values and decision structure. We provide a more detailed formalization—including a problem setting, an identity representation, and consistency/stability constraints—in Appendix A.

## 4.2 Evolving with Specialization

The engineering of autonomous agents has historically focused on improving intent understanding and task completion through external scafolding. From this perspective, prompt engineering injects user intent and constraints into the instruction channel; context engineering injects situational state (documents, observations, and memory) to ground planning; and harness engineering injects executable rules and control policies (e.g., tool gating, verification, retries, and escalation) to stabilize the action loop. These paradigms primarily shape behavior at inference time by conditioning what the agent sees and what it is allowed to do.

A growing challenge, however, is to make agents evolve proactively from interaction with the environment. Recent explorations either (i) construct simulators or games where agents can rehearse trajectories, compare alternatives, and improve through iterative critique [25], or (ii) expose agents to real environments or sandbox [26–28] where they observe state transitions, discover recurring patterns, and learn from sparse but high-fidelity outcomes. Despite their diferences, both lines of work are fundamentally about acquiring efective feedback—signals that evaluate decisions, not just executions—and turning such feedback into better future policies.

For Avatar Intelligence, feedback cannot remain purely generic. Beyond learning in standardized benchmarks or public web contexts, an avatar must specialize within the user's own ecology: their tools, routines, constraints, and risk tolerances. We refer to this user-situated specialization loop as Rehearsal Engineering: a system-level methodology for aligning a standardized, language trained agent to the user's real environment by (1) collecting user- and environment-specific traces and preferences, (2) staging representative scenarios (via simulation, counterfactual rollouts, or controlled sandboxing), (3) evaluating candidate actions with explicit oversight policies, and (4) updating the avatar's memory, skill selection, and decision policy under bounded autonomy.

Situated cognition research in psychology suggests that knowledge is closely tied to the contexts in which it is used [29]. By rehearsing within the user’s own work and life contexts, an agent may learn which contextual cues shape the user’s choices, and how those cues change the user’s trade-ofs across situations. In this view, rehearsal is not merely practice; it is the mechanism by which an avatar internalizes user-specific feedback while remaining safe, auditable, and corrigible.

## 4.3 Safety

Safety is a prerequisite for Avatar Intelligence because avatars are persistent, stateful, and increasingly entrusted with system access and delegated decision authority. We view avatar safety as two complementary layers distinguished by function rather than agent capability. Technical safety continuously enforces delegated authority, reduces the likelihood of failures, and limits their consequences. Agent Insurance complements these controls by allocating residual financial risk and compensating covered losses under explicit contractual terms. Both layers support delegation across capability levels: greater capability does not remove the need for technical control, and insurance provides defined financial recourse rather than a guarantee of correct or user-aligned execution.

Layer I: Technical safety for continuous control. Across capability levels and throughout deployment, safety should be engineered end-to-end around intent fidelity, bounded autonomy, and adversarial robustness:

• Intent alignment and corrigibility. Maintain synchronization between the avatar's policy and the user's intent through explicit preference/constraint representations, intent-drift detection, and structured escalation (ask/confirm) at high stakes.

• Runtime governance (harness-level control). Enforce least privilege, sandboxing, action preconditions, irreversible-action verification, checkpointing/rollback, and deterministic replanning to constrain the action loop even when the model is wrong.

• Security and integrity hardening. Continuously monitor tool usage and environment interactions; detect prompt injection and anomalous behaviors; and protect long-term memory, credentials, and dependencies via sanitization and supply-chain scanning.

Layer II: Agent Insurance for contractually defined loss compensation. Delegated tasks such as fund-moving, code changes, and account operations can expose users to substantial losses when execution fails. Technical safeguards reduce these risks, but residual uncertainty remains in open-world environments. Inspired by recent work on financial risk management for AI agents [30], we argue that Agent Insurance, supported by auditable settlement mechanisms, will become a core infrastructure component for delegation at scale. Service fees can be escrowed and released upon verified delivery; providers can post collateral; and users can optionally purchase underwriting for losses arising from specified failures, such as non-delivery, misexecution, or intent misalignment. Coverage must define verifiable failure conditions, compensation limits, exclusions, and claims procedures. Crucially, compensation requires auditable evidence, including execution traces and evaluation artifacts, together with dispute processes. These mechanisms make responsibility for covered losses explicit and provide contractual recourse, while technical controls remain necessary to prevent harm, including harms that financial compensation cannot reverse.

Summary. Taken together, the open problems in personalization, rehearsal-driven specialization, and safety/assurance constitute the missing infrastructure for advancing from L2 (checkpointed delegation) to L3 (full-pipeline autonomy within delegated domains). Identity-continuous personalization provides a stable decision prior, while Rehearsal Engineering supplies user-situated feedback loops to adapt that prior to real environments. Technical safety continuously constrains delegated actions and limits failure propagation. Agent Insurance provides infrastructure for pricing and allocating residual financial risk and compensating contractually covered losses. Together, continuous technical control and defined financial recourse support trustworthy delegation at scale as avatars assume broader responsibilities with less routine human supervision.

## 5 Conclusion

Avatar Intelligence frames persistent personal agents around a continuing relationship of decision delegation. Its value lies in enabling users to pursue their goals across tasks and over time with less routine supervision and efective control over consequential choices. We outlined a user-centric architecture, a delegation roadmap, and the Agentic Oversight Index to connect this objective with system design and evaluation. The central research agenda is to learn user representations that transfer across contexts, improve decisions through user-situated rehearsal, and support delegation with technical control, accountability, and recourse. Progress should be demonstrated through the decisions users can reliably delegate and the efort required to obtain outcomes they endorse.

## References

[1] OpenAI, “Introducing gpt-5.5,” 2026. [Online]. Available: https://openai.com/index/ introducing-gpt-5-5/

[2] G. DeepMind, “Gemini 3 pro: Best for complex tasks and bringing creative concepts to life,” 2025. [Online]. Available: https://deepmind.google/models/gemini/pro/

[3] DeepSeek, “Deepseek-v4: Towards highly eficient million-token context intelligence,” 2026. [Online]. Available: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/ DeepSeek\_V4.pdf

[4] C. Zhou, H. Chai, W. Chen, Z. Guo, R. Shan, Y. Song, T. Xu, Y. Yang, A. Yu, W. Zhang et al., “Externalization in llm agents: A unified review of memory, skills, protocols and harness engineering,” arXiv preprint arXiv:2604.08224, 2026.

[5] Z. Zhang and A. Zhang, “You only look at screens: Multimodal chain-of-action agents,” arXiv preprint arXiv:2309.11436, 2023.

[6] P. F. Christiano, J. Leike, T. B. Brown, M. Martic, S. Legg, and D. Amodei, “Deep reinforcement learning from human preferences,” in Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, I. Guyon, U. von Luxburg, S. Bengio, H. M. Wallach, R. Fergus, S. V. N. Vishwanathan, and R. Garnett, Eds., 2017, pp. 4299–4307. [Online]. Available: https://proceedings.neurips.cc/paper/2017/hash/ d5e2c0adad503c91f91df240d0cd4e49-Abstract.html

[7] J. Yao, Z. Dou, J. Xu, and J.-R. Wen, “Rlper: A reinforcement learning model for personalized search,” in Proceedings of the Web Conference 2020, 2020, pp. 2298–2308.

[8] S. Lee, S. H. Park, S. Kim, and M. Seo, “Aligning to thousands of preferences via system message generalization,” in Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, A. Globersons, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. M. Tomczak, and C. Zhang, Eds., 2024. [Online]. Available: http://papers.nips.cc/paper\_ files/paper/2024/hash/86c9df30129f7663ad4d429b6f80d461-Abstract-Conference.html

[9] X. Li, Z. C. Lipton, and L. Leqi, “Personalized language modeling from personalized human feedback,” CoRR, vol. abs/2402.05133, 2024. [Online]. Available: https://doi.org/10.48550/arXiv.2402.05133

[10] S. Jin, J. Zhang, Y. Liu, X. Zhang, Y. Zhang, F. Jiang, G. Yin, W. Lin, and R. Yan, “Tagging the thought: Unlocking personalization reasoning via reinforcement learning,” arXiv preprint arXiv:2509.23140, 2025.

[11] R. Chen, X. Zhang, M. Luo, W. Chai, and Z. Liu, “Pad: Personalized alignment of llms at decoding-time,” in The Thirteenth International Conference on Learning Representations, 2025.

[12] M. Kim, K. Lee, S. Joo, H. Lee, T. Thonet, and K. Jung, “Drift: Decoding-time personalized alignments with implicit user preferences,” in Findings of the Association for Computational Linguistics: EMNLP 2025, Suzhou, China, November 4-9, 2025, C. Christodoulopoulos, T. Chakraborty, C. Rose, and V. Peng, Eds. Association for Computational Linguistics, 2025, pp. 6107–6126. [Online]. Available: https://aclanthology.org/2025.findings-emnlp.324

[13] H. Bu, C. Jung, M. Kang, and J. Kim, “Personalized llm decoding via contrasting personal preference,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 2025, pp. 33 946–33 966.

[14] Z. Tan, J. Yan, I.-H. Hsu, R. Han, Z. Wang, L. Le, Y. Song, Y. Chen, H. Palangi, G. Lee et al., “In prospect and retrospect: Reflective memory management for long-term personalized dialogue agents,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025, pp. 8416–8439.

[15] J. Kang, M. Ji, Z. Zhao, and T. Bai, “Memory os of ai agent,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 2025, pp. 25 972–25 981.

[16] H. Sun, Z. Zhang, and S. Zeng, “Preference-aware memory update for long-term llm agents,” arXiv preprint arXiv:2510.09720, 2025.

[17] xAI, “Grok bot: Overview,” 2026. [Online]. Available: https://docs.x.ai/grok-bot/overview

[18] Meta, “Muse: Ai that gets it done,” 2026. [Online]. Available: https://ai.meta.com/muse/

[19] Cue, “Your personal agents, on cue,” 2026. [Online]. Available: https://cue.im/

[20] OpenAI, “Introducing dots,” 2026. [Online]. Available: https://openai.com/index/ introducing-dots/

[21] E. VanDerHorn and S. Mahadevan, “Digital twin: Generalization, characterization and implementation,” Decision support systems, vol. 145, p. 113524, 2021.

[22] F. Wu, Z. Wu, J. Wang, Y. Chen, R. Pei, H. Huang, X. Liao, X. Lou, H. Deng, Z. Fu et al., “Colorecosystem: Powering personalized, standardized, and trustworthy agentic service in massive-agent ecosystem,” arXiv preprint arXiv:2510.21566, 2025.

[23] G. Liu, P. Zhao, L. Liu, Z. Chen, Y. Chai, S. Ren, H. Wang, S. He, and W. Meng, “Learnact: Few-shot mobile gui agent with a unified demonstration benchmark,” arXiv preprint arXiv:2504.13805, 2025.

[24] Z. Wu, H. Huang, Y. Yang, Y. Song, X. Lou, W. Liu, W. Zhang, J. Wang, and Z. Zhang, “Quick on the uptake: Eliciting implicit intents from human demonstrations for personalized mobile-use agents,” arXiv preprint arXiv:2508.08645, 2025.

[25] G. Wang, Y. Xie, Y. Jiang, A. Mandlekar, C. Xiao, Y. Zhu, L. Fan, and A. Anandkumar, “Voyager: An open-ended embodied agent with large language models,” Transactions on Machine Learning Research, 2024.

[26] Y. Zhai, S. Tao, C. Chen, A. Zou, Z. Chen, Q. Fu, S. Mai, L. Yu, J. Deng, Z. Cao et al., “Agentevolver: Towards eficient self-evolving agent system,” arXiv preprint arXiv:2511.10395, 2025.

[27] T. Xue, C. Peng, M. Huang, L. Guo, T. Han, H. Wang, J. Wang, X. Zhang, X. Yang, D. Zhao et al., “Evocua: Evolving computer use agents via learning from scalable synthetic experience,” arXiv preprint arXiv:2601.15876, 2026.

[28] Z. Xi, Y. Ding, W. Chen, B. Hong, H. Guo, J. Wang, D. Yang, C. Liao, X. Guo, W. He et al., “AgentGym: Evolving large language model-based agents across diverse environments,” arXiv preprint arXiv:2406.04151, 2024.

[29] J. S. Brown, A. Collins, P. Duguid et al., “Situated cognition and the culture of learning,” 1989, vol. 18, no. 1, pp. 32–42, 1989.

[30] W. Hua, T. Peng, C. Wang, I. Kaufman, B. Lim, and C. Fang, “Quantifying trust: Financial risk management for trustworthy ai agents,” arXiv preprint arXiv:2604.03976, 2026.

## A Formalization of Avatar Personalization

This appendix provides an optional formalization of the personalization objective in Section 4.1. We use it to make precise what we mean by identity-continuous personalization and the tension between specialization and stability.

Problem setting. We model interaction for a user u over time steps $t = 1 , \dots , T$ . At time $t ,$ the system observes a context $x _ { t }$ and the user’s behavior $o _ { t } ,$ while the avatar produces an action $a _ { t } \sim \pi ( \cdot \mid x _ { t } , \mathcal { T } _ { u } ^ { t } )$ . The key object of interest is the identity representation $\mathcal { L } _ { u } ^ { t }$ , a computable proxy for the user's cognition.

Identity representation. We write the avatar's identity representation as $\mathcal { T } _ { u } = ( z _ { u } , \mathcal { K } _ { u } )$ , where $z _ { u }$ captures relatively stable personality structure (e.g., values, decision style, risk preference) and $\kappa _ { u }$ is a structured store of user-specific knowledge and experience. This decomposition separates how a user thinks from what a user knows.

Personality mining as inverse cognition modeling. A key challenge is to infer latent personality from observed behavior, $z _ { u } = f \Big ( \{ ( x _ { t } , o _ { t } ) \} _ { t = 1 } ^ { T } \Big )$ , which can be viewed as an inverse modeling problem over cognition: rather than matching outputs locally, the goal is to recover a latent structure that renders behavior computable and predictable across contexts.

Cross-context consistency as a predictability constraint. A defining requirement is that the same identity should induce consistent decision structure across heterogeneous tasks. We formalize this as:

$$
\begin{array} { r l } { \underset { \pi , \mathcal { T } _ { u } } { \operatorname* { m a x } } } & { \mathbb { E } _ { ( x , o ) \sim \mathcal { D } _ { u } } \left[ \log \pi ( o \mid x , \mathcal { T } _ { u } ) \right] } \\ { \mathrm { s . t . } } & { \mathcal { C } ( \pi , \mathcal { T } _ { u } ) \leq \epsilon , } \end{array}\tag{3}
$$

where $\mathcal { C } ( \pi , \mathcal { T } _ { u } )$ measures identity inconsistency. Intuitively, the constraint prevents behavior from degenerating into context-local optimization.

Co-evolution and stability. Personalization is inherently dynamic: both the avatar and the user adapt over time, $( \mathcal { T } _ { u } ^ { t } , \tilde { \mathcal { T } } _ { u } ^ { t } )  ( \mathcal { T } _ { u } ^ { t + 1 } , \tilde { \mathcal { T } } _ { u } ^ { t + 1 } )$ . To prevent drift, collapse, or unintended identity substitution, we impose a stability condition, $D \left( \mathcal { T } _ { u } ^ { t } , \tilde { \mathcal { T } } _ { u } ^ { t } \right) \leq \delta .$ which requires the learned identity to remain a faithful and stable proxy of the user.

## B One Instantiation of AOI

This appendix records one possible instantiation of the Agentic Oversight Index (AOI) used in Section 2.3. We score a concrete avatar A over a task distribution τ with operational demand Ω. The functional forms are illustrative; the key requirements are that net value is normalized and that human oversight is represented once, as a monotonic penalty.

Human oversight burden. Let $H ( A , \tau )$ aggregate the human work needed to use, supervise, and take responsibility for an avatar:

$$
H ( A , \tau ) = w _ { f } F ( A , \tau ) + w _ { t } T ( A , \tau ) + w _ { d } D ( A , \tau ) + w _ { a } Q ( A , \tau ) ,\tag{4}
$$

where $F$ is intervention frequency, $T$ is supervision time, D is the cognitive complexity of required decisions, and $Q$ is accountability or audit load. The nonnegative weights encode the evaluation setting. We normalize this burden against $H _ { \mathrm { h u m a n } } ( \tau )$ , the burden incurred when a human performs the task without an avatar:

$$
P ( A , \tau ) = \frac { H ( A , \tau ) } { H _ { \mathrm { h u m a n } } ( \tau ) } .\tag{5}
$$

Thus $P = 0$ denotes no required human oversight, $P = 1$ denotes human efort equal to the human-only baseline, and $P > 1$ explicitly identifies an avatar that creates more human workload than a human-only workflow. Human oversight is represented exclusively by $P ,$ rather than also being included in agent-side operational cost.

Delivered value. Let $V ( I , \Omega )$ denote the total user value produced by an agent with execution capacity $I ,$ decomposed into task value $A _ { \mathrm { t a s k } } ( I , \Omega )$ and optional experiential value $E ( I )$

$$
V ( I , \boldsymbol \Omega ) = A _ { \mathrm { t a s k } } ( I , \boldsymbol \Omega ) + E ( I ) .\tag{6}
$$

We model task value using a saturation form:

$$
A _ { \mathrm { t a s k } } ( I , \Omega ) = \frac { I ^ { \alpha } } { I ^ { \alpha } + \beta \cdot \Omega ^ { \gamma } } \cdot \theta ( \Omega ) .\tag{7}
$$

An optional experiential term can capture satisfaction and calibrated trust without conflating them with human oversight:

$$
E ( I ) = \omega \cdot \left( 1 - \exp \left( - \frac { I } { I _ { 0 } } \right) \right) .\tag{8}
$$

Operational cost. We separate agent-side cost from human oversight. Operational cost comprises fixed execution consumption $C _ { T }$ and learning cost $T _ { L }$ :

$$
C _ { \mathrm { s y s } } ( I , \Omega ) = C _ { T } ( I , \Omega ) + T _ { L } ( I ) ,\tag{9}
$$

where one simple parameterization is

$$
C _ { T } ( I , \Omega ) = c _ { 0 } \cdot \Omega ^ { \eta } \cdot I ^ { - \lambda _ { c } } ,\tag{10}
$$

$$
T _ { L } ( I ) = T _ { 0 } \cdot I ^ { - \rho } \cdot \exp \left( \psi \cdot \ln \left( \frac { \bar { I } } { I } \right) \right) .\tag{11}
$$

AOI. Given a reference scale $s _ { \tau } > 0$ in the same value units as $V , C _ { \mathrm { s y s } }$ , and $R ,$ normalized net value and AOI are computed as:

$$
U ( A , \tau ) = \operatorname { t a n h } \left( \frac { V ( A , \tau ) - C _ { \mathrm { s y s } } ( A , \tau ) } { s _ { \tau } } \right) ,\tag{12}
$$

$$
A O I ( A , \tau ) = U ( A , \tau ) - \lambda P ( A , \tau ) , \qquad \lambda > 0 .\tag{13}
$$

The tanh normalization keeps $U \in ( - 1 , 1 )$ while preserving the ordering among diferent net-value outcomes. This additive form avoids the singularity of a ratio at $P = 0$ . Holding delivered net value fixed, it also guarantees that $\partial A O I / \partial P = - \lambda < 0$ , so a larger oversight burden always lowers the score, even when an agent has negative net value. Unlike a clipped [0, 1] participation rate, $P > 1$ preserves the fact that an agent can impose more human workload than a human-only workflow.