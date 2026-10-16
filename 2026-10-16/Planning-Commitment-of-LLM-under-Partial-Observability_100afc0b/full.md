# Planning Commitment of LLM under Partial Observability

Cassie Huang Zhanwei Cao Yeawon Lee Nathalia Gomez Zhuoyi Yang

Ian G. Harris Tiffany Do Jiayuan Mao Li Zhang

Drexel University Northeastern University University of California, Irvine University of Pennsylvania {Cassie.Huang|Harry.Zhang}@drexel.edu

## Abstract

LLM-based planning agents commit to plans of different lengths before interacting with a partially observable environment. Some aggres sively commit to a long plan based on assumptions about unobserved states, while others conservatively plan only over what they have observed. Both behaviors emerge implicitly in pre-trained models, whether the LLM outputs a plan directly or generates a PDDL formulation for a classical solver. Aggressive commitment risks costly failures, while conservative commitment risks excessive LLM calls. This choice pits safety against efficiency, a trade-off that has been neither characterized systematically nor made controllable. We investigate aggressive and conservative LLM planners and formaliz ers on two partially observable domains, where priors are provided as either explicit statistics or common-sense knowledge. Across three LLMs, both strategies reach similar accuracy. Conservative methods achieve fewer failed attempts (safety) at the cost of more API calls (efficiency). Aggressive methods fail more often but replan efficiently only when priors are informative. Interestingly, analysis shows that LLMs treat the provided priors nearly as facts, which risks failure when priors are unreliable. We also explore hybrid strategies with tunable hyperparameters that adjust the balance between safety and efficiency.

## 1 Introduction

Large language model agents have been used for planning in two ways. In the LLM-as-planner paradigm, the model directly generates action plans from natural-language goals and observations, offering flexibility in open-ended environments and the ability to adapt rapidly when execution deviates from expectation (Wei et al., 2025). In the LLMas-formalizer paradigm, the model instead translates natural-language tasks and observations into an explicit symbolic representation that is solved by an external planner, offering much more formal guarantee, trust, and consistency especially desired in high-stakes applications (Tantakoun et al., 2025). Under partial observability, these paradigms are typically extended through iterative interaction: planner-based agents interleave reasoning, acting, and replanning as new observations arrive, while formalizer-based agents incrementally update their symbolic model and invoke the solver.

How long a plan should an LLM commit to before each execution? In the LLM-as-planner paradigm, this choice is often implicit: some approaches encourage aggressive commitment by generating an overall plan upfront and relying on replanning when execution encounters unexpected obstacles (Erdogan et al., 2025), whereas others adopt a more conservative strategy by interleaving reasoning, acting, and observation at each step (Yao et al., 2023). In contrast, the LLM-as-formalizer paradigm is often bound by the closed-world assumption of formal frameworks like PDDL/STRIPS (Fikes and Nilsson, 1971), making it difficult to represent uncertain beliefs directly, so existing iterative formalization approaches primarily operate conservatively by formalizing and planning over observed information (Gong et al., 2026). Each choice has a cost: aggressive commitment risks failed actions when assumptions prove wrong, compromising safety, while conservative commitment requires more rounds of LLM calls, compromising efficiency. Despite the prevalence of these behaviors across LLM planning systems, planning commitment and its safety–efficiency trade-off have been neither systematically studied nor made controllable.

We take the first step toward studying the planning commitment of LLMs by constructing two controlled, partially observable, text-based environments built on well-known domains of navigation and object manipulation. For Coin Collector (Yuan et al., 2018), we construct the floor plans to follow certain patterns and supply curated statistical priors (e.g., 83% of the time, room\_3 is to the west of room\_8) in a domain with arbitrary layouts. For ALFWorld (Shridhar et al., 2021), we design realistic room contents that let the model draw on its own commonsense knowledge. On both domains, we perform a four-way evaluation of aggressive and conservative LLM planners and formalizers. We instantiate each combination with a representative design from prior work: the conservative planner interleaves reasoning and acting as in ReAct (Yao et al., 2023), the aggressive planner generates a full plan and repairs it as in Plan-and-Act (Erdogan et al., 2025), and the conservative formalizer iteratively grows its PDDL from observations as in PDDLego+ (Gong et al., 2026). We introduce an aggressive formalizer that instead writes assumed facts directly into the PDDL problem, letting us test whether formalizers can benefit from committing beyond what they observe. We find that when priors or common sense are available, aggressive methods are more efficient, while conservative methods are usually safer, with far fewer failed attempts. We also find that aggressive formalizers treat provided priors nearly as fact, and that hybrid strategies offer a middle ground between the two. From these findings, we recommend a level of commitment depending on priorities and environment, ranging from low-stakes, well-characterized settings to high-stakes, unfamiliar ones.

![](images/0966e1722b2d029936ea774a5bb8e121191b17dc54be1cb0bbac34f5b7879dc6.jpg)  
Figure 1: In partially observable environments, models are required to either only make plans from observations when interacting with the environment (Conservative) or assume missing information and replan based on environment interactions (Aggressive).

We find that both strategies reach similar success rates but differ sharply in cost: conservative methods are safer, with far fewer failed attempts, while aggressive methods are more efficient when priors or common sense are available. We also find that LLMs treat provided priors nearly as facts, which risks failure when priors are unreliable. Finally, hybrid strategies with tunable hyperparameters let practitioners control the balance between safety and efficiency. From these findings, we recommend aggressive commitment for low-stakes, well-characterized settings and conservative commitment for high-stakes, unfamiliar ones.

Our contributions are as follows:

• We frame planning commitment as a design axis independent of the planner–formalizer distinction, and introduce an aggressive formalizer to complete a four-way comparison.

• We build two partially observable environments, with priors given as explicit statistics, as common sense, or not at all.

• We show that aggressive and conservative commitment reach similar success rates but trade safety for efficiency, and that LLMs treat probabilistic priors nearly as facts.

• We show that hybrid strategies with tunable hyperparameters control this trade-off, and derive practical recommendations for choosing a level of commitment.

## 2 Related Work

Conservative LLM planners. ReAct interleaves reasoning with environment actions and incorporates each observation before choosing the next action (Yao et al., 2023). Robotic and agentic variants such as SayCan, Inner Monologue, Reflexion, and LLaMAR keep this closed loop (Ahn et al., 2022; Huang et al., 2022; Shinn et al., 2023; Nayak et al., 2025). These agents adapt well and gather information as they go, but they incur repeated inference cost and can behave myopically over long horizons (Wang et al., 2026).

Aggressive LLM planners. ProgPrompt, LLM-Planner, AdaPlanner, DEPS, and Plan-and-Act first generate a long-horizon plan and repair it when execution deviates (Singh et al., 2023; Song et al., 2023; Sun et al., 2023; Wang et al., 2023; Erdogan et al., 2025). This gives global coherence, but early modeling errors can corrupt entire plans, and most methods rely on implicit parametric knowledge rather than a calibrated distribution over unobserved facts.

Bounded-lookahead planners. LATS, LLMguided Monte Carlo planning, and FLARE search over short horizons of hypothesized trajectories before committing to the next action (Zhou et al., 2024; Shi et al., 2025; Hao et al., 2023; Zhao et al., 2023; Wang et al., 2026). They sit between our two extremes; we treat this intermediate level of commitment as outside our scope.

Conservative LLM formalizers. LLM+P and NL2Plan translate fully observable tasks into PDDL for an external planner (Liu et al., 2023; Gestrin et al., 2024; Tantakoun et al., 2025; Huang and Zhang, 2025). Under partial observability, PDDLego and PDDLego+ incrementally update the PDDL from observations (Zhang et al., 2024;

Gong et al., 2026), and WorldCoder learns transition models through interaction (Tang et al., 2024). These approaches inherit the closed-world assumption of symbolic models, which makes uncertainty hard to represent.

Aggressive LLM formalizers. Conformant, contingent, and FOND planners, PPDDL, and POMDP solvers such as SARSOP, POMCP, and DESPOT support committing under uncertainty, but they require complete, hand-specified models (Hoffmann and Brafman, 2006, 2005; Albore et al., 2009; Muise et al., 2012; Mattmüller et al., 2010; Younes and Littman, 2004; Kurniawati et al., 2008; Silver and Veness, 2010; Somani et al., 2013). Few systems pair this machinery with LLM-generated models. Tru-POMDP, closest to our aggressive formulation, generates explicit hypotheses about unobserved objects and goals before belief-space search (Tang et al., 2026a); LLM-DP maintains a distribution over sampled worlds instead of a single hypothesis (Dagan et al., 2023); and PO-PDDL learns symbolic POMDP models from demonstrations (Tang et al., 2026b). None of these study how the presence or strength of priors affects planning.

## 3 Task Formulation and Methodologies

Planning tasks. We first define the task of textbased planning. A planning problem is a tuple $\langle S , \mathcal { A } , \mathcal { T } , s _ { 0 } , g \rangle$ , where S is the set of world states, $\mathcal { A }$ is the set of actions the agent can execute, $\tau : S \times \mathcal { A }  S$ is a deterministic transition function, $s _ { 0 } ~ \in ~ S$ is the initial state, and $g$ is a goal condition. The task is to output a sequence of actions $\pi = [ a _ { 0 } , \ldots , a _ { m } ]$ such that the state reached by applying $\pi$ from $s _ { 0 }$ under T satisfies $g .$ The model never sees this tuple directly. Instead, it is given a natural language domain description (entity types and what each action does), a problem description (the objects, $^ { g , }$ and the known facts about s<sub>0</sub>), and the signatures of the actions in ${ \mathcal { A } } .$ In Coin Collector, for example, g is to hold the coin located in a named room, and A consists of move and take. Optionally, the model also receives a set of statistical priors K about the hidden facts, stated in natural language (e.g., 86% of the time, room\_2 is connected directly to the east of room\_1; Section 4) as a substitute for the common sense the domain lacks.

Partial observability. Under full observability, the problem description states every fact needed to plan (e.g., To the south of room\_1 is room\_9).

![](images/c1f01e15686b718bf37c7b025a032487d025b578ae618224a454a249e0969dfc.jpg)  
Figure 2: Overview of all planning and formalization methodologies. While conservative methods (dark green) plan using only observations and subgoals, aggressive methods (orange) come up with the entire plan by assuming missing information. All approaches are evaluated using a simulator.

Under partial observability, some facts about $s _ { 0 }$ are simply not given. For example in Coin Collector, the model is told the current room and where the coin is located, but none of the connections between the rooms. A missing fact is revealed only when the agent acts in the environment; after each action it receives a text observation of its new state, or an error message if the action could not be executed.

Interaction and success. The agent solves a task over a sequence of planning rounds. In round k, the model receives the task and the interaction history $h _ { k }$ (all actions taken and observations received so far, including failed attempts) and produces a plan segment $\pi _ { k }$ . The segment is executed until it ends or an action is rejected, and the resulting observations are appended to $h _ { k }$ . A task is solved if the goal is reached within a certain number of rounds; correctness is judged only by executing in the simulator, never by comparison to a reference plan, so a successful run may include rejected actions followed by replanning.

## 3.1 Planning Methodologies

We vary two independent choices, summarized in Figure 2.

Planner vs. formalizer. An LLM-as-planner writes the plan segment $\pi _ { k }$ directly as a sequence of actions. An LLM-as-formalizer instead writes a PDDL domain and problem file $\Phi _ { k }$ describing what it believes about the environment, and an off-theshelf classical planner computes $\pi _ { k } = { \mathrm { S o L V E } } { \big ( } \Phi _ { k } { \big ) }$ . If the PDDL cannot be parsed or solved, the error is returned to the model for revision.

Conservative vs. aggressive commitment. The two strategies differ only in which facts a plan is allowed to rely on. A conservative agent relies only on facts stated in the problem description or observed in $h _ { k } .$ , and ignores $K$ even when it is given. Because the goal is usually not reachable from known facts alone, it plans toward an intermediate subgoal (e.g., moving into an unexplored room), and extends the plan once the segment succeeds. An aggressive agent fills in every missing fact with an assumed value—drawn from its own background knowledge, or from K when provided—and plans all the way to the goal; it replans only when an action is rejected. In both cases, replanning continues from the current state rather than restarting. Under this definition, any use of unobserved information, including choosing which room to explore based on a belief about where it leads, constitutes commitment. Conservative agents are therefore instructed to act only on observed facts and are not given K, although their choices among observed options may still be implicitly shaped by the LLM’s knowledge.

Each combination corresponds to a family of existing approaches (Section 2). A formal statement of the execution semantics as well as definitions for all planning methodologies is given in Appendix A.

## 4 Data

We construct benchmarks based on two widely used planning domains to evaluate aggressive and conservative methods under partial observability: Coin Collector (Yuan et al., 2018) is a domain to navigate a 2D grid of rooms in order to find a coin. It involves two actions: move and take. The domain has no inherent common sense, so the priors serve as a controlled proxy for it. In order to control the statistical priors of the dataset, we create our own problems and environment simulator. To generate our dataset, we over-generate floor plans of 9 rooms, and select a subset of problems where 10 of the room pair statistics are considered significant. A room pair statistic is the number of times that a pair of rooms appear together in a given direction. It is considered significant if it is at least 80% or at most 20%. We then select a subset of problems such that 10 room pair statistics over the resulting dataset are significant. All significant statistics are given to the model as priors and accurately reflect the dataset; we do not consider settings where the provided priors are inaccurate. We also evaluate a no-prior setting in which these statistics are withheld. The problems are then used to create natural language problem descriptions. All problem information is given in the fully observable setup, which serves as a control group, while information on room connections is omitted for partially observable problems. To make partially observable problem descriptions, we remove information on all connections in the floor plan. In total we have 100 problems, each consisting of a fully observable and partially observable description. Data examples can be found in Appendix Section D.

ALFWorld household tasks. ALFWorld (Shridhar et al., 2021) is a widely used domain for robotic household tasks. Each game contains several household objects that agents can interact with in either a 3D environment or a natural language environment. To simplify the domain and better support our experiments, we created our own controlled ALFWorld dataset and simulator with natural language interaction only. In each problem, the agent must move a specified object from its initial receptacle to a target receptacle. It involves the following actions: navigate between rooms, inspect receptacles, open and close containers, pick up an object, and place it in or on another receptacle. To generate our dataset, we build a pipeline based on the original ALFWorld environment and generated 100 source worlds. Each final problem contains 5 rooms and 3 complete receptacle-object pairs in every room. The source and target receptacles required by the goal are always kept, and every room contains at least one surface and one openable con tainer. We then assign every problem a unique connected five-room tree. In total, every problem has 5 rooms, 15 receptacles, 15 movable objects, and one cross-room placement goal. ALFWorld consists of only a partially observable dataset where room connectivity and room contents are omitted. We do not provide statistical priors for this domain, rather the problem information was generated to imitate realistic, commonsense, room contents so that the model must use the knowledge in its training data to solve the problem. Because these realistic room contents inherently carry commonsense priors, ALFWorld has no no-prior setting. More details and a data example can be found in Appendix Section E and Appendix Figure 6.

## 5 Experimental Setup

We evaluate the four methods of Section 3.1. All methods receive the same domain description and initial observation. On Coin Collector, aggressive methods are run with and without the statistical priors. Conservative methods never receive them, because using priors to guide any decision, even which observed room to explore next, would itself be a form of commitment. A conservative agent with priors would represent an intermediate level of commitment, which we leave to future work. Our with-priors comparison therefore contrasts the two ends of the commitment spectrum rather than isolating prior access. ALFWorld provides no priors.

Under full observability, each method has one planning attempt. Under partial observability, the attempt limit is the worst case a logical agent could need: k + 1 attempts on Coin Collector, where k is the number of hidden connections, and 20 on ALFWorld (Appendix Section B). Formalizers may revise PDDL syntax errors up to 3 times, and plans are computed with metric-FF. Plans are executed in our own simulators (Appendix Section G.1).

We report planning accuracy (the percentage of problems solved) and, averaged per problem, the number of failed attempts, API calls, and output tokens. An attempt is a planning round that ends in success or a rejected action. API calls count every LLM call, including conservative plan extensions and PDDL revisions, and serve as our primary efficiency cost measure (Appendix Section B). We evaluate Gemini-3-Flash and DeepSeek-Flash through their APIs and Qwen3.8-27B with vLLM on one NVIDIA H100 GPU (Appendix Section B). Prompts are in Appendix Section F.

## 6 Results

Conservative vs. Aggressive Methods Figure 3 displays the planning accuracy, average number of failed attempts, average number of API Calls and average number of output tokens for Coin Collector across all four methods and three models. It is evident that in most cases the performance of aggressive and conservative methods are identical in fully observable environments. This is expected as the two methods differ in how they handle uncertainty, and full observability removes the uncertainty entirely, reducing both to the same behavior. The remaining variations in behavior stem from prompt differences across methodologies and model randomness. The performance drop for Conservative Formalizer comes from not updating the agent’s location in its PDDL (Appendix G.2), a state-tracking error rather than an effect of commitment.

![](images/790b3298be20e6ce6fcdcec83b6434c4e301ac008ecac910536d092dd373de48.jpg)  
Figure 3: Results on Coin Collector and ALFWorld for Aggressive vs. Conservative Planner and Formalizer using Planning Accuracy, Average Number of Failed Attempts, Average Number of API Calls and Average Output Tokens on DeepSeek-Flash (D), Gemini-3-Flash (G) and Qwen3.8-27B (Q). For ALFWorld, Gemini-3-Flash is not included due to budget constraints.

When uncertainty is introduced, conservative methods are safer and aggressive methods never require more API calls than conservative methods, when no statistical priors are introduced. However, the savings are modest while failed attempts are far higher. Because conservative methods do not assume any information and use only information from the environment, the average number of failed attempts is minimal, but the model will need many API calls to solve the problem. Meanwhile, because aggressive methods must output the entire plan, the number of API calls is the same as the total number of planning attempts needed to solve the problem. The number of failed attempts is much larger. Moreover, because the model is not given any statistical priors, the aggressive methods must blindly assume information that is usually incorrect. Therefore the number of API calls for aggressive methods is similar to their conservative counterparts. When statistical priors are introduced, aggressive methods are much more efficient than conservative methods. Specifically, the average number of API Calls reduces by around 40% in comparison to conservative methods. Appendix Figures 11 and 12 display illustrative examples of Aggressive and Conservative, Planner and Formalizer on the same example, comparing the number of attempts and API Calls.

In regards to output token count, the more costsaving method depends on the output type. Because the aggressive planner methods needs to output the entire plan, the average number of output tokens is always greater than its conservative counterpart. Meanwhile, the output tokens for conservative formalizer always exceeds the number for aggressive formalizer.

Partial Observability under Commonsense Environments Because we artificially introduce common sense in Coin Collector using statistical priors, we now explore the performance of aggressive and conservative methods on a truly commonsense environment. From Figure 3 we see that performance findings are upheld in a more commonsense environment. Conservative Planner had fewer failed attempts than Aggressive Planner and Aggressive Planner took fewer API Calls on average. There are many more failed attempts for Formalizer methods, most likely due to the difficulty of the domain. Moreover, unlike Coin Collector, the average number of failed attempts for Aggressive Formalizer is lower than Conservative Formalizer and Aggressive Formalizer is much more efficient than its conservative counterpart. For token count, conservative methods took more output tokens on average than aggressive methods. The increased efficiency for aggressive methods suggests that the models used the common sense encoded in its training data to make correct assumptions about the data and solve the problems faster than conservative methods.

![](images/d489b62e01f4a060846b333641c5432ee0b15c5af64605b820fc8ecf07219e53.jpg)  
Figure 4: Percentage of the times connected lines in problem files do not appear from observations in the environment, but are given in the statistical priors, for Aggressive Formalizer on Coin Collector.

Are conservative methods actually conservative? We next seek to answer the question whether conservative methods actually only use the information given from the environment. From Appendix Figure 7 we see that conservative methods plan and formalize using only information from the environment. All models and conservative methods used close to zero connections that were not observed from the environment, while connections that are unobserved can be seen in more than half the plans generated by aggressive planner and at least 30% of connections encoded in PDDL for aggressive formalizer.

How do aggressive formalizers treat statistical priors? Since the provided statistical priors for

Coin Collector were curated to include different percentages of connections, this raises the question whether models treat statements of high percentage different from statements of low percentage. To answer this, we count the number of times a connection appears in a problem file for PDDL that was not seen in environment observations, but found in the statistical priors given to the model. From Figure 4 it is evident that aggressive formalizers over-rely on statistical priors. The percentage of unobserved connections coming from statistical priors increases when the priors are provided to all three models, and statistical priors of higher probability are much more relied upon.

The over-reliance on statistical priors is consistent with the model preferring provided information over using its training data and performing its own reasoning to solve a problem. While the statistical priors given to the model in this scenario were correct, if a user provides incorrect information to models, this can lead to detrimental consequences when applied to real-world problems.

Method Tradeoff From previous sections, it is evident that conservative methods are safer in terms of failed attempts at the cost of efficiency. Meanwhile, aggressive methods have many more failed attempts before solving the problem, but can utilize commonsense knowledge to increase efficiency. This naturally prompts the question whether we can find a middle ground between the safety and efficiency benefits of each method.

To answer this, we explore two types of hybrid reasoning methods for both Planner and Formalizer: free, where the model chooses between reasoning conservatively or aggressively, and fixed where model starts with aggressive reasoning and then turns conservative after a certain number of failed attempts.

![](images/0538faf0e0db1b80280c21c7487db0936d4e918b32ecd6372adc056e39d471b3.jpg)  
Figure 5: Tradeoff on all evaluation metrics across all methods on partially observable Coin Collector data. Only Gemini results are shown, for all models see Appendix Figure 8.

We evaluated the performance of these hybrid methods on partially observable Coin Collector problems. From Appendix Figure 8, hybrid methods provide a compromise between aggressive and conservative methods. Specifically, Hybrid Planner (fixed) reduces the number of failed attempts and has close to the same API Calls as Aggressive Planner. Gemini-3-Flash and Qwen3.8- 27B as Hybrid Formalizer (fixed) also see a good middle ground between the safety of conservative methods and efficiency of Aggressive methods.

The cases where a hybrid method has more failed attempts than both aggressive and conservative methods are DeepSeek-Flash as hybrid formalizer (fixed) and Gemini-3-Flash as hybrid formalizer (free). Both methods perform closely to Aggressive Formalizer, but are worse than Conservative Formalizer. For Gemini-3-Flash, its performance and error types are the same as Aggressive Formalizer, including hallucinated connections that come from statistical priors. Meanwhile for DeepSeek-Flash, the main issue is the model is not tracking the current room of the agent correctly (Appendix G.2). While this issue appeared in Conservative Formalizer, this rate of incorrect start rooms for Hybrid Formalizer (fixed) is much higher.

In general, the right level of commitment depends on two costs: the cost of an API call and the cost of a failed action in the environment. When no priors or commonsense knowledge are available, we recommend conservative commitment. In this setting, aggressive methods save a modest number of API calls while failing far more often. When reliable priors or commonsense knowledge are available, the choice becomes a genuine tradeoff. Aggressive methods need far fewer API calls at similar accuracy, but still fail more often than conservative methods. If failures are cheap to recover from, our results suggest that aggressive commitment is preferable; web navigation over a known site structure, where a wrong click can be undone, is one such setting. If failures are costly or irreversible, as in automating a lab protocol, the additional API calls of conservative commitment are likely worth the reduction in failed actions. When both costs matter, hybrid methods offer a middle ground: Hybrid Planner (fixed) reduces failed attempts while keeping API calls close to those of Aggressive Planner. Validating these recommendations in such real-world settings remains future work.

Additional Experiments Additional experiments including multiple runs and non-LLM baselines for Coin Collector and a small subset of ALF-World Gemini experiments can be found in Appendix Section C.

## 7 Conclusion

We introduce two controlled, partially observable environments to compare aggressive and conservative LLM planners and formalizers. Both strategies reach similar accuracy but trade off safety against efficiency. When priors or common sense are available, aggressive methods need far fewer API calls; without them, they save little while failing more often. Conservative methods generally fail less often, though not for formalizers on ALFWorld. We therefore recommend aggressive commitment when reliable priors are available and failures are cheap to recover from, conservative commitment when priors are lacking or failures are costly, and hybrid methods when both matter.

## 8 Limitations

This work focuses on two text-based domains, Coin Collector and ALFWorld (under the MIT License), which are both artificial. While we try to make the domains as realistic as possible, performance in this work may not reflect performance on other real-world domains. Moreover, our recommendations are derived from two controlled text-based domains; whether they transfer to real-world settings such as web navigation or lab automation is untested. This opens potential safety risks when using these methodologies in the real world, such as inefficient spending in real world tasks and consequences when errors occur. Nevertheless, this opens future work to explore aggressive methods further and on larger, more realistic domains.

Priors are computed from the evaluation set itself, so they are perfectly calibrated; we do not test miscalibrated or held-out priors. The risk of incorrect priors is extrapolated from the observed over-reliance, not measured directly. Because using priors is itself a form of commitment in our framework, we do not evaluate agents that use priors only to guide exploration. Such intermediatecommitment agents would help separate the value of prior information from the value of committing to it.

One of our evaluation metrics, planning accuracy, may induce false positives where the plan may happen to be correct, but the generated PDDL files do not actually correctly describe the environment that satisfies the constraint, leading to concerns in interpretability and faithfulness. However, to the best of our knowledge there is no feasible alternative. Nevertheless we found no false positives in a 20-sample manual inspection. While there exists methods to evaluate generated PDDL problem files (Zuo et al., 2024) or domain files (Coulter et al., 2022) against a ground-truth, fundamentally there exists non-unique ways to formulate the same constraint.

Output token counts exclude hidden reasoning tokens, which are not reported consistently across providers, so they underestimate the true generation cost.

## References

Michael Ahn, Anthony Brohan, Noah Brown, et al. 2022. Do as i can, not as i say: Grounding language in robotic affordances. In CoRL.

Alexandre Albore, Héctor Palacios, and Héctor Geffner. 2009. A translation-based approach to contingent planning. In Proceedings of the 21st International Joint Conference on Artificial Intelligence, IJCAI’09, page 1623–1628, San Francisco, CA, USA. Morgan Kaufmann Publishers Inc.

Alex Coulter, Teo Ilie, Renee Tibando, and Christian Muise. 2022. Theory alignment via a classical encoding of regular bisimulation. In Workshop on Knowledge Engineering for Planning and Scheduling (KEPS).

Gautier Dagan, Frank Keller, and Alex Lascarides. 2023. Dynamic planning with a llm. Preprint, arXiv:2308.06391.

Lutfi Eren Erdogan, Hiroki Furuta, Sehoon Kim, Nicholas Lee, Suhong Moon, Gopala Anumanchipalli, Kurt Keutzer, and Amir Gholami. 2025. Plan-and-act: Improving planning of agents for longhorizon tasks. In Forty-second International Conference on Machine Learning.

Richard E. Fikes and Nils J. Nilsson. 1971. Strips: A new approach to the application of theorem proving to problem solving. Artificial Intelligence, 2(3– 4):189–208.

Elliot Gestrin, Marco Kuhlmann, and Jendrik Seipp. 2024. NL2Plan: Robust LLM-driven planning from minimal text descriptions. In ICAPS 2024 Workshop on Human-Aware and Explainable Planning (HAXP).

Liancheng Gong, Wang Bill Zhu, Jesse Thomason, and Li Zhang. 2026. Iterative formalization and planning in partially observable environments. In Findings of the Associationfor Computational Linguistics: ACL 2026, pages 12755–12783, San Diego, California, United States. Association for Computational Linguistics.

Shibo Hao, Yi Gu, Haodi Ma, Joshua Hong, Zhen Wang, Daisy Wang, and Zhiting Hu. 2023. Reasoning with language model is planning with world model. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 8154–8173, Singapore. Association for Computational Linguistics.

Jörg Hoffmann and Ronen I. Brafman. 2005. Contingent planning via heuristic forward search with implicit belief states. In Proceedings ofthe Fifteenth International Conference on International Conference on Automated Planning and Scheduling, ICAPS’05, page 71–88. AAAI Press.

Jörg Hoffmann and Ronen I. Brafman. 2006. Conformant planning via heuristic forward search: A new approach. Artificial Intelligence, 170(6):507–541.

Cassie Huang and Li Zhang. 2025. On the limit of language models as planning formalizers. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 4880–4904, Vienna, Austria. Association for Computational Linguistics.

Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, Pierre Sermanet, Tomas Jackson, Noah Brown, Linda Luu, Sergey Levine, Karol Hausman, and brian ichter. 2022. Inner monologue: Embodied reasoning through planning with language models. In 6th Annual Conference on Robot Learning.

Hanna Kurniawati, David Hsu, and Wee Lee. 2008. Sarsop: Efficient point-based pomdp planning by approximating optimally reachable belief spaces.

Bo Liu, Yuqian Jiang, Xiaohan Zhang, Qiang Liu, Shiqi Zhang, Joydeep Biswas, and Peter Stone. 2023. Llm+ p: Empowering large language models with optimal planning proficiency. arXiv preprint arXiv:2304.11477.

Robert Mattmüller, Manuela Ortlieb, Malte Helmert, and Pascal Bercher. 2010. Pattern database heuristics for fully observable nondeterministic planning. In Proceedings ofthe 20th International Conference on Automated Planning and Scheduling (ICAPS 2010), pages 105–112.

Christian Muise, Sheila A. McIlraith, and J. Christopher Beck. 2012. Improved non-deterministic planning by exploiting state relevance. In Proceedings of the Twenty-Second International Conference on International Conference on Automated Planning and Scheduling, ICAPS’12, page 172–180. AAAI Press.

Siddharth Nayak, Adelmo Morrison Orozco, Marina Ten Have, Vittal Thirumalai, Jackson Zhang, Darren Chen, Aditya Kapoor, Eric Robinson, Karthik Gopalakrishnan, James Harrison, Brian Ichter, Anuj Mahajan, and Hamsa Balakrishnan. 2025. Llamar: Long-horizon planning for multi-agent robots in partially observable environments. Preprint, arXiv:2407.10031.

Zijing Shi, Meng Fang, and Ling Chen. 2025. Monte carlo planning with large language model for textbased game agents. Preprint, arXiv:2504.16855.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik R Narasimhan, and Shunyu Yao. 2023. Reflexion: language agents with verbal reinforcement learning. In Thirty-seventh Conference on Neural Information Processing Systems.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. 2021. ALFWorld: Aligning text and embodied environments for interactive learning. In International Conference on Learning Representations (ICLR).

David Silver and Joel Veness. 2010. Monte-carlo planning in large pomdps. In Advances in Neural Information Processing Systems, volume 23. Curran Associates, Inc.

Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox,

Jesse Thomason, and Animesh Garg. 2023. Progprompt: program generation for situated robot task planning using large language models. volume 47, page 999–1012, USA. Kluwer Academic Publishers.

Adhiraj Somani, Nan Ye, David Hsu, and Wee Sun Lee. 2013. Despot: Online pomdp planning with regularization. In Advances in Neural Information Processing Systems, volume 26. Curran Associates, Inc.

Chan Hee Song, Jiaman Wu, Clayton Washington, Brian M. Sadler, Wei-Lun Chao, and Yu Su. 2023. Llm-planner: Few-shot grounded planning for embodied agents with large language models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 2998–3009.

Haotian Sun, Yuchen Zhuang, Lingkai Kong, Bo Dai, and Chao Zhang. 2023. Adaplanner: Adaptive planning from feedback with language models. In Advances in Neural Information Processing Systems, volume 36, pages 58202–58245. Curran Associates, Inc.

Hao Tang, Darren Key, and Kevin Ellis. 2024. Worldcoder, a model-based llm agent: Building world models by writing code and interacting with the environment. In Advances in Neural Information Processing Systems.

Wenjing Tang, Xinyu He, Yongxi Huang, Yunxiao Xiao, Cewu Lu, and Panpan Cai. 2026a. Tru-pomdp: Task planning under uncertainty via tree of hypotheses and open-ended pomdps. Preprint, arXiv:2506.02860.

Wenjing Tang, Xuanjin Jin, Yuan Liu, Renming Huang, Cewu Lu, and Panpan Cai. 2026b. Po-pddl: Learning symbolic pomdps from visual demonstrations for robot planning under uncertainty. Preprint, arXiv:2606.15654.

Marcus Tantakoun, Christian Muise, and Xiaodan Zhu. 2025. LLMs as planning formalizers: A survey for leveraging large language models to construct automated planning models. In Findings of the Association for Computational Linguistics: ACL 2025, pages 25167–25188, Vienna, Austria. Association for Computational Linguistics.

Zehong Wang, Fang Wu, Hongru Wang, Xiangru Tang, Bolian Li, Zhenfei Yin, Yijun Ma, Yiyang Li, Weixiang Sun, Xiusi Chen, and Yanfang Ye. 2026. Why reasoning fails to plan: A planning-centric analysis of long-horizon decision making in llm agents. Preprint, arXiv:2601.22311.

Zihao Wang, Shaofei Cai, Guanzhou Chen, Anji Liu, Xiaojian (Shawn) Ma, and Yitao Liang. 2023. Describe, explain, plan and select: Interactive planning with llms enables open-world multi-task agents. In Advances in Neural Information Processing Systems, volume 36, pages 34153–34189. Curran Associates, Inc.

Hui Wei, Zihao Zhang, Shenghua He, Tian Xia, Shijia Pan, and Fei Liu. 2025. PlanGenLLMs: A modern survey of LLM planning capabilities. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 19497–19521, Vienna, Austria. Association for Computational Linguistics.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR).

Håkan L. S. Younes and Michael L. Littman. 2004. Ppddl 1 . 0 : An extension to pddl for expressing planning domains with probabilistic effects.

Xingdi Yuan, Marc-Alexandre Côté, Alessandro Sordoni, Romain Laroche, Remi Tachet des Combes, Matthew Hausknecht, and Adam Trischler. 2018. Counting to explore and generalize in text-based games. arXiv preprint arXiv:1806.11525.

Li Zhang, Peter Jansen, Tianyi Zhang, Peter Clark, Chris Callison-Burch, and Niket Tandon. 2024. PDDLEGO: Iterative planning in textual environments. In Proceedings ofthe 13th Joint Conference on Lexical and Computational Semantics (\*SEM 2024), pages 212–221, Mexico City, Mexico. Association for Computational Linguistics.

Zirui Zhao, Wee Sun Lee, and David Hsu. 2023. Large language models as commonsense knowledge for large-scale task planning. Preprint, arXiv:2305.14078.

Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, and Yu-Xiong Wang. 2024. Language agent tree search unifies reasoning acting and planning in language models. Preprint, arXiv:2310.04406.

Max Zuo, Francisco Piedrahita Velez, Xiaochen Li, Michael L Littman, and Stephen H Bach. 2024. Planetarium: A rigorous benchmark for translating text to structured planning languages. arXiv preprint arXiv:2407.03321.

## A Formal Task and Methodology Formulation

We formulate interactive text-based planning as decision-making over a discrete-time simulator. Following classical planning paradigms and recent text-based conventions, an agent must reach a target goal state by interacting with an environment. Unlike standard static planning benchmarks, the validity and correctness of a generated plan are determined strictly via online simulator execution rather than offline comparison to ground-truth specification files.

## A.1 Interactive Planning Environment

We define an interactive planning task instance as a quintuple:

$$
I = ( D _ { d } , D _ { p } ^ { \rho } , \mathcal { A } ^ { \prime } , o _ { 0 } , K )\tag{1}
$$

where:

1. $D _ { d }$ is a natural language domain description detailing relevant entity types, relations, and action semantics.

2. $D _ { p } ^ { \rho }$ is the problem description under information setting $\rho \in \{ \mathrm { f u l l }$ , partial}, specifying objects, goal specifications, and initial contextual information.

3. A<sup>′</sup> is the set of action signatures defining valid action names and parameter formats accepted by the simulator.

4. o<sub>0</sub> $= \mathcal { O } ( s _ { 0 } )$ is the initial console observation generated from the simulator’s true underlying state $s _ { 0 } \in { \mathcal { S } }$ via the observation function $\mathcal { O }$ : ${ \mathcal { S } }  \Omega .$

5. K is an optional set of auxiliary common-sense prior statements. When no prior domain knowledge is supplied, $K = \emptyset$

Observability and Information Settings. The simulator maintains a complete underlying state $s _ { t } \in S$ at every step t, regardless of what the agent observes. The information setting $\rho$ determines how $D _ { p } ^ { \rho }$ is instantiated:

• Full Observability $( \rho = { \mathrm { f u l l } } ) \colon$ The model receives $D _ { p } ^ { \mathrm { f u l l } }$ , which contains all facts necessary to construct a complete solution a priori.

• Partial Observability $( \rho = \mathrm { p a r t i a l } ) \colon$ The model receives $D _ { p } ^ { \mathrm { p a r t i a l } }$ , which omits some facts about s<sub>0</sub> (all room connections in Coin Collector; room connections and entity locations in ALFWorld). Omitted facts are revealed only through observations.

Execution Dynamics and Plan Correctness. At execution step $t ,$ given a state $s _ { t } ~ \in ~ S$ and a grounded action $a _ { t } \in \mathcal A$ the simulator evaluates executability via a binary function:

$$
{ \mathrm { E x e c } } : S \times { \mathcal { A } }  \{ { \mathrm { T r u e , F a l s e } } \}\tag{2}
$$

Where $s$ is the state space and $\mathcal { A }$ is the true action space, each element additionally carrying semantics that $\mathcal { A } ^ { \prime }$ omits. If Exec $\left( s _ { t } , a _ { t } \right) \ =$ True, the environment updates the state as $s _ { t + 1 } = T ( s _ { t } , a _ { t } )$ and yields the observation $o _ { t + 1 } = \mathcal { O } ( s _ { t + 1 } ) . \ \mathrm { O t h - }$ erwise, the state remains unchanged $( s _ { t + 1 } = s _ { t } )$

![](images/daea5dc1bdf71bcf4593607c21878ef56f229a71b1862c48eb09f311ec40013e.jpg)

Figure 6: Examples of inputs and plans for Coin Collector and ALFWorld. Note that the images provided are for illustration purposes only and our input is purely textual.  
![](images/80e63743e9dcea2a74dbee61f4953f55f87a3a2cbb307d8ba69453e6cf4aa31b.jpg)

(b) Problem-file connected facts without support  
![](images/3a82afa134f6fbc32848a92ed2afddd3fcd36375a265c5301d1509bda90c599a.jpg)  
Figure 7: (a) The percentage of plans from planner methods that used a connection in Coin Collector that was not observed in the environment. (b) The percentage of connected facts in formalizer methods that contained a connection that did not appear in the environment.

and the simulator returns an execution-error observation $o _ { t + 1 } \in \Omega$ describing why the action was rejected.

The objective is to produce an action sequence $\pi = [ a _ { 0 } , \ldots , a _ { m } ]$ yielding an execution trajectory $\tau ~ = ~ [ o _ { 0 } , a _ { 0 } , o _ { 1 } , a _ { 1 } , \dots , a _ { m } , o _ { m + 1 } ]$ . A plan segment, π, is executable if every action in the segment is accepted by the simulator. An interaction run is successful if its terminal state satisfies $s _ { m + 1 } \in S _ { g } ,$ where $\mathcal { S } _ { g } \subset \mathcal { S }$ denotes the set of goal states. A successful run may include rejected actions followed by replanning.

![](images/5ac90364e3f8b76f5320fba0a42a2b53bdbd267ffc146adcf5c2f49befa90481.jpg)  
Figure 8: Tradeoff on all evaluation metrics across Conservative, Aggressive and Hybrid (both fixed and free) Planner and Formalizer on partially observable Coin Collector data. Only results when statistics are provided are shown.

## A.2 Planning and Reasoning Methodologies

We index execution steps by t and planning round by k; round k begins at step $t _ { k } .$ . To solve instances given by I, we categorize LLM-based planning methodologies along two core axes: the generation strategy (LLM-as-Planner vs. LLM-as-Formalizer) and the reasoning paradigm under uncertainty (Conservative vs. Aggressive Reasoning).

Axis 1: Generation Strategy (Planner vs. Formalizer).

• LLM-as-Planner: The LLM directly operates as an end-to-end policy. Given the task specification I and interaction trajectory $h _ { k } =$ $[ o _ { 0 } , a _ { 0 } , \ldots , o _ { t _ { k } } ]$ , the model directly generates a sequence of executable grounded actions:

$$
\pi _ { k } = [ a _ { t _ { k } } , \dots , a _ { t _ { k + j } } ] = \mathrm { L L M } ( I , h _ { k } )\tag{3}
$$

where $j$ is determined by the reasoning paradigm (Axis 2): bounded by verified information under conservative reasoning, or spanning the full remaining plan to the goal under aggressive reasoning.

• LLM-as-Formalizer: The LLM acts as a translator that maps the natural language inputs into a formal symbolic representation $\Phi _ { k }$ (e.g., PDDL domain and problem specifications):

$$
\Phi _ { k } = \mathrm { L L M } ( I , h _ { k } )\tag{4}
$$

An external deterministic symbolic solver Σ is then invoked to compute a plan that is sound with respect to $\Phi _ { k }$

$$
\pi _ { k } = \Sigma ( \Phi _ { k } )\tag{5}
$$

Axis 2: Reasoning Paradigm under Uncertainty (Conservative vs. Aggressive). Under partial observability $( \rho = \mathrm { p a r t i a l } )$ , the initial problem description $D _ { p } ^ { \mathrm { p a r t i a l } }$ lacks complete state facts. Models navigate this information gap using two distinct paradigms.

Both paradigms are given the same task I, which may or may not include common-sense priors K $( K = \emptyset$ when none are supplied). Conservative reasoning is characterized purely by $h _ { k } \mathbf { \cdot }$ : it commits only to what is verified in the observation history, never drawing on K even when available. Aggressive reasoning instead forms an implicit hypothesis $\hat { s } _ { k } \approx s _ { k }$ over unobserved variables. It draws on the LLM’s own parametric priors when $K = \emptyset$ and additionally on supplied common-sense priors when $K \neq \emptyset$ , allowing it to commit to the full problem regardless of what $h _ { k }$ supports. This hypothesis is never materialized as a separate output. In both paradigms, if execution fails at some step t, the agent replans: $h _ { k }$ , now including the record of the failed attempt, is used to generate a new plan for the remaining problem rather than restarting from $a _ { 0 }$ . The full executed trajectory $\pi$ is the concatenation of successive round-level outputs $\pi _ { k }$ across any replanning or extension steps.

• Conservative Reasoning: The agent operates strictly over verifiably grounded facts collected in the observation history $h _ { k }$ . It refrains from making unverified assumptions about the unobserved state space. Under the LLM-as-Planner strategy, this yields a bounded action sequence extending only as far as grounded facts support, which is a subgoal-directed segment rather than a completed solution. If this segment executes successfully but the goal is not yet reached, the agent extends the plan: $h _ { k }$ is updated with the new observations and a further segment is generated. Under the LLM-as-Formalizer strategy, this yields a symbolic representation containing only observed facts, similarly grown incrementally as $h _ { k }$ expands:

$$
\pi _ { k } ^ { \mathrm { c o n s } } = [ a _ { t _ { k } } , \cdot \cdot \cdot , a _ { t _ { k } + j } ] = \mathrm { L L M } ( I \backslash K , h _ { k } )\tag{6a}
$$

$$
\Phi _ { k } = \operatorname { L L M } ( I \setminus K , h _ { k } )\tag{6b}
$$

where $j$ is bounded by the extent of information available in $h _ { k }$

• Aggressive Reasoning: Unlike the bounded, evidence-limited commitment above, the agent commits to the full problem regardless of what is currently grounded, substituting hypothesized assumptions for missing facts where $h _ { k }$ is insufficient. Under the LLM-as-Planner strategy, this yields a complete trajectory to the goal. Under the LLM-as-Formalizer strategy, the external solver Σ always returns a complete plan from $\Phi _ { k } ,$ so the distinction from the conservative lies in the informational content of $\Phi _ { k }$ , which now contains hypothesized facts alongside observed ones:

$$
\pi _ { k } ^ { \mathrm { a g g } } = [ a _ { t _ { k } } , \dots , a _ { m } ] = \mathrm { L L M } ( I , h _ { k } )
$$

$$
\Phi _ { k } = \mathrm { L L M } ( I , h _ { k } )\tag{7a}
$$

(7b)

## B Additional Computation Information and Experimental Setup

Attempt limits. The model is given one planning attempt when the problem is fully observable. Under partial observability, the attempt limit represents the worst case a logical agent could need given the missing information, rather than a budget on API calls. For Coin Collector, the model is given k + 1 attempts, where k is the number of connections omitted from the problem description (8–11 per problem). For ALFWorld, the model is given 3 receptacles × 5 rooms + 4 room connections + 1 final attempt = 20 attempts. Formalizers are additionally given up to 3 revision attempts per output to fix PDDL syntax errors.

Cost counters. An attempt is a planning round that ends either in success or in a rejected action; a failed attempt is any attempt that does not reach the goal. For aggressive methods, each attempt is a single LLM output (a full plan or PDDL program). For conservative methods, one attempt can contain several LLM outputs, because the agent extends its plan after every successful segment, so a new attempt starts only after a rejected action. API calls count every LLM call, including conservative plan extensions and PDDL revision calls. This is why API calls can exceed attempts even under full observability. Output tokens are counted as reported by each provider; reasoning tokens are not exposed consistently across our three models and are not included. Since every call re-sends the task and the interaction history, input cost grows with the number of API calls, which we treat as our primary cost measure.

DeepSeek-Flash was called with default parameters using the default OpenAI API. The following is model information for the model we used:

• Model Name: deepseek-flash Gemini-3-Flash was called using the Agent Platform API. The following is information about the model and parameters we used:

• Model Name: gemini-3-flash-preview

• temperature: 0.4

![](images/3ad5d9bac7c644c7ebc9f81090b770df98bb7df47f37f428ea20bf32f4b59e27.jpg)  
Figure 9: Multiple runs of DeepSeek-Flash on Coin Collector data. Note because Conservative methods were not given statistical priors, results are repeated.

• top\_p: 0.95

• max\_output\_tokens: 20,000

Qwen3.8-27B was called using vLLM. The following is the information about the model and parameters we used:

• Model Name: Qwen/Qwen3.8-27B

• max\_model\_len: 60,000

• tensor\_parallel\_size: 1

• reasoning\_parser: qwen3

• temperature: 0.4

• top\_p = 0.95

## C Additional Experiments

## C.1 Multiple Runs

Because sampling temperature was set to 0.4 for Gemini-3-Flash and Qwen3.8-27B and left at the default for DeepSeek-Flash, model outputs are inherently stochastic. To check that our findings are robust to this randomness, we repeated the Coin Collector experiments three times with DeepSeek-Flash. Figure 9 shows that, although individual metrics vary across runs, this variation is small relative to the differences between methods, and the qualitative comparisons between methods are consistent across all three runs. Due to time and budget constraints, Gemini-3-Flash and Qwen3.8- 27B were run once.

<table><tr><td>Method</td><td></td><td>Acc.↑ Fail.↓</td><td>API↓</td><td>Tok.↓</td></tr><tr><td>Aggr. planner</td><td>100</td><td>4.0</td><td>5.0</td><td>178</td></tr><tr><td>Cons. planner</td><td>100</td><td>1.5</td><td>4.7</td><td>197</td></tr><tr><td>Aggr. formalizer</td><td>90</td><td>6.1</td><td>7.5</td><td>7,603</td></tr><tr><td>Cons. formalizer</td><td>100</td><td>5.7</td><td>18.6</td><td>16,149</td></tr></table>

Table 1: Gemini-3-Flash on ALFWorld, 10 problems per method. Aggr. and Cons. denote aggressive and conservative. Acc. is accuracy (%), Fail. the average number of failed attempts per problem, API the average number of API calls, and Tok. the average number of output tokens

## C.2 ALFWorld Gemini Experiments

Due to budget constraints, we only performed experiments on a 10-problem subset of ALFWorld using Gemini-3-Flash. From Table 1 we see that performance is similar to that of DeepSeek-Flash and Qwen3.8-27B. The only exception is Conservative Planner is more efficient in terms of API Calls than Aggressive Planner, which may be due to the small number of problems experimented with.

## C.3 Non-LLM Coin Collector Baselines

We implemented four non-LLM baselines, where three use no priors; the probability-based baseline samples from the dataset-level connection frequencies.

Oracle BFS baseline. The oracle breadth-first search (BFS) baseline assumes full observability of the environment. It is given the complete room topology, the direction of every connection, the initial room, and the exact location of the coin. BFS is performed over action-level states. The baseline generates a complete shortest plan before execution and follows it without replanning. Because it has access to information hidden from the other planners, this baseline represents an oracle upper bound rather than a directly comparable partially observable planning method.

Frontier BFS baseline. The frontier BFS baseline assumes no prior access to the hidden map, coin location, or statistical connection information. It begins with the agent’s initial observation and incrementally constructs a graph containing only observed rooms and connections. At each planning point, BFS finds the shortest route through the discovered graph to the nearest exploration frontier, defined as an observed but unvisited room. The complete route to that frontier is generated and executed before the graph is updated using newly obtained observations. This process continues until the coin is found, no reachable frontier remains, or the action budget is exhausted. Thus, the baseline replans between frontier-reaching routes, rather than after every action.

Aggressive BFS baseline. The aggressive BFS baseline receives the partial problem description and initial observation but does not access the hidden environment map or statistical priors. It constructs a hypothetical complete map by preserving observed connections and filling missing connections according to fixed room and direction orderings. BFS then generates a complete plan to the known goal on this hypothetical map before execution begins. The plan is followed unchanged after successful actions and is terminated at the first invalid action. The resulting observations are used to remove contradicted assumptions and construct a revised hypothetical map, from which another complete plan is generated. Although this method is deterministic and reproducible, its unobserved connections are unsupported assumptions and therefore function as deterministic guesses.

In addition, we also implemented a probability based baseline. It uses dataset-level connection frequencies to define a stochastic movement policy. At each step, it samples a direction according to the current room’s probability distribution, removes directions contradicted by environment feedback, and renormalizes the remaining probabilities. It operates without access to the hidden map or coin location and selects actions incrementally rather than generating a complete plan. Multiple random seeds are used to account for sampling variance.

<table><tr><td>Baseline</td><td>Att.</td><td>Steps</td><td>Acc.</td><td>1-Att.</td></tr><tr><td>Oracle BFS</td><td>1.00</td><td>4.09</td><td>100</td><td>100</td></tr><tr><td>Frontier BFS</td><td>6.09</td><td>7.15</td><td>100</td><td>0</td></tr><tr><td>Aggressive BFS</td><td>5.99</td><td>13.60</td><td>100</td><td>3</td></tr><tr><td>Probability</td><td>5.34</td><td>62.75</td><td>100</td><td>53</td></tr></table>

Table 2: Non-LLM baselines on Coin Collector. Att. and Steps are the average number of attempts and actions per problem, Acc. is planning accuracy (%), and 1-Att. is the percentage of problems solved on the first attempt.

![](images/7465484a7e965e57a63889625a396692ad2357f5946dc39f07ef53d3dd40c0ff.jpg)  
Figure 10: Failed attempts and API calls by shortest start-to-coin path length on Coin Collector without connections, averaged over DeepSeek-Flash, Gemini-3-Flash, and Qwen3.8-27B (run 1). n is the number of problems.

From Table 2 we see that besides the Oracle BFS, the non-LLM baselines need more attempts to complete than all LLM methods. Finally, aggressive methods when compared to non-LLM baselines are more efficient or comparable to all non-LLM baselines.

## D Coin Collector Data

We create our own Coin Collector simulation environment and create problems where the statistical priors regarding the dataset are significant. Here we include an example domain and problem descriptions for this environment, as well as the priors and action heads of the dataset. Listing 2 is the action heads for the Coin Collector domain and Listing

<table><tr><td>Method</td><td>Fail. PDDL Loc. Known Assumed Rev.</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Full observability</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DeepSeek-Flash</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Aggressive</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td></td><td>0.00</td></tr><tr><td>Aggressive + priors</td><td>0.06</td><td>0.01</td><td>0.00</td><td>0.05</td><td></td><td>0.01</td></tr><tr><td>Conservative</td><td>0.31</td><td>0.00</td><td>0.30</td><td>0.01</td><td></td><td>0.04</td></tr><tr><td>Gemini-3-Flash</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Aggressive</td><td>0.02</td><td>0.02</td><td>0.00</td><td>0.00</td><td></td><td>0.00</td></tr><tr><td>Aggressive + priors</td><td>0.09</td><td>0.03</td><td>0.00</td><td>0.06</td><td></td><td>0.00</td></tr><tr><td>Conservative</td><td>0.05</td><td>0.05</td><td>0.00</td><td>0.00</td><td></td><td>0.06</td></tr><tr><td>Qwen3.8-27B</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Aggressive</td><td>0.17</td><td>0.17</td><td>0.00</td><td>0.00</td><td></td><td>0.83</td></tr><tr><td>Aggressive + priors</td><td>0.17</td><td>0.16</td><td>0.00</td><td>0.01</td><td>一</td><td>0.85</td></tr><tr><td>Conservative</td><td>0.20</td><td>0.19</td><td>0.01</td><td>0.00</td><td>一</td><td>0.88</td></tr><tr><td>Missing all connections</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DeepSeek-Flash</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Aggressive</td><td>3.11</td><td>0.04</td><td>0.29</td><td>0.67</td><td>2.11</td><td>0.10</td></tr><tr><td>Aggressive + priors</td><td>2.03</td><td>0.01</td><td>0.19</td><td>0.60</td><td>1.23</td><td>0.10</td></tr><tr><td>Conservative</td><td>1.59</td><td>0.06</td><td>1.52</td><td>0.01</td><td>0.00</td><td>0.12</td></tr><tr><td>Gemini-3-Flash</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Aggressive</td><td>2.89</td><td>0.03</td><td>0.01</td><td>0.47</td><td>2.38</td><td>0.01</td></tr><tr><td>Aggressive + priors 2.11</td><td></td><td>0.01</td><td>0.02</td><td>0.79</td><td>1.29</td><td>0.01</td></tr><tr><td>Conservative</td><td>0.29</td><td>0.12</td><td>0.15</td><td>0.02</td><td>0.00</td><td>0.51</td></tr><tr><td>Qwen3.8-27B</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Aggressive</td><td>2.88</td><td>0.25</td><td>0.47</td><td>0.49</td><td>1.67</td><td>1.06</td></tr><tr><td>Aggressive + priors 2.10</td><td></td><td>0.21</td><td>0.48</td><td>0.47</td><td>0.94</td><td>0.97</td></tr><tr><td>Conservative</td><td>0.32</td><td>0.25</td><td>0.07</td><td>0.00</td><td>0.00</td><td>1.22</td></tr></table>

Table 3: Causes of failed attempts for formalizers on Coin Collector (run 1), averaged per problem. Fail. is the total and equals the sum of PDDL (no solvable PDDL after revisions, or an unparsable action), Loc. (wrong agent location in the problem file), Known (contradicts a given or observed connection), and Assumed (relies on an unobserved connection). Rev. is the average number of PDDL revision calls per problem.

1 is the statistical priors of the dataset. Listings 4, 6 and 7 are domain description, fully observable problem description and partially observable problem description respectively.

## E ALFWorld Household Data

To create the household dataset, we downsample 100 larger source worlds. For every problem, we keep the rooms containing the source and target receptacles and the agent’s initial room. The remaining room slots are selected from the same source world. Within every selected room, we keep three complete receptacle-object pairs, including the source and target receptacles required by the goal. The remaining pairs are selected so that every room has at least one surface and one openable container. We then assign every problem a distinct labeled five-room tree and use the four edges as the room connectivity graph. All goals require moving an object between rooms.

To make partially observable problem descriptions, we remove all four room connections and all object and receptacle locations. An observation only reveals the receptacles in the current room and the names of adjacent rooms. The agent may need to approach and open a container to observe its contents. The maximum of 20 planning attempts consists of 1 initial attempt, 15 possible receptacle inspections, and 4 hidden room connections. We validate that every room graph is a connected tree, all goal entities and initial rooms are kept, and every room contains exactly three receptacles and three objects. In total, all 100 worlds, natural language goals, and labeled tree topologies are unique. The problems span 51 room sets and 60 source-target room pairs.

Listings 3 and 5 are the ALFWorld action heads and domain description provided to the model. Listing 8 is an example of a partially observable problem description. It contains the available entities and the task, while the room graph and entity locations are omitted.

## F Prompts

Listings 9, 10, 11, 12 13, 14, 15 and 16 are the prompts for the Aggressive Planner, Aggressive Formalizer, Conservative Planner, Conservative Formalizer, Hybrid Planner (free), Hybrid Planner (fixed), Hybrid Formalizer (free) and Hybrid Formalizer (fixed) methods respectively. Statistical priors are presented to the model under the header "Common Sense Knowledge".

## G Additional Analysis

## G.1 Direction-Based Move Execution

In Coin Collector without connections (run 1, all three models), a move by an aggressive method reached a room other than the one named in its plan in 809 of 3,953 planning rounds (20%). In 805 of these (99.5%), the round still ended in a rejected action later in the same plan, so direction-based execution delays the detection of a wrong assumption by a step rather than hiding it. Only 4 of 1,141 successful aggressive rounds (0.4%) contained such a mismatch. For conservative methods, mismatches occurred in 100 of 2,693 plan segments, and only 2 of 575 successful segments contained one.

## G.2 Formalizer Error Analysis

To separate the effect of commitment from errors in producing PDDL, we classify every failed attempt of the formalizer methods on Coin Collector (run 1) by its cause (Table 3):

<table><tr><td rowspan="2">Method</td><td colspan="3">Length 2 (n=42)</td><td colspan="3">Length 3 (n=30)</td><td colspan="3">Length 4 (n=13)</td><td colspan="3">Length 5–6 (n=15)</td></tr><tr><td>Acc.</td><td>Fail.</td><td>API</td><td>Acc.</td><td>Fail.</td><td>API</td><td>Acc.</td><td>Fail.</td><td>API</td><td>Acc.</td><td>Fail.</td><td>API</td></tr><tr><td>Conservative planner</td><td>100</td><td>0.02</td><td>3.14</td><td>100</td><td>0.06</td><td>4.33</td><td>100</td><td>0.03</td><td>5.82</td><td>100</td><td>0.07</td><td>6.62</td></tr><tr><td>Aggressive planner</td><td>100</td><td>1.93</td><td>2.93</td><td>98</td><td>2.78</td><td>3.76</td><td>100</td><td>4.00</td><td>5.00</td><td>100</td><td>4.24</td><td>5.24</td></tr><tr><td>+ priors</td><td>100</td><td>0.83</td><td>1.83</td><td>100</td><td>1.20</td><td>2.20</td><td>100</td><td>3.05</td><td>4.05</td><td>100</td><td>2.87</td><td>3.87</td></tr><tr><td>Conservative formalizer</td><td>91</td><td>0.61</td><td>3.98</td><td>94</td><td>0.67</td><td>5.26</td><td>90</td><td>1.08</td><td>6.51</td><td>89</td><td>0.91</td><td>6.98</td></tr><tr><td>Aggressive formalizer</td><td>92</td><td>1.94</td><td>3.14</td><td>94</td><td>3.01</td><td>4.27</td><td>82</td><td>4.46</td><td>5.69</td><td>91</td><td>4.40</td><td>5.64</td></tr><tr><td>+ priors</td><td>90</td><td>1.13</td><td>2.37</td><td>89</td><td>1.60</td><td>2.86</td><td>87</td><td>4.28</td><td>5.33</td><td>93</td><td>3.78</td><td>4.80</td></tr></table>

Table 4: Coin Collector results without connections by shortest path length, averaged over the three models (run 1). Acc. is planning accuracy (%), Fail. the average number of failed attempts, and API the average number of API calls.

• PDDL: no solvable PDDL was produced after 3 revisions, or the plan contained an action the simulator cannot parse.

• Location: the agent’s location in the problem file differs from its true location, typically because the model copies the initial location instead of updating it.

• Known: the failing move contradicts a connection that was given in the description or already observed.

• Assumed: the failing move relies on a connection the agent had not yet observed, which is the failure that aggressive commitment deliberately risks.

The breakdown sums to the failed attempts in Tables 5–7.

Three patterns stand out. First, most failures of aggressive formalizers are wrong assumptions, as intended, and priors reduce these by 42–46%, while other causes stay roughly constant. Second, conservative formalizers almost never fail on assumptions, but DeepSeek-Flash fails mostly by keeping a stale agent location in its problem file (96–97% of its failures). This also explains its lower accuracy under full observability, where a conservative formalizer has a single attempt and one stale location ends the run; the drop therefore reflects state tracking, not commitment. Third, Qwen3.8-27B’s formalizer failures under full observability are almost entirely PDDL errors, and it needs about one revision call per problem, compared with at most 0.12 for DeepSeek-Flash. Its lower formalizer accuracy is thus a formalization problem shared by both strategies.

Priors also slightly increase contradictions with known facts for Gemini-3-Flash (0.47 to 0.79 per problem): in some cases the model reasserts a highprobability connection that it had already observed to be absent, which is consistent with the over-

reliance on priors.

## G.3 Results by Path Length

The shortest path from the start room to the coin room is 2 moves in 42 problems, 3 in 30, 4 in 13, and 5–6 in 15 (mean 3.09). The number of hidden connections, which sets the attempt limit, ranges from 8 to 11 and does not vary with path length. Figure 10 and Table 4 break down the results on Coin Collector without connections by path length, averaged over the three models (run 1).

API calls grow with path length for all methods, and fastest for conservative methods, which must observe each room on the way. Failed attempts grow with path length for aggressive methods but stay near zero for conservative planners. The safety–efficiency tradeoff therefore holds at every path length, and both sides of it widen on longer paths. For planners, priors reduce API calls at every length (by 19–41%). For formalizers, the savings are largest on short paths and nearly vanish at length 4, where few problems are available.

## H Full Results

Tables 5, 6, 7, 8, 9, and 10 display the full results for all experiments performed in the main body of this paper.

![](images/3f7e1d95cfe93de6ccdb0408ed187130acd1073a2f072dd05677579f820baaa5.jpg)  
Figure 11: Example of Gemini-3-Flash as Aggressive and Conservative Planner. While Conservative Planner is able to solve the problem in one attempt, Aggressive Planner is able to solve the problem in fewer API Calls.

![](images/839162e03fc53417f1a22ca01c1d87962362bf77cdf61e736d8df3cf1dfd07dd.jpg)  
Figure 12: Example of Gemini-3-Flash as Aggressive and Conservative Formalizer. When statistical priors are given under partial observability, the Aggressive Formalizer can leverage these priors and come up with the correct plan with deductive reasoning.

Listing 1: Statistical Priors for Coin Collector Dataset  
86% of the time, room\_2 is connected directly to the east of room\_1.   
16% of the time, room\_3 is connected directly to the north of room\_2.   
84% of the time, room\_1 is connected directly to the north of room\_9.   
16% of the time, room\_9 is connected directly to the west of room\_4.   
81% of the time, room\_7 is connected directly to the east of room\_6.   
19% of the time, room\_6 is connected directly to the north of room\_5.   
81% of the time, room\_5 is connected directly to the east of room\_8.   
18% of the time, room\_6 is connected directly to the east of room\_4.   
83% of the time, room\_3 is connected directly to the west of room\_8.   
8% of the time, room\_3 is connected directly to the south of room\_5.

Listing 2: Action Heads for Coin Collector  
(move room\_1 room\_2 direction)   
(take item room)

## Listing 3: Action Heads for ALFWorld

; The text after each semicolon is the complete executable command template.   
; Replace only the angle-bracket placeholder with one exact entity name.   
; End the command immediately after the entity name. Never append a room qualifier or explanatory   
text.   
(look agent location) ; look   
(inventory agent) ; inventory   
(examineReceptacle agent receptacle) ; examine <receptacle>   
(examineObject agent object) ; examine <object>   
(GotoLocation agent from\_location to\_location receptacle) ; go to <receptacle>   
(GoToRoom agent from\_location to\_location room) ; go to <room>   
(OpenObject agent location receptacle) ; open <receptacle>   
(CloseObject agent location receptacle) ; close <receptacle>   
(PickupObject agent location object receptacle) ; take <object> from <receptacle>   
(PutObject agent location object receptacle object\_type receptacle\_type) ; move <object> to <   
receptacle>

Listing 4: Domain Description for Coin Collector  
You are standing in an environment with many rooms. Your task is to search the environment and find   
the coin. Once you find the coin, take it.   
The following actions are allowed: move from one room to another and take an item.

Listing 5: Domain Description for ALFWorld  
![](images/48dac1c7436c591dd0351fca7015e657ea9295089ae0b37d04d9a413857f09d3.jpg)

4. To take an object, you must be at the receptacle containing it. If that receptacle is closed, open   
it first.   
5. You can carry at most one movable object at a time.   
6. To move an object to a receptacle, you must be holding the object and be at the target receptacle.   
If the target receptacle is closed, open it first.   
7. Objects inside a closed receptacle may only become visible after you approach and open that   
receptacle.   
8. Use only rooms, receptacles, and objects given in the problem description or revealed by   
environment observations. Do not invent additional entities.   
9. Every command must end immediately after its final entity name. Do not append qualifiers or   
explanatory text.   
Use the problem description, environment observations, and available action heads to choose actions   
that satisfy these rules.

Listing 6: Fully Observable Problem Description for Coin Collector   
There are 9 rooms in this environment: room\_1, room\_2, room\_3, room\_4, room\_5, room\_6, room\_7, room\_8   
and room\_9.   
You are in room\_4.   
To the south of room\_1 is room\_9.   
To the south of room\_2 is room\_6.   
To the east of room\_3 is room\_8.   
To the north of room\_3 is room\_7.   
To the west of room\_4 is room\_9.   
To the east of room\_5 is room\_9.   
To the west of room\_5 is room\_8.   
To the east of room\_6 is room\_7.   
There is a coin in room\_7.

## Listing 7: Partially Observable Problem Description for Coin Collector

There are 9 rooms in this environment: room\_1, room\_2, room\_3, room\_4, room\_5, room\_6, room\_7, room\_8   
and room\_9.   
You are in room\_4.   
There is a coin in room\_7.

## Listing 8: Partially observable household problem description

The rooms are: living room, bedroom, dining room,   
guest room, hallway.   
The receptacles are: bar cart, bed, bookcase cabinet,   
buffet cabinet, closet, end table, entryway drawer,   
guest nightstand, guest nightstand drawer, hall cabinet,   
luggage rack, serving cart basket, shoe rack top,   
tv stand, wardrobe drawer.   
The objects are: jeans, jewelry box, key fob, magazine,   
raincoat, reading lamp, scarf, shoe polish, slippers pair,   
sugar bowl, tablecloth, teapot, video game case,   
water bottle, water glass.   
You are in the dining room.   
Your task is to put slippers pair in the hall cabinet.

## Listing 9: Prompts for Aggressive Planner

initial\_plan\_random = """Given the information below, output the plan that would solve the problem.   
Note that there might be information missing, in which case you should infer what is missing.   
The output should be a list of strings written inside a JSON object in the following format:   
{{   
"plan": ...   
}}   
Domain Description:

{domain\_description}   
Problem Description:   
{problem\_description}   
Available Actions:   
{available\_actions}   
Initial Observation:   
{initial\_observation}   
n n n   
initial\_plan\_statistics = """Given the information below, output the plan that would solve the   
problem. Note that there might be information missing, in which case you should infer what is   
missing.   
The output should be a list of strings written inside a JSON object in the following format:   
{{   
"plan": ...   
}}   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Available Actions:   
{available\_actions}   
Common Sense Knowledge:   
{statistics\_knowledge}   
Initial Observation:   
{initial\_observation}   
n nn   
re\_plan\_random = """In the following problem, the generated plan is incorrect either due to incorrect   
assumptions or other mistakes. Given only the information below, write a new plan, starting   
from the failed step, that would solve the problem. Do not start from the initial state. Note   
that there might be information missing, in which case you should infer what is missing.   
The output should be a list of strings written inside a JSON object in the following format:   
{{   
"plan": ...   
}}   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Available Actions:   
{available\_actions}   
Incorrect Generated Plan:   
{previous\_plan}   
Trajectory for Incorrect Plan:   
{previous\_trajectory}   
"""   
re\_plan\_statistics = """In the following problem, the generated plan is incorrect either due to   
incorrect assumptions or other mistakes. Given only the information below, write a new plan,   
starting from the failed step, that would solve the problem. Do not start from the initial state.   
Note that there might be information missing, in which case you should infer what is missing.   
The output should be a list of strings written inside a JSON object in the following format:   
{{

```python
"plan": ...
}}
Domain Description:
{domain_description}
Problem Description:
{problem_description}
Available Actions:
{available_actions}
Common Sense Knowledge:
{statistics_knowledge}
Incorrect Generated Plan:
{previous_plan}
Trajectory for Incorrect Plan:
{previous_trajectory}"""
```

## Listing 10: Prompts for Aggressive Formalizer

initial\_formalize\_random = """You are a PDDL expert.   
Given the information below, write the domain and problem files in minimal PDDL. Note that there   
might be information missing, in which case you should infer what is missing.   
Return a JSON object in the following format:   
{{   
"domain file": ...,   
"problem file":...   
}}   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Available Actions:   
{available\_actions}   
Initial Observation:   
{initial\_observation}   
11 11 11   
initial\_formalize\_statistics = """You are a PDDL expert.   
Given the information below, write the domain and problem files in minimal PDDL. Note that there   
might be information missing, in which case you should infer what is missing.   
Return a JSON object in the following format:   
{{   
"domain file": ...,   
"problem file":...   
}}   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Available Actions:   
{available\_actions}   
Common Sense Knowledge:   
{statistics\_knowledge}

Initial Observation:   
{initial\_observation}   
" n n   
revision\_random = """You are a PDDL expert.   
The following example and generated PDDL contains an error. Using only the information below, revise   
the PDDL to remove the error. Note there might be information missing, in which case you should   
infer what is missing.   
Return a JSON object in the following format:   
{{   
"domain file": ...,   
"problem file":...   
}}   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Available Actions:   
{available\_actions}   
Generated Domain File:   
{previous\_domain\_file}   
Generated Problem File:   
{previous\_problem\_file}   
Error Message:   
{error\_message}   
n n n   
revision\_statistics = """You are a PDDL expert.   
The following example and generated PDDL contains an error. Using only the information below, revise   
the PDDL to remove the error. Note there might be information missing, in which case you should   
infer what is missing.   
Return a JSON object in the following format:   
{{   
"domain file": ...,   
"problem file":...   
}}   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Available Actions:   
{available\_actions}   
Common Sense Knowledge:   
{statistics\_knowledge}   
Generated Domain File:   
{previous\_domain\_file}   
Generated Problem File:   
{previous\_problem\_file}   
Error Message:   
{error\_message}   
n n n

re\_formalize\_random = """You are a PDDL expert.   
In the following problem, the found plan from the domain and problem files is incorrect either due to   
incorrect assumptions or other mistakes. Given only the information below, write a new domain   
and problem file in minimal PDDL that would solve the problem, starting from the failed step. Do   
not start from the initial state. Note there might be information missing, in which case you   
should infer what is missing.   
Return a JSON object in the following format:   
{{   
"domain file": ...,   
"problem file":...   
}}   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Available Actions:   
{available\_actions}   
Generated Domain File:   
{previous\_domain\_file}   
Generated Problem File:   
{previous\_problem\_file}   
Incorrect Found Plan:   
{previous\_plan}   
Trajectory for Incorrect Plan:   
{previous\_trajectory}   
n n n   
re\_formalize\_statistics = """You are a PDDL expert.   
In the following problem, the found plan from the domain and problem files is incorrect either due to   
incorrect assumptions or other mistakes. Given only the information below, write a new domain   
and problem file in minimal PDDL that would solve the problem, starting from the failed step. Do   
not start from the initial state. Note there might be information missing, in which case you   
should infer what is missing.   
Return a JSON object in the following format:   
{{   
"domain file": ...,   
"problem file":...   
}}   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Available Actions:   
{available\_actions}   
Common Sense Knowledge:   
{statistics\_knowledge}   
Generated Domain File:   
{previous\_domain\_file}   
Generated Problem File:   
{previous\_problem\_file}   
Incorrect Found Plan:   
{previous\_plan}

Trajectory for Incorrect Plan:   
{previous\_trajectory}""

Listing 11: Prompts for Conservative Planner  
get\_action\_prompt\_json = """You are a Planning expert.   
Using ONLY the information below, write an action or a series of actions that would solve the problem   
. Do NOT assume any information. Note the problem may not be solvable with the given information,   
in which case you should come up with a tangible subgoal for the problem. The actions must be   
the same format as the available actions or you will be asked to try again or revert to the   
beginning.   
The output should be a list of strings written inside a JSON object in the following format:   
{{   
"plan": ...   
}}   
Do NOT add extra text.   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Current Trajectory:   
{trajectory}   
Available Actions:   
{available\_actions}   
11 11 11   
re\_plan\_prompt\_json = """You are a Planning expert.   
Using ONLY the information below, write an action or a series of actions that would solve the problem   
, starting from the failed step. Do not start from the initial state. Do NOT assume any   
information. Note the problem may not be solvable with the given information, in which case you   
should come up with a tangible subgoal for the problem. The actions must be the same format as   
the available actions or you will be asked to try again or revert to the beginning.   
The output should be a list of strings written inside a JSON object in the following format:   
{{   
"plan": ...   
}}   
Do NOT add extra text.   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Current Trajectory:   
{trajectory}   
Available Actions:   
{available\_actions}   
n n "

Listing 12: Prompts for Conservative Formalizer  
```python
initial_observation_prompt_json = """You are a PDDL expert.
Using ONLY the information below, write the domain and problem file in minimal PDDL. Do NOT assume
any information. Note the problem may not be solvable with the given information, in which case
```

you should come up with a tangible subgoal for the problem.   
Return ONLY a JSON object in the following format:   
{{   
"domain file": ...,   
"problem file":...   
}}   
Do NOT add extra text.   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Trajectory:   
{trajectory}   
Available Actions:   
{available\_actions}   
n n n   
revision\_prompt\_json = """You are a PDDL expert.   
The following generated PDDL contains an error. Using ONLY the information below, revise the PDDL to   
remove the error. Do NOT assume any information. Note the problem may not be solvable with the   
given information, in which case you should come up with a tangible subgoal for the problem.   
Return ONLY a JSON object in the following format:   
{{   
"domain file": ...,   
"problem file":...   
}}   
Do NOT add extra text.   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Trajectory:   
{trajectory}   
Available Actions:   
{available\_actions}   
Domain File:   
{domain\_file}   
Problem File:   
{problem\_file}   
Error:   
{error}   
n n n   
re\_generate\_prompt\_json = """You are a PDDL expert.   
In the following problem, the found plan from the domain and problem files is incorrect. Given only   
the information below, write a new domain and problem file in minimal PDDL that would solve the   
problem, starting from the failed step. Do not start from the initial state. Do NOT assume any   
information. Note the problem may not be solvable with the given information, in which case you   
should come up with a tangible subgoal for the problem.   
Return a JSON object in the following format:   
{{   
"domain file": ...,

"problem file":...   
}}   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Available Actions:   
{available\_actions}   
Generated Domain File:   
{previous\_domain\_file}   
Generated Problem File:   
{previous\_problem\_file}   
Incorrect Found Plan:   
{previous\_plan}   
Trajectory for Incorrect Plan:   
{previous\_trajectory}"""   
grow\_pddl\_prompt\_json = """You are a PDDL expert.   
Using ONLY the following PDDL files and plan trajectory, modify the PDDL files based on the new   
observations. Do NOT assume any information. Note the problem may not be solvable with the given   
information, in which case you should come up with a tangible subgoal for the problem.   
Return ONLY a JSON object in the following format:   
{{   
"domain file": ...,   
"problem file":...   
}}   
Do NOT add extra text.   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Available Actions:   
{available\_actions}   
Domain File:   
{domain\_file}   
Problem File:   
{problem\_file}   
Trajectory:   
{trajectory}"""

Listing 13: Prompts for Hybrid Planner (Free)  
```python
get_action_prompt = """You are a Planning expert.
Using the information below, write an action or a series of actions that would solve the problem.
Note the problem may not be solvable with the given information, in which case you should come
up with a tangible subgoal for the problem OR infer what is missing. The actions must be the
same format as the available actions or you will be asked to try again or revert to the
beginning.
The output should be a list of strings written inside a JSON object in the following format:
{{
"plan": ...
```

}}   
Do NOT add extra text.   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Current Trajectory:   
{trajectory}   
Available Actions:   
{available\_actions}   
11 11 11   
re\_plan\_prompt = """You are a Planning expert.   
Using the information below, write an action or a series of actions that would solve the problem,   
starting from the failed step. Do not start from the initial state. Note the problem may not be   
solvable with the given information, in which case you should come up with a tangible subgoal   
for the problem OR infer what is missing. The actions must be the same format as the available   
actions or you will be asked to try again or revert to the beginning.   
The output should be a list of strings written inside a JSON object in the following format:   
{{   
"plan":   
}}   
Do NOT add extra text.   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Current Trajectory:   
{trajectory}   
Available Actions:   
{available\_actions}   
n1 n n   
get\_action\_prompt\_statistics = """You are a Planning expert.   
Using the information below, write an action or a series of actions that would solve the problem.   
Note the problem may not be solvable with the given information, in which case you should come   
up with a tangible subgoal for the problem OR infer what is missing. The actions must be the   
same format as the available actions or you will be asked to try again or revert to the   
beginning.   
The output should be a list of strings written inside a JSON object in the following format:   
{{   
"plan": .   
}}   
Do NOT add extra text.   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Common Sense Knowledge:   
{statistics\_knowledge}

Current Trajectory:   
{trajectory}   
Available Actions:   
{available\_actions}   
11 11 11   
re\_plan\_prompt\_statistics = """You are a Planning expert.   
Using the information below, write an action or a series of actions that would solve the problem,   
starting from the failed step. Do not start from the initial state. Note the problem may not be   
solvable with the given information, in which case you should come up with a tangible subgoal   
for the problem OR infer what is missing. The actions must be the same format as the available   
actions or you will be asked to try again or revert to the beginning.   
The output should be a list of strings written inside a JSON object in the following format:   
{{   
"plan": ...   
}}   
Do NOT add extra text.   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Common Sense Knowledge:   
{statistics\_knowledge}   
Current Trajectory:   
{trajectory}   
Available Actions:   
{available\_actions}   
n n n

## Listing 14: Prompts for Hybrid Planner (fixed)

generate\_plan\_aggressive\_random = """Given the information below, output the plan that would solve   
the problem. Note that there might be information missing, in which case you should infer what   
is missing.   
The output should be a list of strings written inside a JSON object in the following format:   
{{   
"plan": ...   
}}   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Available Actions:   
{available\_actions}   
Initial Observation:   
{initial\_observation}   
n n n   
generate\_plan\_aggressive\_statistics = """Given the information below, output the plan that would   
solve the problem. Note that there might be information missing, in which case you should infer   
what is missing.   
The output should be a list of strings written inside a JSON object in the following format:   
{{

"plan": ...   
}}   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Available Actions:   
{available\_actions}   
Common Sense Knowledge:   
{statistics\_knowledge}   
Initial Observation:   
{initial\_observation}   
11 n1 11   
re\_plan\_aggressive\_random = """In the following problem, the generated plan is incorrect either due   
to incorrect assumptions or other mistakes. Given only the information below, write a new plan,   
starting from the failed step, that would solve the problem. Do not start from the initial state.   
Note that there might be information missing, in which case you should infer what is missing.   
The output should be a list of strings written inside a JSON object in the following format:   
{{   
"plan": ...   
}}   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Available Actions:   
{available\_actions}   
Incorrect Generated Plan:   
{previous\_plan}   
Trajectory for Incorrect Plan:   
{previous\_trajectory}   
11 n n1   
re\_plan\_aggressive\_statistics = """In the following problem, the generated plan is incorrect either   
due to incorrect assumptions or other mistakes. Given only the information below, write a new   
plan, starting from the failed step, that would solve the problem. Do not start from the initial   
state. Note that there might be information missing, in which case you should infer what is   
missing.   
The output should be a list of strings written inside a JSON object in the following format:   
{{   
"plan": ...   
}}   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Available Actions:   
{available\_actions}   
Common Sense Knowledge:   
{statistics\_knowledge}   
Incorrect Generated Plan:   
{previous\_plan}

Trajectory for Incorrect Plan:   
{previous\_trajectory}   
n n n   
generate\_plan\_conservative\_prompt = """You are a Planning expert.   
Using ONLY the information below, write an action or a series of actions that would solve the problem   
. Do NOT assume any information. Note the problem may not be solvable with the given information,   
in which case you should come up with a tangible subgoal for the problem. The actions must be   
the same format as the available actions or you will be asked to try again or revert to the   
beginning.   
The output should be a list of strings written inside a JSON object in the following format:   
{{   
"plan": ...   
}}   
Do NOT add extra text.   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Current Trajectory:   
{trajectory}   
Available Actions:   
{available\_actions}   
n n n   
re\_plan\_conservative\_prompt = """   
You are a Planning expert.   
Using ONLY the information below, write an action or a series of actions that would solve the problem   
, starting from the failed step. Do not start from the initial state. Do NOT assume any   
information. Note the problem may not be solvable with the given information, in which case you   
should come up with a tangible subgoal for the problem. The actions must be the same format as   
the available actions or you will be asked to try again or revert to the beginning.   
The output should be a list of strings written inside a JSON object in the following format:   
{{   
"plan": .   
}}   
Do NOT add extra text.   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Current Trajectory:   
{trajectory}   
Available Actions:   
{available\_actions}   
n n n

Listing 15: Prompts for Hybrid Formalizer (free)  
initial\_observation\_prompt = """You are a PDDL expert.   
Using the information below, write the domain and problem file in minimal PDDL. Note the problem may   
not be solvable with the given information, in which case you should come up with a tangible   
subgoal for the problem OR infer what is missing.

Return ONLY a JSON object in the following format:   
{{   
"domain file": ...,   
"problem file":...   
}}   
Do NOT add extra text.   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Trajectory:   
{trajectory}   
Available Actions:   
{available\_actions}   
n n n   
revision\_prompt = """You are a PDDL expert.   
The following generated PDDL contains an error. Using the information below, revise the PDDL to   
remove the error. Note the problem may not be solvable with the given information, in which case   
you should come up with a tangible subgoal for the problem OR what is missing.   
Return ONLY a JSON object in the following format:   
{{   
"domain file": ...,   
"problem file":...   
}}   
Do NOT add extra text.   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Trajectory:   
{trajectory}   
Available Actions:   
{available\_actions}   
Domain File:   
{domain\_file}   
Problem File:   
{problem\_file}   
Error:   
{error}   
n n n   
re\_generate\_prompt = """You are a PDDL expert.   
In the following problem, the found plan from the domain and problem files is incorrect. Given the   
information below, write a new domain and problem file in minimal PDDL that would solve the   
problem, starting from the failed step. Do not start from the initial state. Note the problem   
may not be solvable with the given information, in which case you should come up with a tangible   
subgoal for the problem OR infer what is missing.   
Return a JSON object in the following format:   
{{   
"domain file": ...,   
"problem file":...

```python
}}
Domain Description:
{domain_description}
Problem Description:
{problem_description}
Available Actions:
{available_actions}
Generated Domain File:
{previous_domain_file}
Generated Problem File:
{previous_problem_file}
Incorrect Found Plan:
{previous_plan}
Trajectory for Incorrect Plan:
{previous_trajectory}"""
grow_pddl_prompt = """You are a PDDL expert.
Using the following PDDL files and plan trajectory, modify the PDDL files based on the new
observations. Note the problem may not be solvable with the given information, in which case you
should come up with a tangible subgoal for the problem OR infer what is missing.
Return ONLY a JSON object in the following format:
{{
"domain file": ...,
"problem file":...
}}
Do NOT add extra text.
Domain Description:
{domain_description}
Problem Description:
{problem_description}
Available Actions:
{available_actions}
Domain File:
{domain_file}
Problem File:
{problem_file}
Trajectory:
{trajectory}"""
initial_observation_prompt_statistics = """You are a PDDL expert.
Using the information below, write the domain and problem file in minimal PDDL. Note the problem may
not be solvable with the given information, in which case you should come up with a tangible
subgoal for the problem OR infer what is missing.
Return ONLY a JSON object in the following format:
{{
"domain file": ...,
"problem file":...
}}
Do NOT add extra text.
```

Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Common Sense Knowledge:   
{statistics\_knowledge}   
Trajectory:   
{trajectory}   
Available Actions:   
{available\_actions}   
n1 n n   
revision\_prompt\_statistics = """You are a PDDL expert.   
The following generated PDDL contains an error. Using the information below, revise the PDDL to   
remove the error. Note the problem may not be solvable with the given information, in which case   
you should come up with a tangible subgoal for the problem OR what is missing.   
Return ONLY a JSON object in the following format:   
{{   
"domain file": ...,   
"problem file":...   
}}   
Do NOT add extra text.   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Common Sense Knowledge:   
{statistics\_knowledge}   
Trajectory:   
{trajectory}   
Available Actions:   
{available\_actions}   
Domain File:   
{domain\_file}   
Problem File:   
{problem\_file}   
Error:   
{error}   
n n n   
re\_generate\_prompt\_statistics = """You are a PDDL expert.   
In the following problem, the found plan from the domain and problem files is incorrect. Given the   
information below, write a new domain and problem file in minimal PDDL that would solve the   
problem, starting from the failed step. Do not start from the initial state. Note the problem   
may not be solvable with the given information, in which case you should come up with a tangible   
subgoal for the problem OR infer what is missing.   
Return a JSON object in the following format:   
{{   
"domain file": ...,   
"problem file":...   
}}

Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Common Sense Knowledge:   
{statistics\_knowledge}   
Available Actions:   
{available\_actions}   
Generated Domain File:   
{previous\_domain\_file}   
Generated Problem File:   
{previous\_problem\_file}   
Incorrect Found Plan:   
{previous\_plan}   
Trajectory for Incorrect Plan:   
{previous\_trajectory}"""   
grow\_pddl\_prompt\_statistics = """You are a PDDL expert.   
Using the following PDDL files and plan trajectory, modify the PDDL files based on the new   
observations. Note the problem may not be solvable with the given information, in which case you   
should come up with a tangible subgoal for the problem OR infer what is missing.   
Return ONLY a JSON object in the following format:   
{{   
"domain file": ...,   
"problem file":...   
}}   
Do NOT add extra text.   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Common Sense Knowledge:   
{statistics\_knowledge}   
Available Actions:   
{available\_actions}   
Domain File:   
{domain\_file}   
Problem File:   
{problem\_file}   
Trajectory:   
{trajectory}"""

Listing 16: Prompts for Hybrid Formalizer (fixed)  
```python
initial_formalize_random = """You are a PDDL expert.
Given the information below, write the domain and problem files in minimal PDDL. Note that there
might be information missing, in which case you should infer what is missing.
Return a JSON object in the following format:
{{
"domain file": ...,
```

```jinja
"problem file":...
}}
Domain Description:
{domain_description}
Problem Description:
{problem_description}
Available Actions:
{available_actions}
Initial Observation:
{initial_observation}
11 1 11
initial_formalize_statistics = """You are a PDDL expert.
Given the information below, write the domain and problem files in minimal PDDL. Note that there
might be information missing, in which case you should infer what is missing.
Return a JSON object in the following format:
{{
"domain file": ...,
"problem file":...
}}
Domain Description:
{domain_description}
Problem Description:
{problem_description}
Available Actions:
{available_actions}
Common Sense Knowledge:
{statistics_knowledge}
Initial Observation:
{initial_observation}
n n n
revision_random = """You are a PDDL expert.
The following example and generated PDDL contains an error. Using only the information below, revise
the PDDL to remove the error. Note there might be information missing, in which case you should
infer what is missing.
Return a JSON object in the following format:
{{
"domain file": ...,
"problem file":...
}}
Domain Description:
{domain_description}
Problem Description:
{problem_description}
Available Actions:
{available_actions}
Generated Domain File:
{previous_domain_file}
Generated Problem File:
```

{previous\_problem\_file}   
Error Message:   
{error\_message}   
n nn   
revision\_statistics = """You are a PDDL expert.   
The following example and generated PDDL contains an error. Using only the information below, revise   
the PDDL to remove the error. Note there might be information missing, in which case you should   
infer what is missing.   
Return a JSON object in the following format:   
{{   
"domain file": ...,   
"problem file":...   
}}   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Available Actions:   
{available\_actions}   
Common Sense Knowledge:   
{statistics\_knowledge}   
Generated Domain File:   
{previous\_domain\_file}   
Generated Problem File:   
{previous\_problem\_file}   
Error Message:   
{error\_message}"""   
re\_formalize\_random = """You are a PDDL expert.   
In the following problem, the found plan from the domain and problem files is incorrect either due to   
incorrect assumptions or other mistakes. Given only the information below, write a new domain   
and problem file in minimal PDDL that would solve the problem, starting from the failed step. Do   
not start from the initial state. Note there might be information missing, in which case you   
should infer what is missing.   
Return a JSON object in the following format:   
{{   
"domain file": ...,   
"problem file":...   
}}   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Available Actions:   
{available\_actions}   
Generated Domain File:   
{previous\_domain\_file}   
Generated Problem File:   
{previous\_problem\_file}   
Incorrect Found Plan:

{previous\_plan}   
Trajectory for Incorrect Plan:   
{previous\_trajectory}   
n n n   
re\_formalize\_statistics = """You are a PDDL expert.   
In the following problem, the found plan from the domain and problem files is incorrect either due to   
incorrect assumptions or other mistakes. Given only the information below, write a new domain   
and problem file in minimal PDDL that would solve the problem, starting from the failed step. Do   
not start from the initial state. Note there might be information missing, in which case you   
should infer what is missing.   
Return a JSON object in the following format:   
{{   
"domain file": ...,   
"problem file":...   
}}   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Available Actions:   
{available\_actions}   
Common Sense Knowledge:   
{statistics\_knowledge}   
Generated Domain File:   
{previous\_domain\_file}   
Generated Problem File:   
{previous\_problem\_file}   
Incorrect Found Plan:   
{previous\_plan}   
Trajectory for Incorrect Plan:   
{previous\_trajectory}   
  
initial\_observation\_prompt = """You are a PDDL expert.   
Using ONLY the information below, write the domain and problem file in minimal PDDL. Do NOT assume   
any information. Note the problem may not be solvable with the given information, in which case   
you should come up with a tangible subgoal for the problem.   
Return ONLY a JSON object in the following format:   
{{   
"domain file": ...,   
"problem file":...   
}}   
Do NOT add extra text.   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Trajectory:   
{trajectory}

Available Actions:   
{available\_actions}   
n n n   
revision\_prompt = """You are a PDDL expert.   
The following generated PDDL contains an error. Using ONLY the information below, revise the PDDL to   
remove the error. Do NOT assume any information. Note the problem may not be solvable with the   
given information, in which case you should come up with a tangible subgoal for the problem.   
Return ONLY a JSON object in the following format:   
{{   
"domain file": ...,   
"problem file":...   
}}   
Do NOT add extra text.   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Trajectory:   
{trajectory}   
Available Actions:   
{available\_actions}   
Domain File:   
{domain\_file}   
Problem File:   
{problem\_file}   
Error:   
{error}   
n n n   
re\_generate\_prompt = """You are a PDDL expert.   
In the following problem, the found plan from the domain and problem files is incorrect. Given only   
the information below, write a new domain and problem file in minimal PDDL that would solve the   
problem, starting from the failed step. Do not start from the initial state. Do NOT assume any   
information. Note the problem may not be solvable with the given information, in which case you   
should come up with a tangible subgoal for the problem.   
Return a JSON object in the following format:   
{{   
"domain file": ...,   
"problem file":...   
}}   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Available Actions:   
{available\_actions}   
Generated Domain File:   
{previous\_domain\_file}   
Generated Problem File:   
{previous\_problem\_file}

Incorrect Found Plan:   
{previous\_plan}   
Trajectory for Incorrect Plan:   
{previous\_trajectory}   
n n n   
grow\_pddl\_prompt = """You are a PDDL expert.   
Using ONLY the following PDDL files and plan trajectory, modify the PDDL files based on the new   
observations. Do NOT assume any information. Note the problem may not be solvable with the given   
information, in which case you should come up with a tangible subgoal for the problem.   
Return ONLY a JSON object in the following format:   
{{   
"domain file": ...,   
"problem file":...   
}}   
Do NOT add extra text.   
Domain Description:   
{domain\_description}   
Problem Description:   
{problem\_description}   
Available Actions:   
{available\_actions}   
Domain File:   
{domain\_file}   
Problem File:   
{problem\_file}   
Trajectory:   
{trajectory}"""

<table><tr><td rowspan="2">Method</td><td colspan="5">Without statistics</td><td colspan="5">With statistics</td></tr><tr><td>Acc.↑</td><td>Att.</td><td>Fail.↓</td><td>API↓</td><td>Tok.↓</td><td>Acc.↑</td><td>Att.</td><td>Fail.↓</td><td>API↓</td><td>Tok.↓</td></tr><tr><td colspan="9">Full observability</td><td></td></tr><tr><td>Planner</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Aggressive</td><td>100</td><td>1.00</td><td>0.00</td><td>1.00</td><td>35 35</td><td>100</td><td>1.00</td><td>0.00</td><td>1.00</td><td>35</td></tr><tr><td>Conservative</td><td>98</td><td>1.00</td><td>0.02</td><td>1.00</td><td>35</td><td>1 94</td><td></td><td></td><td>一 1.00</td><td>一 35</td></tr><tr><td>Hybrid (free)</td><td>100</td><td>1.00</td><td>0.00</td><td>1.00</td><td></td><td></td><td>1.00</td><td>0.06</td><td></td><td></td></tr><tr><td>Hybrid (fixed)</td><td>100</td><td>1.00</td><td>0.00</td><td>1.00</td><td>35</td><td>100</td><td>1.00</td><td>0.00</td><td>1.00</td><td>34</td></tr><tr><td>Formalizer</td><td></td><td></td><td></td><td></td><td></td><td></td><td>1.00</td><td>0.06</td><td>1.01</td><td>468</td></tr><tr><td>Aggressive</td><td>100</td><td>1.00</td><td>0.00</td><td>1.02</td><td>485</td><td>94</td><td></td><td></td><td></td><td></td></tr><tr><td>Conservative</td><td>69</td><td>1.00</td><td>0.31</td><td>1.77</td><td>727</td><td></td><td></td><td></td><td>一</td><td></td></tr><tr><td>Hybrid (free)</td><td>93</td><td>1.00</td><td>0.07</td><td>1.01</td><td>466</td><td>90</td><td>1.00</td><td>0.10</td><td>1.01</td><td>456</td></tr><tr><td>Hybrid (fixed)</td><td>100</td><td>1.00</td><td>0.00</td><td>1.01</td><td>483</td><td>92</td><td>1.00</td><td>0.08</td><td>1.04</td><td>490</td></tr><tr><td colspan="9">Missing all connections</td><td></td><td></td></tr><tr><td>Planner</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Aggressive</td><td>99</td><td>4.09</td><td>3.10</td><td>4.09</td><td>111</td><td>100</td><td>2.47</td><td>1.47</td><td>2.47</td><td>86</td></tr><tr><td>Conservative</td><td>100</td><td>1.03</td><td>0.03</td><td>4.23</td><td>52</td><td></td><td></td><td>一</td><td>一</td><td>一</td></tr><tr><td>Hybrid (free)</td><td>100</td><td>2.08</td><td>1.08</td><td>4.07</td><td>73</td><td>100</td><td>2.32</td><td>1.32</td><td>2.48</td><td>80</td></tr><tr><td>Hybrid (fixed)</td><td>100</td><td>2.71</td><td>1.71</td><td>4.22</td><td>90</td><td>100</td><td>2.01</td><td>1.01</td><td>2.71</td><td>71</td></tr><tr><td>Formalizer</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Aggressive</td><td>99</td><td>4.10</td><td>3.11</td><td>4.20</td><td>1,515</td><td>97</td><td>3.00</td><td>2.03</td><td>3.10</td><td>1,211</td></tr><tr><td>Conservative</td><td>99</td><td>2.58</td><td>1.59</td><td>5.84</td><td>1,947</td><td>99</td><td></td><td></td><td>3.11 一</td><td></td></tr><tr><td>Hybrid (free)</td><td>97</td><td>4.27</td><td>3.30</td><td>4.60</td><td>1,500</td><td></td><td>2.93</td><td>1.94</td><td></td><td>1,108</td></tr><tr><td>Hybrid (fixed)</td><td>94</td><td>4.62</td><td>3.68</td><td>6.31</td><td>2,212</td><td>100</td><td>3.24</td><td>2.24</td><td>4.23</td><td>1,632</td></tr></table>

Table 5: Coin Collector results for DeepSeek-Flash (run 1). Acc. is planning accuracy (%), Att. the average number of attempts per problem, Fail. the average number of failed attempts per problem (attempts minus accuracy), API the average number of API calls per problem, and Tok. the average number of output tokens per problem. Conservative methods are not given statistics (–). All rows use 100 problems.

<table><tr><td rowspan="2">Method</td><td colspan="5">Without statistics</td><td colspan="5">With statistics</td></tr><tr><td>Acc.↑</td><td>Att.</td><td>Fail.↓</td><td>API↓</td><td>Tok.↓</td><td>Acc.↑</td><td>Att.</td><td>Fail.↓</td><td>API↓</td><td>Tok.↓</td></tr><tr><td colspan="9">Full observability</td><td></td></tr><tr><td>Planner</td><td></td><td></td><td>0.00</td><td>1.00</td><td>35</td><td>98</td><td>1.00</td><td>0.02</td><td>1.00</td><td>34</td></tr><tr><td>Aggressive</td><td>100</td><td>1.00 1.00</td><td>0.00</td><td>1.00</td><td>35</td><td>一</td><td></td><td></td><td></td><td>一</td></tr><tr><td>Conservative Hybrid (free)</td><td>100</td><td>1.00</td><td>0.00</td><td>1.00</td><td>35</td><td>99</td><td>1.00</td><td>0.01</td><td>1.00</td><td>34</td></tr><tr><td>Hybrid (fixed)</td><td>100 100</td><td>1.00</td><td>0.00</td><td>1.00</td><td>35</td><td>99</td><td>1.00</td><td>0.01</td><td>1.00</td><td>34</td></tr><tr><td colspan="9">Formalizer</td><td></td></tr><tr><td>Aggressive</td><td>98</td><td>1.00</td><td>0.02</td><td>1.00</td><td>438</td><td>91</td><td>1.00</td><td>0.09</td><td>1.00</td><td>442</td></tr><tr><td>Conservative</td><td>95</td><td>1.00</td><td>0.05</td><td>1.67</td><td>634</td><td>一</td><td></td><td></td><td>一</td><td></td></tr><tr><td>Hybrid (free)</td><td>73</td><td>1.00</td><td>0.27</td><td>1.00</td><td>420</td><td>76</td><td>1.00</td><td>0.24</td><td>1.01</td><td>427</td></tr><tr><td>Hybrid (fixed)</td><td>96</td><td>1.00</td><td>0.04</td><td>1.01</td><td>440</td><td>88</td><td>1.00</td><td>0.12</td><td>1.02</td><td>454</td></tr><tr><td colspan="9">Missing all connections</td><td></td></tr><tr><td>Planner</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Aggressive</td><td>99</td><td>3.68</td><td>2.69 0.09</td><td>3.68 4.43</td><td>106 54</td><td>100</td><td>2.51</td><td>1.51</td><td>2.51</td><td>85</td></tr><tr><td>Conservative</td><td>100</td><td>1.09</td><td>2.16</td><td></td><td>100</td><td>100</td><td></td><td>一</td><td>一 2.47</td><td>一 84</td></tr><tr><td>Hybrid (free)</td><td>100</td><td>3.16</td><td></td><td>3.63</td><td></td><td>100</td><td>2.47</td><td>1.47</td><td>2.90</td><td>74</td></tr><tr><td>Hybrid (fixed)</td><td>100</td><td>2.65</td><td>1.65</td><td>3.92</td><td>80</td><td></td><td>2.09</td><td>1.09</td><td></td><td></td></tr><tr><td colspan="9">Formalizer</td><td></td><td>3.11</td></tr><tr><td>Aggressive</td><td>100</td><td>3.89</td><td>2.89</td><td>3.90</td><td>1,432</td><td>99</td><td>3.10</td><td>2.11</td><td></td><td>1,248</td></tr><tr><td>Conservative</td><td>100</td><td>1.29</td><td>0.29 2.70</td><td>4.91</td><td>1,572 1,248</td><td>99 1</td><td>3.25</td><td>2.26</td><td></td><td></td></tr><tr><td>Hybrid (free)</td><td>100</td><td>3.70</td><td></td><td>3.70</td><td></td><td></td><td></td><td></td><td>3.26</td><td>1,241</td></tr><tr><td>Hybrid (fixed)</td><td>100</td><td>3.34</td><td>2.34</td><td>4.98</td><td>1,711</td><td>100</td><td>2.82</td><td>1.82</td><td>4.14</td><td>1,557</td></tr><tr><td>Planner</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Aggressive</td><td>100</td><td>1.00</td><td>0.00</td><td>1.00</td><td>35 35</td><td>98</td><td>1.00</td><td>0.02</td><td>1.00</td><td>34 一</td></tr><tr><td>Conservative</td><td>98</td><td>1.00</td><td>0.02</td><td>1.00</td><td>35</td><td>1 98</td><td>1.00</td><td>0.02</td><td>一 1.00</td><td>35</td></tr><tr><td>Hybrid (free)</td><td>97 99</td><td>1.00 1.00</td><td>0.03 0.01</td><td>1.00 1.00</td><td>35</td><td>99</td><td>1.00</td><td>0.01</td><td>1.00</td><td>34</td></tr><tr><td colspan="9">Hybrid (fixed)</td><td></td></tr><tr><td>Formalizer</td><td>83</td><td></td><td>0.17</td><td>1.67</td><td>831</td><td>83</td><td>1.00</td><td>0.17</td><td>1.69</td><td>816</td></tr><tr><td>Aggressive Conservative</td><td>80</td><td>1.00 1.00</td><td>0.20</td><td>1.93</td><td>821</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Hybrid (free)</td><td></td><td></td><td>0.22</td><td>1.74</td><td>837</td><td>一 72</td><td>1.00</td><td>0.28</td><td>1.88</td><td>885</td></tr><tr><td>Hybrid (fixed)</td><td>78 84</td><td>1.00 1.00</td><td>0.16</td><td>1.69</td><td>843</td><td>77</td><td>1.00</td><td>0.23</td><td>1.80</td><td>872</td></tr><tr><td colspan="9">Missing all connections</td><td></td></tr><tr><td>Planner</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Aggressive</td><td>100</td><td>3.61</td><td>2.61</td><td>3.61</td><td>93</td><td>100</td><td>2.62</td><td>1.62</td><td>2.62</td><td>89</td></tr><tr><td>Conservative</td><td>100</td><td>1.00</td><td>0.00</td><td>4.45</td><td>53</td><td></td><td></td><td></td><td>1</td><td>一</td></tr><tr><td>Hybrid (free)</td><td>100</td><td>2.18</td><td>1.18</td><td>3.96</td><td>69</td><td>100</td><td>2.39</td><td>1.39</td><td>2.50</td><td>82</td></tr><tr><td>Hybrid (fixed)</td><td>100</td><td>2.63</td><td>1.63</td><td>4.11</td><td>79</td><td>100</td><td>2.06</td><td>1.06</td><td>2.99</td><td>74</td></tr><tr><td colspan="9">Formalizer</td><td></td><td></td></tr><tr><td>Aggressive</td><td>75</td><td>3.63</td><td>2.88</td><td>4.46</td><td>1,464</td><td>73</td><td>2.83</td><td>2.10</td><td>3.59</td><td>1,277</td></tr><tr><td>Conservative</td><td>76</td><td>1.08</td><td>0.32</td><td>4.68</td><td>1,477</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Hybrid (free)</td><td>76</td><td>3.61</td><td>2.85</td><td>4.70</td><td>1,369</td><td>70</td><td>2.63</td><td>1.93</td><td>3.83</td><td>1,188</td></tr><tr><td>Hybrid (fixed)</td><td>73</td><td>3.32</td><td>2.59</td><td>5.58</td><td>1,861</td><td>71</td><td>2.07</td><td>1.36</td><td>3.97</td><td>1,531</td></tr></table>

Table 6: Coin Collector results for Gemini-3-Flash (run 1). Acc. is planning accuracy (%), Att. the average number of attempts per problem, Fail. the average number of failed attempts per problem (attempts minus accuracy), API the average number of API calls per problem, and Tok. the average number of output tokens per problem. Conservative methods are not given statistics (–). All rows use 100 problems.

Table 7: Coin Collector results for Qwen3.8-27B (run 1). Acc. is planning accuracy (%), Att. the average number of attempts per problem, Fail. the average number of failed attempts per problem (attempts minus accuracy), API the average number of API calls per problem, and Tok. the average number of output tokens per problem. Conservative methods are not given statistics (–). All rows use 100 problems.

<table><tr><td rowspan="2">Method</td><td colspan="5">Without statistics</td><td colspan="5">With statistics</td></tr><tr><td>Acc.↑</td><td>Att.</td><td>Fail.↓</td><td>API↓</td><td>Tok.↓</td><td>Acc.↑</td><td>Att.</td><td>Fail.↓</td><td>API↓</td><td>Tok.↓</td></tr><tr><td colspan="9">Full observability</td><td></td></tr><tr><td>Planner</td><td></td><td></td><td></td><td></td><td>35</td><td>100</td><td>1.00</td><td>0.00</td><td>1.00</td><td>35</td></tr><tr><td>Aggressive</td><td>100</td><td>1.00</td><td>0.00 0.00</td><td>1.00 1.00</td><td>35</td><td></td><td></td><td></td><td></td><td>一</td></tr><tr><td>Conservative</td><td>100</td><td>1.00</td><td>0.00</td><td>1.00</td><td>35</td><td>一 99</td><td>1.00</td><td>0.01</td><td>1.00</td><td>35</td></tr><tr><td>Hybrid (free)</td><td>100 100</td><td>1.00 1.00</td><td>0.00</td><td>1.00</td><td>35</td><td>100</td><td>1.00</td><td>0.00</td><td>1.00</td><td>35</td></tr><tr><td colspan="9">Hybrid (fixed) Formalizer</td><td></td></tr><tr><td>Aggressive</td><td>98</td><td>1.00</td><td>0.02</td><td>1.03</td><td>494</td><td>96</td><td>1.00</td><td>0.04</td><td>1.00</td><td>472</td></tr><tr><td>Conservative</td><td>75</td><td>1.00</td><td>0.25</td><td>1.73</td><td>717</td><td>1</td><td></td><td></td><td></td><td></td></tr><tr><td>Hybrid (free)</td><td>95</td><td>1.00</td><td>0.05</td><td>1.03</td><td>479</td><td>93</td><td>1.00</td><td>0.07</td><td>1.03</td><td>473</td></tr><tr><td>Hybrid (fixed)</td><td>99</td><td>1.00</td><td>0.01</td><td>1.00</td><td>478</td><td>91</td><td>1.00</td><td>0.09</td><td>1.00</td><td>469</td></tr><tr><td colspan="9">Missing all connections</td><td></td></tr><tr><td>Planner</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">Aggressive</td><td>2.32</td></tr><tr><td>Conservative</td><td>100 100</td><td>3.98 1.01</td><td>2.98 0.01</td><td>3.98 4.28</td><td>107 55</td><td>100</td><td>2.32</td><td>1.32</td><td></td><td>78</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>一 100</td><td>2.31</td><td>一 1.31</td><td>一 2.45</td><td>一 78</td></tr><tr><td>Hybrid (free)</td><td>100</td><td>2.15</td><td>1.15</td><td>4.00</td><td>72 78</td><td>100</td><td></td><td></td><td></td><td>76</td></tr><tr><td>Hybrid (fixed)</td><td>100</td><td>2.59</td><td>1.59</td><td>3.74</td><td></td><td></td><td>2.06</td><td>1.06</td><td>2.81</td><td></td></tr><tr><td colspan="9">Formalizer</td><td></td><td></td></tr><tr><td>Aggressive</td><td>98</td><td>4.27</td><td>3.29</td><td>4.35</td><td>1,594</td><td>100</td><td>2.93</td><td>1.93</td><td>3.03</td><td>1,166</td></tr><tr><td>Conservative</td><td>96</td><td>2.87</td><td>1.91</td><td>6.39</td><td>2,127</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Hybrid (free)</td><td>98</td><td>4.25</td><td>3.27</td><td>4.42</td><td>1,443</td><td>97</td><td>2.94</td><td>1.97</td><td>3.17</td><td>1,133</td></tr><tr><td>Hybrid (fixed)</td><td>93</td><td>4.71</td><td>3.78</td><td>6.47</td><td>2,283</td><td>96</td><td>3.51</td><td>2.55</td><td>4.57</td><td>1,739</td></tr><tr><td>Planner</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Aggressive</td><td>100</td><td>1.00</td><td>0.00 0.02</td><td>1.00 1.00</td><td>35 35</td><td>100</td><td>1.00</td><td>0.00</td><td>1.00</td><td>35 一</td></tr><tr><td>Conservative</td><td>98</td><td>1.00</td><td></td><td></td><td>35</td><td>1 96</td><td></td><td></td><td>1.00</td><td>35</td></tr><tr><td>Hybrid (free)</td><td>100</td><td>1.00</td><td>0.00</td><td>1.00</td><td></td><td></td><td>1.00</td><td>0.04</td><td></td><td></td></tr><tr><td>Hybrid (fixed)</td><td>100</td><td>1.00</td><td>0.00</td><td>1.00</td><td>35</td><td>100</td><td>1.00</td><td>0.00</td><td>1.00</td><td>35</td></tr><tr><td>Formalizer</td><td></td><td></td><td></td><td></td><td>486</td><td>94</td><td></td><td></td><td>1.00</td><td>476</td></tr><tr><td>Aggressive</td><td>99</td><td>1.00</td><td>0.01</td><td>1.01</td><td></td><td></td><td>1.00</td><td>0.06</td><td></td><td></td></tr><tr><td>Conservative</td><td>69</td><td>1.00</td><td>0.31</td><td>1.63</td><td>679</td><td>一</td><td></td><td></td><td>一</td><td></td></tr><tr><td>Hybrid (free)</td><td>93</td><td>1.00</td><td>0.07</td><td>1.01</td><td>470</td><td>90</td><td>1.00</td><td>0.10</td><td>1.01</td><td>456</td></tr><tr><td>Hybrid (fixed)</td><td>100</td><td>1.00</td><td>0.00</td><td>1.02</td><td>490</td><td>86</td><td>1.00</td><td>0.14</td><td>1.00</td><td>470</td></tr><tr><td colspan="9">Missing all connections</td><td></td><td></td></tr><tr><td>Planner</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Aggressive</td><td>99</td><td>3.81</td><td>2.82</td><td>3.81</td><td>100</td><td>100</td><td>2.40</td><td>1.40</td><td>2.40</td><td>81</td></tr><tr><td>Conservative</td><td>100</td><td>1.00</td><td>0.00</td><td>4.33</td><td>53</td><td>一</td><td></td><td>一</td><td>1</td><td>一</td></tr><tr><td>Hybrid (free)</td><td>100</td><td>2.21</td><td>1.21</td><td>4.10</td><td>74</td><td>100</td><td>2.28</td><td>1.28</td><td>2.47</td><td>78</td></tr><tr><td>Hybrid (fixed)</td><td>100</td><td>2.61</td><td>1.61</td><td>4.03</td><td>83</td><td>100</td><td>2.07</td><td>1.07</td><td>2.82</td><td>75</td></tr><tr><td>Formalizer</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Aggressive</td><td>97</td><td>4.52</td><td>3.55</td><td>4.65</td><td>1,682</td><td>98</td><td>2.98</td><td>2.00</td><td>3.00</td><td>1,168</td></tr><tr><td>Conservative</td><td>100</td><td>2.67</td><td>1.67</td><td>6.07</td><td>2,012</td><td>一</td><td></td><td></td><td></td><td></td></tr><tr><td>Hybrid (free)</td><td>99</td><td>4.02</td><td>3.03</td><td>4.30</td><td>1,352</td><td>99</td><td>2.91</td><td>1.92</td><td>3.04</td><td>1,085</td></tr><tr><td>Hybrid (fixed)</td><td>94</td><td>4.56</td><td>3.62</td><td>6.05</td><td>2,109</td><td>96</td><td>3.46</td><td>2.50</td><td>4.42</td><td>1,712</td></tr></table>

Table 8: Coin Collector results for DeepSeek-Flash (run 2). Acc. is planning accuracy (%), Att. the average number of attempts per problem, Fail. the average number of failed attempts per problem (attempts minus accuracy), API the average number of API calls per problem, and Tok. the average number of output tokens per problem. Conservative methods are not given statistics (–). All rows use 100 problems.

Table 9: Coin Collector results for DeepSeek-Flash (run 3). Acc. is planning accuracy (%), Att. the average number of attempts per problem, Fail. the average number of failed attempts per problem (attempts minus accuracy), API the average number of API calls per problem, and Tok. the average number of output tokens per problem. Conservative methods are not given statistics (–). All rows use 100 problems.

<table><tr><td>Method</td><td>Acc.↑</td><td>Att.</td><td>Fail.↓</td><td>API↓</td><td>Tok.↓</td></tr><tr><td>DeepSeek-Flash</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Aggressive planner</td><td>98</td><td>3.81</td><td>2.83</td><td>3.81</td><td>146</td></tr><tr><td>Conservative planner</td><td>100</td><td>2.33</td><td>1.33</td><td>5.69</td><td>159</td></tr><tr><td>Aggressive formalizer</td><td>75</td><td>11.40</td><td>10.65</td><td>12.34</td><td>10,240</td></tr><tr><td>Conservative formalizer</td><td>63</td><td>12.21</td><td>11.58</td><td>24.41</td><td>16,718</td></tr><tr><td>Qwen3.8-27B</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Aggressive planner</td><td>100</td><td>3.67</td><td>2.67</td><td>3.67</td><td>137</td></tr><tr><td>Conservative planner</td><td>100</td><td>1.14</td><td>0.14</td><td>5.89</td><td>178</td></tr><tr><td>Aggressive formalizer</td><td>26</td><td>3.33</td><td>3.07</td><td>6.20</td><td>6,609</td></tr><tr><td>Conservative formalizer</td><td>30</td><td>5.59</td><td>5.29</td><td>18.26</td><td>11,371</td></tr></table>

Table 10: ALFWorld results for DeepSeek-Flash and Qwen3.8-27B, 100 problems per method, without statistics. Acc. is planning accuracy (%), Att. the average number of attempts per problem, Fail. the average number of failed attempts per problem (attempts minus accuracy), API the average number of API calls per problem, and Tok. the average number of output tokens per problem. All rows use 100 problems.