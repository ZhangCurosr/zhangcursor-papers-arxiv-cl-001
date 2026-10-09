# Project Greenhouse: Progress Toward Fully Open and Sovereign Agentic Search

Jimmy Lin Sahel Sharifymoghaddam Lingwei Gu Nour Jedidi

David R. Cheriton School of Computer Science University of Waterloo

## Abstract

Project Greenhouse represents our exploration of a simple thesis: We believe that it is possible to build fully open and sovereign models for agentic search with only modest computational resources. As a first milestone, we describe how to build a competitive pointwise decoder-only reranker using a simple two-step recipe comprising pre-training from scratch followed by supervised fine-tuning, starting only from commonly available datasets. Contrary to the dominant approach in the literature, we do not rely on existing open-weight backbones from third parties, and thus we are fully in control of model training, from end to end. We were able to accomplish the bulk of our experiments using no more than a handful of GPUs. This report articulates the importance and benefits of our approach, and we share artifacts that enable transparent, independent reproduction of all aspects of model training. Beyond data, code, and configurations that capture our efforts, we also release checkpoints for our family of Gaggle models, demonstrating the feasibility of our approach and providing a first step toward validating our broader thesis.

## 1 Introduction

Project Greenhouse represents our exploration of a simple thesis:

We believe that it is possible to build fully open and sovereign models for agentic search with only modest computational resources.

This report describes our first project milestone. As the headline result, we show that it is possible to build a competitive pointwise decoder-only reranker using a simple two-step recipe comprising pre-training from scratch followed by supervised fine-tuning, starting only from commonly available datasets. Most notably, we do not rely on existing open-weight backbones to initialize our reranker, which has been the dominant approach described in the literature. Models that we pre-trained ourselves from publicly available raw corpora appear to be sufficient to produce competitive rerankers after supervised fine-tuning using a publicly available dataset. Our training recipe currently does not require synthetic data or teacher-provided knowledge from any third-party sources (proprietary or open-weight). Thus, we retain full control over all dependencies in the entire training pipeline, from end to end. Furthermore, we were able to accomplish the bulk of our experiments using only a handful of GPUs, making it feasible for others to reproduce our results.

In addition to code, data, configurations, and other artifacts necessary to reproduce our results, we also release Gaggle, a family of fully open and sovereign model checkpoints that encapsulate our efforts. These artifacts and the experiences we share here offer an “existence proof” of one initial important step in our thesis, laying the foundation for future explorations.

To better circumscribe and contextualize our claims, let us be more precise:

• Our definition of “agentic search” is discussed in Section 2. We are explicit about scoping our efforts around this limited goal, as opposed to training models that are capable of general reasoning, coding, etc.

• Our definition of “fully open and sovereign” is discussed in Section 3, but intuitively, we desire that every step of the model training process (data, code, configurations) is captured in publicly available artifacts.

The remainder of this report describes exactly what we’ve accomplished, starting with the headline result (Section 4) and continuing with the details of our pre-training and supervised fine-tuning steps (Sections 5 and 6). Contrastive conditions and variants of our main approach are presented in Section 7, before we conclude with a discussion of related work and limitations.

Project Greenhouse is intended to be a long-running project, and this report captures a concrete first milestone that we deem worthwhile to share with the community. We view the present contributions of this work as follows:

• We provide a model of LLM training as a directed property hypergraph that allows us to precisely characterize what we mean by “fully open and sovereign”. This formulation not only describes our efforts, but can be extended to characterize an open ecosystem and to coordinate distributed efforts by research groups beyond our institution.

• We describe a small step toward supporting the veracity of our thesis. Although our ambitions encompass agentic search in its entirety, a competitive reranker represents an important stepping stone. This report shares our experiences and interesting findings along the way.

• We share our Gaggle family of decoder-only pointwise rerankers and model checkpoints that capture important steps in our pre-training and fine-tuning journey.

## 2 Toward Agentic Search

Given our thesis, it makes sense to be more precise about what we mean by “agentic search”. Stripping away unhelpful anthropomorphization and deliberate attempts at making things sound more complicated than they actually are, agents are simply models using tools in loops. Anthropic engineers characterize agents as “LLMs using tools based on environmental feedback in a loop”.<sup>1</sup> In modern parlance, the “thing” that does the “looping” is typically referred to as the harness. The harness invokes the model, executes tool calls, returns their results, and maintains the context needed to continue (as well as participating in the decision to stop when the user task is complete).

## 2.1 Scoping Our Efforts

Thus, agentic search is simply agents doing “searchy stuff”. For these applications, models use “search tools” in loops, broadly defined to include, for example, grep, BM25, dense vector search, external APIs that perform live web search, etc. Of course, agentic search can involve non-search tools as well, for example, to read a retrieved document, to execute an analysis, or to capture intermediate findings. Typically, a model would decompose a complex information need into smaller components, and for each, the model might issue a query, examine retrieved results, identify missing information, and follow up with reformulated queries. In many cases, the model is also responsible for synthesizing relevant information into a coherent whole that addresses the user’s information need. Our definition is intentionally broad to encompass the information-seeking aspects of a variety of downstream tasks, from automated data science to writing detailed research reports.

Since agentic search covers quite a bit of ground, we are cognizant of the need to appropriately scope our efforts. We focus only on information seeking: finding, assessing, and synthesizing information from a variety of (possibly heterogeneous) sources in response to a user request. To be clear, we are not attempting to build a general-purpose frontier model; for example, coding and general reasoning are outside our immediate scope. It would be unreasonable to organize our research around competing directly with frontier labs across the full range of capabilities they pursue. Instead, we believe that a deliberately scoped effort can yield useful, measurable advances while contributing to knowledge in a rigorous fashion. For us, the focus is on agentic search using fully open and sovereign models.

Our conception of agentic search does not prescribe a particular system architecture. An informationseeking task might be handled by a single (powerful) model driven by a simple harness, or by multiple agents that explore different sources in parallel and coordinate autonomously to exchange findings. A tool call can be as simple as a system command (e.g., grep), a REST API invocation, or may itself be a composite system involving LLMs (e.g., BM25 search followed by LLM-based reranking). These alternatives represent design choices to be explored empirically.

While our ultimate goal is to build fully open and sovereign models for agentic search in its entirety, our approach emphasizes incremental progress: building useful components, rigorously understanding their behavior and limitations under controlled conditions, and composing them into increasingly capable systems. Achieving full openness and sovereignty while maintaining output quality is an unrealistic goal at the outset: instead, initial systems are likely to be hybrids that integrate both proprietary and open components—for example, a proprietary frontier model coordinator that synthesizes relevant information while delegating searches to fully open and sovereign models. Consistent with our core thesis, we seek to make individual components fully open and sovereign so that others can inspect their development, reproduce our experiments, and build on the results. Over time, we hope to gradually expand our scope to encompass more and more of agentic search.

## 2.2 Pointwise Reranking

Against this backdrop, our first project milestone—that we have successfully accomplished, as described in this report—focuses on a decoder-only pointwise reranker. While rerankers are quite common today, let us be precise about the problem we are tackling, building on the definition of text retrieval (ranking) from Lin [2021]:

The formulation of text retrieval (alternatively, text ranking)—what information retrieval researchers more precisely call ad hoc retrieval—is typically defined as follows: Given an information need expressed as a query q, the text retrieval task is to return a ranked list of k document $\mathrm { ~ \bar { ~ } { ~ } { ~ } ^ { 2 } ~ } \{ d _ { 1 } , d _ { 2 } \dotsb \cdot d _ { k } \}$ from an arbitrarily large but finite collection of documents $\mathcal { D } = \{ \grave { d } _ { i } \}$ that maximizes a metric of interest, for example, nDCG, AP, etc. These metrics vary, but they all aim to quantify the “goodness” of the results with respect to the information need; in some cases, metrics can be understood more formally in terms of the utility that a user would derive from consuming the results. The retrieval task is also called top-k retrieval (or ranking), where k is the length of the ranked list (also known as the retrieval or ranking depth).

Given this setup, a reranker is a component that takes as input an initial list of documents (typically from top-k retrieval, often called the first stage) and reorders the list (i.e., generates a permutation of the input) with the aim to improve its quality in terms of the metric of interest. A pointwise reranker accomplishes this one $( q , d _ { n } )$ pair at a time, where q is the query and $d _ { n }$ is a document returned by top-k retrieval above, $n \in \{ 1 \ldots k \}$ . For every such pair, which is considered in isolation independently, the reranker generates a score $s _ { n } , \mathrm { i . e . , } \mathcal { R } ( q , d _ { n } ) = s _ { n } .$ , such that sorting by these scores produces a permutation of the input ranked list that is intended to increase the metric of interest.<sup>3</sup>

In this report, we describe a fully open and sovereign decoder-only LLM for pointwise reranking that achieves a level of effectiveness competitive with alternatives reported in the literature built on open-weight models. Beyond model checkpoints, we share all artifacts (data, code, configurations) that are necessary to reproduce our work, end to end.

## 2.3 Why Pointwise Reranking?

A natural next question: Why start with a pointwise reranker as the first project milestone? We provide three answers:

Ease of iteration. Velocity is an important consideration in our project, and a pointwise reranker enables rapid iteration during model development. At inference time, the reranker is simply a function that maps a query–document pair $( q , d )$ to a scalar score. Each candidate can be scored independently, and the resulting scores determine the ranking. This gives us a simple interface for training and evaluation, allowing us to isolate the effects of changes to training data, training recipes, model architectures, configurations, etc.

With pointwise rerankers, we can evaluate model checkpoints on a fixed set of candidate documents without processing the entire corpus each time. Compared to pairwise [Pradeep et al., 2021] and listwise [Pradeep et al., 2023a,b] variants, pointwise rerankers are easier to build because there are fewer design choices to explore. With respect to other reasonable options as a first project milestone, we considered but did not pursue training embedding models, which typically involve much more effort. Assessing the quality of an updated model usually requires recomputing embeddings for the entire corpus before evaluating top-k retrieval. For large collections, this additional step can substantially slow the development cycle. We feel that training agentic search models is even less appropriate as a first step, as such efforts involve substantial complexity, including generating and evaluating multi-step trajectories and, depending on the approach, collecting rollouts to feed reinforcement learning. In the end, we targeted pointwise reranking, which gives us a more tightly scoped experimental setting with fewer interacting components. We believe that it is a simple but useful capability that can provide an interesting existence proof for our core thesis.

Usefulness in isolation. A reranker is useful as a standalone component in a broader retrieval pipeline. It can be applied to candidate documents generated by any number of approaches: BM25, a dense retrieval model, an external search API, or a hybrid of these. Reranking can be especially helpful in cases where one does not have direct access to the raw corpus, such as when searching via a web API. In all cases, the interface remains the same: given a query and a list of candidate documents, score each pair and reorder the candidates. The design makes it possible to incorporate a reranker into an existing system with minimal effort.

The modularity of rerankers is particularly attractive for agentic search. At each iteration, a search tool may return more content than the agent can reasonably inspect. A reranker can help prioritize which results to read or include in the model’s context. This means that a reranker can be inserted into an agentic search application at multiple points, thus increasing its applicability.

The existence of commercial reranking services provides further evidence that this is a useful capability in its own right. Examples include Microsoft’s semantic ranker in Azure AI Search,<sup>4</sup> Google Cloud’s Ranking API,<sup>5</sup> NVIDIA’s NeMo Retriever Reranking NIM,<sup>6</sup> and dedicated APIs from Cohere,<sup>7</sup> Voyage $\mathbb { A } \breve { \mathbb { I } } _ { * } ^ { 8 }$ and Jina $\mathrm { A I . ^ { 9 } }$ These offerings span hosted APIs, integrated search services, and deployable inference microservices, illustrating the utility of reranking across different retrieval architectures. The recent release of the Jev “decision model”<sup>10</sup> further confirms the utility of reranking as an important independent capability: although the model was designed for structured outputs in general, reranking has emerged as a popular use case.

Usefulness as a stepping stone. A reranker is also an important stepping stone towards more sophisticated models for agentic search. Beyond ordering results at inference time to prioritize what is fed to the model (as discussed above), a reranker can be used to assess relevance, i.e., used as an LLM judge [Zheng et al., 2023, Upadhyay et al., 2024, 2025]. A pointwise reranker $\mathcal { R } ( q , d _ { n } ) = s _ { n } ,$ if properly calibrated, can provide scores that estimate the probability of relevance, $\mathrm { i . e . , } s _ { n } \approx P ( y _ { n } = 1 \mid q , d _ { n } )$ , where $y _ { n }$ denotes binary relevance. For search agents, a good relevance judge might be a helpful capability when inspecting retrieved results, particularly in cases where they only have access to “weak” tools such as grep [Li et al., 2026] or BM25 search [Hsu et al., 2026].

Modern search agents can be trained with reinforcement learning to acquire search policies [Zhuang et al., 2025, Jin et al., 2025], and relevance scores can contribute to reward signals by providing feedback on the relevance of the material retrieved by their actions [Mao et al., 2024, Li et al., 2025]. Alternatively, a relevance judge could be used as a component of a verifier to assess the quality of rollouts as part of rejection sampling [Wu et al., 2025, Luo et al., 2026b]. There are many possibilities, and it is not difficult to see how a reranker could serve as an important component in agentic search, either at training time or inference time.

To be clear, this report captures our present findings within a larger effort. Beyond pointwise reranking, we have plans to build models for listwise reranking, query reformulation, relevance feedback, steering search trajectories, etc. towards full agentic search. A reranker is a good first milestone that provides an existence proof of one initial aspect of our broader vision. Our reranking model is of immediate utility and provides a stepping stone for subsequent research.

## 3 Fully Open and Sovereign Models

Next, let us precisely define what we mean by fully open and sovereign models.

## 3.1 LLM Training as Hypergraphs

Abstractly, we can model LLM training as a directed property hypergraph $\mathcal { G } = ( V , E )$ where the vertices $( V )$ represent artifacts (for example, a raw corpus, training examples, model weights, etc.) and the edges (E) represent computations (including training recipes and associated configuration data) that consume upstream artifacts to generate downstream artifacts. Following standard notation, a (directed) hyperedge $e \in E$ is formally an ordered pair $( T _ { e } , H _ { e } )$ , where $T _ { e } , \bar { H _ { e } } \subseteq V$ , commonly written as $T _ { e } \longrightarrow H _ { e \cdot } ^ { - } \mathrm { \bf ~ A }$ hyperedge describes a computation, and we can think of an edge traversal as someone running that computation. Traversing an edge that already exists can be viewed as a reproducibility attempt.

In this directed hypergraph representation, both vertices and hyperedges are annotated with properties in the form of arbitrary key–value pairs. One simple possibility is that a vertex is annotated with the Hugging Face dataset that captures the artifact and a hyperedge is annotated with a command invocation identified by a commit id in a particular GitHub repo. Both vertices and edges might be associated with publications. Each hyperedge might be further annotated with compute requirements, for example, quantifying the approximate number of GPU hours that an edge traversal will require.

For example, pre-training starts from a raw corpus of texts $T _ { 1 } = \{ { \mathcal { C } } _ { 1 } \}$ to produce pre-trained model weights $\grave { H } _ { 1 } = \left\{ \mathcal { M } _ { 1 } \right\}$ via $e _ { 1 }$ , a pre-training recipe (Section 5). With appropriate property annotations, it would be possible for someone with sufficient compute resources to reproduce model pre-training, $\mathrm { i . e . , }$ to traverse $T _ { 1 } \longrightarrow H _ { 1 }$ . In this example, we assume that the pre-training recipe associated with $e _ { 1 }$ includes not only the actual source code but also all relevant configuration data.

Hyperedges provide a natural representation to capture other aspects of LLM training because they allow multiple tail vertices. For example, a fine-tuning hyperedge $e _ { 2 }$ might start from a pre-trained backbone and some training set $( T _ { 2 } ^ { \bf { \bar { \alpha } } } = \{ \mathcal { M } _ { 1 } , \mathcal { S } _ { 1 } \} )$ to produce model weights $H _ { 2 } = \left\{ { \mathcal { M } } \backslash { \mathcal { M } } _ { 2 } \right\}$ via a particular fine-tuning recipe (Section 6). In principle, the head vertex set can contain multiple elements as well, representing more than one output artifact.

Definition: Given a hypergraph representation of model training, we define a fully open and sovereign model as a vertex (model weights) where all upstream vertices and edges are publicly available and do not contain proprietary dependencies.

In other words, a model is fully open and sovereign if and only if it is possible to independently derive the model weights from a set of publicly available sources via computational steps (represented by edge traversals) that are also captured in public artifacts.

Of course, the tracing of upstream sources must “stop” somewhere, because otherwise it is easy to arrive at absurdities. For example, a standard pre-training corpus $( \mathrm { e . g . }$ ., FineWeb-Edu or ClimbMix) lies downstream of potentially complex processes that generated it. These include stages that involve web crawling, curation, cleaning, parsing, etc., and the texts ultimately derive from a multitude of authors on the web. It would be impractical to trace LLM training all the way back to these authors. Similarly, standard training datasets (e.g., MS MARCO) are the output of some annotation process (in this case, by Microsoft). As we trace sources upstream, where this process “ends” necessarily involves making some judgment calls, which we indeed do, based on commonly accepted datasets in the field. For example, we consider starting with the ClimbMix corpus or the MS MARCO dataset fully open and sovereign, given their widespread availability and familiarity among researchers.

Based on our definition, any model that is derived from open weights, but whose pre-training corpus and pre-training recipes are unavailable, does not qualify as being fully open and sovereign. While open-weight models are undoubtedly useful and provide high-quality backbones for subsequent finetuning, our work shows that they are not necessary to produce a competitive reranker. In Section 8, we discuss related work with respect to our definitions.

## 3.2 The LLM Training Ecosystem

The directed hypergraph representation of model training proposed above provides a precise way to capture provenance: a model can be characterized as a specific traversal of the training hypergraph, with intermediate artifacts captured along the way. Such a representation can help a group of collaborating researchers more accurately track ablations and contrastive conditions. A data ablation can be thought of as a hyperedge with fewer tail vertices. A code ablation or an alternative training recipe can be thought of as different hyperedges with the same tail vertices.

We can extend this representation of LLM training beyond a single research group to characterize the entire ecosystem comprised of a multitude of researchers around the world working in a distributed fashion. All that would be needed is a common machine-readable encoding of the representation proposed here and some method of aggregating such representations across many researchers. With such a characterization of the field, we could, for example, trace all models that are fine-tuned versions of a particular Qwen backbone, or all models that have used MS MARCO or BEIR. Some of this information is already captured in Hugging Face metadata, but we can imagine an agent semi-automatically enriching the ecosystem description by scanning the latest literature to extract vertex and hyperedge information.

This representation could support agents that identify unexplored combinations of artifacts and recipes, propose new experiments, and execute feasible candidates subject to resource and evaluation constraints. We could further close the loop with automatic evaluation harnesses and give agents autonomy in “growing” the hypergraph according to some high-level goal. This might provide a concrete substrate for automated AI scientists or for recursive self improvement within a narrowly specified scope.

## 3.3 Other Considerations

Who traverses the edges? Viewing model training as traversing hyperedges from tails to heads does not necessarily mean that we have exhaustively traversed every edge ourselves. Furthermore, it also does not necessarily mean that a specific hyperedge traversal is guaranteed to be “successful” (more discussion below). It is important to untangle questions about openness and sovereignty (the first) from questions about reproducibility (the second), which are orthogonal.

Openness and sovereignty focus on the availability of resources (data, code, etc.) to transform one artifact into another. Whether the data and recipes “work as advertised” is a (different) question of reproducibility. They are completely independent. To see why, consider the hypothetical case in which an organization shares its data and code with another under a non-disclosure agreement. In this case, the results may be reproducible (by the second organization), but not open since the data and code are not publicly available.

The success of a specific hyperedge traversal is a question of reproducibility, and in this work we make judgment calls. As a specific example, in one set of experiments we start with a model checkpoint shared by Andrej Karpathy, which he asserts as having been pre-trained on a particular dataset with code in a specific repo. Although we have not actually run his code from the dataset, we trust that his complete recipe works as claimed. That is, we have not independently verified reproducibility, but in this case, we feel that trust is warranted.

Similarly, readers of our report or those wishing to use our results do not necessarily need to traverse every hyperedge themselves. Instead, they can take advantage of any vertex along a traversal (i.e., directly use the model weights or any intermediate artifacts). As discussed above, whether a particular code repository accurately reflects the putative hyperedge is a question of reproducibility. Someone can indeed “trust us”, for example, that our claimed approach to fine-tuning does indeed yield a particular level of effectiveness on benchmark datasets, or not trust our claims and try out the method independently (i.e., a reproduction attempt). This is possible since by our definition every vertex and hyperedge must be publicly available. Thus, a fully open and sovereign ecosystem is maximally flexible. Anyone can take advantage of an artifact directly (i.e., download model weights and perform inference) or trace an artifact upstream to rerun the experiments that generated it. In an open ecosystem, anyone can consider alternatives by branching off another hyperedge and sharing the resulting artifacts, which represents a discrete contribution to our collective knowledge.

What’s our standard of reproducibility? In the ideal case, repeated traversals of the same hyperedge should yield identical results. This is a realistic assumption for deterministic transformations. However, we expect that, in many cases, repeated traversals of the same hyperedge yield “substantively” similar, but not necessarily identical, results.

For model training, even with the same inputs, code, configuration, random seed, etc., unavoidable differences in hardware, software environments, and nondeterministic operations in training will likely lead to different model weights (see explorations in Section 6.2). Thus, our notion of reproducibility does not require output artifacts to be byte-wise identical.<sup>11</sup> Strictly speaking, multiple hyperedge traversals are unlikely to produce the same exact output artifact(s); vertices in our LLM training hypergraph thus capture the output of a specific execution instance.

Instead, we adopt a more “relaxed” assumption about reproducibility. Accepting that byte-identical artifacts are unlikely, rerunning a computation should nevertheless reproduce its associated experimental claims substantively, for example, comparable effectiveness on the reported benchmarks, within a reasonable range of variation. Our use of vague weasel terms such as “substantive”, “reasonable”, etc. is deliberate, as we will need to exercise judgment, but the community will ultimately assess the veracity of our claims. Along the way, we will adopt commonly accepted standards, such as quantifying effectiveness variations through multiple trials and referencing existing evaluations already documented in the literature. Ultimately, the public availability of all artifacts (vertices) and code (hyperedges) makes third-party audits of our claims possible.

What about the use of synthetic data, proprietary teachers, etc.? Synthetic data, teacher guidance, and other techniques can significantly increase model effectiveness. The obvious concern is that they introduce dependencies (both vertices and hyperedges) on proprietary APIs or models, or on models that are themselves not fully open and sovereign. What do we do? Here, we attempt to adopt a practical position and exercise judgment on a case-by-case basis—but this is an issue that we frequently grapple with throughout the project.

Consider a concrete case: Is the use of synthetic data compatible with our notion of openness and sovereignty? Let’s assume that the dataset is made publicly available, but was generated by a proprietary model. Let’s even assume that the associated code was made available (seeds, prompt templates, etc.) but that the data generation process itself requires calling a proprietary API. Would we consider a model derived from such a dataset fully open and sovereign? Does the answer to the above question change if the dataset were generated by an open-weight model, in which case it would be possible to recreate the dataset independently? The same question applies to teacher guidance and other side information, including labels, rankings, scores, and logits.

In short, we are not committing to a specific conclusion in this report, because the actual answer is likely to be organization- and context-dependent. Nevertheless, we suspect that an important consideration is whether the use of proprietary or non-open models represents a one-time process (e.g., whose results are captured statically in a dataset) or requires continued access during training (e.g., to determine rewards). However, our commitment to making vertices and hyperedges publicly available allows others to reach their own conclusions about openness and sovereignty. Furthermore, the transparency of our approach allows others to examine the impact of proprietary and non-open components via ablation studies.

## 3.4 Why is this important? The Organizational Perspective

Sovereign AI encompasses many overlapping and closely related concepts, centered on the idea that organizations—and, by extension, countries—should retain control over their use and governance of AI. One aspect of sovereignty is the ability to independently build, deploy, and maintain AI infrastructure without requiring the continued cooperation of an external provider. It is this aspect of sovereignty, focused specifically on models, that we focus on here.

The ability of organizations to build and deploy their own models enables autonomy and mitigates risks arising from external dependencies. Reliance on an external model provider exposes an organization to changes in pricing, terms of service, permitted uses, and even continued access. This is not an abstract concern; recent episodes have raised alarms with many leaders. On June 12, 2026, Anthropic suspended access to Fable 5 and Mythos 5 for all customers following a US government directive restricting access by foreign nationals, including those within the United States.<sup>12</sup> Although these export controls were subsequently lifted,<sup>13</sup> the disruption illustrates how organizations can lose access to critical capabilities through decisions outside their control.

Geopolitical risks aside, an organization can be at the mercy of an external provider that determines the availability of certain capabilities for particular applications. Although such dependencies invariably exist with any external service, the increasing importance of AI in all aspects of an organization’s operations increases the risk from not “owning the means of production”. To provide a concrete example: At its launch, Anthropic’s Fable 5 model routed requests flagged by its safeguards— including requests concerning biology and cybersecurity—to a less capable model.<sup>14</sup> Regardless of the motivation, such restrictions mean that an organization’s access to certain capabilities may depend on the provider’s policy, over which the organization may have limited control.

Use of an external model provider also raises questions about control over an organization’s proprietary knowledge in the form of prompts, retrieved documents, and other input that may contain confidential information. Palantir [2026] advises that an organization should insist that model providers operate under a Zero Data Retention (ZDR) agreement, a privacy concept whereby none of an organization’s data in the form of prompts, outputs, and telemetry is retained beyond the ephemeral processing required to fulfill the request. However, the report goes on to argue that ZDR may not be sufficient as there are other channels for “leakage”, and that ZDR does not guarantee continued access, tying in to the other concerns expressed above.

In this context, our notion of fully open and sovereign models gives an organization greater control over all aspects of a model’s lifecycle, from end to end. Access to model weights alone (i.e., building on existing open-weight models) may not be sufficient to achieve a desired level of control that an organization would be comfortable with. Developers of open-weight models make checkpoints available, but they do not necessarily disclose the raw pre-training corpora or other important aspects of the training pipeline, including data processing, alignment, the mid- and post-training phases, etc. While open-weight models certainly provide useful starting points for deployment, they leave important upstream dependencies unresolved.

We identify two main challenges with open-weight models:

First, incomplete provenance makes it difficult to assess whether the model was trained in a way that satisfies an organization’s requirements in terms of data licensing, data privacy, data quality, etc. Furthermore, it can be challenging to assess alignment and whether model behavior is consistent with an organization’s values. For example, it is well known that LLMs can exhibit systematic political and cultural biases associated with their training data and development choices [Feng et al., 2023, Buyl et al., 2024]. For example, a Chinese open-weight model may deny the historic reality of the 1989 Tiananmen Square protests and massacre to reflect the position of the Chinese Communist Party. More insidious are backdoors that can be embedded in LLMs [Kurita et al., 2020, Xu et al., 2024, Hubinger et al., 2024]. Both of these issues persist even if an organization performs model inference itself in isolation, eliminating the possibility that the model communicates with the original developers. Although researchers have developed techniques to “re-align” open-weight models [Chen et al., 2026], successful realignment on evaluated behaviors does not establish that all unwanted behaviors or hidden backdoors have been removed. The unknown unknowns remain a concern.

Second, there is no guarantee that the original developers of an open-weight model will continue to release improved checkpoints under the same license. Future releases might remain proprietary, adopt more restrictive terms, or prioritize capabilities that do not match an organization’s needs. An organization can continue to use an existing checkpoint and improve it through further training, but access to weights alone severely restricts the types of experiments and modifications that can be performed. For example, an organization might wish to rebuild the model using a different data mixture, change the pre-training objective, or investigate a different model architecture. Such changes require access to upstream artifacts and training recipes that the original developers may not have released. Thus, an organization whose development strategy relies primarily on continued training or fine-tuning successive third-party checkpoints might find itself in the model equivalent of a “dead end” if those releases cease or become unsuitable. In this case, further independent progress becomes difficult without substantial effort.

As discussed above, the increasingly critical role that models play in an organization requires effective leaders to carefully weigh the benefits and costs of external dependencies, and to prepare risk mitigation strategies in advance. In summary, fully open and sovereign models alleviate many concerns about open-weight models and allow organizations to retain control “of their own destiny”.

## 3.5 Why is this important? The Academic Perspective

The previous narrative focuses on openness and sovereignty from the perspective of organizational control. While the ability to build, deploy, and govern AI without external interference is undoubtedly important, such discussions can seem abstract from the academic perspective.<sup>15</sup> Building on previous discussions, we outline two concrete ways in which fully open and sovereign models are important:

Rigorous understanding requires transparency. As an academic research group, we are committed to increasing our understanding (of the world, more broadly, and LLMs, more narrowly) in a rigorous manner, and often that cannot be accomplished without transparency into the artifacts that we are studying.

In our recent NanoKnow project [Gu et al., 2026], we asked a simple question: How do LLMs know what they know? This question is difficult to answer because pre-training data is either unknown or inaccessible, for both proprietary models and many open-weight models as well. In a “closed-book” setting, we do not know if an LLM knows a fact because it simply memorized the fact from the pre-training corpus, or perhaps the model made a multi-hop inference based on other memorized facts. In a RAG setting, where additional context is injected into the model’s prompt, we cannot disentangle the effects of parametric knowledge from the grounding text provided to the model. Similar questions apply to Hypothetical Document Embeddings (HyDE) [Gao et al., 2023], a popular technique where an LLM is asked to generate a “fake” hypothetical document that is then used as a pivot for searching “real” documents in an actual collection. When evaluating models using a benchmark test collection, we don’t know if HyDE is effective because it generalized relevance across different information needs or merely because it had memorized relevant content for the test queries [Yoon et al., 2025].

To summarize, rigorously understanding how models answer questions is often not possible without transparency—in these cases, access to the pre-training corpus. Transparency provides a starting point to understand the interplay between memorization and generalization in LLMs. While transparency alone is insufficient, access to the pre-training corpus enables controlled interventions that connect data to observed behavior. These experiments are not possible with proprietary or even open-weight models. Unless we are able to train models from scratch, many lines of investigation will remain closed, and along with them, a complete, rigorous understanding of LLM behavior.

Transparency naturally translates into interventions that address many concerns with proprietary and open-weight LLMs. Does a portion of the pre-training corpus violate an organization’s requirements in terms of data licensing, privacy, data quality, etc.? If so, the solution is straightforward: remove the data and retrain a new model. Does the dataset used for instruction fine-tuning contain instances that are inconsistent with an organization’s values? If so, the solution is straightforward: remove the offending instances and retrain a new model. Of course, it is not possible to assure that such ablations maintain model output quality, but these interventions give an organization confidence that specifically identified problems are addressed. Such options are typically not available for proprietary and open-weight models. Transparency with fully open and sovereign models allows one to study the impact of interventions in a controlled manner, thereby contributing to a rigorous understanding of model behavior.

End-to-end training provides valuable educational opportunities. Another cornerstone of the academic mission is to train the next generation of scholars, scientists, and researchers who will shape the future. Richard Feynman’s famous quote is apt here: “What I cannot create, I do not understand”. Our hypergraph organization of LLM training and the view of reproducibility as the traversal of hyperedges provide valuable educational opportunities for students. There is no substitute for “doing it oneself”, and as previously articulated elsewhere [Lin, 2022]: reproducibility is the starting point for “side quests” that yield greater understanding of the phenomena under examination. Our hypergraph formulation provides a precise way for students to think through individual steps in a much larger end-to-end training recipe and to explore how local differences may affect overall quality (e.g., in terms of retrieval metrics) via ablations and other contrastive conditions.

Starting from finished checkpoints hides many decisions that are pedagogically important, which we hope that students learn to recognize and question. How was the pre-training corpus assembled? How was the final pre-training data mix arrived at? Where did the training examples come from? What is the quality of the manually provided judgments? What is the impact of filtering training examples? Working through the pipeline and methodically traversing each hyperedge gives students opportunities to question decisions that are captured in existing artifacts and ponder the consequences of alternatives. A small model trained end to end can provide educational value that is difficult to obtain from fine-tuning a much more capable model whose development is inaccessible.

The educational value also extends beyond successfully executing a recipe. When a reproduction fails to match the reported result, students must formulate hypotheses, inspect intermediate artifacts, and design experiments that distinguish competing explanations. They must learn to separate implementation errors from differences in configuration, data, or random variation. With limited computing resources, they must also decide which experiments are potentially most informative. These are important research skills, and a transparent training pipeline provides a concrete setting in which to develop them.

Finally, our hypergraph formulation supports collaboration across students and projects, and even potentially across institutions. One student might investigate a data-processing step, while another studies a training objective, and a third explores a variant model architecture. Shared intermediate artifacts allow students to build on each other’s work without independently repeating every computation—yet retaining the ability to examine the impact of one’s decision in an end-to-end task. This provides a possible answer to the question posed in Section 3.3: Who traverses the hyperedges? The answer: individual students, potentially across projects, advisors, and even institutions—in a distributed but coordinated fashion. The resulting artifacts become both potential research contributions and educational resources for the next group of students.

## 3.6 Openness and Sovereignty

Thus far, we have advocated for fully open and sovereign models as a coherent, singular goal. We conclude this section by distinguishing openness from sovereignty and explaining why Project Greenhouse pursues both. The two concepts are related, but neither implies the other. An organization can retain control over its models without making its data, code, or training recipes public. A frontier lab can be sovereign in this sense by controlling the entire model development process from end to end while keeping its methods proprietary. Conversely, the public availability of artifacts does not by itself ensure that another organization has the compute, expertise, or other resources needed to independently rebuild and modify a model. Openness concerns accessibility, whereas sovereignty concerns control.

Project Greenhouse is formulated around openness and sovereignty because our goals depend simultaneously on both. Openness ensures accessibility, which is a prerequisite for the transparency needed to facilitate explorations that lead to rigorous understanding. As an academic institution, we are committed to sharing findings from our efforts and contributing to publicly accessible knowledge in an open manner. On the other hand, sovereignty concerns control over dependencies, which affect the bounds of what is knowable for us. The inability to inspect certain upstream artifacts restricts the types of explorations that we can pursue. Access to all stages of the model development process enables the rigorous understanding of model behavior that we desire.

Together, openness and sovereignty support the rigorous understanding and educational opportunities discussed above. Our emphasis on modest computational requirements further seeks to make these opportunities feasible for others. However, we are cognizant of the limitations of our approach: it is unrealistic to imagine competing with an industry lab across a diverse range of capabilities such as general-purpose reasoning, coding, drug discovery, financial analysis, etc. Instead, we have decided to focus on agentic search, which is broad enough to “matter” but yet narrowly scoped enough that we can make meaningful contributions. We present one such small, initial contribution.

## 4 Headline Results

This report describes how we built Gaggle (base reranker), our pointwise decoder-only reranker, by pre-training a causal language model from scratch on the ClimbMix corpus [Diao et al., 2025] and then fine-tuning the base checkpoint with publicly available query–document relevance annotations curated in RLHN-250K [Thakur et al., 2025] using localized contrastive estimation (LCE) [Gao et al., 2021, Pradeep et al., 2022]. This two-step recipe depends only on commonly available datasets and gives us end-to-end control over model training from initialization through relevance fine-tuning, providing a concrete step toward fully open and sovereign search models.

We were able to accomplish everything described in this report with modest computational resources, using only a handful of GPUs. All of our supervised fine-tuning experiments can run on a single modern GPU; we typically use one or two GPUs in practice. For pre-training, the most resourceintensive part of our efforts, we were able to procure limited compute cycles on a single server with 8× H100 GPUs.

Our headline result involves these two model checkpoints:

• gaggle-nanochat-pretrained-climbmix-20260924 is our pre-trained causal language model checkpoint based on the nanochat codebase, detailed in Section 5.

• gaggle-reranker-20261005 is the checkpoint of Gaggle (base reranker), a model soup [Wortsman et al., 2022] formed by averaging model weights from four fine-tuning trials using the same supervised fine-tuning recipe, detailed in Section 6.

All headline results presented in this section for Gaggle (base reranker) are from evaluating the single model soup checkpoint.

## 4.1 Experimental Setup

We implemented a relatively standard multi-stage ranking pipeline [Nogueira et al., 2019] comprised of first-stage (top-k) retrieval that returns a ranked list of |k| documents from the corpus of interest. In our experiments, we used BM25 [Robertson and Zaragoza, 2009] with k = 100 for first-stage retrieval, as implemented with Lucene in the Pyserini toolkit [Lin et al., 2021]. Candidate documents from first-stage retrieval are then reranked, as detailed below.

We adopt the formal definition of pointwise reranking in Section 2.2. The model can be viewed as a function $\mathcal { R } ( q , d _ { n } ) = s _ { n }$ that scores each $( q , d _ { n } )$ pair independently, where $q$ is the query and $d _ { n }$ is a document returned by the first-stage retriever, $\bar { n } \in \{ 1 \ldots \bar { k } \}$ . The reranking model is trained such that reordering the list of |k| documents by $s _ { n }$ is likely to improve effectiveness as measured by some standard metric.

For evaluation, we followed standard practice and measured nDCG@10 on test collections from the TREC Deep Learning Tracks 2019–2023 [Craswell et al., 2019, 2020, 2021, 2022, 2023] (DL19–23 for short) and seven BEIR collections [Thakur et al., 2021]: TREC-COVID, TREC-News, Robust04, NFCorpus, SciFact, SCIDOCS, and FiQA. We evaluated each model once per collection, and for each evaluated model, we computed an unweighted mean of its collection-level scores within each benchmark group (TREC DL and BEIR). In our experiments, the documents and queries were truncated to 512 and 128 tokens, respectively.

We fine-tuned our reranker on RLHN-250K [Thakur et al., 2025], which includes MS MARCO, FiQA, and SCIDOCS-RR among its seven training sources. Following the original RLHN paper, four out of our seven BEIR evaluation collections—TREC-COVID, TREC-News, Robust04, and NFCorpus—can be considered out of domain with respect to this supervised fine-tuning dataset.

We compared our Gaggle (base reranker) with the following conditions:

• BM25: the first-stage retrieval baseline using BM25 scores [Robertson and Zaragoza, 2009], implemented with Pyserini [Lin et al., 2021], evaluated without reranking.

• Oracle reranker: we sorted each top-100 BM25 candidate list by decreasing relevance grade from the query relevance judgments (qrels), producing an ordering in perfect agreement with the available relevance judgments. Unjudged documents received grade zero, and documents with equal grades retained their BM25 order. This provides the maximum nDCG@10 score achievable by reranking those candidates under the same qrels.

• Out-of-the-box listwise rerankers: Gemma-4-26B-A4B [Gemma Team, 2026] (26B total parameters, 4B active), Qwen3.8-27B [Qwen Team, 2026b] (27B parameters), and GPT-6.1 Sol were used as listwise rerankers applied to the top-100 BM25 candidates. For Gemma and Qwen, we used zero-shot, non-thinking generation with windows of 20 candidates, a stride of 10, and a 16,384-token context. For GPT-6.1 Sol, we used zero-shot generation with minimal reasoning, windows of 20 candidates, and a stride of 10. Appendix A provides the RankLLM [Sharifymoghaddam et al., 2025] prompt used for these baselines and the Qwen-specific directive omitted for Gemma and Sol.

• Jev (shared-rubric scoring): we used the jev-latest model from TypeSafe AI<sup>16</sup> to score documents using the four-level TREC relevance-grade rubric, supplying the query, documents, and rubric together and sorting by each document’s expected grade. We supplied all top-100 BM25 candidates in a single request per query. Jev appears only in the TREC DL comparisons; Appendix B provides the prompt and scoring rule.

• Jina-Reranker-v3: a 0.6B-parameter fine-tuned listwise reranker based on Qwen3 [Wang et al., 2025].<sup>17</sup> The model jointly encodes the query and candidate documents with causal attention and computes relevance scores using contextual embeddings from each document’s final token. We supplied all top-100 BM25 candidates to the public reranking implementation, which processed them in groups of up to 64 documents.

• Other fine-tuned listwise rerankers: RankVicuna [Pradeep et al., 2023a], RankZephyr [Pradeep et al., 2023b], and FirstMistral [Chen et al., 2025], each with 7B parameters, were applied to rerank top-100 BM25 candidates. RankVicuna and RankZephyr were fine-tuned on MS MARCO queries and candidate passages using teacher-generated rankings: GPT-3.5 for RankVicuna, and GPT-3.5 followed by GPT-4 for RankZephyr. FirstMistral ranked candidates by sorting the logits of their identifier tokens at the first output position. These models appear only in the TREC DL comparisons, with results copied from RankLLM.

• MonoT5: we used the 3B-parameter checkpoint of MonoT5 [Nogueira et al., 2020],<sup>18</sup> fine-tuned on MS MARCO for 10,000 steps. MonoT5 scored the top-100 BM25 candidates independently with a 512-token input limit.

• Rank0 and Rank1: two Qwen2.5-14B pointwise rerankers trained on the MS MARCO data mix from Rank1 [Weller et al., 2025],<sup>19</sup> which contains query–passage pairs paired with target reasoning chains. Rank1 was trained to generate a reasoning chain before computing a relevance score, while Rank0 [Jedidi et al., 2026] was trained on the same data but without the reasoning chains. Both models rerank the top-100 BM25 candidates. These models appear only in the TREC DL comparisons, with results copied from Jedidi et al. [2026].

• Qwen3-Reranker: 4B and 8B pointwise rerankers fine-tuned from Qwen3 backbones [Zhang et al., 2025b], applied to rerank top-100 BM25 candidates.

• RLHN Qwen2.5-3B: the pointwise reranker released by Thakur et al. [2025],<sup>20</sup> applied to rerank top-100 BM25 candidates. The model was trained on the RLHN-680K data, which we discuss in Section 7.2.

<table><tr><td>Model</td><td></td><td>Size</td><td>DL19</td><td>DL20</td><td>DL21</td><td>DL22</td><td>DL23</td><td>Mean</td></tr><tr><td>1 BM25 [Lin et al., 2021]</td><td></td><td></td><td>0.506</td><td>0.480</td><td>0.446</td><td>0.269</td><td>0.263</td><td>0.393</td></tr><tr><td>2</td><td>Oracle reranker</td><td></td><td>0.892</td><td>0.871</td><td>0.835</td><td>0.678</td><td>0.669</td><td>0.789</td></tr><tr><td colspan="9">Out-of-the-box listwise rerankers</td></tr><tr><td>3</td><td>Gemma-4 [Gemma Team, 2026]</td><td>26B</td><td>0.743</td><td>0.705</td><td>0.723</td><td>0.516</td><td>0.497</td><td>0.637</td></tr><tr><td></td><td>4 Qwen3.8 [Qwen Team, 2026b]</td><td>27B</td><td>0.732</td><td>0.705</td><td>0.716</td><td>0.520</td><td>0.492</td><td>0.633</td></tr><tr><td>5 Jev</td><td></td><td>?</td><td>0.736</td><td>0.715</td><td>0.721</td><td>0.505</td><td>0.476</td><td>0.631</td></tr><tr><td></td><td>6 GPT-6.1 Sol</td><td>?</td><td>0.750</td><td>0.729</td><td>0.725</td><td>0.548</td><td>0.513</td><td>0.653</td></tr><tr><td colspan="9">Fine-tuned listwise rerankers</td></tr><tr><td></td><td>7† RankVicuna [Pradeep et al., 2023a]</td><td>7B</td><td>0.672</td><td>0.655</td><td>0.625</td><td>0.435</td><td>0.419</td><td>0.561</td></tr><tr><td></td><td>8† RankZephyr [Pradeep et al., 2023b]</td><td>7B</td><td>0.741</td><td>0.711</td><td>0.701</td><td>0.511</td><td>0.442</td><td>0.621</td></tr><tr><td></td><td>9† FirstMistral [Chen et al., 2025]</td><td>7B</td><td>0.725</td><td>0.701</td><td>0.685</td><td>0.489</td><td>0.443</td><td>0.609</td></tr><tr><td></td><td>10 Jina-Reranker-v3 [Wang et al., 2025]</td><td>0.6B</td><td>0.715</td><td>0.696</td><td>0.689</td><td>0.503</td><td>0.453</td><td>0.611</td></tr><tr><td colspan="11">Fine-tuned pointwise rerankers</td></tr><tr><td></td><td>11 MonoT5 [Nogueira et al., 2020]</td><td>3B</td><td>0.719</td><td>0.689</td><td>0.665</td><td>0.497</td><td>0.450</td><td>0.604</td></tr><tr><td></td><td>12† Rank0 [Jedidi et al., 2026] 13† Rank1 [Weller et al., 2025]</td><td>14B</td><td>0.733</td><td>0.687</td><td>0.707</td><td>0.496</td><td>0.473</td><td>0.619</td></tr><tr><td></td><td></td><td>14B</td><td>0.663</td><td>0.650</td><td>0.631</td><td>0.448</td><td>0.417</td><td>0.562</td></tr><tr><td></td><td>14 Qwen3-Reranker [Zhang et al., 2025b]</td><td>4B</td><td>0.739</td><td>0.693</td><td>0.682</td><td>0.505</td><td>0.456</td><td>0.615</td></tr><tr><td>15 16</td><td>Qwen3-Reranker [Zhang et al., 2025b]</td><td>8B</td><td>0.731</td><td>0.665</td><td>0.663</td><td>0.512</td><td>0.454</td><td>0.605</td></tr><tr><td>17</td><td>RLHN Qwen2.5 [Thakur et al., 2025]</td><td>3B</td><td>0.742</td><td>0.686</td><td>0.706</td><td>0.487</td><td>0.456</td><td>0.615</td></tr><tr><td></td><td>Gaggle (base reranker)</td><td>3B</td><td>0.757</td><td>0.723</td><td>0.714</td><td>0.533</td><td>0.480</td><td>0.641</td></tr></table>

Table 1: Headline results on TREC Deep Learning 2019–2023, reporting nDCG@10 on reranking top-100 BM25 candidates. Our pointwise reranker—Gaggle (base reranker)—is shown in row 17. A row number annotated with the <sup>†</sup> symbol marks results reported in prior work and copied here directly; unmarked rows capture numbers from our own evaluations. Results in rows 7–9 were taken from RankLLM [Sharifymoghaddam et al., 2025], and those in rows 12 and 13 were taken from Jedidi et al. [2026]. Mean represents the unweighted average across collections. The highest reported score in each column, including ties, is bolded (excluding the oracle reranker).

<table><tr><td>Model</td><td>Size COVID</td><td></td><td>News</td><td>Robust</td><td></td><td>NFC SciFact</td><td>SCID</td><td>FiQA</td><td>Mean</td></tr><tr><td>1 BM25 [Lin et al., 2021]</td><td></td><td>0.595</td><td>0.395</td><td>0.407</td><td>0.322</td><td>0.679</td><td>0.149</td><td>0.236</td><td>0.398</td></tr><tr><td>2 Oracle reranker</td><td></td><td>0.975</td><td>0.827</td><td>0.837</td><td>0.544</td><td>0.927</td><td>0.443</td><td>0.590</td><td>0.735</td></tr><tr><td colspan="10">Out-of-the-box listwise rerankers</td></tr><tr><td>3 Gemma-4 [Gemma Team, 2026]</td><td>26B</td><td>0.846</td><td>0.492</td><td>0.662</td><td>0.387</td><td>0.805</td><td>0.208</td><td>0.439</td><td>0.548</td></tr><tr><td>4 Qwen3.8 [Qwen Team, 2026b]</td><td>27B</td><td>0.849</td><td>0.505</td><td>0.647</td><td>0.390</td><td>0.803</td><td>0.225</td><td>0.462</td><td>0.554</td></tr><tr><td>5 GPT-6.1 Sol</td><td>?</td><td>0.837</td><td>0.526</td><td>0.668</td><td>0.398</td><td>0.821</td><td>0.237</td><td>0.520</td><td>0.572</td></tr><tr><td colspan="10">Fine-tuned listwise rerankers</td></tr><tr><td>6 Jina-Reranker-v3 [Wang et al., 2025]</td><td>0.6B</td><td>0.821</td><td>0.459</td><td>0.543</td><td>0.359</td><td>0.748</td><td>0.215</td><td>0.430</td><td>0.511</td></tr><tr><td colspan="10">Fine-tuned pointwise rerankers</td></tr><tr><td>7 MonoT5 [Nogueira et al., 2020]</td><td>3B</td><td>0.803</td><td>0.487</td><td>0.560</td><td>0.373</td><td>0.763</td><td>0.191</td><td>0.460</td><td>0.520</td></tr><tr><td>8Qwen3-Reranker [Zhang et al., 2025b]</td><td>4B</td><td>0.853</td><td>0.497</td><td>0.589</td><td>0.374</td><td>0.794</td><td>0.235</td><td>0.445</td><td>0.541</td></tr><tr><td>9 Qwen3-Reranker [Zhang et al., 2025b] 10 RLHN Qwen2.5 [Thakur et al., 2025]</td><td>8B</td><td>0.843 0.858</td><td>0.512 0.465</td><td>0.583 0.588</td><td>0.371</td><td>0.794</td><td>0.225</td><td>0.456</td><td>0.541</td></tr><tr><td></td><td>3B</td><td></td><td></td><td></td><td>0.312</td><td>0.768</td><td>0.217</td><td>0.431</td><td>0.520</td></tr><tr><td>11 Gaggle (base reranker)</td><td>3B</td><td>0.859</td><td>0.515</td><td>0.592</td><td>0.380</td><td>0.790</td><td>0.236</td><td>0.455</td><td>0.547</td></tr></table>

Table 2: Headline results on seven BEIR collections, reporting nDCG@10 on reranking top-100 BM25 candidates. Our pointwise reranker—Gaggle (base reranker)—is shown in row 11. All results are from our own evaluations. Mean represents the unweighted average across collections. The highest reported score in each column, including ties, is bolded (excluding the oracle reranker).

## 4.2 Reranker Effectiveness and Implications

Tables 1 and 2 provide our headline results, showing the effectiveness of our pointwise reranker Gaggle (base reranker) on TREC DL and BEIR in terms of nDCG@10. Model sizes in billions of parameters are provided as a point of reference. In the BEIR table, COVID, News, Robust, NFC, and SCID abbreviate TREC-COVID, TREC-News, Robust04, NFCorpus, and SCIDOCS, respectively.

The comparisons include BM25, the oracle, out-of-the-box listwise models, and fine-tuned pointwise and listwise rerankers as described above. Across both benchmark groups (TREC DL and BEIR), Gaggle (base reranker) achieves competitive reranking effectiveness and improves over the BM25 baseline on every collection. Furthermore, the mean nDCG@10 of Gaggle (base reranker) exceeds those of all fine-tuned pointwise and listwise conditions that we compare against. On TREC DL, only GPT-6.1 Sol achieves higher effectiveness. On BEIR, Gaggle (base reranker) achieves effectiveness comparable to Gemma-4, but appears to score lower than Qwen3.8 and GPT-6.1 Sol. These results place our pointwise reranker, at 3B parameters, favorably among the comparison conditions, including larger models used for out-of-the-box listwise reranking. It is not a surprise that our model is not as effective as GPT-6.1 Sol, a frontier model.

The collection-level results reveal variation that is not captured by the benchmark means: Gaggle (base reranker) achieves the highest reported non-oracle nDCG@10 on DL19, even higher than GPT-6.1 Sol. On BEIR, Gaggle (base reranker) has higher nDCG@10 than GPT-6.1 Sol on TREC-COVID, and comparable effectiveness on SCIDOCS under the noise guideline, while Sol scores higher on the remaining collections. Our reranker is among the strongest models on TREC-COVID, TREC-News, and SCIDOCS.

The score variability study in Section 6.2 finds run-to-run sample standard deviations of approximately 0.001 for benchmark means and 0.003 on average for individual datasets. Score differences of roughly this magnitude should be treated as experimental noise.

The oracle upper bound is shown in Row 2 of both result tables, i.e., the maximum possible scores with top-100 BM25 candidate documents. This gap may reflect scope for reranking improvements as well as noise from incomplete or incorrect relevance judgments. Further investigation is needed to distinguish these effects.

In support of our central claim in this report—corresponding to the first project milestone of Project Greenhouse—these results show that supervised fine-tuning using a publicly available dataset can produce a competitive reranker from a model checkpoint that we pre-train from scratch, without requiring an intermediate instruction-tuning or alignment stage, mid- or post-training, teacher supervision, synthetic data, or SFT hyperparameter tuning. Our two-step recipe puts us in control of the entire training process, end to end.

Most importantly—and contrary to the dominant approach adopted by information retrieval researchers today—we show that it does not appear necessary to start from an existing open-weight model backbone to produce a competitive pointwise reranker. The comparisons in Tables 1 and 2 assess overall reranking effectiveness across models with different sizes, data mixtures, and training and inference procedures. Thus, we are not able to isolate any specific factors other than to provide a feasibility demonstration that provides the first step towards validating our broader thesis.

Naturally, this headline result raises many interesting questions about the effects of different backbones, different training datasets, and other architectural design choices. We present contrastive conditions that examine some of these questions in Section 7, but leave many explorations for future work as we continue to pursue the broader objectives of Project Greenhouse.

## 5 Pre-Training

Our pre-trained model backbones were produced with Andrej Karpathy’s nanochat,<sup>21</sup> a compact, full-stack codebase for training decoder-only (GPT-style) language models, using the original implementation in its depth-34 configuration. We kept nanochat’s model definition, optimizer, data pipeline, and evaluation code unchanged; modifications for pre-training our own backbones in this work involved altering the batch-size schedule.

## 5.1 Experimental Setup

Model architecture. We used nanochat’s depth-34 architecture without modification: a decoderonly transformer with 34 layers, model width 2,176, and 17 attention heads of dimension 128, trained with a 2,048-token context.

Our approach combined standard components (rotary position embeddings, QK normalization, parameter-free RMSNorm, a squared-ReLU MLP, untied input and output embeddings, and logit soft-capping) with several lightweight additions from recent speedrun-style designs.<sup>22</sup> Every other layer has a value embedding, which is a per-token lookup table whose output is mixed into the attention values through a learned gate, adding token-level capacity at almost no compute cost. Three of every four layers attend within a 512-token sliding window, while the remaining layers attend over the full context. Learned per-layer scalars rescale the residual stream and mix the input embedding back into every layer.

In total, our model backbone has 3.29B parameters: 1.93B in the transformer blocks, 1.21B in the value-embedding tables, and 0.14B in the input and output embeddings. Because value embeddings are lookups, the total per-token training compute is comparable to that of a dense model with about 2B parameters.

Data and tokenizer. We pre-trained from scratch on the ClimbMix corpus [Diao et al., 2025], using Karpathy’s document-shuffled release (karpathy/climbmix-400b-shuffle). We held out the last of its 6,543 shards for validation and trained on the rest, packing documents into 2,048-token sequences with nanochat’s BOS-aligned best-fit packing, in which every sequence starts at a document boundary. Following nanochat’s recipe, we trained a byte-level BPE tokenizer with a vocabulary of 32,768 entries on the first 2B characters of the training split.

Training. We kept nanochat’s optimizer and learning-rate schedule unchanged: Muon<sup>23</sup> updates the weight matrices of the transformer blocks and AdamW updates the embeddings and scalar parameters. Our only modifications were to change the learning rate and the batch size schedule.

Unlike nanochat, which scales its peak learning rates with the square root of the batch size, we kept their peak values unchanged as the batch grows, so each increase cuts the gradient noise much as a learning rate decay would. Early in pre-training, since the model is not very good, even a small data batch clearly shows the direction of an effective update; in these situations, a larger batch may provide limited additional benefit per token, while consuming more tokens per step. Thus, using small data batches at the beginning of pre-training is more data efficient.

As the loss decreases, however, the signal-to-noise ratio of each batch decreases, so larger batches start to become worthwhile [McCandlish et al., 2018]. Therefore, we doubled the batch size three times, from 262,144 to 524,288, 1,048,576, and 2,097,152 tokens over the course of training, which also reduced the total number of optimizer steps. We kept the learning rates fixed throughout the pre-training process, so each increase of the batch size cuts the gradient noise, much as a learning-rate decay would [Smith et al., 2018]. These batch size increases correspond to the discontinuities in the validation loss plot in Figure 1. For example, we see that the validation loss drops suddenly, with a brief spike at the first batch size increase.

In total, we pre-trained for exactly one pass over the entire corpus (comprising 287B tokens), on a single server with 8× NVIDIA H100 GPUs with FP8 matrix multiplications. The entire process consumed roughly 1,600 GPU-hours in this configuration.

Bidirectional variant. Our GPU budget did not allow for a second pre-training run from scratch, so we adapted the causal model to bidirectional attention following LLM2Vec [BehnamGhader et al., 2024]. Starting from the causal checkpoint at 245B tokens, we continued pre-training on the remaining 42B tokens with masked next-token prediction (MNTP). MNTP predicted each masked token from the preceding position while allowing attention to context on both sides. Each prediction therefore used the same position as in causal next-token training. Since the checkpoint had no saved optimizer state, we raised the learning rate from zero to 0.26 times the peak over the first 200 steps, where the causal run’s schedule had reached, and then followed that schedule’s linear decay to 0.05 times the peak. Adaptation used approximately 19,900 steps, a 20% masking rate, and symmetric attention windows. The adaptation consumed approximately 340 H100-hours.

![](images/5f39a820617dd1b18c560ad9a0c263d72fe19c4b96a9cc25b272c9d05cdae901.jpg)  
Figure 1: Validation loss on held-out ClimbMix during pre-training.

## 5.2 Model Evaluation

It is, of course, desirable to assess the quality of a pre-trained model intrinsically, without consideration of any specific downstream task. To evaluate base pre-trained models without post-training, it is necessary to use benchmarks that do not require instruction following:

• CORE [Li et al., 2024]: 22 in-context learning tasks from DataComp-LM, covering world knowledge, commonsense reasoning, language understanding, reading comprehension, and symbolic problem solving. CORE averages accuracy on each task after rescaling the score so that random guessing scores 0 and perfect accuracy scores 1. We used nanochat’s implementation with up to 500 test questions per task.

• MMLU [Hendrycks et al., 2021]: four-option multiple-choice questions on 57 subjects. We used 5-shot prompts and picked the answer letter (A–D) that the model rates as most likely, using the LM Evaluation Harness.<sup>24</sup>

• MMLU-Pro [Wang et al., 2024]: a harder, ten-option version of MMLU, scored the same way. We do not use the more common chain-of-thought setting, where the model writes out its reasoning before answering, because its prompts do not fit in our 2,048-token context.

Models evaluated locally used the same evaluation code, with each tokenizer’s default special tokens and each model’s own context length. Our bidirectional variant, which would otherwise see the answer, is scored with the answer tokens masked, as during model pre-training. In cases where a model developer directly reports the relevant metric, we use the published scores; otherwise, we performed evaluations ourselves.

To confirm the correctness of our evaluation code, we validated against published results. On seven reference models with published CORE scores, our CORE is within 0.011 of published values. These are: GPT-2 at four sizes [Radford et al., 2019], OLMo-1B [Groeneveld et al., 2024], phi-1.5 [Li et al., 2023], and Qwen2-1.5B [Yang et al., 2024]. Our MMLU and MMLU-Pro scores are within 0.003 of the stock harness, which in turn reproduces published chain-of-thought MMLU-Pro scores to within 0.02. CORE fluctuates slightly late in training, so we released the causal checkpoint with the highest CORE (283B tokens) rather than the final model (287B tokens, 0.004 lower). It is this model checkpoint that we release and evaluate throughout this report.

Table 3 compares our models with well-known base models of up to about 7B parameters released in the past year, grouped by whether their pre-training corpus is public. For reference, the table also includes AI2’s two latest OLMo base models, whose pre-training corpus is publicly available, but which are only available at 7B.

Based on our evaluation, our causal model scores much higher than nanochat-d34, which was trained with an earlier version of the same codebase on 89B tokens of FineWeb-Edu. This contrast shows the importance of the pre-training corpus and token budget. Other interesting observations: Puro 2B achieves a CORE score similar to ours with about five times as many training tokens. Apart from nanochat-d34, Puro 2B, and OPEN-1B, every model, including all models with undisclosed data, scores higher than ours, but each was trained on 1.7T to 34T tokens or does not report its budget. The models shown in Table 3 differ in data, architecture, and training recipe, so the comparison places our model among existing checkpoints rather than isolating any single factor. Our primary goal here is to support the claim that our model checkpoint, pre-trained from scratch ourselves, provides a “reasonable” starting point for subsequent fine-tuning to yield a competitive pointwise reranker.

<table><tr><td>Model</td><td>Data</td><td>Params Tokens</td><td></td><td>CORE</td><td></td><td>MMLU MMLU-Pro</td></tr><tr><td></td><td>1 gaggle-nanochat-pretrained</td><td>ClimbMix</td><td>3.3B</td><td>283B</td><td>0.458</td><td>0.493 0.178</td></tr><tr><td></td><td>2 gaggle-nanochat-pretrained-bidirectional ClimbMix</td><td></td><td>3.3B</td><td>287B 0.339</td><td>0.453</td><td>0.187</td></tr><tr><td colspan="7">Open pre-training data</td></tr><tr><td>3 nanochat-d34</td><td>FineWeb-Edu</td><td>2.2B</td><td>89B</td><td>0.338†</td><td>0.257</td><td>0.115</td></tr><tr><td>4 OPEN-1B [Donaghy et al., 2026]</td><td>OPEN-1B mix</td><td>1.6B</td><td>400B</td><td>0.239</td><td>0.259†</td><td>0.117†</td></tr><tr><td>5 Puro 2B [Luo et al., 2026a]</td><td>Puro mix</td><td>2.0B</td><td>1.4T</td><td>0.464</td><td>0.552</td><td>0.235</td></tr><tr><td>6 Apertus v1.1 4B [Panferov et al., 2026]</td><td>Apertus corpus</td><td>3.8B</td><td>1.7T</td><td>0.495</td><td>0.573</td><td>0.231</td></tr><tr><td>7 K2-Horizon 3.7B</td><td>TxT360-v2 mix</td><td>5.1B</td><td>22.9T</td><td>0.554</td><td>0.664†</td><td>0.380</td></tr><tr><td>8 MiniCPM5-2B</td><td>Ultra-FineWeb mix</td><td>2.5B</td><td></td><td>0.500</td><td>0.653</td><td>0.384</td></tr><tr><td colspan="7">Open pre-training data, 7B (latest OLMo, for reference)</td></tr><tr><td>9 Olmo 3 7B [Team Olmo, 2025]</td><td>Dolma 3</td><td>7.3B</td><td>6T</td><td>0.562</td><td>0.669†</td><td>0.373</td></tr><tr><td>10 Olmo Hybrid 7B [Merrill et al., 2026]</td><td>Dolma 3</td><td>7.4B</td><td>6T</td><td>0.579</td><td>0.685</td><td>0.417†</td></tr><tr><td colspan="7"></td></tr><tr><td>11 Qwen3.5-4B [Qwen Team, 2026a]</td><td>Undisclosed pre-training data Undisclosed</td><td>4.2B</td><td></td><td>0.589</td><td></td><td></td></tr><tr><td>12 Qwen3.5-2B [Qwen Team, 2026a]</td><td>Undisclosed</td><td>1.9B</td><td></td><td>0.484</td><td>0.731 0.540</td><td>0.487</td></tr><tr><td>13 Gemma-4-E2B [Gemma Team, 2026]</td><td>Undisclosed</td><td>4.6B</td><td></td><td>0.507</td><td>0.577</td><td>0.367 0.247</td></tr><tr><td>14 Granite 4.1 3B</td><td>Undisclosed</td><td>3.4B</td><td>15T</td><td>0.555</td><td>0.665†</td><td>0.336</td></tr><tr><td>15 LFM2.5-2.6B</td><td>Undisclosed</td><td>2.7B</td><td>34T</td><td>0.485</td><td>0.637</td><td>0.371</td></tr></table>

Table 3: Overview of a selection of recent pre-trained base models. Rows 1 and 2 are our causal model trained from scratch and its bidirectional variant, adapted from the causal checkpoint with masked next-token prediction. For all scores, higher is better. <sup>†</sup> indicates scores reported by the model developer and copied here directly; other scores are from our own evaluations. <sup>‡</sup> indicates scoring with masked prediction. Params include embedding tables.

## 5.3 Released Artifacts

The pre-training phase of our project concluded with two model checkpoints that we share and build on in the supervised fine-tuning step:

• gaggle-nanochat-pretrained-climbmix-20260924 is our causal pre-trained checkpoint, as detailed in this section.

• gaggle-nanochat-pretrained-climbmix-bidirectional-20260929 is the bidirectional variant described above (starting with causal modeling and switching to bidirectional modeling). We also release the modified attention code needed to perform inference using this variant.

## 6 Supervised Fine-Tuning

This section details the supervised fine-tuning recipe applied to our pre-trained checkpoint to produce a competitive pointwise reranker. We provide an exploration of model score variability across four training runs and describe the final model soup Gaggle (base reranker) used for the headline results in Section 4.

## 6.1 Experimental Setup

Our reranker adopts the formal definition of pointwise reranking from Section 2.2 and the setup described in our headline results from Section 4.1. We do not repeat the formalism here.

![](images/10e0c3110bf583b831c42d2b85e24af6c5a49e8e372502a31704a69ba3cc9783.jpg)  
(a) Standard prompt

![](images/49479290108fffcb1a823a6657de8a971b68efd68362bc77d5d6424a39e116f2.jpg)  
(b) QPQ prompt

![](images/b6ba0ec247aff4a3253db15791bf6cdbc8d87fa8867ec08f4856d238b9080db2.jpg)  
(c) Mask-appended prompt  
Figure 2: Prompts used for both reranker supervised fine-tuning and evaluation. (a) The standard prompt is used for Gaggle (base reranker) and all other contrastive studies by default. (b) The QPQ prompt is used for the causal query-repetition condition; both occurrences contain the same truncated query. (c) The mask-appended prompt is used for the Gaggle (MNTP) mask condition; <|output\_end|> is the MNTP mask token. With the single-position readout, relevance is scored using the ␣true minus $\sqcup ^ { \underline { { \mathbf { f } } } \cdot \mathbf { a } \mathbf { 1 } }$ se logits at the colon after Relevant:, immediately before the mask in (c).

Model and scoring. The gaggle-nanochat-pretrained-climbmix-20260924 checkpoint, our 3.29B-parameter causal language model pre-trained on ClimbMix (see Section 5), provides a starting point for Gaggle (base reranker). The reranker scores each query–passage pair independently, $\mathcal { R } ( q , d _ { n } ) = s _ { n }$ , using the “standard” prompt in Figure 2, panel (a). Our training prompts used the label Passage: for the candidate text. However, we use “document” and “passage” interchangeably throughout the paper. The relevance score s is computed as the difference between the next-token logits for ␣true and ␣false, each a single token that includes the leading space shown as ␣:

$$
s ( q , d ) = \ell _ { \mathrm { t r u e } } ( q , d ) - \ell _ { \mathrm { f a l s e } } ( q , d ) .\tag{1}
$$

We retained the pre-trained language-model output rows for these two tokens (2×2176 = 4,352 parameters) and fine-tuned them jointly with the backbone. The remaining output rows were discarded, leaving approximately 3.22B parameters in the reranker, compared with 3.29B parameters in the original language model. No classification head was added, and all retained parameters were initialized from the pre-trained checkpoint. Queries and passages were truncated to 128 and 512 tokens, respectively, during both training and evaluation.

Training data. It is well known [Nogueira and Cho, 2019] that a high-quality pointwise reranker can be fine-tuned from a pre-trained transformer backbone using a collection of query–document relevance pairs. We adopt this approach, but the biggest variable remains the data recipe: exactly what training data and what data mix.

In this work, we used RLHN-250K, a curated subset of seven retrieval datasets from the BGE training collection [Thakur et al., 2025]. Its construction uses a small LLM to identify potential false negatives and a stronger LLM to verify them; confirmed false negatives are relabeled as positives, and instances with an excessive number of false negatives are discarded. The baseline dataset used in training Gaggle (base reranker) therefore already used filtered and relabeled data; further experiments in Section 7.4 examined the effects of additional curation.

Training objective. We used localized contrastive estimation (LCE), introduced by Gao et al. [2021] and subsequently studied by Pradeep et al. [2022]. Both studies found LCE more effective than pointwise cross-entropy (CE), which treats each query–document pair independently. LCE applies a softmax over one positive passage and K sampled hard negatives for the same query:

$$
\mathcal { L } _ { \mathrm { L C E } } = - \log \frac { \exp ( s ( q , d ^ { + } ) / \tau ) } { \exp ( s ( q , d ^ { + } ) / \tau ) + \sum _ { i = 1 } ^ { K } \exp ( s ( q , d _ { i } ^ { - } ) / \tau ) } .\tag{2}
$$

Our comparison with pointwise cross-entropy in Section 7.5 further supports this choice under the tested configuration. We fixed the softmax temperature at τ = 1 in all LCE training runs.

Configuration. We fine-tuned the reranker for one epoch using AdamW [Loshchilov and Hutter, 2019] with 150 steps of linear warm-up to a peak learning rate of $5 \times 1 0 ^ { - 5 }$ , followed by cosine decay. We used a common optimization recipe for the contrastive studies, changing the factors specified by each experiment. We did not perform hyperparameter tuning for the baseline or the contrastive studies in Section 7, except for the diagnostic comparison of peak learning rates for Gemma and MiniCPM in Section 7.1. Appendix C records data sampling and microbatching, detailed optimization and implementation settings, and the shared and per-run configurations.

<table><tr><td>Model</td><td>Seed</td><td>DL19</td><td>DL20</td><td>DL21</td><td>DL22</td><td>DL23</td><td>Mean</td></tr><tr><td colspan="10">Individual runs</td></tr><tr><td>1 Baseline – run 1</td><td>17</td><td>0.755</td><td>0.720</td><td>0.715</td><td>0.533</td><td>0.485</td><td>0.641</td><td></td></tr><tr><td>2 Baseline – run 2</td><td>18</td><td>0.758</td><td>0.723</td><td>0.707</td><td>0.531</td><td>0.480</td><td></td><td>0.640</td></tr><tr><td>3 Baseline – run 3</td><td>17</td><td>0.755</td><td>0.720</td><td>0.714</td><td>0.533</td><td>0.481</td><td></td><td>0.641</td></tr><tr><td>4 Baseline – run 4</td><td>17</td><td>0.750</td><td>0.720</td><td>0.715</td><td>0.528</td><td>0.480</td><td>0.638</td><td></td></tr><tr><td colspan="10">Soup checkpoint</td></tr><tr><td>5 Gaggle (base reranker)</td><td></td><td>0.757</td><td>0.723</td><td>0.714</td><td>0.533</td><td></td><td>0.480</td><td>0.641</td></tr><tr><td>Mean of runs 1–4</td><td></td><td>Stats summary</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">0.754 0.721</td></tr><tr><td>Sample SD of runs 1–4</td><td></td><td>0.003</td><td>0.001</td><td>0.712 0.004</td><td>0.531 0.002</td><td>0.481 0.002</td><td>0.640 0.001</td><td></td></tr><tr><td colspan="10">(a) TREC Deep Learning</td></tr><tr><td>Model</td><td></td><td>Seed COVID</td><td>News</td><td>Robust</td><td>NFC</td><td>SciFact</td><td>SCID</td><td>FiQA</td><td>Mean</td></tr><tr><td colspan="10">Individual runs</td></tr><tr><td>1 Baseline – run 1</td><td>17</td><td>0.858</td><td>0.510</td><td>0.591</td><td>0.381</td><td>0.784</td><td>0.233</td><td>0.451</td><td>0.544</td></tr><tr><td>2 Baseline – run 2</td><td>18</td><td>0.852</td><td>0.512</td><td>0.584</td><td>0.383</td><td>0.781</td><td>0.235</td><td>0.448</td><td>0.542</td></tr><tr><td>3 Baseline – run 3</td><td>17</td><td>0.862</td><td>0.516</td><td>0.591</td><td>0.380</td><td>0.784</td><td>0.234</td><td>0.452</td><td>0.545</td></tr><tr><td>4 Baseline – run 4</td><td>17</td><td>0.860</td><td>0.515</td><td>0.592</td><td>0.377</td><td>0.783</td><td>0.234</td><td>0.454</td><td>0.545</td></tr><tr><td colspan="10">Soup checkpoint</td></tr><tr><td>5 Gaggle (base reranker)</td><td></td><td>0.859</td><td>0.515</td><td>0.592</td><td>0.380</td><td>0.790</td><td>0.236</td><td>0.455</td><td>0.547</td></tr><tr><td colspan="10">Stats summary</td></tr><tr><td>Mean of runs 1–4</td><td>0.858</td><td></td><td>0.513</td><td>0.589</td><td>0.380</td><td>0.783</td><td>0.234</td><td>0.451</td><td>0.544</td></tr><tr><td>Sample SD of runs 1–4</td><td>0.004</td><td></td><td>0.003</td><td>0.003</td><td>0.002</td><td>0.001</td><td>0.001</td><td>0.002</td><td>0.001</td></tr></table>

(b) BEIR  
Table 4: Analysis of run-to-run variability and model averaging (nDCG@10). All models have 3.22B parameters; the four individual runs used the same training recipe, with differences in random seed, GPU hardware, microbatch token budget, and resulting update count. All models rerank top-100 BM25 candidates. Rows 1–4 represent individual trials; row 5 is the soup checkpoint used in our headline results. Summary rows report the mean and sample standard deviation (SD) across rows 1–4 only. The highest score in each column is marked in bold (including ties, excluding summary rows).

## 6.2 Run-to-Run Variability and Model Averaging

We performed four baseline supervised fine-tuning runs to characterize run-to-run variation in ranking effectiveness. Each run started from the same causal pre-trained checkpoint and used the RLHN-250K data, following the same recipe detailed here.

To provide more details: runs 1, 3, and 4 used seed 17, while run 2 used seed 18. Runs 1–3 used two RTX PRO 6000 Blackwell GPUs; run 4 used two H100 GPUs with gradient caching. The runs shared the optimization recipe detailed here but differed in execution environment and microbatch configuration, as explained in Appendix C. Our comparison therefore measures variation across multiple trials that include changes in seed and execution settings.

Finally, we formed a model soup [Wortsman et al., 2022] by averaging all parameters across the final model checkpoints from the four individual runs above, with a weight of 0.25 per checkpoint. The model soup corresponds to the checkpoint gaggle-reranker-20261005, which is used to produce our headline result in Section 4.

Our model soup and individual models are evaluated in exactly the same way with results shown in Table 4. To be explicit: the soup scores were obtained from this single averaged model (i.e., this is not so-called results averaging). The variability of individual runs is summarized in terms of the sample standard deviation across the four individual models.

The four individual runs achieve similar ranking effectiveness: means of 0.640±0.001 on TREC DL and 0.544±0.001 on BEIR, where ± denotes the sample standard deviation. We also computed the sample standard deviation separately for each test collection; these values range from 0.001 to 0.004, with an average of 0.003 across the 12 collections. We use these standard deviation measurements as our guideline for run-to-run noise when interpreting differences in the contrastive studies. That is, we consider differences that are roughly below this threshold to be experimental noise.

The soup model achieved a mean nDCG@10 of 0.641 on TREC DL and 0.547 on BEIR, exceeding the corresponding averages of the individual runs by 0.001 and 0.003. Its TREC DL mean matched the strongest individual run, while its BEIR mean exceeded all four individual runs. Although these gains are close to the observed run-to-run noise scale, the results suggest a potential benefit from parameter averaging, even when all model instances follow the same training recipe. This appears consistent with previous studies exploring model soups [Wortsman et al., 2022].

## 7 Contrastive Conditions

In this section, we examine how modeling and training choices affect reranking effectiveness. Our experiments considered a number of different factors against baseline run 1 of our Gaggle (base reranker), as described in Section 6.2. To be clear, we do not compare against the model soup, which would not be methodologically fair.

We interpret score differences descriptively using the per-trial variability measured in Section 6.2 as a reference. Specifically, we treat changes on the order of 0.001 in benchmark means or 0.003 on individual collections as experimental noise. Comparisons use the unrounded evaluation scores.

## 7.1 The Effects of Different Backbones

## 7.1.1 Experimental Setup

Our headline result demonstrates that we are able to fine-tune a pointwise reranker (Section 6) from our own pre-trained model checkpoint (Section 5). A natural follow-up question is: What about other model backbones? How much of our success is attributable to the choice of model backbones?

To answer this question, we applied exactly the same supervised fine-tuning recipe (as described in Section 6) but starting from different open-weight backbones: nanochat-d34,<sup>25</sup> Gemma-4-E2B,<sup>26</sup> Qwen3.5-4B-Base,<sup>27</sup> and MiniCPM5-2B-Base.<sup>28</sup>

All training runs used causal attention, the standard prompt, the LCE loss function, and the last non-padded token for scoring. We fine-tuned each model’s text parameters on RLHN-250K, matching the optimization hyperparameters and input-length limits used for our headline results.

Additional details for different backbones: For MiniCPM, we used openbmb/MiniCPM5-2B-Base, the model checkpoint with pre-training only, rather than the mid-trained release. For the multimodal backbones Gemma-4-E2B and Qwen3.5-4B-Base, we discarded the vision encoders and Gemma’s audio encoder, leaving approximately 4.6B and 4.2B trainable parameters, respectively; Gemma’s count includes its per-layer embedding tables. Reported model parameter sizes in our results table exclude untied output layers, since scoring uses only the two answer-token rows; for example, nanochat-d34 has 2.1B parameters under this convention and 2.2B if we include its output layer. Each backbone retained its native architecture and tokenizer.

We first fine-tuned every backbone with the peak learning rate used in our headline results, which is $5 \times 1 0 ^ { - 5 }$ . Additional analyses showed that this peak was too high for Gemma-4-E2B and MiniCPM, so we reran both with a peak learning rate of $\mathrm { \bar { 1 } } \times 1 0 ^ { - 5 }$ , keeping the warm-up and decay schedule unchanged (details below).

## 7.1.2 Results and Discussion

Experimental results of reranker effectiveness with different backbones are shown in Table 5. Using the shared supervised fine-tuning recipe, baseline run 1 of Gaggle (base reranker) achieves a higher mean nDCG@10 than the other backbones on both TREC DL and BEIR, with differences exceeding the run-to-run noise guideline. Outperforming nanochat-d34 did not come as a surprise, since our backbone was pre-trained from a better data mix, but results for the other backbones were surprising. In particular, given that Gemma and MiniCPM achieve higher CORE, MMLU, and MMLU-Pro scores (see Table 3), we found it perplexing that using these backbones as a starting point for supervised fine-tuning resulted in a less effective reranker.

<table><tr><td>Backbone</td><td>Size LR</td><td>TREC DL</td><td>BEIR</td></tr><tr><td>1 gaggle-nanochat-pretrained</td><td> $3 . 2 \mathrm { B } \ 5 \times 1 0 ^ { - 5 }$ </td><td>0.641</td><td>0.544</td></tr><tr><td>2 nanochat-d34</td><td> $2 . 1 8 \ \mathrm { ~ 5 } { \times } 1 0 ^ { - 5 }$ </td><td>0.621</td><td>0.521</td></tr><tr><td>3 Gemma-4-E2B</td><td> $4 . 6 \mathrm { B } \ 5 \times 1 0 ^ { - 5 }$ </td><td>0.627</td><td>0.537</td></tr><tr><td>4Qwen3.5-4B-Base</td><td> $4 . 2 \mathrm { B } \ 5 \times 1 0 ^ { - 5 }$ </td><td>0.626</td><td>0.530</td></tr><tr><td>5 MiniCPM5-2B-Base</td><td> $2 . 2 \mathrm { B } \ 5 \times 1 0 ^ { - 5 }$ </td><td>0.627</td><td>0.530</td></tr><tr><td>6 Gemma-4-E2B</td><td> $4 . 6 \mathrm { B } 1 \times 1 0 ^ { - 5 }$ </td><td>0.640</td><td>0.546</td></tr><tr><td>7MiniCPM5-2B-Base</td><td> $2 . 2 \mathbf { B } 1 \times 1 0 ^ { - 5 }$ </td><td>0.642</td><td>0.551</td></tr></table>

Table 5: Effects of different backbones (mean nDCG@10). All configurations are trained with the same supervised fine-tuning recipe. LR denotes the peak learning rate; other hyperparameters follow the headline recipe, and hardware and microbatch differences are documented in Appendix C. All configurations rerank top-100 BM25 candidates. Row 1 is baseline run 1 of Gaggle (base reranker). The highest score in each column is marked in bold.

![](images/8a5b82235730e50b068fe613d742f85410d72e6d986734cf17586888591a44dc.jpg)  
(a) All backbones at $5 \times 1 0 ^ { - 5 }$

![](images/ba1dea7834d17cce8612ede0610eb5e5a79460993e7293038d83cccef8924e27.jpg)  
(b) Gemma and MiniCPM at $1 \times 1 0 ^ { - 5 }$  
gaggle-nanochat-pretrained Gemma-4-E2B MiniCPM Qwen3.5-4B-Base  
Figure 3: Training loss on identical batches at peak learning rates $5 \times 1 0 ^ { - 5 } \ ( \mathrm { a } )$ and $1 \times 1 0 ^ { - 5 } \left( \mathsf { b } \right)$ ; bins widen over training to smooth per-batch noise.

We followed up and examined the training dynamics of Gemma and MiniCPM to better understand their lower reranking effectiveness with the shared recipe. All backbones initially used the same peak learning rate of $5 \times \mathrm { 1 0 ^ { - 5 } }$ , which we had used for our headline results without hyperparameter tuning. On identical batches, training losses for Gemma and MiniCPM increased as the learning rate reached its peak, while losses from our model continued to decrease, as shown in Figure 3a.

We considered differences in weight scale as a possible explanation. Our backbone was pre-trained with Muon and had weight matrices approximately five times larger in scale than those of Gemma and MiniCPM, which were pre-trained with AdamW. We hypothesized that the shared learning rate produced larger updates relative to the smaller weights in Gemma and MiniCPM, contributing to their increases in training loss. In other words, the learning rate appeared to be too large.

To test this explanation, we repeated supervised fine-tuning for Gemma and MiniCPM, but with a smaller peak learning rate of $\mathrm { \bar { 1 } \times 1 0 ^ { - 5 } }$ , keeping the other training settings unchanged. The lower rate eliminated the increase in training loss, seen in Figure 3b. Both backbones reached TREC DL effectiveness comparable to baseline run 1, while MiniCPM achieves a higher BEIR mean beyond the noise guideline. These results are consistent with the proposed weight-scale explanation and suggest that learning-rate sensitivity should be considered when comparing backbones under a shared training recipe.

We find that, starting from MiniCPM, it is possible to train a pointwise reranker that is more effective than our Gaggle (base reranker). However, the fact that there are “better” pre-trained checkpoints available to fine-tune does not detract from our core thesis.<sup>29</sup> Our claims focus on sufficiency, that our own model checkpoint pre-trained from scratch can produce a competitive reranker, which remains true. In other words, at least for the task of pointwise reranking, it appears that our pre-training recipe is “good enough”. This allows us to retain full end to end control of the training pipeline.

<table><tr><td>Training data</td><td>Instances</td><td>TREC DL</td><td>BEIR</td></tr><tr><td>1 RLHN-250K</td><td>247,534</td><td>0.641</td><td>0.544</td></tr><tr><td>2 RLHN-680K</td><td>648,762</td><td>0.646</td><td>0.545</td></tr><tr><td>3 Tevatron MS MARCO</td><td>399,600</td><td>0.618</td><td>0.470</td></tr></table>

Table 6: Effects of training data (mean nDCG@10). All rerankers have 3.22B parameters and use the same one-epoch training recipe, with differences in training data (size and source mix), GPU hardware, microbatch token budget, and resulting update count. All configurations rerank top-100 BM25 candidates. Row 1 is baseline run 1 of Gaggle (base reranker). The highest score in each column is marked in bold.

## 7.2 The Effects of Training Data

## 7.2.1 Experimental Setup

It is well known, across diverse machine-learning applications, that model effectiveness depends on the size of the training dataset and the composition of the training instances. In our next series of experiments, we attempted to isolate the effects of the data recipe.

One obvious starting point is to examine the variant datasets prepared by Thakur et al. [2025] using the same initialization and training recipe: RLHN-250K (247,534 instances) versus RLHN-680K (648,762 instances). Since we applied exactly the same training recipe, training models under both data conditions for one epoch, the larger dataset increases the number of instances by a factor of 2.62 and the number of optimization steps from 1,931 to 5,060.

Note that the number of training instances is not the only difference between these variants; the source distribution also changes. In moving from RLHN-250K to RLHN-680K, MS MARCO increases from 55.2% to 70.2% of the data and HotpotQA from 12.1% to 13.0%. The share of Natural Questions decreases from 12.0% to 8.9%, although RLHN-680K contains more Natural Questions training instances than RLHN-250K. FEVER, SCIDOCS-RR, FiQA, and ArguAna retain approximately the same instance counts and consequently account for smaller fractions of the total dataset size.

To provide a point of contrast on data composition, we also trained on 399,600 prepared instances from the Tevatron MS MARCO dataset,<sup>30</sup> using the same causal backbone, LCE objective, and one-epoch training recipe. Note that Tevatron contains only MS MARCO data, whereas RLHN-250K combines seven collections and includes LLM-assisted relabeling of false negatives [Thakur et al., 2025], so the datasets differ in terms of size, composition, and preparation. The Tevatron and RLHN-250K datasets each provide approximately 30 negative passages per query on average. The larger Tevatron dataset required 3,112 optimizer updates, compared with 1,931 for RLHN-250K.

## 7.2.2 Results and Discussion

Effectiveness results are shown in Table 6. Training on the larger RLHN dataset increases mean nDCG@10 on TREC DL from 0.641 to 0.646, an improvement larger than the benchmark-level noise guideline. Gains on four of the five TREC DL collections exceed the typical collection-level noise scale, while the DL21 difference falls below the threshold. The BEIR mean changes from 0.544 to 0.545, which we treat as experimental noise. This average hides larger collection-specific changes, including an increase of 0.017 on TREC-News and a decrease of 0.015 on TREC-COVID. The TREC DL gains are consistent with the increased emphasis on MS MARCO in the larger dataset, but this comparison does not isolate the effects of dataset size, source distribution, and additional optimization steps.

<table><tr><td></td><td colspan="4">Training queries</td></tr><tr><td>Benchmark</td><td>50K</td><td>100K</td><td>150K</td><td>200K 250K</td></tr><tr><td>TREC DL</td><td>0.628</td><td>0.616</td><td>0.634</td><td>0.637 0.638</td></tr><tr><td>BEIR</td><td>0.525</td><td>0.533</td><td>0.542 0.543</td><td>0.545</td></tr></table>

Table 7: Effectiveness of baseline run 4 across RLHN-250K training checkpoints (mean nDCG@10). All configurations rerank top-100 BM25 candidates.

On TREC DL, training on RLHN-250K yields higher reranking effectiveness than training on Tevatron MS MARCO, with a difference of 0.023 in mean nDCG@10. Both datasets included MS MARCO, but Tevatron provides more training instances and optimizer updates. The hard-negative mining and LLM-assisted relabeling in RLHN-250K may provide higher-quality negatives with fewer false-negative labels, contributing to this difference. The effectiveness gap is larger on BEIR, at 0.074 mean nDCG@10, consistent with the expected benefit of broader training-domain coverage. Tevatron contains no BEIR training collections, whereas RLHN-250K includes FiQA and SCIDOCS-RR, covering two of the seven BEIR evaluation domains.

Table 7 tracks effectiveness over the course of RLHN-250K training, using the same top-100 BM25 reranking protocol as our headline results. We see that most of the improvements in effectiveness arrive early. By roughly 150K queries, results on both benchmarks are close to their final scores, and the remaining instances contribute only small gains. However, the two benchmarks appear to behave differently along the way. Effectiveness on BEIR improved steadily throughout, but effectiveness on the TREC DL collections declined mid-training while the learning rate remained high but scores recovered as the learning rate decayed. This pattern suggests sensitivity to the learning-rate schedule, consistent with the observations in Section 7.1.

## 7.3 The Effects of Causal versus Bidirectional Modeling

## 7.3.1 Experimental Setup

In these experiments, we compared causal and bidirectional attention, prompt changes, and mean pooling. All variants followed the basic experimental setup described in Section 6.1, but with changes to the starting checkpoint, attention pattern, prompt, and readout, specified below.

The two starting checkpoints were:

• gaggle-nanochat-pretrained-climbmix-20260924 (the causal variant), and

• gaggle-nanochat-pretrained-climbmix-bidirectional-20260929 (the bidirectional variant).

The bidirectional variant was adapted from an earlier causal checkpoint using MNTP, as described in Section 5.1. The two backbones differed in their final pre-training but were trained on roughly the same number of tokens.

We also tested a query–passage–query (QPQ) prompt under causal attention. The first query conditions passage representations, while the repeated query enables the model to attend to the passage before scoring. This placement is motivated by Jina-Reranker-v3, which places a query before and after the candidate documents [Wang et al., 2025]. This prompt design is also related to echo embed dings, in which the prompt repeats an input and extracts embeddings from the second occurrence, whose tokens can attend to the complete first occurrence under causal attention [Springer et al., 2025]. In our reranker, we repeat only the same truncated query around a single passage and retain the baseline scoring function. The short context window of our pre-trained checkpoint prohibits direct adoption of Jina’s multi-document recipe, although repeating the query is feasible for our model.

Figure 2 shows the three prompts used in supervised fine-tuning and evaluation:

• Standard: the prompt in our headline results, tested with both causal and bidirectional attention.

• QPQ: the same truncated query appears before and after the passage, with causal attention.

• Appended mask: the MNTP mask token, <|output\_end|>, follows Relevant:, with bidirectional attention. Scoring uses the preceding colon, matching MNTP’s prediction position.

<table><tr><td>Configuration</td><td>TREC DL</td><td>BEIR</td></tr><tr><td>1 Causal, standard prompt</td><td>0.641</td><td>0.544</td></tr><tr><td>2 Causal, QPQ prompt</td><td>0.647</td><td>0.551</td></tr><tr><td>3 Causal, mean pooling</td><td>0.642</td><td>0.543</td></tr><tr><td>4 MNTP, standard prompt</td><td>0.643</td><td>0.544</td></tr><tr><td>5 MNTP, appended mask</td><td>0.643</td><td>0.543</td></tr><tr><td>6 MNTP, mean pooling</td><td>0.638</td><td>0.538</td></tr></table>

Table 8: Effects of causal versus bidirectional modeling (mean nDCG@10). All rerankers have 3.22B parameters and use the same training recipe, with differences in pre-trained initialization, attention, prompt, and readout. All configurations rerank top-100 BM25 candidates. Row 1 is baseline run 1 of Gaggle (base reranker). The highest score in each column is marked in bold.

For both backbones, we additionally tested mean pooling with the standard prompt. We averaged final hidden states over all non-padding tokens and applied the same two pre-trained scoring rows, fine-tuning them jointly with the backbone. No new classification head was introduced.

## 7.3.2 Results and Discussion

Table 8 compares the effects of attention pattern, prompt, and pooling on ranking effectiveness. With the standard prompt, the causal baseline and the bidirectional reranker initialized from the MNTP checkpoint achieve comparable effectiveness, with differences on the scale of run-to-run noise (rows 1 and 4). Appending the mask token produces benchmark means comparable to the standard bidirectional prompt (rows 4 and 5).

Compared to the standard causal prompt (row 1), the causal QPQ prompt (row 2) improves mean nDCG@10 from 0.641 to 0.647 on TREC DL and from 0.544 to 0.551 on BEIR, with both changes larger than the noise guideline. Repetition may help by giving the model query representations that have been conditioned by the candidate document before the final scoring position, without requiring any changes to its pre-trained causal attention pattern. This is a possible explanation for the observed effectiveness improvement.

To understand why bidirectional adaptation yields ranking effectiveness comparable to the causal baseline, we tested whether the adapted rerankers attended to later tokens. We selected DL19, DL20, and SciFact to cover both TREC DL and BEIR under the in-domain classification used in Section 4.1. Their small query sets (43, 54, and 300 queries, respectively) kept the diagnostic evaluation inexpensive. With model weights fixed, blocking attention to later tokens reduced mean nDCG@10 across these three collections from 0.753 to 0.700 for the standard prompt and from 0.750 to 0.688 for the mask-appended prompt.

These decreases show that the adapted models rely on bidirectional attention. However, the baseline trained and evaluated with causal attention achieves comparable ranking effectiveness, suggesting that the model has learned alternative ways to combine query and passage information.

For the bidirectional model, replacing the single-position readout (row 4) with mean pooling (row 6) reduces mean nDCG@10 from 0.643 to 0.638 on TREC DL and from 0.544 to 0.538 on BEIR, with both declines larger than the noise guideline. However, for the causal model, the differences between the last non-padded token and mean pooling readout are within the noise guideline, suggesting comparable effectiveness (rows 1 and 3). Together, these results show that the adapted models do make use of bidirectional attention, but without a clear reranking advantage over the causal baseline. Similarly, replacing their single-position readout with mean pooling did not appear to improve reranking effectiveness.

A possible explanation is that MNTP improves masked-token prediction without sufficiently improving the representations used for relevance scoring, and hence has minimal impact on reranking. We leave pre-training a BERT-style bidirectional model [Devlin et al., 2019] from scratch and exploring other pooling methods to future work.

<table><tr><td>Training configuration</td><td>TREC DL</td><td>BEIR</td></tr><tr><td>1 LCE</td><td>0.641</td><td>0.544</td></tr><tr><td>2 LCE + self-filtering</td><td>0.641</td><td>0.548</td></tr><tr><td>3 LCE + self-filtering + self-distillation</td><td>0.640</td><td>0.547</td></tr></table>

Table 9: Effects of self-filtering and self-distillation (mean nDCG@10). All rerankers have 3.22B parameters and use the same pre-trained initialization and training recipe, except for self-filtering, the additional distillation loss, and the resulting update count. All configurations rerank top-100 BM25 candidates. Row 1 is baseline run 1 of Gaggle (base reranker). The highest score in each column is marked in bold (including ties).

## 7.4 The Effects of Self-Filtering and Self-Distillation

## 7.4.1 Experimental Setup

In these experiments, we compared the baseline trained on RLHN-250K with two variants: additional self-filtering of the training data and self-distillation on the self-filtered data. The first comparison examined the effect of filtering; the second examined the added value of score supervision from the trained baseline checkpoint.

We used the model checkpoint from baseline run 1 to identify potentially noisy positives in RLHN-250K. For each training query, we scored all positive and negative passages and removed a positive if more than $\operatorname* { m a x } ( 9 , 0 . 3 7 5 N )$ negatives received a higher score, where N is the total number of negatives available for that instance before sampling. Queries with no remaining positive relevance judgments were then discarded. We did not apply this procedure to ArguAna, which was one of the seven training sources in RLHN-250K [Thakur et al., 2025], because its task requires retrieving counterarguments, which may not align with the model’s general relevance judgments.

This filtering procedure removed 14,887 positive passages (2.6%) and 1,484 queries (0.6%), leaving 246,050 queries in the dataset. We then trained a new reranker on the retained data using the same LCE objective and training recipe as our baseline run 1. The checkpoint from baseline run 1 supplied the filtering decisions, while the new reranker was initialized from the original pre-trained checkpoint.

We also explored whether the baseline checkpoint can provide useful supervision beyond data filtering. One common ranking-distillation objective is Margin-MSE [Hofstätter et al., 2020], which minimizes the squared difference between teacher and student score margins for pairs of passages. We did not select this objective, but instead used Kullback–Leibler (KL) divergence to match LCE’s groupwise softmax formulation: LCE supervises the documents with positive labels, while KL divergence transfers the teacher’s relative preferences over the sampled negatives to the student. For a fixed teacher, minimizing KL divergence is equivalent to minimizing cross-entropy with soft targets. This motivates our objective formulation here, without making any empirical claims about effectiveness compared to Margin-MSE.

Putting everything together, on the self-filtered data, the distillation objective is:

$$
p _ { T } = \mathrm { s o f t m a x } ( \mathbf { s } _ { - } ^ { T } / \tau _ { \mathrm { K D } } ) ,
$$

$$
p _ { S } = \mathrm { s o f t m a x } ( \mathbf { s } _ { - } ^ { S } / \tau _ { \mathrm { K D } } ) ,\tag{3}
$$

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { L C E } } + 0 . 3 \tau _ { \mathrm { K D } } ^ { 2 } D _ { \mathrm { K L } } ( p _ { T } \Vert p _ { S } ) .\tag{4}
$$

Both teacher and student use $\tau _ { \mathrm { K D } } = 2 ,$ , while LCE retains $\tau = 1$ . The factor $\tau _ { \mathrm { K D } } ^ { 2 }$ compensates for the temperature’s effect on the gradient scale. Here, $\mathbf { s } _ { - } ^ { T }$ and $\mathbf { s } _ { - } ^ { S }$ are teacher and student scores over the K negatives. The positive documents are excluded from distillation, leaving their supervision to LCE. The teacher is fixed; the student has the same architecture and starts from the original pre-trained initialization. Distillation was not performed for ArguAna.

## 7.4.2 Results and Discussion

Experimental results are shown in Table 9. We see that self-filtering improves mean BEIR nDCG@10 from 0.544 to 0.548, beyond the benchmark-level noise guideline, while TREC DL effectiveness remains within run-to-run noise. The largest observed gain was on TREC-News, increasing from 0.510 to 0.529 and exceeding the typical collection-level noise threshold. Adding self-distillation with matched temperatures produces benchmark means comparable to self-filtering alone; both difference are within the noise guideline. Collection-level changes appear mixed: FiQA increased by 0.004, but DL23 and SciFact decreased by 0.007 and 0.008, respectively. These results suggest that the trained checkpoint can provide useful filtering decisions, but that score supervision from a teacher of the same capability appears to offer no clear additional benefits in reranking effectiveness.

<table><tr><td>Objective</td><td>TREC DL</td><td>BEIR</td></tr><tr><td>1 LCE</td><td>0.641</td><td>0.544</td></tr><tr><td>2 Pointwise CE</td><td>0.628</td><td>0.528</td></tr></table>

Table 10: The effects of different loss functions (mean nDCG@10). Both variants start from the same 3.29B-parameter pre-trained checkpoint; the rerankers retain 3.22B parameters for LCE and 3.29B for CE. All configurations rerank top-100 BM25 candidates. Row 1 is baseline run 1 of Gaggle (base reranker). The highest score in each column is marked in bold.

A common alternative, naturally, is to distill supervision from a stronger teacher. In line with our focus on sovereign AI, we opted for self-filtering and self-distillation using variants of our own models to avoid an additional external teacher over which we may not have control. We leave alternative formulations of distillation for future work.

## 7.5 The Effects of Different Loss Functions

## 7.5.1 Experimental Setup

In these final experiments, we compared LCE with pointwise cross-entropy (CE), implemented as answer-token prediction. The CE experiment used the same sampled positive and negative query–document pairs and the same optimizer-step plan as LCE. This setting also matched the pre-trained initialization, standard prompt, optimization settings, random seed, training precision, and input-length limits.

With the CE loss function, we trained the model to take each pair as input to predict a single ␣true or ␣false answer token. This approach retains the full pre-trained language-model output layer and averages answer-token cross-entropy over the pairs, with normalization over the entire vocabulary. With LCE, the model instead retains only the two scoring rows and averages contrastive losses over query groups. This comparison therefore evaluates standard supervised fine-tuning against our LCE recipe under matched training exposure.

## 7.5.2 Results and Discussion

Experimental results are shown in Table 10. We see that LCE achieves higher mean nDCG@10 than CE on both benchmarks: 0.641 versus 0.628 on TREC DL and 0.544 versus 0.528 on BEIR. The absolute improvements of 0.013 and 0.016, respectively, are larger than the run-to-run noise guideline. These results support LCE for our decoder-based reranker under the tested recipe and are consistent with the encoder-based findings of Pradeep et al. [2022].

## 8 Related Work

## 8.1 AI Sovereignty

The idea of AI sovereignty is not new. Canada’s national AI strategy from 2017 already emphasized retaining domestic talent and economic benefits from AI.<sup>31</sup> In a speech dating to March 2018, French President Emmanuel Macron linked AI infrastructure and development to French sovereignty and retaining economic value in France and Europe.<sup>32</sup> The concept has been steadily gaining prominence: In February 2024, NVIDIA founder and CEO Jensen Huang asserted at the World Governments Summit in Dubai that every country needs to own the production of their own intelligence.<sup>33</sup> Sovereign AI has also attracted substantial attention from major consulting firms such as Deloitte, McKinsey,

Accenture, and BCG, all of which have opined on various aspects of this complex topic. The Linux Foundation [Gerosa et al., 2025] identified open source as a foundation for sovereignty through transparency, adaptability, and reducing dependence on individual vendors. Although initially articulated through the lens of national concerns, sovereignty has increasingly been discussed at the level of individual organizations. Recent episodes concerning access to frontier models from Anthropic (discussed in Section 3.4) have thrust questions about AI sovereignty into the spotlight.

Alex Karp, the CEO of Palantir, has prominently advocated for AI sovereignty in recent months, emphasizing the importance of organizations retaining control over their institutional knowledge and their “means of production”. These arguments were more precisely laid out in August 2026,<sup>34</sup> but they can be traced back to public interviews in July. He warns of model providers misappropriating customers’ business data and capturing their “alpha” (an organization’s distinctive knowledge and capabilities that give it a competitive advantage). Another concern he raised is the cost associated with LLMs and organizations not necessarily deriving value from token spend. LLM inference costs provide another rationale for the pursuit of sovereign AI: any organization dependent on external model providers will have limited control over pricing charged by third parties. Thus, as part of sovereignty efforts, many organizations are exploring whether smaller, and therefore cheaper, task-specific models might be just as effective as frontier models (more discussion below).

In a white paper from July 2026 to complement many of Karp’s arguments, Palantir [2026] argues for sovereignty from the perspective of an organization owning the value created through AI and the freedom to pursue new opportunities. The choices that an organization makes with respect to AI model usage can narrow or broaden these economic rights. The report laid out recommendations spanning models, compute, and operational control. It argues that zero data retention alone is insufficient: organizations must also preserve the ability to switch providers and retain the knowledge and improvements generated through their use of AI.

While perhaps not as outspoken as Alex Karp, Microsoft CEO Satya Nadella in July 2026 made similar arguments about AI sovereignty in what he calls the “Reverse Information Paradox”.<sup>35</sup> He asserted that, with AI, organizations pay for intelligence twice, once with money, and again with something even more valuable: proprietary knowledge that an organization must reveal to make that intelligence useful. Over time, model providers learn about their customers (via prompts, confidential information, agent tools, human feedback, usage statistics, etc.), making it imperative for organizations to establish a trust boundary to prevent misappropriation. Nadella actually cites Karp, saying that organizations “want to know they own the means of production, and it’s not being transferred to someone else”.

Our work is very much inspired by this line of thinking, and we share concerns about institutional autonomy. To this discussion, we additionally contribute two less frequently mentioned issues from the academic perspective: the important role of transparency in developing a rigorous understanding of model behavior and the valuable educational opportunities that arise from the ability to build fully open and sovereign models (Section 3.5). From a technical perspective, we focus specifically on the transparency and reproducibility of model development, including access to the data and recipes needed to independently build various artifacts. We make available everything necessary for an organization to reproduce our results, to inspect each step in the pipeline, and to make any modifications as necessary. Our contribution is to demonstrate what can be achieved with limited computational resources in a fully open and sovereign manner.

## 8.2 Open Models and Related Efforts

We compare the goals of Project Greenhouse and our current contributions to a representative selection of other open models and related efforts.

OLMo and Tülu. Building on earlier open efforts such as BLOOM [BigScience Workshop, 2022] and Pythia [Biderman et al., 2023], AI2 announced the OLMo (Open Language Model) project in May 2023.<sup>36</sup> The goal was to develop fully open language models to benefit the research community by providing access for scientific study and education around all aspects of model creation. AI2 accomplished this by releasing not only model checkpoints, but also training data, code, and evaluation resources. The first OLMo models were released in February 2024 [Groeneveld et al., 2024], followed by OLMo 2 in November 2024 [Team OLMo, 2024] and Olmo 3 in November 2025 [Team Olmo, 2025]. This seminal line of work has provided an important reference point for open research in building language models and establishes an important foundation for our approach: transparency throughout model development enables independent experimentation, reproducibility, and rigorous understanding.

Nevertheless, there are important differences between the OLMo project and our efforts: OLMo aspires to be a family of state-of-the-art models with a diverse set of capabilities, including longcontext reasoning, function calling, coding, instruction following, general chat, and knowledge recall. In contrast, our ambitions are much more tightly scoped, focused only on agentic search. Due to its frontier aspirations, OLMo required far more compute than typically available to academic institutions: The developers reported approximately 56 days from the start of training to the evaluation of Olmo 3 Think 32B on a cluster with 1,024 H100 GPUs [Team Olmo, 2025]. The model was pre-trained on roughly 6T tokens, approximately 20× larger than our corpus.

Despite this transparency, reproducing the larger OLMo models from scratch remains beyond the resources of most individual research groups. In contrast, our reranker models are far more restricted in capability but can be reproduced using only a handful of GPUs. We therefore view our efforts as complementary: alongside transparency in how models are developed, we emphasize making end-to-end experimentation practical under modest compute budgets. This accessibility matters because researchers need to do more than reproduce a single checkpoint; they must also be able to modify the data, test alternative recipes, and run controlled ablations.

Tülu is a complementary project [Lambert et al., 2024], also by AI2, that focuses on open post-training methods for existing models. The effort partially overlaps with the OLMo project and its recipes informed the development of instruction-following OLMo models. For example, Tülu 3 builds on Llama 3.1 base models, with releases that include data, code, and recipes used for post-training. DR Tulu [Shao et al., 2026] is an agentic search model directly trained for open-ended, long-form deep research. Starting from Qwen3-8B, the authors performed supervised fine-tuning on research trajectories, including examples generated from GPT-5, followed by reinforcement learning with evolving rubrics (RLER). These rubrics incorporate retrieved evidence and are updated using the model’s rollouts, allowing the evaluation criteria to adapt to newly discovered information. The project released model weights, training data, code, and search-agent infrastructure.

While DR Tulu shares some of the same end goals as our efforts, there are substantial differences in scope and approach. Project Greenhouse aspires to tackle agentic search in its entirety (which includes deep research), but our approach is more incremental: we plan to develop and evaluate specific capabilities before integrating them into more complex systems. Our first milestone is a pointwise reranker, which is far less ambitious in scope than DR Tulu. However, we require that the resulting models be fully open and sovereign under our definition. Although DR Tulu makes its agent-training pipeline available, it begins with an open-weight backbone whose pre-training data and recipes are not fully available. Thus, it does not qualify as fully open and sovereign under our definition. We will not repeat arguments for why we feel that open-weight models are not enough, but see Section 3.4 for the organizational perspective and Section 3.5 for the academic perspective.

Marin. The purpose of the Marin project is the discovery and promulgation of the “process knowledge” necessary to create frontier-level artificial intelligence.<sup>37</sup> It originated at Stanford University in 2024, but the Marin models are now primarily developed by Open Athena, a nonprofit that accelerates academia with capabilities from the AI frontier, with community contributions.<sup>38</sup> According to the Marin website, “open” means that everything is shared: all discussions, all code, including models, infrastructure, etc., model weights, the sources of all the data in the training corpus, the results of successful and unsuccessful experiments, and tools that make it easy to learn from this effort. As of October 2026, the project is training a mixture-of-experts model with 535 billion parameters total, 23 billion parameters active, on 18 trillion tokens of data.

Marin articulates a laudable vision of model development, but Project Greenhouse differs in several important respects. Whereas Marin explicitly aims for frontier-level intelligence, we have far more tightly scoped ambitions focused on agentic search. Aiming for the frontier requires substantial centralized computational resources; for example, the hero run referenced above is being trained on 11× GB200 NVL72 racks. In contrast, our approach allows for more decentralization, as most individual experiments described in this paper can be reproduced with only one or two GPUs. Thus, we view our efforts as complementary, but sharing the same goal of expanding public knowledge.

Thomson. In August 2026, Chen et al. [2026] contributed to the sovereign AI discussion through Thomson, a new general-purpose frontier model family trained with an enhanced focus on high-stakes professional work. Although not technically open by our definition, discussions of sovereignty figured prominently in the narrative around the model and its associated technical report, making a focused discussion of the Thomson effort worthwhile. The research team demonstrated that frontier performance is achievable by a wide range of institutions through continual learning on existing open-weight models (specifically, Qwen3.6-35B and Qwen3.5-397B). Model weights for the small 35B variant of Thomson are publicly available.

While we share their emphasis on institutional autonomy and on making model development transparent, there are several important differences between the Thomson effort and Project Greenhouse. Again, like Marin and OLMo discussed above, Thomson aims for the frontier, which we do not. However, instead of starting “from scratch”, i.e., from pre-training, Thomson starts from Qwen open-weight models. Thus, the model does not meet our definition of fully open and sovereign as upstream details remain unavailable. In working with Qwen as a starting point, the Thomson researchers expended considerable effort in “value re-alignment” before they could proceed with continued pre-training to focus on desired capabilities. Furthermore, the resources required for their efforts are beyond most organizations: the model family was “developed by a technical team not exceeding three dozen engineers and scientists on a modest compute cluster with no more than 368 B200 GPUs available at any stage of experimentation”. While the project represents an important example of a viable alternative approach, its resource requirements remain beyond the budget of most academic institutions.

Other open models. The discussions above focus on prominent, high-profile efforts, but researchers have released many open models over the years. Earlier contributions to transparent model development include LLM360’s Amber and CrystalCoder in 2023 [Liu et al., 2023], followed by MAP-Neo [Zhang et al., 2024] and DCLM [Li et al., 2024] in 2024, which made training data and development artifacts available alongside model weights.

Subsequent efforts have pursued several complementary directions: SmolLM [Allal et al., 2025] emphasizes small models and transparent training recipes, while Comma, introduced with Common Pile in 2025 [Kandpal et al., 2025], emphasizes public-domain and openly licensed training data. Continuing development is evident in LLM360’s subsequent K2 models [K2 Team, 2025], the expanding Pleias family [Langlais et al., 2025], including Baguettotron, and multilingual initiatives such as Apertus [Project Apertus, 2025] and EuroLLM [Ramos et al., 2026], with further releases and technical reports appearing in 2025–2026. Most recently, Gensyn’s Open-1B [Donaghy et al., 2026], announced in September 2026, explores independently auditable training through the release of data, checkpoints, and a framework for replaying individual training steps.<sup>39</sup>

Qwen and other open-weight models. While there are many open-weight models (too numerous to enumerate), the Qwen family of models [Yang et al., 2024, Zhang et al., 2025b, Qwen Team, 2026a,b] deserves special recognition as the starting point most commonly used by information retrieval researchers. However, as we have argued throughout this report, the availability of model weights is insufficient from both the perspective of organizational control and academic transparency.

Other organization- and task-specific models. In part as a reaction to the sovereignty considerations discussed throughout this report, organizations are increasingly adapting models to their own tasks, seeking better performance, lower costs, and greater control over model behavior. These efforts share the common theme of attempting to reduce dependence on proprietary frontier models by substituting open-weight backbones.

A blog post in June 2026 shared details on a collaboration between Thinking Machines and Bridgewater AIA Labs to build small models that capture human expertise on six information filtering tasks.<sup>40</sup> Not surprisingly, they demonstrated that small custom task-specific models, fine-tuned on high-quality proprietary datasets labeled by expert investors, can exceed frontier performance.

The legal AI startup Harvey introduced in August 2026 their Tenet model,<sup>41</sup> post-trained from a Kimi K3 base in collaboration with Fireworks research for long-horizon legal work. The blog post indicated promising results for both performance and cost-efficiency, and significant gains on complex, document-intensive tasks like M&A Diligence. The company followed up in September 2026 with another blog post<sup>42</sup> about training a recursive language model [Zhang et al., 2025a] harness in collaboration with Baseten.

Shopify provides another example of an organization training small models for specialized use cases, as disclosed by their CEO in September 2026.<sup>43</sup> Together, all of the instances discussed above represent public case studies of organizations grappling with AI sovereignty—control over their AI model usage from multiple perspectives (performance, cost, dependence, etc.).

## 9 Limitations

Don’t like this work? Or perhaps there are too many limitations? We’ll pre-empt criticisms in the form of anticipated objections in an attempt to self-own:

Little novelty. Yes, the often-used criticism and the favorite of reviewer 2. Project Greenhouse lays out an overall vision and in this report we deliver on the first project milestone. We concede that there is nothing particularly novel about each of our steps, which build on well-known methods from the literature, except that to our knowledge, we are the first to have put everything together in this way.

Reranking is too easy. Yes, it could indeed be the case that reranking is too easy for any modern LLM, and this is why we are able to demonstrate competitive effectiveness with such a simple two-step recipe. Even if true, this finding is entirely orthogonal to the claims we make in this paper, which is focused on building fully open and sovereign models. We believe that our goals are possible with limited computational resources, which dictates the model sizes that we can practically explore. The fact that competitive rerankers are possible using small models is not inconsistent with our core thesis. We will follow up with additional explorations to better characterize the interplay between model size and task difficulty.

Agentic search is too hard. While we have demonstrated the viability of fully open and sovereign models for reranking, it might be the case that the full suite of agentic capabilities is beyond what can be achieved in a fully open and sovereign manner with limited computational resources.<sup>44</sup> In other words, a competitive reranker does not establish a path to a competitive agent.

Note that as discussed in Section 2, we do not prescribe any particular architecture, and it could be the case that non-open and non-sovereign models are still needed for at least some aspects of agentic search. For example, a frontier or near-frontier model might still be necessary for overall coordination, while delegating other aspects of search to smaller open and sovereign models. If this is the case, our core thesis will not be fully validated. This may very well be true, but we won’t know unless we try—and along the way, we’re sure much will be learned. We’ll try it and report back.

You’ve just moved the sovereignty boundary upstream. Based on our model of LLM training as hypergraph traversal (Section 3), we defined sovereignty in terms of tracing provenance to publicly available datasets such as ClimbMix [Diao et al., 2025] and RLHN-250K [Thakur et al., 2025]. As we conceded, this is based on judgment, but we might be criticized for drawing arbitrary boundaries. For example, datasets can contain undisclosed filtering, synthetic generation, annotation decisions, and dependencies on other models.

A skeptical reader could ask why treating a dataset as a trusted starting point is acceptable while treating a model checkpoint that way is not. We argue that this distinction is defensible because access to individual training instances permits inspection and removal, which is not possible with access to model weights alone. Perhaps a refinement of our sovereignty claim is the necessity of explicitly declaring provenance boundaries.

You could have just started with an existing fully open backbone. While we have argued that open-weight models are not sufficient to achieve our goals, these arguments do not necessitate our specific development path, specifically, pre-training our own backbone from scratch (Section 5). Fine-tuning an existing small, fully open backbone might provide the same upstream transparency and perhaps better effectiveness or lower development cost.

This may perhaps be the case, but we make several important counterpoints: Our claims primarily center around providing an “existence proof” of what’s possible (i.e., sufficiency); we never claimed that our path is uniquely necessary. From the perspective of reproducibility and pedagogy, however, it was desirable for us to pre-train a backbone from scratch: the backbone described in Section 5 is simply what we ended up with. Indeed, it would be interesting future work to extend the experiments in Section 7.1 to include other open backbones.

You didn’t try this . . . Yes, indeed. There are many data and training conditions that we did not have time to explore and document in this report. We attempted to strike a balance between velocity and achieving our milestone objective, separating the experiments that are necessary to support our claims from the “nice to have” interesting extensions. Of course, more experiments can improve our understanding of LLM behavior. However, we believe that our headline result provides a small step in demonstrating the veracity of our core thesis. We invite the community to contribute to our efforts and believe that our hypergraph model of LLM training provides a lightweight mechanism for coordinating experimental activity across diverse research groups around the world.

You’re overfitting. Perhaps. While some of our experiments can be considered out of domain (see Section 4.1), they are not as thorough as we would have liked, and this is an admitted weakness of our present results. However, this limitation does not detract from the central claims in this report about model openness and sovereignty. The standard tricks that the community has developed to promote generalization and guard against overfitting remain applicable to our training pipeline, and there is no reason why we can’t also adopt those techniques. We will follow up in subsequent work to explicitly explore this question of out-of-domain generalization and report back.

You’ll never win. A provider of proprietary or non-sovereign models can always distill down to whatever model size and scope of capabilities that we’re targeting. In that way, we will always be dominated along the Pareto frontier and will never achieve “best in class” status. This is likely true, but our objective was never to “win”. Our thesis centers on demonstrating capabilities in a fully open and sovereign manner, with the bar of achieving “competitive” effectiveness (as opposed to achieving the state of the art). In this way, our models will support something that no proprietary or non-sovereign model can: transparency and the rigorous understanding that comes along with it.

## 10 Conclusion

Project Greenhouse intends to be an ongoing, long-term project and this report lays out our goals at a high level, including arguments on why we believe our efforts are important and potentially impactful. Our goals are ambitious but, we feel, achievable, and here we deliver the first project milestone. We describe how to build a competitive pointwise decoder-only reranker with a simple two-step recipe comprising pre-training from scratch and supervised fine-tuning, starting only from commonly available datasets. As we do not rely on existing open-weight backbones from third parties, we are fully in control of model training, from end to end. The modest computational requirements of our experiments mean that it is feasible for others to reproduce and to build on our results.

The path forward is clear for us: having laid out our vision and delivered the first project milestone, we’ll simply continue and do what we said we would. We extend an invitation for other like-minded academic research groups, organizations, and independent researchers to join us on our journey. Otherwise, stay tuned for the next installment!

## Acknowledgments and Disclosure of Funding

This research was supported in part by the Natural Sciences and Engineering Research Council of Canada (NSERC) and GPUs from the Digital Research Alliance of Canada.

## References

Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, Andrés Marafioti, Hynek Kydlícek, Agustín Piqueres Lajarín, Vaibhav Srivastav,ˇ Joshua Lochner, Caleb Fahlgren, Xuan-Son Nguyen, Clémentine Fourrier, Ben Burtenshaw, Hugo Larcher, Haojun Zhao, Cyril Zakka, Mathieu Morlon, Colin Raffel, Leandro von Werra, and Thomas Wolf. SmolLM2: When Smol goes big – data-centric training of a small language model. arXiv:2502.02737, 2025.

Parishad BehnamGhader, Vaibhav Adlakha, Marius Mosbach, Dzmitry Bahdanau, Nicolas Chapados, and Siva Reddy. LLM2Vec: Large language models are secretly powerful text encoders. In Proceedings ofthe First Conference on Language Modeling, 2024.

Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika, and Oskar Van Der Wal. Pythia: A suite for analyzing large language models across training and scaling. In Proceedings of the 40th International Conference on Machine Learning, pages 2397–2430, 2023.

BigScience Workshop. BLOOM: A 176B-parameter open-access multilingual language model. arXiv:2211.05100, 2022.

Maarten Buyl, Alexander Rogiers, Sander Noels, Iris Dominguez-Catena, Edith Heiter, Raphael Romero, Iman Johary, Alexandru-Cristian Mara, Jefrey Lijffijt, and Tijl De Bie. Large language models reflect the ideology of their creators. arXiv:2410.18417, 2024.

Shengzhuang Chen, Jerrod Parker, Yejin Bang, Andrew M. Bean, Nabeel Seedat, Stefan Winzeck, Daniil Glazko, Jannik Zgraggen, Fangyi Yu, Scott Arnott, Dietrich Trautmann, Luca Ciuffreda, Guglielmo Bonifazi, Davide Romano, Bradley Bell, Kirsty Fielding, Daniele Giofrè, Tom Zielund, Ipshita Chatterjee, Sneha Murthy Ghantasala, Manpreet Nanreh, John Scoville, Maciej Sakowicz, Wassim Seifeddine, Lukas Thede, and Jonathan Richard Schwarz. Thomson: Continual learning of frontier models for SovereignAI. arXiv:2608.27147, 2026.

Zijian Chen, Ronak Pradeep, and Jimmy Lin. Accelerating listwise reranking: Reproducing and enhancing FIRST. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 3165–3172, Padua, Italy, 2025.

Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, and Ellen M. Voorhees. Overview of the TREC 2019 Deep Learning Track. In Proceedings of the Twenty-Eighth Text REtrieval Conference (TREC 2019), Gaithersburg, Maryland, 2019.

Nick Craswell, Bhaskar Mitra, Emine Yilmaz, and Daniel Campos. Overview of the TREC 2020 Deep Learning Track. In Proceedings of the Twenty-Ninth Text REtrieval Conference (TREC 2020), Gaithersburg, Maryland, 2020.

Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, and Jimmy Lin. Overview of the TREC 2021 Deep Learning Track. In Proceedings of the Thirtieth Text REtrieval Conference (TREC 2021), Gaithersburg, Maryland, 2021.

Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, Jimmy Lin, Ellen M. Voorhees, and Ian Soboroff. Overview of the TREC 2022 Deep Learning Track. In Proceedings ofthe Thirty-First Text REtrieval Conference (TREC 2022), Gaithersburg, Maryland, 2022.

Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Hossein A. Rahmani, Daniel Campos, Jimmy Lin, Ellen M. Voorhees, and Ian Soboroff. Overview of the TREC 2023 Deep Learning Track. In Proceedings of the Thirty-Second Text REtrieval Conference (TREC 2023), Gaithersburg, Maryland, 2023.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota, 2019.

Shizhe Diao, Yu Yang, Yonggan Fu, Xin Dong, Dan Su, Markus Kliegl, Zijia Chen, Peter Belcak, Yoshi Suhara, Hongxu Yin, Mostofa Patwary, Yingyan (Celine) Lin, Jan Kautz, and Pavlo Molchanov. Nemotron-CLIMB: Clustering-based iterative data mixture bootstrapping for language model pre-training. In Advances in Neural Information Processing Systems 38 (NeurIPS 2025) Datasets and Benchmarks Track, San Diego, California, 2025.

John Donaghy, Brian Wilcox, Oguzhan Ersoy, Shikhar Rastogi, Adam St Arnaud, Alexey Titov,˘ Jordan Greenberg, Ben Fielding, and Harry Grieve. OPEN-1B: A fully auditable training run. arXiv:2609.17380, 2026.

Shangbin Feng, Chan Young Park, Yuhan Liu, and Yulia Tsvetkov. From pretraining data to language models to downstream tasks: Tracking the trails of political biases leading to unfair NLP models. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 11737–11762, Toronto, Canada, July 2023.

Luyu Gao, Zhuyun Dai, and Jamie Callan. Rethink training of BERT rerankers in multi-stage retrieval pipeline. In Advances in Information Retrieval, pages 280–286, 2021.

Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. Precise zero-shot dense retrieval without relevance labels. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1762–1777, Toronto, Canada, 2023.

Gemma Team. Gemma 4 technical report. arXiv:2607.02770, 2026.

Marco Gerosa, Anna Hermansen, Anni Lai, and Adrienn Lawson. The state of sovereign AI: Exploring the role of open source projects and global collaboration in global AI strategy, 2025.

Dirk Groeneveld, Iz Beltagy, Evan Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, Shane Arora, David Atkinson, Russell Authur, Khyathi Chandu, Arman Cohan, Jennifer Dumas, Yanai Elazar, Yuling Gu, Jack Hessel, Tushar Khot, William Merrill, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew Peters, Valentina Pyatkin, Abhilasha Ravichander, Dustin Schwenk, Saurabh Shah, William Smith, Emma Strubell, Nishant Subramani, Mitchell Wortsman, Pradeep Dasigi, Nathan Lambert, Kyle Richardson, Luke Zettlemoyer, Jesse Dodge, Kyle Lo, Luca Soldaini, Noah Smith, and Hannaneh Hajishirzi. OLMo: Accelerating the science of language models. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15789–15809, Bangkok, Thailand, August 2024.

Lingwei Gu, Nour Jedidi, and Jimmy Lin. NanoKnow: How to know what your language model knows. In Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 3213–3219, Melbourne, Australia, 2026.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In Proceedings of the Ninth International Conference on Learning Representations (ICLR 2021), 2021.

Sebastian Hofstätter, Sophia Althammer, Michael Schröder, Mete Sertkan, and Allan Hanbury. Improving efficient neural ranking models with cross-architecture knowledge distillation. arXiv:2010.02666, 2020.

Tz-Huan Hsu, Jheng-Hong Yang, and Jimmy Lin. Rethinking agentic search with Pi-Serini: Is lexical retrieval sufficient? arXiv:2605.10848, 2026.

Evan Hubinger, Carson Denison, Jesse Mu, Mike Lambert, Meg Tong, Monte MacDiarmid, Tamera Lanham, Daniel M. Ziegler, Tim Maxwell, Newton Cheng, Adam Jermyn, Amanda Askell, Ansh Radhakrishnan, Cem Anil, David Duvenaud, Deep Ganguli, Fazl Barez, Jack Clark, Kamal Ndousse, Kshitij Sachan, Michael Sellitto, Mrinank Sharma, Nova DasSarma, Roger Grosse,

Shauna Kravec, Yuntao Bai, Zachary Witten, Marina Favaro, Jan Brauner, Holden Karnofsky, Paul Christiano, Samuel R. Bowman, Logan Graham, Jared Kaplan, Sören Mindermann, Ryan Greenblatt, Buck Shlegeris, Nicholas Schiefer, and Ethan Perez. Sleeper agents: Training deceptive LLMs that persist through safety training. arXiv:2401.05566, 2024.

Nour Jedidi, Yung-Sung Chuang, James R. Glass, and Jimmy Lin. Don’t “overthink” pointwise reranking: Is reasoning truly necessary? In Findings of the Association for Computational Linguistics: EMNLP 2026, Budapest, Hungary, 2026.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Ö. Arık, Dong Wang, Hamed Zamani, and Jiawei Han. Search-R1: Training LLMs to reason and leverage search engines with reinforcement learning. In Proceedings ofthe Second Conference on Language Modeling (COLM 2025), 2025.

K2 Team. K2-V2: A 360-open, reasoning-enhanced LLM. arXiv:2512.06201, 2025.

Nikhil Kandpal, Brian Lester, Colin Raffel, Sebastian Majstorovic, Stella Biderman, Baber Abbasi, Luca Soldaini, Enrico Shippole, A. Feder Cooper, Aviya Skowron, John Kirchenbauer, Shayne Longpre, Lintang Sutawika, Alon Albalak, Zhenlin Xu, Guilherme Penedo, Loubna Ben Allal, Elie Bakouch, John David Pressman, Honglu Fan, Dashiell Stander, Guangyu Song, Aaron Gokaslan, Tom Goldstein, Brian R. Bartoldson, Bhavya Kailkhura, and Tyler Murray. The Common Pile v0.1: An 8TB dataset of public domain and openly licensed text. arXiv:2506.05209, 2025.

Keita Kurita, Paul Michel, and Graham Neubig. Weight poisoning attacks on pretrained models. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 2793–2806, 2020.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, and Hannaneh Hajishirzi. Tulu 3: Pushing frontiers in open language model post-training. arXiv:2411.15124, 2024.

Pierre-Carl Langlais, Pavel Chizhov, Mattia Nee, Carlos Rosas Hinostroza, Matthieu Delsart, Irène Girard, Othman Hicheur, Anastasia Stasenko, and Ivan P. Yamshchikov. Even small reasoners should quote their sources: Introducing the Pleias-RAG model family. arXiv:2504.18225, 2025.

Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Gadre, Hritik Bansal, Etash Guha, Sedrick Keh, Kushal Arora, Saurabh Garg, Rui Xin, Niklas Muennighoff, Reinhard Heckel, Jean Mercat, Mayee Chen, Suchin Gururangan, Mitchell Wortsman, Alon Albalak, Yonatan Bitton, Marianna Nezhurina, Amro Abbas, Cheng-Yu Hsieh, Dhruba Ghosh, Josh Gardner, Maciej Kilian, Hanlin Zhang, Rulin Shao, Sarah Pratt, Sunny Sanyal, Gabriel Ilharco, Giannis Daras, Kalyani Marathe, Aaron Gokaslan, Jieyu Zhang, Khyathi Chandu, Thao Nguyen, Igor Vasiljevic, Sham Kakade, Shuran Song, Sujay Sanghavi, Fartash Faghri, Sewoong Oh, Luke Zettlemoyer, Kyle Lo, Alaaeldin El-Nouby, Hadi Pouransari, Alexander Toshev, Stephanie Wang, Dirk Groeneveld, Luca Soldaini, Pang Wei Koh, Jenia Jitsev, Thomas Kollar, Alexandros G. Dimakis, Yair Carmon, Achal Dave, Ludwig Schmidt, and Vaishaal Shankar. DataComp-LM: In search of the next generation of training sets for language models. arXiv:2406.11794, 2024.

Yuan Li, Qi Luo, Xiaonan Li, Bufan Li, Qinyuan Cheng, Bo Wang, Yining Zheng, Yuxin Wang, Zhangyue Yin, and Xipeng Qiu. R3-RAG: Learning step-by-step reasoning and retrieval for LLMs via reinforcement learning. arXiv:2505.23794, 2025.

Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. Textbooks are all you need II: phi-1.5 technical report. arXiv:2309.05463, 2023.

Zhuofeng Li, Haoxiang Zhang, Cong Wei, Pan Lu, Ping Nie, Yi Lu, Yuyang Bai, Shangbin Feng, Hangxiao Zhu, Ming Zhong, Yuyu Zhang, Jianwen Xie, Yejin Choi, James Zou, Jiawei Han, Wenhu Chen, Jimmy Lin, Dongfu Jiang, and Yu Zhang. Beyond semantic similarity: Rethinking retrieval for agentic search via direct corpus interaction. arXiv:2605.05242, 2026.

Jimmy Lin. A proposed conceptual framework for a representational approach to information retrieval. arXiv:2110.01529, 2021.

Jimmy Lin. Building a culture of reproducibility in academic research. arXiv:2212.13534, 2022.

Jimmy Lin, Xueguang Ma, Sheng-Chieh Lin, Jheng-Hong Yang, Ronak Pradeep, and Rodrigo Nogueira. Pyserini: A Python toolkit for reproducible information retrieval research with sparse and dense representations. In Proceedings of the 44th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2356–2362, 2021.

Zhengzhong Liu, Aurick Qiao, Willie Neiswanger, Hongyi Wang, Bowen Tan, Tianhua Tao, Junbo Li, Yuqi Wang, Suqi Sun, Omkar Pangarkar, Richard Fan, Yi Gu, Victor Miller, Yonghao Zhuang, Guowei He, Haonan Li, Fajri Koto, Liping Tang, Nikhil Ranjan, Zhiqiang Shen, Xuguang Ren, Roberto Iriondo, Cun Mu, Zhiting Hu, Mark Schulze, Preslav Nakov, Tim Baldwin, and Eric P. Xing. LLM360: Towards fully transparent open-source LLMs. arXiv:2312.06550, 2023.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In Proceedings of the Seventh International Conference on Learning Representations (ICLR 2019), 2019.

Kairong Luo, Jiarui Cui, Yaorui Yin, Shengqi Chen, Yiming Yang, Linxiang Gao, Yanmohan Wang, Chengxia Li, Mingzhe Zhang, Kaifeng Lyu, and Wenguang Chen. Puro-2B: Poor Lab’s Qwen2- 1.5B trained on RTX 5090 within \$5090. arXiv:2608.27370, 2026a.

Kun Luo, Hongjin Qian, Zheng Liu, Ziyi Xia, Shitao Xiao, Zhao Cao, Siqi Bao, Jun Zhao, and Kang Liu. Reinforcing agentic search via reward density optimization. In Proceedings ofthe 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10261–10283, San Diego, California, 2026b.

Shengyu Mao, Yong Jiang, Boli Chen, Xiao Li, Peng Wang, Xinyu Wang, Pengjun Xie, Fei Huang, Huajun Chen, and Ningyu Zhang. RaFe: Ranking feedback improves query rewriting for RAG. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 884–901, Miami, Florida, November 2024.

Sam McCandlish, Jared Kaplan, Dario Amodei, and OpenAI Dota Team. An empirical model of large-batch training. arXiv:1812.06162, 2018.

William Merrill, Yanhong Li, Tyler Romero, Anej Svete, Caia Costello, Pradeep Dasigi, Dirk Groeneveld, David Heineman, Bailey Kuehl, Nathan Lambert, Chuan Li, Kyle Lo, Saumya Malik, DJ Matusz, Benjamin Minixhofer, Jacob Morrison, Luca Soldaini, Finbarr Timbers, Pete Walsh, Noah A. Smith, Hannaneh Hajishirzi, and Ashish Sabharwal. Olmo Hybrid: From theory to practice and back. arXiv:2604.03444, 2026.

Rodrigo Nogueira and Kyunghyun Cho. Passage re-ranking with BERT. arXiv:1901.04085, 2019.

Rodrigo Nogueira, Wei Yang, Kyunghyun Cho, and Jimmy Lin. Multi-stage document ranking with BERT. arXiv:1910.14424, 2019.

Rodrigo Nogueira, Zhiying Jiang, Ronak Pradeep, and Jimmy Lin. Document ranking with a pretrained sequence-to-sequence model. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 708–718, 2020.

Palantir. Institutional sovereignty in the age of AI, 2026.

Andrei Panferov, Davit Melikidze, Martin Jaggi, and Dan Alistarh. Apertus LLM family expansion via distillation and quantization. arXiv:2605.29128, 2026.

Ronak Pradeep, Rodrigo Nogueira, and Jimmy Lin. The Expando-Mono-Duo design pattern for text ranking with pretrained sequence-to-sequence models. arXiv:2101.05667, 2021.

Ronak Pradeep, Yuqi Liu, Xinyu Zhang, Yilin Li, Andrew Yates, and Jimmy Lin. Squeezing water from a stone: A bag of tricks for further improving cross-encoder effectiveness for reranking. In Advances in Information Retrieval, pages 655–670, 2022.

Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy Lin. RankVicuna: Zero-shot listwise document reranking with open-source large language models. arXiv:2309.15088, 2023a.

Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy Lin. RankZephyr: Effective and robust zero-shot listwise reranking is a breeze! arXiv:2312.02724, 2023b.

Project Apertus. Apertus: Democratizing open and compliant LLMs for global language environments. arXiv:2509.14233, 2025.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026a.

Qwen Team. Qwen3.8-Max: A new bar for coding and cowork, August 2026b.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners, 2019.

Miguel Moura Ramos, Duarte M. Alves, Hippolyte Gisserot-Boukhlef, João Alves, Pedro Henrique Martins, Patrick Fernandes, José Pombal, Nuno M. Guerreiro, Ricardo Rei, Nicolas Boizard, Amin Farajian, Mateusz Klimaszewski, José G. C. de Souza, Barry Haddow, François Yvon, Pierre Colombo, Alexandra Birch, and André F. T. Martins. EuroLLM-22B: Technical report. arXiv:2602.05879, 2026.

Stephen Robertson and Hugo Zaragoza. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 3(4):333–389, 2009.

Rulin Shao, Akari Asai, Shannon Zejiang Shen, Hamish Ivison, Varsha Kishore, Jingming Zhuo, Xinran Zhao, Molly Park, Samuel G. Finlayson, David Sontag, Tyler Murray, Sewon Min, Pradeep Dasigi, Luca Soldaini, Faeze Brahman, Wen-tau Yih, Tongshuang Wu, Luke Zettlemoyer, Yoon Kim, Hannaneh Hajishirzi, and Pang Wei Koh. DR Tulu: Reinforcement learning with evolving rubrics for deep research. In Proceedings of the 43rd International Conference on Machine Learning, pages 110046–110101, 2026.

Sahel Sharifymoghaddam, Ronak Pradeep, Andre Slavescu, Ryan Nguyen, Andrew Xu, Zijian Chen, Yilin Zhang, Yidi Chen, Jasper Xian, and Jimmy Lin. RankLLM: A Python package for reranking with LLMs. In Proceedings ofthe 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 3681–3690, Padua, Italy, 2025.

Samuel L. Smith, Pieter-Jan Kindermans, Chris Ying, and Quoc V. Le. Don’t decay the learning rate, increase the batch size. In Proceedings ofthe Sixth International Conference on Learning Representations (ICLR 2018), 2018.

Jacob Springer, Suhas Kotha, Daniel Fried, Graham Neubig, and Aditi Raghunathan. Repetition improves language model embeddings. In Proceedings of the Thirteenth International Conference on Learning Representations (ICLR 2025), 2025.

Team OLMo. 2 OLMo 2 furious. arXiv:2501.00656, 2024.

Team Olmo. Olmo 3. arXiv:2512.13961, 2025.

Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In Proceedings ofthe Neural Information Processing Systems Track on Datasets and Benchmarks, 2021.

Nandan Thakur, Crystina Zhang, Xueguang Ma, and Jimmy Lin. Hard negatives, hard lessons: Revisiting training data quality for robust information retrieval with LLMs. In Findings of the Associationfor Computational Linguistics: EMNLP 2025, pages 9064–9083, 2025.

Shivani Upadhyay, Ronak Pradeep, Nandan Thakur, Nick Craswell, and Jimmy Lin. UMBRELA: UMbrela is the (open-source reproduction of the) Bing RELevance Assessor. arXiv:2406.06519, 2024.

Shivani Upadhyay, Ronak Pradeep, Nandan Thakur, Daniel Campos, Nick Craswell, Ian Soboroff, and Jimmy Lin. A large-scale study of relevance assessments with large language models using UMBRELA. In Proceedings of the 2025 International ACM SIGIR Conference on Innovative Concepts and Theories in Information Retrieval, pages 358–368, Padua, Italy, 2025.

Feng Wang, Yuqing Li, and Han Xiao. jina-reranker-v3: Last but not late interaction for listwise document reranking. arXiv:2509.25085, 2025.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. MMLU-Pro: A more robust and challenging multi-task language understanding benchmark. In Advances in Neural Information Processing Systems 37 (NeurIPS 2024) Datasets and Benchmarks Track, 2024.

Orion Weller, Kathryn Ricci, Eugene Yang, Andrew Yates, Dawn Lawrie, and Benjamin Van Durme. Rank1: Test-time compute for reranking in information retrieval. In Proceedings ofthe Second Conference on Language Modeling (COLM 2025), 2025.

Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and Ludwig Schmidt. Model soups: Averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In Proceedings ofthe 39th International Conference on Machine Learning, pages 23965–23998, 2022.

Jialong Wu, Baixuan Li, Runnan Fang, Wenbiao Yin, Liwen Zhang, Zhengwei Tao, Dingchu Zhang, Zekun Xi, Gang Fu, Yong Jiang, Pengjun Xie, Fei Huang, and Jingren Zhou. WebDancer: Towards autonomous information seeking agency. arXiv:2505.22648, 2025.

Jiashu Xu, Mingyu Ma, Fei Wang, Chaowei Xiao, and Muhao Chen. Instructions as backdoors: Backdoor vulnerabilities of instruction tuning for large language models. In Proceedings of the 2024 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3111–3126, Mexico City, Mexico, 2024.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. Qwen2 technical report. arXiv:2407.10671, 2024.

Yejun Yoon, Jaeyoon Jung, Seunghyun Yoon, and Kunwoo Park. Hypothetical documents or knowledge leakage? Rethinking LLM-based query expansion. In Findings of the Association for Computational Linguistics: ACL 2025, pages 19170–19187, Vienna, Austria, 2025.

Alex L. Zhang, Tim Kraska, and Omar Khattab. Recursive language models. arXiv:2512.24601, 2025a.

Ge Zhang, Scott Qu, Jiaheng Liu, Chenchen Zhang, Chenghua Lin, Chou Leuang Yu, Danny Pan, Esther Cheng, Jie Liu, Qunshu Lin, Raven Yuan, Tuney Zheng, Wei Pang, Xinrun Du, Yiming Liang, Yinghao Ma, Yizhi Li, Ziyang Ma, Bill Lin, Emmanouil Benetos, Huan Yang, Junting Zhou, Kaijing Ma, Minghao Liu, Morry Niu, Noah Wang, Quehry Que, Ruibo Liu, Sine Liu, Shawn Guo, Soren Gao, Wangchunshu Zhou, Xinyue Zhang, Yizhi Zhou, Yubo Wang, Yuelin Bai, Yuhan Zhang, Yuxiang Zhang, Zenith Wang, Zhenzhu Yang, Zijian Zhao, Jiajun Zhang, Wanli Ouyang, Wenhao Huang, and Wenhu Chen. MAP-Neo: Highly capable and transparent bilingual large language model series. arXiv:2405.19327, 2024.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv:2506.05176, 2025b.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems 36 (NeurIPS 2023) Datasets and Benchmarks Track, pages 46595–46623, New Orleans, Louisiana, 2023.

Shengyao Zhuang, Xueguang Ma, Bevan Koopman, Jimmy Lin, and Guido Zuccon. Rank-R1: Enhancing reasoning in LLM-based document rerankers via reinforcement learning. arXiv:2503.06034, 2025.

## A Listwise Reranking Prompt

Figure 4 reproduces the RankLLM prompt used for the Gemma-4, Qwen3.8, and GPT-6.1 Sol listwise reranker baselines in Section 4, taken from Sharifymoghaddam et al. [2025]. We used the qwen\_non\_thinking\_template.yaml template from RankLLM revision 443adf3.<sup>45</sup>

Qwen3.8 used the system message as shown. For Gemma-4, we omitted the Qwen-specific /no\_think line and disabled thinking through the model’s chat template. For GPT-6.1 Sol, we implemented our listwise reranker using Pi, concatenating the RankLLM template’s system message prefix (omitting the /no\_think line), passage entries, and suffix into a single user message and invoking Pi with thinking minimal. In Pi, we disabled all tools, extensions, skills, and automatic context-file loading, so the model does not have access to tools for reading files, executing commands, or accessing external resources. For each sliding window, we substituted its passage count for {num}, the query for {query}, and each passage’s identifier and text for {rank} and {candidate}. The user message concatenates the prefix, one body entry per passage, and the suffix.

## B Jev Shared-Rubric Prompt

Figure 5 shows the shared-rubric request used for Jev’s TREC DL results in Section 4. We used the jev-latest model from TypeSafe AI and supplied the query, all top-100 BM25 candidates in retrieval order, and the four-level TREC relevance-grade rubric in a single request, with one score question per passage.

We ranked passages by the API’s returned expected grade, $\begin{array} { r } { s _ { i } = \sum _ { q = 0 } ^ { 3 } g p _ { i } ( g ) } \end{array}$ , where $p _ { i } ( g )$ is the probability assigned to relevance grade g for passage i. Score ties retained the original BM25 order. We used the same query and passage truncation limits as Gaggle (base reranker), 128 and 512 tokens, respectively, with cl100k\_base tokenization.

## C Supervised Fine-Tuning Configuration

This appendix records detailed configuration settings for the fine-tuning procedure in Section 6.1. Baseline runs 1–4 denote the four individual causal rerankers described in Section 6.2. The soup denotes their parameter-averaged checkpoint, used for the headline results in Section 4.

Specifically:

• gaggle-nanochat-pretrained-climbmix-20260924 provides the initialization for the causal Gaggle rerankers;

• gaggle-nanochat-pretrained-climbmix-bidirectional-20260929 provides the initialization for the Gaggle (MNTP) rerankers.

MNTP identifies the bidirectionally adapted backbone, not a separate relevance-training objective.

Data sampling and microbatching. For each training instance, we sampled one positive passage and K hard negatives from the query’s available passages. We chose K as the largest value in {7, 15, 23} supported by its negative pool. Of the 247,534 baseline instances, 1.93% used seven negatives, 45.85% used 15, and 52.21% used 23, forming groups of 8, 16, or 24 query–passage pairs. Instances were bucketed by K so that each microbatch contained groups of the same size. Each group contributed one contrastive loss, computed jointly over its positive and sampled negatives.

Within each bucket, we packed complete query groups into per-GPU microbatches, padding the query–passage sequences to the longest sequence in the microbatch. Packing was limited by both an estimated padded-token budget and a group-count cap. Token estimates included the prompt, query and passage lengths, and a 30% safety margin. The token budget accounted for GPU memory reserved for model parameters, gradients, and optimizer states, with a cap of 60,000 estimated tokens per GPU. Baseline runs 1, 3, and 4 used microbatch budgets of 39,312 tokens per GPU; run 2 used 39,347. Microbatches contained at most 64, 32, or 21 query groups for K = 7, 15, or 23, respectively, limiting them to approximately 512 query–passage sequences.

![](images/a7527f725541d3fdfb65c573937aa4ef63d7a3cac11fd73390e06c0fc92274e0.jpg)  
Figure 4: Listwise prompt used for zero-shot reranking with Qwen3.8, Gemma-4, and GPT-6.1 Sol. The ellipsis denotes repeated passage entries; Gemma-4 and GPT-6.1 Sol omitted the opening /no\_think directive.

```jsonl
{
"state": {
"query": "<query>",
"passages": {"P1": "<passage 1>", "P2": "<passage 2>"},
"relevance_levels": {
"Irrelevant": "Irrelevant: the passage has nothing to do with the query.",
"Related": "Related: the passage seems related to the query but does not answer
it.",
"Highly relevant": "Highly relevant: the passage has some answer for the query, but
the answer may be a bit unclear, or hidden amongst extraneous information.",
"Perfectly relevant": "Perfectly relevant: the passage is dedicated to the query and
contains the exact answer."
}
},
"questions": {
"P1": {
"type": "score",
"instructions": "How relevant is passage P1 to the query?",
"criteria": ["Irrelevant", "Related", "Highly relevant", "Perfectly relevant"]
},
"P2": {
"type": "score",
"instructions": "How relevant is passage P2 to the query?",
"criteria": ["Irrelevant", "Related", "Highly relevant", "Perfectly relevant"]
}
}
}
```  
Figure 5: Jev reranking prompt using the four-level TREC relevance-grade rubric at evaluation time. Two passages illustrate the request structure; each evaluated request contained all 100 BM25 candidates and one corresponding question per passage. The rubric definitions and question wording are reproduced verbatim, with placeholders for the query and passage text. Passages were ordered by the returned expected relevance grades.

Gradient accumulation across microbatches and GPUs targeted an effective batch of 128 query groups per optimizer update. When a single group exceeded the microbatch budget, gradient caching processed its sequences in smaller chunks while retaining the joint LCE loss over all candidates. Baseline run 4 retained run 1’s packing plan and used gradient caching with execution chunks of up to 16,384 tokens on its H100 GPUs.

Optimization. Unless specified otherwise, we fine-tuned all retained parameters for one epoch using AdamW [Loshchilov and Hutter, 2019] with a learning rate of 5 × 10<sup>−5</sup>, β = (0.9, 0.95), weight decay of 0.01, and gradient clipping at a global norm of 12. The learning rate increased linearly for 150 steps and then followed a cosine decay to zero at the final update. Baseline run 1 used seed 17 and 1,931 updates. Baseline training used two GPUs with gradient checkpointing.

We used a common optimization recipe and changed the factors specified by each experiment. Apart from the diagnostic learning-rate comparison for Gemma and MiniCPM reported in Section 7.1, we did not perform hyperparameter tuning. The comparisons characterize effectiveness under this common recipe.

Evaluation and numerical precision. All SFT comparisons used top-100 BM25 candidates and nDCG@10, with means computed separately over TREC Deep Learning 2019–2023 and the seven BEIR collections (TREC-COVID, TREC-News, Robust04, NFCorpus, SciFact, SCIDOCS, and FiQA). The four baseline runs were evaluated using the same harness and inputs as the soup checkpoint. Table 11 records training and evaluation precision in one place.

Hardware and execution. Baseline runs 1–3 used two NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs (96 GB each), while run 4 used two H100 GPUs (80 GB each). Runs 1, 3, and 4 completed 1,931 updates; run 2 completed 1,933. The self-filtering, self-distillation, causal QPQ, and standard MNTP variants used the same hardware and microbatch budget as baseline run 1. Both self-filtered variants completed 1,918 updates. The RLHN-680K, Tevatron MS MARCO, meanpooling, and mask-appended variants used two H200 NVL GPUs (141 GB each) with budgets of 60,000 tokens per GPU. The RLHN-680K and Tevatron runs completed 5,060 and 3,112 updates, respectively; the prompt and pooling variants completed 1,928–1,931 updates. The nanochat-d34 run used four RTX A6000 GPUs (48 GB each), a budget of 11,121 tokens per GPU, and 1,929 updates. The 5×10<sup>−5</sup> Gemma-4-E2B and Qwen3.5-4B-Base runs used the baseline microbatch budget and completed 1,931 updates, using two RTX PRO GPUs and two H200 NVL GPUs, respectively. The $1 \times 1 \dot { 0 } ^ { - 5 }$ MiniCPM run used the same budget and completed 1,931 updates, on two RTX PRO GPUs for the first 1,683 updates and two H200 NVL GPUs for the rest. The $1 \times 1 0 ^ { - 5 }$ Gemma-4-E2B run used two H200 NVL GPUs, the baseline microbatch budget executed in chunks of at most 8,192 tokens per GPU, and completed 1,931 updates. The pointwise CE run used two NVIDIA RTX PRO 6000 Blackwell GPUs (96 GB each) and completed 1,931 updates, with microbatches of eight query–passage pairs per GPU.

<table><tr><td>Setting</td><td>Value</td></tr><tr><td>1 Fine-tuning</td><td>All retained parameters; no frozen layers or LoRA</td></tr><tr><td>2 Scoring parameters</td><td>Two pre-trained LM output rows: 4,352 parameters; no new head</td></tr><tr><td>3 Implementation</td><td>Native backbone; scaled dot-product attention (SDPA)</td></tr><tr><td>4 Training precision</td><td>FP32 weights, gradients, and optimizer states; TF32 matrix multiplication</td></tr><tr><td>5 Evaluation precision</td><td>BF16 backbone; FP32 scoring rows and scoring computation; final hidden states cast to FP32</td></tr><tr><td>6 Gradient checkpointing Enabled</td><td></td></tr><tr><td>7 Query / passage limits</td><td>128 / 512 tokens</td></tr><tr><td>8 Negatives per group</td><td>K ∈ {7, 15, 23}</td></tr><tr><td>9 Target effective batch</td><td>128 query groups</td></tr><tr><td>10 Microbatch group cap</td><td>max_groups=32, scaled by group size</td></tr><tr><td></td><td>11 Batch-memory estimate 1 GiB per 1,000 estimated tokens</td></tr><tr><td>12 Optimizer</td><td>AdamW, β = (0.9, 0.95)</td></tr><tr><td>13 Learning rate</td><td> $5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>14 Weight decay</td><td>0.01</td></tr><tr><td>15 Gradient clipping</td><td>Global norm 12</td></tr><tr><td>16 Schedule</td><td>150 warm-up updates, then cosine decay to zero</td></tr><tr><td>17 Training epochs</td><td>1 τ = 1</td></tr><tr><td>18 LCE temperature</td><td></td></tr><tr><td>19 Distributed training 20 Data workers</td><td>Two GPUs with distributed data parallelism 6</td></tr><tr><td></td><td>200 updates</td></tr><tr><td>21 Checkpoint interval</td><td></td></tr></table>

Table 11: Default settings for supervised fine-tuning and evaluating individual Gaggle rerankers. Run-specific exceptions are described in Appendix C.

Readout and checkpoint selection. The default readout used the last non-padded token; the maskappended prompt used the token immediately before the mask, and mean pooling averaged final hidden states over all non-padding tokens. The final checkpoint was used for each baseline evaluation; checkpoint-saving intervals were 200 updates for runs 1–2, 50 for run 3, and 100 for run 4. The soup was produced by parameter averaging without further optimization.