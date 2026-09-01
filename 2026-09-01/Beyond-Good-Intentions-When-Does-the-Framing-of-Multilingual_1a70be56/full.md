# Beyond Good Intentions: When Does the Framing of Multilingual and Low-Resource NLP Research Become a Caricature?

Nedjma Ousidhoum<sup>◦</sup>, Noopur Zambare<sup>⋄</sup>, Mohamed Abdalla<sup>⋄</sup> <sup>◦</sup>Cardiff University, <sup>⋄</sup>University of Alberta Contact: OusidhoumN@cardiff.ac.uk,mabdall2@ualberta.ca

## Abstract

Building language technologies and conducting NLP research for low-resource languages— particularly when led by native speakers or involving participatory research practices—are often framed as means of addressing inequality, serving local communities, and, at times, contributing to decolonisation. In this paper, we examine recently published NLP and ML papers, focusing on the narratives used to characterise multilinguality, low-resource languages, and underrepresented cultures. We propose a framework for analysing research framings and identify recurring rhetorical patterns that may hinder accountability and constrain equitable knowledge production for—and by— underserved communities. We further assess the evidential basis of assertions regarding community benefit and find that such statements are often weakly supported or left unsubstan tiated. Although community ownership and participation are frequently presented as key objectives, our analysis, supported by statistics from the ACL Anthology, suggests that research outputs more often prioritise resource creation and benchmarking—important but distinct goals—over evidence of broader structural change. We conclude by offering practical recommendations to help authors, reviewers, and readers critically assess these assertions and avoid potentially misleading framings.<sup>1</sup>

Warning: This work highlights potentially problematic scholarly practices. To avoid misrepresentation or undue targeting, we use fictionalised, deliberately absurd, satirical examples for illustrative purposes. These examples are not intended as commentary on real publications or individuals, particularly students and early-career researchers.

## 1 Introduction

The English-speaking community has long been excludedfrom the technological systems that shape our digital sphere. Without immediate   
intervention, this imbalance risks undermining   
cultural continuity, democratic participation,   
economic opportunity, and the preservation of   
Europe’s rich linguistic heritage. We invited a   
member ofthe affected community to contribute to our research. Through the creation ofa   
benchmark, we take an important step toward   
linguistic justice, digital inclusion, and ensuring that English has a meaningful place in the future of artificial intelligence.

The preceding introduction is intentionally satirical. Yet replacing “English” with languages frequently discussed in multilingual and low-resource NLP makes the underlying rhetorical structure unexpectedly familiar. The satirical premise is not that underrepresented or endangered languages are undeserving of computational attention—they often are (Liu et al., 2022). Rather, it is that multilingual NLP research can frame linguistic underrepresentation through narratives of urgency, moral responsibility, and anticipated social transformation that exceed what the technical contribution and available evidence can establish.

Clarifying research framings—what an algorithm, benchmark, dataset, or language technology is intended to do, why, by whom, and for whom— is essential for accountability (Schlichtkrull et al., 2023). Such framings shape how technical work is justified, how impact is anticipated (Chamoun et al., 2025), and how communities are represented (Liu et al., 2022). In multilingual and low-resource NLP, these choices may be particularly consequential because technical artefacts are often positioned as interventions in larger linguistic, social, cultural, or political issues (Ousidhoum et al., 2025). Papers may invoke goals such as language preservation, democratisation, decolonisation, community empowerment, inequality reduction, or digital inclusion (Bird, 2020). At the same time, uneven language coverage, unequal access to language technologies, and historical marginalisation are legitimate concerns, and non-English NLP work may also face inequities in how it is developed and evaluated (Barkhordar et al., 2026), including negative review bias and expectations that exceed the scope of the claims being made. These concerns, however, do not in themselves establish a causal link between a technical contribution and a societal outcome; the mechanisms connecting technical work to broader social impacts therefore need to be clearly articulated.

In this paper, we investigate recurring framings in multilingual and low-resource NLP, focusing on how motivations, methodological choices, and anticipated societal outcomes are connected (see Figure 1). Our focus is on whether research claims are proportionate to the evidence presented and whether the proposed mechanisms plausibly connect technical contributions to societal impacts. Specifically, we:

• identify recurring narrative structures linking linguistic underrepresentation, technical intervention, and claimed societal benefits;

• propose an analytical framework grounded in epistemic elements and research framings to examine how goals, methods, stakeholders, and anticipated outcomes are articulated;

• apply this framework to multilingual and community-oriented low-resource NLP research, examining assumptions about participation, benefit, and impact;

• investigate common patterns related to incentives, authorship, and the promotion of communityoriented multilingual NLP.

We find that social impact claims are often treated as natural consequences of resource creation, benchmark development, or model improvement. We also find that narratives of linguistic urgency sometimes appear to encourage assumptions about community needs, technological priorities, or anticipated benefits without sufficient corroborating evidence (Liu et al., 2022; Adamu, 2023; Ferreira, 2024). To encourage clearer alignment between technical contributions, supporting evidence, and societal claims, we conclude with an accountability checklist to help authors, reviewers, and readers critically evaluate multilingual and lowresource NLP papers and artefacts while maintaining proportionate expectations about their impact.

![](images/d63a5967b1ae7b5ff36ab57adb4570184e32e8732f415e52b5bc0494299e4fc3.jpg)  
Figure 1: Commonly observed rhetorical and methodological patterns in research framings. Motivations (highlighted in red) are used to establish a moral imperative, the fulfillment of which (in green) is framed as leading to social impact (in blue).

## 2 Related Work

Issues of accountability and responsible research have motivated several analyses of NLP practices, including studies of corporate influence in NLP and AI systems (Abdalla and Abdalla, 2021; Abdalla et al., 2023; Aitken et al., 2024), as well as investigations of how research is framed and how such narratives may shape accountability. For instance, Liu et al. (2023) reviewed summarisation papers and found that fewer than 15% meaningfully engaged with responsible AI concerns. Similarly, Schlichtkrull et al. (2023) formalised the notion of epistemic narratives and analysed automated fact-checking papers, identifying frequent misalignments between goals and methods, underspecified or absent stakeholder definitions, and weak connections between these elements. Chamoun et al. (2025) further extended this line of work by analysing more recent publications on automated fact-checking and hate speech, while also exploring automated approaches to research narrative analysis. Relatedly, Subramonian et al. (2024) examined how democratisation is conceptualised in NLP and ML research, finding limited engagement with established theories of democratisation despite frequent rhetorical invocation of the term.

Such studies reveal problematic research practices, broader trends in the field, and overlooked downstream risks (Birhane et al., 2022b). They also provide guidance for future work by encouraging stronger alignment between goals and methods, while prompting reflection on concerns such as dual use (Leins et al., 2020; Kaffee et al., 2023) and overclaiming (Grodzinsky et al., 2012). More broadly, ethical practices in AI, ML, and NLP have received considerable attention, including work on artefact documentation and transparency (Bender, 2011; Bender and Friedman, 2018; Gebru et al., 2021; Rogers et al., 2021; Mohammad, 2022), as well as best-practice recommendations for responsible research (Hollenstein et al., 2020).

Despite growing attention to participation and equity in multilingual and low-resource NLP (Joshi et al., 2020; Blasi et al., 2022; Dogruöz and˘ Sitaram, 2022), work in this area has shed light on the general state of the field and common practices. This includes challenges in data collection (Yu et al., 2022), the importance of involving speak ers and communities whose languages are being studied (Bird, 2020, 2022; Lent et al., 2022; Liu et al., 2022; Mager et al., 2023; Bird and Yibarbuk, 2024), corporate influence over language technologies and its implications for underrepresented linguistic communities (Held et al., 2023; Bird, 2025), and recent developments in large language model research (Mihalcea et al., 2024). However, relatively little attention has been paid to how research narratives and framings themselves shape method ological choices and justificatory claims in mul tilingual and low-resource NLP. Investigations of problematic epistemic narratives—such as an emphasis on numerical metrics and misalignment with social impact—have tended to focus on NLP and CL more broadly (Kogkalidis and Chatzikyriakidis, 2025). Meanwhile, critical discussions of building NLP artefacts for non-English, multilingual, and low-resource contexts have primarily focused on documentation and actionable recommendations (Bird and Yibarbuk, 2024; Ousidhoum et al., 2025). Efforts to make the review of non-English papers fairer have likewise addressed important aspects of the research process, including biased assumptions and a priori unrealistic expectations (Barkhordar et al., 2026). Nevertheless, we are not aware of any systematic investigation of problematic narratives, exaggerated claims, or research framings specifically in multilingual and low-resource NLP.

Addressing this gap is the primary focus of this paper.

## 3 Assessing Framings and Contributions

We select 100 papers addressing multilinguality broadly and low-resource languages in particular. We examine the claims made in these studies and how they contribute to broader research framings and narratives. Specifically, we analyse their abstracts and introductions, extracting relevant quotations and annotating them according to the elements defined below. Full annotation statistics are reported in the Appendix.

## 3.1 Paper Selection

We first identified well-known and highly cited studies in the field, then expanded the pool through citation chaining, publications from prominent research consortia, and keyword-based searches of the ACL Anthology using terms such as multilingual, low-resource, reviv, revival, decolonization, and decolon. We manually screened the resulting pool to exclude studies that were not relevant to our research questions. We make the list of NLP papers publicly available on GitHub, while annotations are available upon request.

## 3.2 Assessing Claimed Contributions

Following Schlichtkrull et al. (2023) and Chamoun et al. (2025), we examine passages pertaining to each paper’s motivation (why) and research focus (what), primarily extracting quotations from the introductions and consulting abstracts when relevant information is missing. From these excerpts, we identify the means (applications) and ends (goals) of the work and characterise its narrative (framing) based on the relationships among these elements. We additionally assess the extent to which stated goals are supported by evidence (i.e., evidential support in Table 1).

Annotation Process The annotation was conducted in two rounds by two expert annotators, both authors of this paper, following the recommendations of Krippendorff (2018). The process was iterative: we began with categories informed by an initial pilot study and introduced additional labels when needed. We employed a multi-label annotation scheme because individual passages could contain multiple relevant elements. The annotators achieved > 90% agreement. Examples from the final label set are shown in Table 1.

<table><tr><td>Ends (Goals)</td><td>Means (Applications)</td><td>Evidential Support</td></tr><tr><td>• Serving communities: developing artefacts for a specific community • Decolonisation: addressing historical or structural inequalities • Reducing inequality: promoting more equitable representation Improving evaluation: developing or enhancing benchmarks or metrics contributing new insights to research oping benchmarks/evaluation methods • Language revitalisation: preserving, Supporting language learners: e.g.,</td><td>Improving LLM performance: e.g., enhancing multilingual capabilities across tasks NLP applications: e.g., translation, NER, transcription, and other applica- tions • LLM safety: addressing safety and • Anecdotes isolated examples or per- ethical considerations</td><td>• Non-NLP scientific articles evidence drawn from other academic domains • Newspaper articles or polls media reports or collected public opinion • Prior NLP work: evidence drawn from another NLP paper sonal observations used as evidence • Advancing knowledge production: • Improving LLM evaluation: devel- • Expressions of threat: claims of po- tential harm that are not substantiated</td></tr></table>

Table 1: Annotated epistemic elements and main categories used in our analysis. The ends refer to the stated goals of the work, the means refer to how authors intend to use their contributions to achieve those goals, and evidential support refers to the evidence used to justify the claimed contribution.

Identifying the Means and Ends The ends refer to the purpose or ultimate aim of the work—that is, what the authors claim to accomplish. Based on our data, we identify the following main categories of ends:

• Serving communities refers to goals explicitly directed towards benefiting a specific group of people, for example by improving NLP technologies for a particular linguistic community.

• Decolonising refers to goals framed in terms of contributing to the decolonisation of a region or marginalised group.

• Reducing inequality refers, in contrast to serving communities, to broader goals of improving representation, coverage, or equality beyond a specific community. For example, this includes generalising an observed gap from one language or language family to multiple groups.

• Advancing knowledge production refers to contributing resources and insights that support scientific research.

• Improving evaluation includes developing tools, benchmarks, or metrics that support the evaluation of NLP systems.

• Language revitalisation refers to efforts to support the revitalisation of a language through data collection and NLP technologies.

• Other ends include addressing environmental issues, such as reducing the carbon footprint of NLP systems through greater efficiency; building powerful models, including achieving state-ofthe-art performance; and government monitoring, where the stated aim involves monitoring government interests or activities in a particular region.

The means, or applications, refer to how authors propose to use what is developed in a paper to accomplish a particular goal or end. Based on our data, we identify the following main categories of means:

• Improving LLM performance refers to developing resources, methods, or models intended to improve the performance of LLMs for particular languages or tasks.

• NLP applications include developing or improving technologies for specific tasks, such as machine translation, named entity recognition, transcription, and other applications.

• LLM safety and ethical considerations refers to developing or applying methods intended to make language models safer or more appropriate for particular linguistic communities.

• Improving LLM evaluation refers to developing methods for evaluating and comparing the capabilities of LLMs, particularly multilingual models, and identifying their limitations.

• Helping language learners captures applications that use NLP technologies to support people learning one or more languages.

• Other captures applications that do not fit within the categories above such as annotating or collecting data.

Narrative Annotation We identify epistemic narratives or framings based on the relationships among stated means, ends, and presented evidence (see Table 1). We define the following narrative categories:

• Knowledge Production A narrative in which the primary justification is scientific contribution.

The stated goal is to advance understanding or support future research without linking the work to broader societal outcomes.

• Explicit Pathway to Impact A narrative in which both means and ends are clearly articulated. The paper describes a concrete technical contribution (e.g., an automated languageprocessing task) and links it to a defined outcome (e.g., serving a specific community). This category excludes cases where the sole stated end is knowledge production.

• Vague Serving A narrative in which a technical contribution is presented as broadly serving a language community or population without a clearly articulated pathway to impact or supporting evidence.

• Saviourism / Decolonisation A narrative in which a resource or model is presented as saving an endangered language, reviving a historical language, or contributing to decolonisation.

• Tech Solutionism A narrative in which a technical resource or tool is presented as addressing inherently social, political, or structural problems (e.g., regional tensions) without supporting evidence or a demonstrated mechanism, potentially exaggerating the impact of the intervention.

• Tokenism A narrative in which limited acts of inclusion (e.g., involving annotators from underrepresented groups) are presented as meaningfully addressing broader structural inequalities in ways that may simplify or exaggerate the social impact of the work. This narrative is more common in promotional materials.

## 4 The Risk of Exaggerated Claims in Multilingual and Low-Resource NLP

We examine the evidential support and justificatory narratives used in the selected NLP papers. We find 42% of the papers commonly rely on three forms of evidence: (i) perceived threats and anecdotal justifications, (ii) self-citation or weak evidential grounding, and (iii) broad claims regarding inequality reduction or community benefit that are weakly connected to the technical contribution.

## 4.1 Perceived Threats and Anecdotal Justifications as Evidence

We observe that a substantial number of claims rely on limited or insufficiently substantiated forms of justification. In several cases, assumptions, plausible narratives, or normative opinions are presented as scientific claims without clear empirical grounding. This distinction is important because plausible explanations can acquire the appearance of evidence even when the underlying empirical relationship has not been established (Brem and Rips, 2000). More broadly, we observe recurrent narratives of urgency or threat used to justify technical work. These include claims that some language communities risk becoming “second-class” participants in technological progress, concerns about economic or societal “exclusion from AI”, fears of linguistic disappearance, and arguments that language loss will lead to cultural homogenisation. While we acknowledge that some of these concerns are valid, the existence of a socially significant problem does not, by itself, demonstrate that a particular NLP system constitutes an effective response. Invoking such rationales without evidence specific to the population or context under discussion, or without establishing how the proposed intervention would mitigate the identified risk, can position technological researchers— even when they themselves belong to the broader community—as authoritative interpreters of community needs and interests (Adamu, 2023).

For example, statements suggesting that the predominantly spoken nature of some languages makes them particularly suitable for technologies such as speech systems or multimodal large language models are sometimes made without empirical support, potentially representing local communities in agency-depriving ways (Ferreira, 2024). However, the relationship between a language’s predominantly spoken nature and the effectiveness of speech-based technologies is not straightforward: in low-resource settings, limited training and evaluation data can constrain system performance. Therefore, the same scarcity of resources invoked to motivate such technologies may also limit their effectiveness, creating a tension between the proposed justification and the practical feasibility of the intervention. Moreover, the relationship between model capacity, data availability, and performance in low-resource languages is not straightforward, particularly in multilingual settings (Chang et al., 2024).

## The Limits of Citation-Based Justifications

Our claims are grounded in well-cited surveys that appear regardless of topical fit, alongside studies on language endangerment supported by a paper whose title was sufficiently alarming to obviate the needforfurther reading and a small number of self-citations arranged in mutually reinforcing sequence.

The presence of citations does not necessarily imply strong evidential support. In practice, feasibility support frequently consists of self-citation or references to a narrow set of commonly cited NLP papers (e.g., Joshi et al. (2020)), rather than diverse or domain-specific evidence establishing a connection between the proposed technical intervention and the mitigation of stated risks, such as the economic marginalisation of Global Majority populations or linguistic homogenisation. In some cases, references function primarily as rhetorical support for broad claims rather than as evidence demonstrating feasibility or a plausible causal mechanism. For example, concerns about language endangerment are often supported by references to prior NLP work, while leaving unspecified how the proposed technical contribution is expected to influence long-term language vitality (Bird, 2020).

## 4.2 The Gap Between Technical Contributions and Social Impact

An information extraction system trained on English newswire data would simply report the F1 score. We, by contrast, report the F1 score and note that it represents a meaningful step toward the end of inequality in Europe.

Across the examined papers, ‘reducing inequality” (48%) and ‘serving communities” (55%) are frequently stated as end goals. To investigate how such claims are substantiated, we compare 10 papers addressing multilingual or low-resource languages with 10 others focusing exclusively on highresource languages, primarily English and Chinese, across a range of representative NLP tasks, including NLU, NER, NLG, topic classification, multiplechoice QA, multitask language understanding, and language modelling. Specifically, we pair multilingual or low-resource papers with high-resource papers addressing the same task (e.g., a multilingual NER paper with an English or Chinese NER paper). Despite comparable technical contributions, multilingual and low-resource papers in our sample more frequently invoked tech-solutionist narratives and broad societal outcomes—including disaster relief, addressing illiteracy, and reducing ethnic tensions—than their high-resource counterparts, including in cases where the primary contribution was a task-specific benchmark. By contrast, highresource papers predominantly focused on technical objectives such as robustness evaluation and model performance. Notably, in the multilingual and low-resource papers, the pathway from technical contributions to claimed societal outcomes often remained underspecified. This gap was even more pronounced in institutional and project promotional materials, where we identified instances consistent with tokenism and saviourism across 17 collected online articles. We emphasise that this asymmetry should be understood as a difference in scholarly rhetoric rather than as a difference in the value of research on any particular language. It may also reflect structural expectations and pressures placed on non-English NLP research during the review process, rather than researchers’ choices alone (Barkhordar et al., 2026).

## 4.3 Epistemic Risks of Exaggerated Claims

Ultimately, Continental-EU-BERT is not merely a language model. It is a strike a decisive blow against centuries of computational inequity, colonial epistemology, and the lingering hegemony of English. It represents a meaningful step toward the empowerment ofmarginalised communities.

We find that 11% of the articles make unsupported claims about decolonisation or saviourism, while 14% exhibit techno-solutionist narratives. Such rhetoric may obscure the limits of technical interventions by exaggerating what a resource or model can realistically achieve (Reiter, 2025), while potentially undermining the credibility of otherwise valuable technical contributions by weakening the connection between empirical findings and societal outcomes. It may also produce reductive representations of languages and their speakers, particularly when communities are framed primarily in terms of deficit, vulnerability, or technological absence (Bird, 2022). Even when the development of an artefact involves community members, an author’s membership in a community does not, by itself, establish that they represent the experiences, priorities, or preferences of all its speakers. Differences in socioeconomic position, educational opportunities, geographic location, age, and access to technology can produce substantially different experiences within the same linguistic community (Warschauer and Matuchniak, 2010). Statements made on behalf of a social group should therefore be distinguished from evidence concerning the perspectives of the particular groups intended to use or benefit from the proposed technology. Finally, such framing can create implicit moral imperatives around technical development, presenting language technologies as inherently necessary or ethically urgent despite limited evidence of their impact among underrepresented communities (Adamu, 2023).

## 5 On the Risks of Constructing Implicit Moral Imperatives

While we acknowledge the value of building technologies for underserved communities, framings that present technical intervention as inherently leading to greater inclusion of local researchers and increased linguistic representation, often reinforced through narratives of social benefit or empowerment (Mihalcea et al., 2024), should be approached carefully. In our analysis, these framings and forms of evidential support (shown in green and red in Figure 1, respectively) tend to legitimise technical intervention even when its necessity, feasibility, or expected impact remains questionable. In the following, we examine several prominent patterns through which such normative expectations are constructed.

## 5.1 Local Researcher Inclusion and the Idealisation of Participatory Research

Participatory research, particularly in the context of low-resource languages, has increasingly been framed as a pathway towards addressing longstanding inequities in NLP and ML research. While such approaches can foster more inclusive and community-centred practices, they are sometimes presented in idealised terms, with participation associated with broader outcomes such as the adequate representation of marginalised communities (Birhane et al., 2022a). Such framings can implicitly treat participation as sufficient to address persistent power asymmetries, institutional hierarchies, and external pressures from dominant actors in academia and industry.

In some narratives, participatory research is portrayed as a collective responsibility through which researchers are expected to place their language or country “on the map”, implicitly suggesting that greater representation can address deeper structural inequalities. However, approximately 20% of the artefacts we analysed are unambiguously owned by Big Tech companies. This figure does not account for authors’ institutional affiliations, although most papers include at least one author affiliated with a prestigious institution. These patterns suggest that community-oriented research can remain embedded within highly concentrated technological and institutional structures. Greater visibility, therefore, does not necessarily address persistent challenges such as underfunding, exploitative labour dynamics involving vendors and contractors in the Global South (Shahid et al., 2025), the potential misuse of language communities (Ousidhoum et al., 2025), unequal access to computational resources, and broader structural barriers shaping research ecosystems (Png, 2022).

Framing Underrepresentation as a Threat Across multilingual and low-resource NLP papers, we observe recurring rhetorical patterns in the framing of motivations, methodological choices, and anticipated societal outcomes. These patterns often frame underrepresentation as a threat requiring technical intervention, as illustrated in Figure 1. While we do not dispute the existence of disparities in available resources, we highlight cases in which rhetorical framing surpasses the empirical support and may encourage junior researchers who feel a sense of duty toward their language(s) to join projects without adequate compensation or recognition (Ousidhoum et al., 2025).

Collaboration vs. Sustained Engagement Although some papers emphasise collaboration with local communities or contributors, there is often limited evidence of sustained engagement beyond the initial study. In many cases, authors do not appear to continue publishing, raising questions about the continuity and depth of participatory claims. Our analysis of papers in the ACL Anthology shows an increasing number of papers with at least 15 authors in recent years, rising from 9 in 2020 to 37 in 2021 and reaching 135 in 2025. Since 2015, authors of these papers account for approximately 2% of all new authors, despite the papers themselves representing only 0.005% of all publications (see Appendix). In addition, among first-time authors of papers published since 2015, the probability of publishing again is close to zero. This raises questions about the extent to which participatory projects create sustained research engagement, particularly when they are presented as pathways for junior researchers from the Global South (Ousidhoum et al., 2025). Our intention is not to discourage participation in such projects, but to encourage more careful communication by established institutions and more cautious interpretation of participation opportunities by junior researchers from the Global Majority.

## 5.2 Problem Framing and Pathways to Impact

Papers motivated by linguistic revival (7%) often invoke concerns about language extinction, digital exclusion, socioeconomic disadvantage, marginalisation, or safety risks. However, the relationships between these concerns and the proposed interventions often lack feasibility analyses, deployment evidence, or clearly articulated causal mechanisms grounded in relevant non-NLP research. Instead, such relationships are sometimes presented as selfevident, with limited consideration of the conditions under which technical interventions address the problems identified. Urgency is also reinforced through morally charged language, with aspirations such as democratisation $( n = 6 )$ (Subramonian et al., 2024), community benefit, and inequality reduction (> 50%) used to connect technical contributions to broader societal outcomes without systematically establishing the pathways through which these outcomes would be achieved.

Representing Communities: Exceptionalism vs. Overgeneralisation An additional issue arises when a language or community is framed as uniquely underserved or exceptionally neglected without comparative baselines or supporting evidence, potentially contributing to its essentialisation. Stronger claims would ideally include comparisons across factors such as dataset coverage, benchmark availability, model performance, funding, or deployment constraints. Conversely, some work treats broad linguistic regions or large groups of languages as homogeneous, flattening meaningful linguistic, cultural, socioeconomic, and political differences (Adamu, 2023; Keleg, 2025). More meaningful forms of empowerment require representations that are grounded in relevant social science research and empirical evidence and that account for variation within and across communities.

Underspecified Justifications for Artefact Creation Several papers justify new datasets, benchmarks, or models primarily by noting that the resource does not yet exist, while simultaneously linking its creation to broader societal impact. Although exploratory artefact creation is entirely legitimate, the motivation for creating a resource should not be conflated with demonstrated necessity: the absence of a benchmark or dataset does not, by itself, establish either scientific or practical need. Stronger motivations, where such claims are made, should include evidence of demand, utility, deployment relevance, or comparative benefits.

Insufficient Ethical Considerations Ethical concerns may also remain insufficiently addressed when underserved languages or communitycentred approaches are invoked primarily as methodological framing, including relatively straightforward considerations such as annotation compensation (Ousidhoum et al., 2025). This raises important questions regarding whose priorities and values are prioritised, which trade-offs are accepted, and how potential harms are identified and evaluated.

## 6 Guidelines for Ethical Multilingual and Low-Resource NLP Research

Based on our observations, broad claims about linguistic communities, technological needs, or societal outcomes can be problematic when supported by limited or narrowly scoped evidence. Recurring methodological limitations include:

• limited sample sizes;

• inadequate representativeness;

• broad conclusions drawn from exploratory evidence;

• insufficient acknowledgement of methodological limitations.

We therefore propose a checklist for stakeholders engaging with multilingual and low-resource NLP research, including papers, datasets, benchmarks, and industrial releases. The checklist focuses on recurring questions concerning research framing, evidentiary support, and implicit premises. The questions below are intended to guide authors, reviewers, and readers when evaluating papers, blog posts, benchmarks, and product descriptions. They can help identify stereotypical or essentialist representations, as well as cases where claims about low-resource languages or underrepresented communities rely on assumptions that might be considered less acceptable in high-resource settings, such as those involving English, German, or French. They also draw attention to potentially asymmetric evidentiary expectations applied to non-English NLP research, which may contribute to gatekeeping. A useful guiding question across all roles is:

Would this claim be made in the same form in a high-resource language context? If not, whatjustifies the difference?

More broadly, the checklist aims to surface:

• rhetorical overreach;

• paternalistic framing;

• techno-solutionist assumptions; and

• unequal evidentiary standards.

## Context and Language Ecology

• Is exceptionalism being introduced without justification?

– Would similar claims be made for a highresource language?

– If not, are strong claims about a language or community necessary for motivating the work?

– Are common hype cycles or rhetorical patterns in the field being reproduced?

• Are low-resource language contexts treated as homogeneous, or is meaningful variation acknowledged?

• Do authors engage with relevant linguistic, sociolinguistic, anthropological, or regional scholarship where appropriate?

• Are local researchers meaningfully included in the work?

– Has relevant non-English or regional scholarship been consulted?

• Are technical systems implicitly framed as solving structural or societal problems?

## Evidence and Data Quality

• Are claims about language use, demographics, literacy, or infrastructure empirically grounded?

• Is language framed in essentialising terms (e.g., attributing outcomes to “illiteracy” or

cultural deficit explanations)?

• Are dataset sources and provenance clearly documented?

• Has relevant domain expertise or community knowledge been consulted?

## Stakeholder Engagement

• Who are the intended users or affected communities?

– How were their needs identified?

– Was there meaningful engagement, such as interviews, workshops, or field studies?

• Are local priorities reflected in the research agenda?

• Is there evidence of engagement with relevant sociolinguistic or infrastructural contexts?

## Inequality and Harm

• Do research practices avoid extractive dynamics under the framing of “inclusion” or “diversity”?

• Does the system shift dependency or centralise control?

• Are contributors (e.g., annotators) appropriately recognised and compensated?

• Is there space to challenge imported assumptions from dominant research hubs?

• Is there accountability beyond publication or initial deployment?

• Is evaluation conducted in relevant local contexts rather than only abstract or global settings?

• Are local practices and constraints adequately considered?

## 7 Conclusion

We examined how multilingual and low-resource NLP research is framed, with particular attention to equity and integrity. Our analysis suggests that, although these framings are often motivated by well-intentioned goals, the evidence supporting associated claims is frequently limited or unclear. We highlight the need for stronger alignment between technical contributions and evidential support and provide practical guidance to help authors, reviewers, and readers critically assess such claims without imposing unjustifiably higher standards.

We hope this work encourages reflection on how research practices, evaluation norms, and rhetorical conventions shape perceptions of research in the area.

## Limitations

This is not a comprehensive study but a starting point, as such it has a few limitations. The first limitation is that our analysed sample was of only 100 papers. While this sample size is within ranges of past work, it is not complete.

Second, our analysis largely focused on abstracts and introduction. While we believe this decision is reasonable—as our initial more in-depth analyses of full papers did not yield more informative results—it is possible that some papers provided nuanced views or takes further down in the methodology or other sections.

Lastly, we did not engage with the authors of papers to get their opinions or views. We explain our reasoning for this in the next section, but such interaction could have resulted in greater insights.

## Ethical Considerations

The primary risk of this work is that it could inadvertently alienate, shame, or disproportionately affect particular authors, especially junior or underrepresented researchers. To mitigate this risk, our methodology is deliberately anonymised: we do not link findings to individual papers or provide direct quotes from the analysed works, instead using deliberately exaggerated and satirical examples.

As stated in the introduction, while we identify potentially harmful narratives in some low-resource language research, we do not intend to suggest that low-resource languages are unworthy of research or technological development. Rather, our goal is to support the integrity of such efforts by advocating for more transparent, evidence-based, and practically grounded pathways to impact.

AI Use We use privacy-preserving models to assist with proofreading, in line with the ACL guidelines.

## Acknowledgments

We thank anonymous reviewers for their helpful suggestions.

## References

Mohamed Abdalla and Moustafa Abdalla. 2021. The grey hoodie project: Big tobacco, big tech, and the threat on academic integrity. In Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics, and Society, pages 287–297.

Mohamed Abdalla, Jan Philip Wahle, Terry Ruas, Aurélie Névéol, Fanny Ducel, Saif Mohammad, and Karen Fort. 2023. The elephant in the room: Analyzing the presence of big tech in natural language processing research. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers).

Muhammad Sadi Adamu. 2023. No more “solutionism” or “saviourism” in futuring african hci: A manyfesto. ACM Transactions on Computer-Human Interaction, 30(2):1–42.

Will Aitken, Mohamed Abdalla, Karen Rudie, and Catherine Stinson. 2024. Collaboration or corporate capture? quantifying nlp’s reliance on industry artifacts and contributions. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3433–3448.

Ehsan Barkhordar, Abdulfattah Safa, Verena Blaschke, Erika Lombart, Marie-Catherine de Marneffe, and Gözde Gül ¸Sahin. 2026. Are non-english papers reviewed fairly? language-of-study bias in nlp peer reviews.

Emily M Bender. 2011. On achieving and evaluating language-independence in nlp. Linguistic Issues in Language Technology, 6.

Emily M. Bender and Batya Friedman. 2018. Data Statements for Natural Language Processing: Toward Mitigating System Bias and Enabling Better Science. Transactions of the Association for Computational Linguistics, 6:587–604.

Steven Bird. 2020. Decolonising speech and language technology. In Proceedings ofthe 28th International Conference on Computational Linguistics, pages 3504–3519, Barcelona, Spain (Online). International Committee on Computational Linguistics.

Steven Bird. 2022. Local languages, third spaces, and other high-resource scenarios. In Proceedings ofthe 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 7817–7829, Dublin, Ireland. Association for Computational Linguistics.

Steven Bird. 2025. Big ai is accelerating the metacrisis: What can we do? arXiv preprint arXiv:2512.24863.

Steven Bird and Dean Yibarbuk. 2024. Centering the speech community. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 826–839.

Abeba Birhane, William Isaac, Vinodkumar Prabhakaran, Mark Diaz, Madeleine Clare Elish, Iason Gabriel, and Shakir Mohamed. 2022a. Power to the people? opportunities and challenges for participatory ai. In Proceedings ofthe 2nd ACM Conference on Equity and Access in Algorithms, Mechanisms, and Optimization, pages 1–8.

Abeba Birhane, Pratyusha Kalluri, Dallas Card, William Agnew, Ravit Dotan, and Michelle Bao. 2022b. The values encoded in machine learning research. In Proceedings ofthe 2022 ACM conference onfairness, accountability, and transparency, pages 173–184.

Damian Blasi, Antonios Anastasopoulos, and Graham Neubig. 2022. Systematic inequalities in language technology performance across the world’s languages. In Proceedings ofthe 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5486–5505, Dublin, Ireland. Association for Computational Linguistics.

Sarah K Brem and Lance J Rips. 2000. Explanation and evidence in informal argument. Cognitive science, 24(4):573–604.

Eric Chamoun, Nedjma Ousidhoum, Michael Sejr Schlichtkrull, and Andreas Vlachos. 2025. Social good or scientific curiosity? uncovering the research framing behind NLP artefacts. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics.

Tyler A. Chang, Catherine Arnett, Zhuowen Tu, and Benjamin K. Bergen. 2024. When is multilinguality a curse? language modeling for 250 high- and low-resource languages. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 4074–4096.

A. Seza Dogruöz and Sunayana Sitaram. 2022.˘ Language technologies for low resource languages: Sociolinguistic and multilingual insights. In Proceedings of the 1st Annual Meeting of the ELRA/ISCA Special Interest Group on Under-Resourced Languages, pages 92–97, Marseille, France. European Language Resources Association.

Pedro Ferreira. 2024. Examining the" local" in ict4d: A postcolonial perspective on participation. In Proceedings ofthe 2024 CHI Conference on Human Factors in Computing Systems, pages 1–13.

Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé Iii, and Kate Crawford. 2021. Datasheets for datasets. Communications ofthe ACM, 64(12):86– 92.

Frances S Grodzinsky, Keith Miller, and Marty J Wolf. 2012. Moral responsibility for computing artifacts: the rules" and issues of trust. Acm Sigcas Computers and Society, 42(2):15–25.

William Held, Camille Harris, Michael Best, and Diyi Yang. 2023. A material lens on coloniality in nlp. arXiv preprint arXiv:2311.08391.

Nora Hollenstein, Maria Barrett, and Lisa Beinborn. 2020. Towards best practices for leveraging human language processing signals for natural language processing. In Proceedings ofthe Second Workshop on Linguistic and Neurocognitive Resources, pages 15– 27, Marseille, France. European Language Resources Association.

Pratik Joshi, Sebastin Santy, Amar Budhiraja, Kalika Bali, and Monojit Choudhury. 2020. The state and fate of linguistic diversity and inclusion in the NLP world. In Proceedings ofthe 58th Annual Meeting of the Associationfor Computational Linguistics, pages 6282–6293, Online. Association for Computational Linguistics.

Lucie-Aimée Kaffee, Arnav Arora, Zeerak Talat, and Isabelle Augenstein. 2023. Thorny roses: Investigating the dual use dilemma in natural language processing. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 13977–13998.

Amr Keleg. 2025. LLM alignment for the Arabs: A homogenous culture or diverse ones. In Proceedings of the 3rd Workshop on Cross-Cultural Considerations in NLP (C3NLP 2025), pages 1–9.

Konstantinos Kogkalidis and Stergios Chatzikyriakidis. 2025. On tables with numbers, with numbers. In Proceedings of the 1st Workshop on Language Models for Underserved Communities (LM4UC 2025), pages 104–115.

Klaus Krippendorff. 2018. Content analysis: An introduction to its methodology. Sage publications.

Kobi Leins, Jey Han Lau, and Timothy Baldwin. 2020. Give me convenience and give her death: Who should decide what uses of NLP are appropriate, and on what basis? In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 2908–2913, Online. Association for Computational Linguistics.

Heather Lent, Kelechi Ogueji, Miryam de Lhoneux, Orevaoghene Ahia, and Anders Søgaard. 2022. What a creole wants, what a creole needs. In Proceedings of the Thirteenth Language Resources and Evaluation Conference, pages 6439–6449, Marseille, France. European Language Resources Association.

Yu Lu Liu, Meng Cao, Su Lin Blodgett, Jackie Chi Kit Cheung, Alexandra Olteanu, and Adam Trischler. 2023. Responsible AI considerations in text summarization research: A review of current practices. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 6246–6261. Association for Computational Linguistics.

Zoey Liu, Crystal Richardson, Richard Hatcher, and Emily Prud’hommeaux. 2022. Not always about you: Prioritizing community needs when developing

endangered language technology. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3933–3944.

Manuel Mager, Elisabeth Maier, Katharina Kann, and Ngoc Thang Vu. 2023. Ethical considerations for machine translation of indigenous languages: Giving a voice to the speakers. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4871– 4897.

Rada Mihalcea, Oana Ignat, Longju Bai, Angana Borah, Luis Chiruzzo, Zhijing Jin, Claude Kwizera, Joan Nwatu, Soujanya Poria, and Thamar Solorio. 2024. Why ai is weird and should not be this way: Towards ai for everyone, with everyone, by everyone. arXiv preprint arXiv:2410.16315.

Saif Mohammad. 2022. Ethics sheets for ai tasks. In Proceedings of the 60th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 8368–8379.

Nedjma Ousidhoum, Meriem Beloucif, and Saif M. Mohammad. 2025. Building better: Avoiding pitfalls in developing language resources when data is scarce. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 8881–8894, Vienna, Austria. Association for Computational Linguistics.

Marie-Therese Png. 2022. At the tensions of south and north: Critical roles of global south stakeholders in ai governance. In Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency, pages 1434–1445.

Ehud Reiter. 2025. We should evaluate real-world impact. Computational Linguistics, pages 1–13.

Anna Rogers, Timothy Baldwin, and Kobi Leins. 2021. ‘just what do you think you’re doing, dave?’ a checklist for responsible data use in NLP. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 4821–4833, Punta Cana, Dominican Republic. Association for Computational Linguistics.

Michael Schlichtkrull, Nedjma Ousidhoum, and Andreas Vlachos. 2023. The intended uses of automated fact-checking artefacts: Why, how and who. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 8618–8642.

Farhana Shahid, Mona Elswah, and Aditya Vashistha. 2025. Think outside the data: Colonial biases and systemic issues in automated moderation pipelines for low-resource languages. In Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society, volume 8, pages 2331–2344.

Arjun Subramonian, Vagrant Gautam, Dietrich Klakow, and Zeerak Talat. 2024. Understanding “democrati zation” in nlp and ml research. In Proceedings ofthe

2024 Conference on Empirical Methods in Natural Language Processing, pages 3151–3166.

Mark Warschauer and Tina Matuchniak. 2010. New technology and digital worlds: Analyzing evidence of equity in access, use, and outcomes. Review of research in education, 34(1):179–225.

Xinyan Yu, Trina Chatterjee, Akari Asai, Junjie Hu, and Eunsol Choi. 2022. Beyond counting datasets: A survey of multilingual dataset construction and necessary resources. In Findings ofthe Association for Computational Linguistics: EMNLP 2022, pages 3725–3743.

## A Appendix: Statistics

<table><tr><td>Category</td><td>%</td></tr><tr><td>Ends</td><td></td></tr><tr><td>Serve Communities</td><td>55</td></tr><tr><td>Decolonise Fight Inequality</td><td>2 48</td></tr><tr><td>Language Revitalisation</td><td>7</td></tr><tr><td>Advancing knowledge production</td><td>98</td></tr><tr><td>Narratives</td><td></td></tr><tr><td>Explicit Pathway to Impact</td><td>85</td></tr><tr><td>Knowledge Production</td><td>99</td></tr><tr><td>Saviourism/Decolonisation</td><td>11</td></tr><tr><td></td><td></td></tr><tr><td>Tech Solutionism Vague Serving</td><td>14 14</td></tr></table>

Table 2: Distribution of main annotated ends and narrative categories across the analysed papers.

<table><tr><td>Feasibility Support</td><td>Count</td></tr><tr><td>Previous NLP Paper</td><td>76</td></tr><tr><td>Opinion</td><td>28</td></tr><tr><td>Scientific Article/Poll</td><td>9</td></tr><tr><td>Sense of Threat</td><td>11</td></tr></table>

Table 3: Main types of feasibility support identified across the analysed papers.

<table><tr><td>Year</td><td>%New Authors</td><td> $\leq$  15 Authors</td><td> $>$  15 Authors</td><td>% Papers &gt; 15</td></tr><tr><td>2000</td><td>0.1250</td><td>1253</td><td>1</td><td>0.0008</td></tr><tr><td>2001</td><td>0</td><td>813</td><td>1</td><td>0.0012</td></tr><tr><td>2002</td><td>0</td><td>1216</td><td>0</td><td>0</td></tr><tr><td>2003</td><td>0</td><td>1186</td><td>0</td><td>0</td></tr><tr><td>2004</td><td>0.0313</td><td>1922</td><td>2</td><td>0.0010</td></tr><tr><td>2005</td><td>0</td><td>1260</td><td>1</td><td>0.0008</td></tr><tr><td>2006</td><td>0.2941</td><td>2173</td><td>1</td><td>0.0005</td></tr><tr><td>2007</td><td>0.0196</td><td>1598</td><td>3</td><td>0.0019</td></tr><tr><td>2008</td><td>0</td><td>2231</td><td>1</td><td>0.0004</td></tr><tr><td>2009</td><td>0</td><td>2232</td><td>1</td><td>0.0004</td></tr><tr><td>2010</td><td>0.0250</td><td>3067</td><td>2</td><td>0.0007</td></tr><tr><td>2011</td><td>0</td><td>2289</td><td>1</td><td>0.0004</td></tr><tr><td>2012</td><td>0.0625</td><td>3390</td><td>1</td><td>0.0003</td></tr><tr><td>2013</td><td>0.0455</td><td>2822</td><td>2</td><td>0.0007</td></tr><tr><td>2014</td><td>0.0145</td><td>3634</td><td>5</td><td>0.0014</td></tr><tr><td>2015</td><td>0</td><td>2955</td><td>3</td><td>0.0010</td></tr><tr><td>2016</td><td>0.0107</td><td>4209</td><td>11</td><td>0.0026</td></tr><tr><td>2017</td><td>0.0027</td><td>3481</td><td>6</td><td>0.0017</td></tr><tr><td>2018</td><td>0.0343</td><td>4812</td><td>10</td><td>0.0021</td></tr><tr><td>2019</td><td>0.0447</td><td>5065</td><td>9</td><td>0.0018</td></tr><tr><td>2020</td><td>0.0129</td><td>7208</td><td>37</td><td>0.0051</td></tr><tr><td>2021</td><td>0.0185</td><td>7107</td><td>41</td><td>0.0057</td></tr><tr><td>2022</td><td>0.0185</td><td>8604</td><td>45</td><td>0.0052</td></tr><tr><td>2023</td><td>0.0206</td><td>8980</td><td>52</td><td>0.0058</td></tr><tr><td>2024</td><td>0.0264</td><td>12047</td><td>82</td><td>0.0068</td></tr><tr><td>2025</td><td>0.0131</td><td>14751</td><td>135</td><td>0.0091</td></tr></table>

Table 4: Evolution in the number of papers with fewer than and more than 15 authors in the ACL Anthology by year.