# How Different, or Different How? Stance Granularity in Viewpoint Selection for Dialog Systems that Engage Users with Alternative Views

Morita Tarvirdians\* Reeve Lorena\* Catharine Oertel

TU Delft, Delft, The Netherlands

M.Tarvirdians@tudelft.nl reevelorena@gmail.com c.r.m.m.oertel@tudelft.nl

## Abstract

Systems that aim to help a user engage with viewpoints other than their own have to decide how different those viewpoints should be. Viewpoints are mainly represented on a scalar scale, too coarse to distinguish viewpoints that disagree with the user about different things. We introduce a selection policy in which a bandit sets how far viewpoints may sit from the user, and the policy selects the most novel viewpoint within that bound. We vary only its stance encoder, comparing a scalar score with Perspectivized Stance Vectors, which break a position into separate stances on concepts. We integrate it inside a dialog system and run a betweensubjects study (N = 63). Concept-level stance produces more varied exposure sequences and broader coverage of the topic’s positions, but does not change how deeply users justify viewpoints or how accurately they reconstruct them. Finally, we analyze user–viewpoint relations at the concept level, showing that users reconstruct a viewpoint least accurately on concepts that one side took no position on, rather than on concepts where the two disagreed. Our findings open up opportunities for dialog systems that support engagement with alternative viewpoints by acting on which concepts a user and a viewpoint disagree on, rather than on distance alone.

## 1 Introduction

Public discourse increasingly takes place in information environments such as social media feeds, search results and recommender systems, where people selectively encounter views consistent with their existing positions (Frey, 1986; Zarouali et al., 2021). In politically contentious settings this is associated with affective polarization, which increases hostility toward opposing groups and reduces willingness to interact with them (Iyengar and Westwood, 2015; Iyengar et al., 2019). One response is to structure exposure to perspectives other than a user’s own, drawing on perspectivetaking, the process of understanding a situation from another person’s point of view (Reniers et al., 2022).

![](images/af077b20597a1436c6dbea1af63047e69e342b3e2476e9970ba20bb818f355a4.jpg)  
Figure 1: Two viewpoints a single stance score cannot tell apart. Under SCALAR both are rated 2 against the user’s 3, so neither is closer and the system picks arbitrarily. Under PSV they disagree with the user on different concepts (highlighted), so B is selected as the more novel. Both conditions then ask the same Socratic style questions, repeating until k viewpoints are shown.

Providing alternative views is not sufficient on its own, because the relation between a new view and a person’s existing position shapes how they process it (Lewandowsky et al., 2012; Zhou and Shen, 2024). A view close to what a user already holds adds little, while one far from it may be dismissed rather than considered (Sherif and Hovland, 1961). Which views a user sees should therefore depend on where that user stands, and several systems select on this basis, choosing what to show next from the arguments a user has already been exposed to (Aicher et al., 2022, 2023b,a). Such a system compares each candidate viewpoint with the user’s own position, and a viewpoint can differ from that position in two ways. It can differ in degree, by opposing the user more or less strongly, and in kind, by opposing a different aspect of the issue. Which of these differences a system selects on is an open question.

How a system represents stance of a viewpoint decides which of the two it can act on at all. Stance is usually summarised as a single point on an ordinal scale running from opposing to supporting a position, which records degree and discards kind. Two viewpoints that oppose different aspects of an issue can land on the same point of that scale, and a system comparing them with a user sees one value for both. Figure 1 shows this for hunting. Both viewpoints sit two steps from the user’s position, so a system using the SCALAR scale has no basis for preferring either, even though the aspects they disagree with the user about are not the same. A system that cannot see a difference cannot select on it, whatever rule it applies.

Finer representations do exist. Perspectivized Stance Vectors (PSV), for example, record a separate stance toward each of several concepts in an issue (Plenz et al., 2025), so two viewpoints that oppose different aspects receive different vectors (Figure 1, right). What such a representation changes for a user has not been tested, because a system that adopts a new representation normally changes the rule that uses it at the same time, leaving the two effects confounded. We therefore keep selection policy–the rule that decides which viewpoint to show next– fixed and swap only the stance representation underneath it, comparing a single ordinal score (SCALAR) with Perspectivized Stance Vectors (PSV). We implement this policy in a dialog system that presents the selected viewpoints and asks users about each of them.

Isolating the representation in this way lets us ask three questions. RQ1: How does representing stance at the concept level, rather than as a single score, change the viewpoints shown to a user? RQ2: Does concept-level stance representation affect how fully users justify viewpoints and how accurately they reconstruct them? RQ3: How does the concept-level relation between a user’s stance and a viewpoint’s stance relate to reconstruction accuracy? Answering them contributes a viewpoint-selection policy that operates over interchangeable stance representations, an account of what each representation makes available both to users and across the full stance space, and a concept-level analysis of reconstruction that a single aggregate score leaves hidden.

## 2 Related Work

Selecting viewpoints for a particular user draws on three lines of work. Argument retrieval defines how candidates are found. User-conditioned selection makes that choice depend on the recipient and on what they have already seen. Work on stance supplies the signal that relates a candidate to a user. We review each below, asking in each case how the relation between a user and a candidate viewpoint is represented, since that representation is what we vary.

Argument retrieval and user-conditioned selection. Argument retrieval finds arguments relevant to a query, including arguments with particular argumentative properties (Wachsmuth et al., 2018). Argument recommendation conditions that selection on the recipient. Brenneis et al. (Brenneis et al., 2021) introduce a dataset for evaluating argument recommenders from personal attitudes and interactions with arguments, more recent work incorporates user or stakeholder perspectives (Heinisch et al., 2024; Aicher et al., 2022), and PerspectiveArg2024 formulates perspective-aware argument selection as a retrieval task (Falk et al., 2024). These approaches establish user and perspective information as retrieval signals, but differ in how they represent the relation between a user and a candidate argument.

Selection can also unfold over a conversation, with the user’s current state and the arguments already shown affecting what comes next. Aicher et al. (Aicher et al., 2022, 2023a) model exposure histories and user positions in argumentative dialog systems to identify arguments that broaden subsequent exposure, showing that selection can operate as a policy over an evolving interaction rather than as one-shot retrieval. Our policy similarly selects over a session rather than in one shot.

Stance as a retrieval signal. Stance provides a direct signal for distinguishing arguments in controversial discussions. Methods exist for identifying the position expressed toward a topic or target (Durmus et al., 2019; Kobbe et al., 2020), and stance has been incorporated into retrieval and ranking through stance-aware query expansion (Kiesel et al., 2021) and stance-aware re-ranking for comparative retrieval (Reimer et al., 2023). These approaches treat stance as a property attached to an argument. In our setting stance instead defines the space in which candidates are compared, so that the distance between a user’s stance and a candidate determines eligibility and the stance representation determines which differences a policy can distinguish.

Conditioning on stance distance is supported by work on how arguments are processed. Social Judgment Theory models messages relative to a receiver’s existing position (Sherif and Hovland, 1961), and studies of belief revision and misinformation show that the discrepancy between incoming information and prior beliefs affects how that information is processed (Durmus and Cardie, 2018; Zhao and Fink, 2021; Lewandowsky et al., 2012). This literature motivates conditioning selection on user–viewpoint distance without specifying how that distance should be represented.

Fine-grained stance representations. A compact stance representation characterizes an argument’s position toward a target, but collapses distinctions between positions on different aspects of an issue. Plenz et al. (Plenz et al., 2025) address this with Perspectivized Stance Vectors, which represent an argument through its stances toward a shared set of issue-specific concepts, making agreement and disagreement explicit at the level of individual concepts. They use concept-level stance to analyze agreement between arguments after retrieval. We use it instead as the representation over which a viewpoint-selection policy operates.

## 3 Method

We formulate viewpoint selection as a retrieval problem over a fixed corpus of viewpoints. Our methodology represents the stance of the participant and candidate viewpoints, computes their distance in the resulting stance space, and uses this distance together with novelty to select and order five viewpoints. We integrate this selection policy into a dialog system that encodes the participant’s freetext stance and presents the selected viewpoints. Figure 2 illustrates this selection framework. The remainder of this section describes the viewpoint corpus, stance representations, selection policy, and the dialog system.

## 3.1 Viewpoint Corpus

We use the animal-hunting corpus of Plenz et al. (2025), drawn from Debate.org. After filtering, the corpus contains 364 viewpoints (Appendix A). The original corpus provides Perspectivized Stance Vectors (PSVs) over 100 concepts. For selection we reduce this to a fixed five-concept signature, hunt game, hunter, while hunting animals, outdoor activity and good. Concepts were ranked by how much their stance varies across the corpus, how often they take a non-neutral stance, and how well they separate pro from con arguments, and the top five were retained (Appendix B). The reduction preserves the structure of the full 100-concept representation, which yields a closely matching clustering of the corpus (adjusted Rand index .92).

![](images/863bd0426ab75ab0ad1afc9871df8562e278e346340585b4c09bf1f6b237ae91.jpg)  
Figure 2: Viewpoint-selection framework. Only the stance encoder φ varies. It maps the participant stance and the candidate viewpoints into a scalar or a conceptlevel (PSV) space, giving the distance d. The policy π is identical in both conditions. An ε-greedy mechanism sets the bound δ, and the most novel unseen viewpoint within it is selected, repeating five times

## 3.2 Stance Representations

A stance representation encodes a position so that two positions can be compared numerically. We compare a scalar representation with a conceptlevel representation based on PSVs.

SCALAR. A stance is encoded as an ordinal value $k _ { a } \in \{ 1 , \ldots , 5 \}$ , from strongly against to strongly for, following a common stance operationalization (Chuang et al., 2026). The distance between two stances is $d _ { \mathrm { s } } ( a , b ) = | k _ { a } - k _ { b } |$ . This representation distinguishes five overall positions but does not encode which aspects of the issue produce the difference.

PSV. Following Plenz et al. (2025), a stance is encoded as an $N \times 3$ one-hot matrix P, with each row representing a stance (against, neutral, or for) on concept i. Distance is the number of concept– stance changes between two positions:

$$
d _ { \mathrm { P } } ( P _ { a } , P _ { b } ) = \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { 3 } | P _ { a , i j } - P _ { b , i j } | .
$$

With $N = 5$ , the representation can encode $3 ^ { 5 } = { }$ 243 positions, of which 35 occur in our corpus.

## 3.3 Selecting and Ordering Viewpoints

We formulate viewpoint selection as retrieval over a fixed corpus, because rather than generating new content, the system selects one unseen viewpoint at each step. Let V be the corpus (Section 3.1), $\phi : \Sigma ^ { * } \to S$ the stance encoder, and d a distance over the resulting stance space (Section 3.2). A participant’s stance is $u = \phi ( x _ { u } )$ , where $x _ { u }$ is their own view on the topic, written as free text before the session.

Let H denote the viewpoints already shown. At each step, the policy first restricts selection to unseen viewpoints within a distance bound δ of the participant’s stance. This defines the eligible set:

$$
\mathcal { W } = \{ v \in \mathcal { V } \setminus \mathcal { H } : d ( \phi ( v ) , u ) \leq \delta \} .\tag{1}
$$

Distance alone does not determine which eligible viewpoint is selected. To favor viewpoints that add new stance information, we score each candidate by the stance features it introduces that have not appeared in previously shown viewpoints:

$$
\operatorname { n o v } ( v , { \mathcal { H } } ) = { \Big | } \sigma ( v ) \backslash \bigcup _ { h \in { \mathcal { H } } } \sigma ( h ) { \Big | } ,\tag{2}
$$

where $\sigma ( v )$ is the set of stance features associated with v. Under PSV, these are the five concept– stance pairs; under SCALAR, the feature is the single ordinal stance level.

The policy then selects the eligible viewpoint with the highest novelty score:

$$
\pi ( u , \mathcal { H } ) = \underset { v \in \mathcal { W } } { \arg \operatorname* { m a x } } \operatorname { n o v } ( v , \mathcal { H } ) .\tag{3}
$$

Ties are broken by smallest distance and then uniformly at random. If no unseen viewpoint satisfies the distance bound, the nearest unseen viewpoint is selected. Figure 2 illustrates these three stages.

Distance and novelty serve complementary purposes. The distance bound limits how far a viewpoint may differ from the participant, measured as the difference between overall ratings under

SCALAR and the $L _ { 1 }$ distance between conceptlevel stance vectors under PSV. Novelty favors viewpoints that introduce stance features not covered by the session history. Without novelty, the policy could repeatedly select nearby viewpoints with highly similar positions.

The stance representation determines how both criteria operate. SCALAR encodes five overall positions and assigns one feature to each viewpoint, making novelty binary. At any given distance from the participant, at most two scalar positions are possible. PSV represents each viewpoint through five concept–stance pairs, allowing novelty to reflect which aspects of the issue have not yet been encountered.

At the start of each session, an ε-greedy bandit (Sutton and Barto, 2018) selects δ from the distances available in that condition. The bound remains fixed throughout all five viewpoints. The bandit’s arm values are updated between sessions using the depth of participants’ answers (Section 4.3); responses do not change the sequence within the current session.

Dialog system. The policy runs inside a textbased dialog system. Each turn presents a viewpoint unchanged from the corpus, framed as the view of a person identified by an arbitrary unisex first name. Five names are used, with no further information provided about these individuals. The system then asks five fixed Socratic-style questions (Danry et al., 2023; Ang et al., 2023) that scaffold perspective-taking by asking the participant to articulate each position from the speaker’s point of view (Reniers et al., 2022). They move from restating the position, through the reasons and assumptions behind it, to how it relates to the participant’s own view (Appendix C). Participants answer in free text. A session presents k = 5 viewpoints and therefore 25 written answers from each participant. The bandit uses the depth (Section 4.3) of these answers to update its arm values, informing the distance bound selected for subsequent sessions.

## 4 User Study

## 4.1 Study Design and Procedure

We conducted a between-subjects study on Should animal hunting be allowed?. The study was online. Participants first completed a consent form, the perspective-taking subscale of the Interpersonal Reactivity Index (IRI-PT) (Davis, 1983), and wrote their own view on the issue as free-text. Then they interacted with the dialog system (Section 3.3), followed by the reconstruction task (Section 4.3).

## 4.2 Participants

Participants were auto-balanced between the two conditions. Of 79 participants, 63 completed the study, 31 in SCALAR and 32 in PSV condition. They were recruited through personal networks and were not compensated. Conditions did not differ on IRI-PT, gender, or age bracket (Appendix E).

## 4.3 Measures

Exposure (RQ1). From session logs, we count distinct exposure sequences and the concept–stance pairs covered per session. We also replay the policy over the full corpus for every possible participant position.

Justification depth (RQ2). To measure whether participants gave reasons rather than merely restating a viewpoint, each written answer is scored from 0 (bare assertion) to 3 (two or more distinct reasons), following Trénel (2004); Steenbergen et al. (2003). This measure reflects the role of addressing reasons in engaging with opposing positions (Kuhn and Udell, 2007). Scores are averaged across the five answers for each viewpoint and rescaled to [0, 1].

Reconstruction accuracy (RQ2, RQ3). After each session, participants reconstruct one randomly selected viewpoint by labeling each of its five concepts as against, neutral, or for. We compare the reconstructed profile P<sup>ˆ</sup> with the true profile P using $1 - L _ { 1 } ( \hat { P } , P ) / ( 2 N )$ , yielding scores from 0 to 1, with chance at 1/3.

Target specificity (RQ3). We score the reconstruction of one randomly selected viewpoint against all five viewpoints shown in the session. Specificity is the score difference between the target and the other viewpoints, adjusted by the score expected from a reconstruction with no knowledge of the target. It therefore measures whether the reconstruction identifies the intended viewpoint beyond what could be explained by how distinct the session’s viewpoints are.

Concept-level relation (RQ3). For each concept, we classify the participant’s and viewpoint’s stances as matching, opposing, or involving at least one neutral stance. We also test whether errors on differing concepts tend to reflect the participant’s own stance, comparing observed errors with a permutation null.

## 4.4 Analysis

Only the condition comparison of reconstruction accuracy was pre-specified; all other analyses are exploratory. We report estimates with confidence intervals and apply Holm correction within test families. Per-concept outcomes are analyzed using binomial GEE with participant-clustered standard errors, and justification depth using a mixed model with participant random intercepts. The corpus replay covers all participant positions and distance bounds, with 100 seeds per setting.

## 5 Results

## 5.1 Concept-Level Stance Individualizes and Broadens Exposure

RQ1 asks how concept-level stance representation changes the viewpoints shown to participants. Figure 3 shows each viewpoint’s distance from the participant in the representation used for selection. Because the panels use different scales, we compare counts rather than distances across conditions. We count distinct distance sequences, each defined by the five distances in presentation order; sessions with the same distance sequence count as one even if they show different viewpoints.

Sequences. The 31 SCALAR sessions produced 6 distinct distance sequences, compared with 24 across the 32 PSV sessions. Under SCALAR, the first viewpoint matched the participant’s stance in all 31 sessions, the second was one step away in all 31, and the third was one step away in 27. The fourth and fifth returned to the participant’s stance in 23 and 26 sessions, respectively. The sequence is fully determined by the participant’s stance and the distance bound, with each of the 9 observed combinations producing exactly one sequence (Appendix F). Over the condition as a whole, 82 of the 155 viewpoints shown (52.9%) sat at the participant’s own stance. Under PSV the first viewpoint matched the participant on all five concepts in 20/32 sessions. The second sat at the furthest distance the bound allowed in 27/32, after which distance fell across the remaining slots, with means of 7.1, 6.1, 3.6 and 2.5 at slots 2 to 5.

Coverage. The issue has 15 concept–stance combinations, one for each of the three stances on each of the five concepts, and a session covers those its viewpoints take between them. PSV sessions covered 13.7 of the 15 on average (SD 2.2), against 11.0 under SCALAR (SD 2.4). All 15 appeared in 22 of 32 PSV sessions and in 1 of 31 under SCALAR.

Distance and coverage. At each sequence position, participant distance was strongly correlated with the number of newly introduced concept– stance combinations under both PSV $( \rho = . 7 2 – . 9 5 )$ and SCALAR (ρ = .73–1.00), where both measures varied. Thus, more distant viewpoints tended to introduce more new combinations, linking exposure to disagreement with the novelty criterion used by the policy.

Corpus replay. To test whether these patterns depended on the participants sampled, we replayed the deployed policy across the full corpus for all possible participant positions, 5 scalar ratings and 243 PSV profiles. We evaluated every distance bound using 100 seeds. Each SCALAR rating– bound combination produced exactly one sequence. At the widest bound, PSV covered all 15 combinations in every run, compared with a mean of 12.7 (SD 1.2) under SCALAR, which covered all 15 in only 5% of runs (Appendix G). At the narrowest bound, mean coverage was 9.1 under PSV and 8.3 under SCALAR; PSV selected the nearest unseen viewpoint as a fallback in 52.5% of selections. The corpus contained at most two scalar rating levels at any given distance, compared with a median of 6–12 PSV profiles at each $L _ { 1 }$ distance from 4 to 8.

Overall, concept-level stance representation produced more varied distance sequences and broader coverage of concept–stance combinations. The distance bound constrained how far selected viewpoints could differ from the participant, while novelty shaped which viewpoints were selected within that constraint.

## 5.2 Reconstruction Is Hard Under Both Representations

RQ2 asks whether concept-level stance affects justification depth and viewpoint reconstruction.

Participants reconstructed 43.9% of concept stances under SCALAR (68/155) and 48.7% under PSV (78/160), OR 1.22 [0.72, 2.06]. Both conditions sit above the 33.3% chance level. We re-ran the comparison without the 21 “not sure” answers, without the four participants whose stance was unresolved, without each participant in turn, and against the second labelling of the same viewpoints. The odds ratio stayed between 1.10 and 1.41 throughout, so the small PSV advantage is steady and never becomes reliable (Appendix H).

![](images/11b5a219bb6450a7dfc88773d85c5705c73d50dbdd1cc244b5c4746bd6e66640.jpg)

![](images/43d019e6efa33f632e1a0b2c6c29ee54bacee5540234287e67cf8d1a24e60cba.jpg)  
Figure 3: Distance between each viewpoint and the participant by sequence position. Top: SCALAR; bottom: PSV. Scales differ and should not be compared vertically. Lines show sessions; thick lines are condition means, dashed lines the four sessions with unresolved participant stance, and jitter separates identical sequences.

We score each participant’s description not only against the viewpoint they were asked about but also against the other four they saw. It fitted the intended viewpoint best for 24 of 32 participants under PSV and 13 of 31 under SCALAR. That gap opens more easily when a session’s five viewpoints are far apart, because one description can then fit only one of them, while in a session of similar viewpoints the same description fits all five about equally well. SCALAR sessions were the tighter ones (mean distance 5.3 against 6.3). The size of the advantage does not differ between conditions (+0.055 [−0.07, 0.18]), and subtracting what a reconstructor with no knowledge of the target would score on the same sessions leaves it there (Appendix I).

Justification depth, scored on the answers participants wrote during the session, showed no condition difference either. It averaged 0.44 under SCALAR and 0.40 under PSV (b = −.04 [−.15, .07], 1,575 turns), with similar response length (30.7 against 28.6 words) and no condition difference on any question, although the commonground question (Q5, Appendix C) was the hardest in both conditions (0.19 against 0.60 on the question asking for reasons).

Across all 63 participants, the longer someone spent answering during the session, the better their reconstruction accuracy afterwards (OR 1.45 per SD of time on task [1.09, 1.93]). However, justification depth during the session fell from one viewpoint to the next, by 0.013 of the scale per slot, or about 0.05 between the first viewpoint and the fifth $( p = . 0 0 1 )$ . It did not depend on how far a viewpoint sat from the participant’s own stance in either condition, so participants did not write shallower answers about the viewpoints they disagreed with most.

Stance representation therefore changed which viewpoints participants saw (Section 5.1) without changing how well they understood them.

## 5.3 Neutral Positions, Not Disagreement, Are Hardest to Reconstruct

Participants were right on 55.6% of concepts where they and the viewpoint took the same side (79/142), 51.2% where they took opposite sides (22/43), and 34.9% where at least one side was neutral (44/126), against chance at 33.3% (Table 1). Accuracy was higher where they agreed than where either side was neutral (OR 2.09 [1.20, 3.65], Holm $p = . 0 2 9 )$ , and no different where they disagreed (OR 1.19 [0.59, 2.42], $p = . 6 3 )$ . Disagreeing with a viewpoint therefore did not make it harder to read.

The shortfall sat where the participant was neutral, at 24% (11/45), below chance and below agreement (OR 0.28 [0.13, 0.61], p = .001). Where only the viewpoint was neutral it was 42% (22/53), no different from agreement. Participant-neutral labels were also the least stable when re-measured, 10 of 25 changing, so we treat this split as exploratory. The pattern held in both conditions $( p \ = \ . 8 5 )$ and under offline re-measurement (OR 1.86, Holm $p = . 0 4 8 )$ .

Within PSV, accuracy fell as participant and viewpoint differed on more concepts, OR 0.82 per $L _ { 1 }$ unit [0.72, 0.92], about 0.67 per concept, over 29 participants (Figure 4). The slope stayed between 0.74 and 0.88 across adjustments for text features, participant extremity, slot, offline stance labels, concept relation, and dropping each participant in turn. It was steepest where participant and viewpoint agreed (OR 0.68 [0.50, 0.93] against 0.98 where either was neutral), so distance cost accuracy even where they agreed. Under SCALAR we found no such association (OR 0.96 [0.84, 1.11]). The interaction was not reliable $( p = . 3 3 )$ and the two distance scales are not comparable, so this is not a condition difference.

<table><tr><td>Relation</td><td>Scalar</td><td>Psv</td><td>All</td></tr><tr><td>Same side</td><td>35/65 (.54)</td><td>44/77 (.57)</td><td>79/142 (.56)</td></tr><tr><td>Opposite sides</td><td>9/19 (.47)</td><td>13/24 (.54)</td><td>22/43 (.51)</td></tr><tr><td>Either neutral</td><td>24/70 (.34)</td><td>20/56 (.36)</td><td>44/126 (.35)</td></tr><tr><td>participant</td><td>5/28 (.18)</td><td>6/17 (.35)</td><td>11/45 (.24)</td></tr><tr><td>viewpoint</td><td>11/27 (.41)</td><td>11/26 (.42)</td><td>22/53 (.42)</td></tr><tr><td>both</td><td>8/15 (.53)</td><td>3/13 (.23)</td><td>11/28 (.39)</td></tr></table>

Table 1: Reconstruction accuracy by participant– viewpoint stance relation, shown as correct/total (accuracy). Chance is .33. Participant stances were extracted at runtime under PSV and post hoc under SCALAR.

Where participant and viewpoint agreed, 33 answers were still wrong. Of those, 21 said the viewpoint held no position and 12 gave the opposite one. Nine of the 21 were “not sure” answers, and without them the split is even, so we cannot say participants tended to leave a stance unattributed rather than reverse it.

Participants might have substituted their own stance for the viewpoint’s when reconstructing it. When their stance differed from the viewpoint’s, two incorrect labels were possible, only one of which matched their own stance, giving a chance rate of 0.5. Participants used their own stance in 21 of 35 PSV errors and 23 of 48 SCALAR errors. These proportions were close to the permutationnull rates of 47% in both conditions $\left( \boldsymbol { p } = . 0 6 3 \right.$ and $p = . 4 3 9$ , respectively). The PSV result points in the predicted direction, but the difference amounts to only about four errors across 19 participants. With so few errors, we cannot distinguish this pattern from chance, so we treat it as inconclusive.

## 6 Discussion

We investigated whether the stance representation a policy uses to select viewpoints changes what a user encounters and what they take away from it.

In RQ1, we investigated whether representing stance at the concept level, rather than as a single score, changes which viewpoints are selected. It does. Both conditions ran the same selection policy, so the difference came from the encoder alone, PSV produced 24 distinct exposure sequences against 6 under SCALAR, and participants encountered more of the concept–stance combinations available on the topic. Replaying the deployed policy over the full corpus and all possible participant positions shows the same pattern. While prior work has either surfaced diverse opinions without selecting relative to the user (Freelon et al., 2012; Kim et al., 2021), or selected relative to the user while treating the stance encoding as given (Aicher et al., 2022, 2023a), our results indicate that the stance representation itself shapes selection behavior by defining which distinctions the policy can make between viewpoints.

![](images/6d01654408786cbf113194d680c90dc9600ed890e24e71ca2fafc36b7e0c1d92.jpg)  
Participant-to-viewpoint L1 distance over five concepts  
Figure 4: Reconstruction accuracy by participant– viewpoint $L _ { 1 }$ distance over five concepts. Marker area indicates the number of participants at each distance, lines are binomial GEE fits with 95% bands, and the dashed line marks chance (1/3).

Our second research question (RQ2) asked whether this difference in exposure carried through to how deeply participants justified the viewpoints they encountered and how accurately they reconstructed them. It did not, on either measure, but our study was powered to detect only large effects. This is consistent with (Saveski et al., 2022), who found that exposure to opposing political content did not improve participants’ reported understanding of why others held those views. In our study, exposure was deliberately structured and every viewpoint was followed by fixed Socratic style questions, yet reconstruction still did not differ between conditions. Showing participants a wider range of viewpoints did not help them reconstruct those viewpoints more accurately.

In RQ3, we investigated which parts of a viewpoint participants reconstructed correctly, and how this depended on their own stance. Participants were about as accurate on concepts where they opposed the viewpoint as on concepts where they agreed with it, and markedly less accurate where either side held no position. Accuracy declined as the overall stance distance increased, consistent with the discrepancy account of Sherif and Hovland (1961). However, the most frequent reconstruction error was to label an opposing stance as neutral, and neutral positions were the hardest to reconstruct. This suggests that stance discrepancy may affect not only how a viewpoint is judged, but also how its specific claims are understood and remembered.

Implications These findings have implications for dialog systems that select viewpoints for a user. Breadth and stance discrepancy should be treated separately, since novelty relative to the user can favor the most distant viewpoints. A policy could instead target a particular stance distance and select the most novel viewpoint within that range, provided the representation distinguishes multiple viewpoints at the same distance. Our policy keeps the distance bound fixed within each session. Future work could explore within-session adaptation, updating this bound in response to user interaction to balance novelty and discrepancy dynamically.

Societal Impact Polarization on contested issues is a long-standing societal problem. With this work we aim to contribute to supporting users in encountering and engaging with viewpoints other than their own. We integrate this support into a dialog system, a type of system many people are now familiar with, and show that the stance representation a policy uses determines which disagreements a user is able to meet. This opens up opportunities for dialog systems that broaden what users encounter on contested issues while remaining accountable for which dimensions of disagreement they act upon on.

## 7 Conclusion

We presented a viewpoint-selection policy that operates over interchangeable stance representations, allowing us to vary the representation while keeping the selection rule fixed. Evaluating it with 63 participants and by replaying it across the full stance space, we showed that representation shapes which differences the system can act on. Conceptlevel selection produced more varied exposure and broader topic coverage, but did not change justification depth or overall reconstruction accuracy. Concept-level analysis showed that participants reconstructed neutral concepts less accurately than concepts on which they agreed or disagreed with a viewpoint, a distinction obscured by overall accuracy. Our findings establish stance representation as a design choice in viewpoint selection. This opens a direction for systems that control how far a viewpoint is from a user separately from which specific disagreements it raises.

## Limitations

Our findings have several limitations. First, the study focused on a single contested issue, one language, and a corpus of user-contributed viewpoints, so the results may not generalize to other topics or content sources. Second, the concept-level representation used a fixed signature of five concepts selected during system development; different concepts or issues may produce different selection patterns. Third, reconstruction required participants to report a viewpoint’s stance on each concept, which may conflate their understanding of its position with their recall of its content. Finally, the study involved 63 participants and was powered to detect only large effects in the condition comparison. The absence of differences in justification depth and reconstruction accuracy therefore does not establish that the representations are equivalent on these outcomes.

## Ethics Statement

This study was approved by the Human Research Ethics Committee of our university. All participants provided consent before participation. The dialog system offered no opinions or advice of its own and generated no arguments. That said, both selecting viewpoints close to the user’s position and selecting ones far from them can have their own consequences, and deployment of our proposed approach on more controversial and sensitive topics and in real-world settings may therefore need further ethical considerations and approvals.

## Gen AI Declaration

Generative AI tools were used to assist with copyediting and coding, but the authors are fully accountable for the content and ideas.

## Acknowledgments

We gratefully acknowledge the support of the Delft AI Initiative and the Designing Intelligence (DI) lab. This research was partially funded by the Hybrid Intelligence Center, a 10-year programme funded by the Dutch Ministry of Education, Culture and Science through the Netherlands Organisation for Scientific Research, Hybrid Intelligence Centre, grant number 024.004.022.

## References

Annalena Aicher, Daniel Kornmueller, Yuki Matsuda, Stefan Ultes, Wolfgang Minker, and Keiichi Yasumoto. 2023a. Towards breaking the self-imposed filter bubble in argumentative dialogues. In Proceedings of the 24th Annual Meeting of the Special Interest Group on Discourse and Dialogue, pages 593– 604, Prague, Czechia. Association for Computational Linguistics.

Annalena Aicher, Wolfgang Minker, and Stefan Ultes. 2022. Towards modelling self-imposed filter bubbles in argumentative dialogue systems. In Proceedings of the Thirteenth Language Resources and Evaluation Conference, pages 4126–4134, Marseille, France. European Language Resources Association.

Annalena Bea Aicher, Daniel Kornmüller, Wolfgang Minker, and Stefan Ultes. 2023b. Self-imposed Filter Bubble Model for Argumentative Dialogues. In Proceedings ofthe 5th International Conference on Conversational User Interfaces, pages 1–11, Eindhoven Netherlands. ACM.

Beng Heng Ang, Sujatha Das Gollapalli, and See Kiong Ng. 2023. Socratic question generation: A novel dataset, models, and evaluation. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pages 147–165.

Markus Brenneis, Maike Behrendt, and Stefan Harmeling. 2021. How will I argue? a dataset for evaluating recommender systems for argumentations. In Proceedings ofthe 22nd Annual Meeting ofthe Special Interest Group on Discourse and Dialogue, pages 360–367, Singapore and Online. Association for Computational Linguistics.

Yun-Shiuan Chuang, Ruixuan Tu, Chengtao Dai, You Li, Smit Vasani, Binwei Yao, Michael Henry Tessler, Sijia Yang, Dhavan Shah, Robert Hawkins, Junjie Hu, and Timothy T. Rogers. 2026. Debate: A largescale benchmark for evaluating opinion dynamics in role-playing llm agents. Preprint, arXiv:2510.25110.

Valdemar Danry, Pat Pataranutaporn, Yaoli Mao, and Pattie Maes. 2023. Don’t just tell me, ask me: AI systems that intelligently frame explanations as questions improve human logical discernment accuracy over causal AI explanations. In Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems, pages 1–13. ACM.

Mark H. Davis. 1983. Measuring individual differences in empathy: Evidence for a multidimensional approach. Journal of Personality and Social Psychology, 44(1):113–126.

Esin Durmus and Claire Cardie. 2018. Exploring the role of prior beliefs for argument persuasion. In Proceedings ofthe 2018 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 1035–1045.

Esin Durmus, Faisal Ladhak, and Claire Cardie. 2019. Determining relative argument specificity and stance for complex argumentative structures. In Proceedings ofthe 57th Annual Meeting ofthe Associationfor Computational Linguistics, pages 4630–4641, Florence, Italy. Association for Computational Linguistics.

Neele Falk, Andreas Waldis, and Iryna Gurevych. 2024. Overview of PerspectiveArg2024 the first shared task on perspective argument retrieval. In Proceedings of the 11th Workshop on Argument Mining (ArgMining 2024), pages 130–149, Bangkok, Thailand. Association for Computational Linguistics.

Deen G. Freelon, Travis Kriplean, Jonathan Morgan, W. Lance Bennett, and Alan Borning. 2012. Facilitating diverse political engagement with the living voters guide. Journal ofInformation Technology & Politics, 9(3):279–297.

Dieter Frey. 1986. Recent research on selective exposure to information. volume 19 of Advances in Experimental Social Psychology, pages 41–80. Academic Press.

Isabelle Guyon and André Elisseeff. 2003. An introduction to variable and feature selection. Journal of machine learning research, 3(Mar):1157–1182.

Peter E Hart, David G Stork, and RO Duda. 2001. Pattern classification. Wiley Hoboken.

Philipp Heinisch, Lorik Dumani, Philipp Cimiano, and Ralf Schenkel. 2024. “tell me who you are and I tell you how you argue”: Predicting stances and arguments for stakeholder groups. In Findings ofthe Associationfor Computational Linguistics: NAACL 2024, pages 1968–1982, Mexico City, Mexico. Association for Computational Linguistics.

Lawrence Hubert and Phipps Arabie. 1985. Comparing partitions. Journal ofClassification, 2(1):193–218.

Shanto Iyengar, Yphtach Lelkes, Matthew Levendusky, Neil Malhotra, and Sean J. Westwood. 2019. The Origins and Consequences of Affective Polarization in the United States. Annual Review of Political Science, 22(1):129–146.

Shanto Iyengar and Sean J. Westwood. 2015. Fear and Loathing across Party Lines: New Evidence on Group Polarization. American Journal of Political Science, 59(3):690–707. \_eprint: https://onlinelibrary.wiley.com/doi/pdf/10.1111/ajps.1

Johannes Kiesel, Nico Reichenbach, Benno Stein, and Martin Potthast. 2021. Image retrieval for arguments using stance-aware query expansion. In Proceedings

of the 8th Workshop on Argument Mining, pages 36– 45, Punta Cana, Dominican Republic. Association for Computational Linguistics.

Hyunwoo Kim, Haesoo Kim, Kyung Je Jo, and Juho Kim. 2021. StarryThoughts: Facilitating diverse opinion exploration on social issues. Proceedings of the ACM on Human-Computer Interaction, 5(CSCW1):1–29.

Jonathan Kobbe, Ioana Hulpus, and Heiner Stuckenschmidt. 2020. Unsupervised stance detection for arguments from consequences. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 50–60, Online. Association for Computational Linguistics.

Deanna Kuhn and Wadiya Udell. 2007. Coordinating own and other perspectives in argument. Thinking & Reasoning, 13(2):90–104.

Stephan Lewandowsky, Ullrich K. H. Ecker, Colleen M. Seifert, Norbert Schwarz, and John Cook. 2012. Misinformation and its correction: Continued influence and successful debiasing. Psychological Science in the Public Interest, 13(3):106–131. PMID: 26173286.

Moritz Plenz, Philipp Heinisch, Janosch Gehring, Philipp Cimiano, and Anette Frank. 2025. From argumentation to deliberation: Perspectivized stance vectors for fine-grained (dis)agreement analysis. In Findings ofthe Associationfor Computational Linguistics: NAACL 2025, pages 1525–1553, Albuquerque, New Mexico. Association for Computational Linguistics.

Jan Heinrich Reimer, Alexander Bondarenko, Maik Fröbe, and Matthias Hagen. 2023. Stance-aware re-ranking for non-factual comparative queries. In Proceedings ofthe 10th Workshop on Argument Mining, pages 45–51, Singapore. Association for Computational Linguistics.

Renate L. E. P. Reniers, Ahmad Abu-Akel, and Ana Seara-Cardoso. 2022. Editorial: Cognitive Empathy and Perspective Taking: Understanding the Mechanisms of Normal and Abnormal Experiences and Abilities. Frontiers in Psychiatry, 13:945258.

Peter J. Rousseeuw. 1987. Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. Journal ofComputational and Applied Mathematics, 20:53–65.

Martin Saveski, Nabeel Gillani, Ann Yuan, Prashanth Vijayaraghavan, and Deb Roy. 2022. Perspectivetaking to reduce affective polarization on social media. Proceedings ofthe International AAAI Confer-<sub>152.</sub> ence on Web and Social Media, 16:885–895.

Muzafer Sherif and Carl I. Hovland. 1961. Social Judgment: Assimilation and Contrast Effects in Communication and Attitude Change. Yale University Press.

Marco R Steenbergen, André Bächtiger, Markus Spörndli, and Jürg Steiner. 2003. Measuring political deliberation: A discourse quality index. Comparative European Politics, 1(1):21–48.

Richard S. Sutton and Andrew G. Barto. 2018. Reinforcement Learning: An Introduction, 2 edition. MIT Press, Cambridge, MA, USA.

Matthias Trénel. 2004. Measuring the deliberativeness of online discussions. coding scheme 2.4. Unpublished paper, October, 18.

Henning Wachsmuth, Shahbaz Syed, and Benno Stein. 2018. Retrieval of the best counterargument without prior topic knowledge. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 241–251, Melbourne, Australia. Association for Computational Linguistics.

Brahim Zarouali, Mykola Makhortykh, Mariella Bastian, and Theo Araujo. 2021. Overcoming polarization with chatbot news? investigating the impact of news content containing opposing views on agreement and credibility. European Journal ofCommunication, 36(1):53–68.

Xinyan Zhao and Edward L. Fink. 2021. Proattitudinal versus counterattitudinal messages: Message discrepancy, reactance, and the boomerang effect. Communication Monographs, 88(3):286–305.

Yanmengqian Zhou and Lijiang Shen. 2024. Processing of misinformation as motivational and cognitive biases. Frontiers in Psychology, 15:1430953.

## A Corpus Quality Filtering

Eight deterministic criteria were applied to the 408 raw arguments. Table 2 lists them with the number of entries each flagged and the pattern used. Criteria B–F are case-insensitive, and E and F also match across line breaks. The criteria are not mutually exclusive, so the per-criterion counts sum to 45 while 41 unique entries were flagged. Three further entries were gibberish that no criterion matched, because criterion A requires eight or more consecutive single-character tokens and the noise in these entries is interleaved with longer ones. We removed these three by hand and record their content hashes in the filtering script, so the reduction from 408 to 364 is reproducible.

Description Flagged Pattern   
A Incoherent or gibberish 5 (\b\w\b[ \t]+){8,}   
B External URL references 5 https?://|www\.   
C Strong profanity 7 \bfuck\w\*\b|\bshit\w\*\b|\basshole\w\*\b   
D Mild profanity 17 \bdamn\b|\bhell\b|\bcrap\b|\bbitch\w\*\b|\bbastard\w\*\b   
E Repetitive meme or spam 5 (\b\w+[ \t]+\w+[ \t]+\w+[ \t]+\w+[ \t]+\w+\b)(?:.\*?\1){3,}   
F Student essay artefacts 3 Name:.\*?Teacher:|Hunting Essay|Class Period:   
G Excessive whitespace 1 \n{5,}| {15,}   
H Repeated symbols 2 (>[>\s]\*){6,}|\*{4,}  
Table 2: Corpus quality-filtering criteria, the number of entries each flagged, and the regular expression used.

## B Choosing the Five Concepts

Concepts were ranked by three criteria. Stance variance ranks concepts by the variance of their discrete stance values (Guyon and Elisseeff, 2003). Non-neutrality rate ranks by the proportion of arguments taking a non-neutral stance, which operationalises the granularity principle of Plenz et al. (2025). The Fisher discriminant quantifies each concept’s power to separate the pro and con groups, $F _ { i } = ( \mu _ { \mathrm { P R O } , i } - \mu _ { \mathrm { C O N } , i } ) ^ { 2 } / ( \sigma _ { \mathrm { P R O } , i } ^ { 2 } + \sigma _ { \mathrm { C O N } , i } ^ { 2 } + \varepsilon )$ (Hart et al., 2001). All five deployed concepts rank within the top nine by stance variance and the top ten by the Fisher criterion.

We evaluate the deployed signature against a reference clustering in the full 100-concept space. Arguments were clustered with hierarchical average linkage under $L _ { 1 }$ distance, and the number of clusters was chosen by maximising the silhouette score (Rousseeuw, 1987) over $K \in \{ 2 , \ldots , 6 \}$ which selects $K = 2$ . Agreement between the reduced-space and full-space partitions, measured by the Adjusted Rand Index (Hubert and Arabie, 1985), is 0.92 on the filtered corpus and 0.88 on the raw corpus.

Selecting the top concepts by any single criterion performs worse at the same size. Fisher-based selection at N = 5 reaches ARI 0.84 on the raw corpus, against 0.88 for the deployed signature, and stance variance ranking needs six concepts to reach 0.90.

## C Question Bank

Table 3 gives the five question templates verbatim.   
The order is fixed and identical in both conditions.

<table><tr><td colspan="2">Template</td></tr><tr><td>Q1</td><td>Can you explain in your own words what you under- stand this perspective to be saying?</td></tr><tr><td>Q2</td><td>Why do you think someone might hold this view about animal hunting?</td></tr><tr><td>Q3</td><td>What evidence or assumptions might underlie this</td></tr><tr><td>Q4</td><td>perspective? How does this perspective compare to your own view on {topic}?</td></tr><tr><td>Q5</td><td>What common ground, if any, do you see between this perspective and your own?</td></tr></table>

Table 3: The five question templates, in presentation order.

Q2 to Q5 are shown alone, with only the topic substituted. Q1 is shown as part of a longer (first) message that introduces the viewpoint and then poses Q1.

Viewpoints are named Alex, Jordan, Morgan, Riley and Casey in order of appearance.

## D Prompts

Figure 5 reproduces the three runtime prompts as they appear in the deployed code. All are zero-shot, sent as a single user message with no system prompt and no exemplars. All three use claude-haiku-4-5-20251001 with temperature unset, so at the API default.

<table><tr><td></td><td>SCALAR (n = 31)</td><td>PsV (n = 32)</td></tr><tr><td>Gender (p = .79)</td><td></td><td></td></tr><tr><td>Woman</td><td>11 (35.5)</td><td>10 (31.3)</td></tr><tr><td>Man</td><td>18 (58.1)</td><td>20 (62.5)</td></tr><tr><td>Non-binary</td><td>1 (3.2)</td><td>0 (0.0)</td></tr><tr><td>Not reported</td><td>1 (3.2)</td><td>2 (6.3)</td></tr><tr><td>Age bracket (p = .34)</td><td></td><td></td></tr><tr><td>19-24</td><td>13 (41.9)</td><td>14 (43.8)</td></tr><tr><td>25-34</td><td>13 (41.9)</td><td>13 (40.6)</td></tr><tr><td>35-44</td><td>4 (12.9)</td><td>1 (3.1)</td></tr><tr><td>45-54</td><td>1 (3.2)</td><td>4 (12.5)</td></tr><tr><td>IRI-PT, 0–28 (p = .65)</td><td></td><td></td></tr><tr><td>Mean (SD)</td><td>15.5 (4.2)</td><td>16.0 (4.1)</td></tr><tr><td>Range</td><td>8-25</td><td>9-24</td></tr></table>

Table 4: Condition balance for the 63 analysed participants, as counts with column percentages. No participant selected a bracket above 45–54. p values are Fisher–Freeman–Halton exact tests for gender and age, with declined responses (not reported) excluded, and a Welch t-test for IRI-PT (t(60.9) = 0.46).

## E Participant Demographics

Age, gender, and the perspective-taking subscale of the Interpersonal Reactivity Index (IRI-PT) (Davis, 1983) were collected in the pre-task survey. Table 4 reports all three for the 63 analysed participants.

Gender and age bracket are tested with the Fisher–Freeman–Halton exact test, which extends Fisher’s exact test to r × 2 tables, and IRI-PT with a Welch two-sample t-test. With these cell counts the tests have limited power.

![](images/533fd1b446e14a1ed15dcc36b6f0e9d72daba516c8c1fa75b6408be62c844f13.jpg)  
Figure 5: The three runtime prompts.

## F Determinism of the Scalar Sequences

Section 5.1 reports that under SCALAR the sequence of distances follows from the participant’s stance and the distance bound. This appendix derives that sequence and checks it against the sessions.

Let s be the participant’s stance level and δ the bound. Under SCALAR a viewpoint carries one stance feature, its level, so novelty is binary and the policy of Section 3.3 reduces to three steps. First, the eligible levels are those within δ of s. Second, the policy shows an eligible level that has not been shown yet, taking the nearest one when several remain. Third, once every eligible level has been shown, novelty is zero for all candidates and the tie-break on distance returns the level nearest s, which is s itself. Several corpus viewpoints share each level, so which text a slot shows varies, but its distance does not.

Table 5 lists every combination of s and δ in the data with the observed sequence and the one this derivation gives. The two agree in all 9 combinations, covering 31 of 31 sessions and 155 of 155 slots.

<table><tr><td>S</td><td>δ</td><td>n</td><td>observed</td><td>derived</td></tr><tr><td>1</td><td>1</td><td>2</td><td>0,1,0,0,0</td><td>0,1,0,0,0</td></tr><tr><td>2</td><td>1</td><td>4</td><td>0,1,1,0,0</td><td>0,1,1,0,0</td></tr><tr><td>3</td><td>1</td><td>10</td><td>0,1,1,0,0</td><td>0,1,1,0,0</td></tr><tr><td>4</td><td>1</td><td>5</td><td>0,1,1,0,0</td><td>0,1,1,0,0</td></tr><tr><td>1</td><td>2</td><td>2</td><td>0,1,2,0,0</td><td>0,1,2,0,0</td></tr><tr><td>2</td><td>2</td><td>3</td><td>0,1,1,2,0</td><td>0,1,1,2,0</td></tr><tr><td>3</td><td>2</td><td>3</td><td>0,1,1,2,2</td><td>0,1,1,2,2</td></tr><tr><td>3</td><td>3</td><td>1</td><td>0,1,1,2,2</td><td>0,1,1,2,2</td></tr><tr><td>4</td><td>4</td><td>1</td><td>0,1,1,2,3</td><td>0,1,1,2,3</td></tr></table>

Table 5: Observed and derived distance sequences under SCALAR, by participant stance s and distance bound δ. n is the number of sessions in each combination.

Two things limit what the agreement shows. First, every combination holds a single observed sequence, so the evidence is 9 combinations rather than 31 sessions, and two of them hold one session each. Second, slots 1 and 2 are identical in every session and therefore carry no information about the derivation.

## G Simulation Results

The exposure results in Section 5.1 hold for the stances and bounds the 63 participants happened to produce. To separate the policy from that sample, we replay the deployed selection code over the full corpus for every participant stance each representation admits, the 5 scalar positions and the 243 PSV profiles, every distance bound available in that condition and 100 random seeds per setting. Each run returns the five viewpoints the policy would select. We score a run by coverage, the number of the 15 concept–stance combinations its five viewpoints take between them, which is counted the same way in both conditions and is therefore comparable across them.

Coverage rises with the bound in both conditions. Under PSV it reaches all 15 combinations before the bound is exhausted. Under SCALAR it flattens below 15, because the corpus holds at most two scalar positions at each distance from the participant, so a wider bound adds few further positions to choose from.

![](images/d50c9c66c99a9db283518bb539e3f5427d58db10ff989a1e5bab1f1b7de344d3.jpg)  
Figure 6: Simulated coverage of the 15 concept–stance combinations by distance bound, over the full corpus and every possible participant position, 100 seeds per setting. Each condition’s bound is rescaled to its own range (SCALAR 1–4, PSV 2–10) and labelled with its actual value. Bands show ±1 SD across runs. The two horizontal scales are not commensurable, so only the shapes of the lines should be compared.

## H Label Agreement

Participant stance profiles and corpus labels are both assigned automatically (Section 3.2), so we check them in two ways. First, stability. We run the extraction twice over the same participant texts and compare the two profiles, 157 concept cells over the 32 PSV participants, with three cells excluded where one run returned no decision. Second,

agreement with a second labelling. We label the 47 viewpoints that served as reconstruction targets a second time and compare the result to the corpus labels, 235 concept cells. Table 6 reports Cohen’s κ per concept for both checks.
<table><tr><td>Concept</td><td>Participant</td><td>Viewpoint</td><td>Neutral</td></tr><tr><td>hunt game</td><td>0.84</td><td>0.74</td><td>6</td></tr><tr><td>hunter</td><td>0.81</td><td>0.78</td><td>7</td></tr><tr><td>while hunting animals</td><td>0.66</td><td>0.74</td><td>6</td></tr><tr><td>outdoor activity</td><td>0.72</td><td>0.52</td><td>12</td></tr><tr><td>good</td><td>0.60</td><td>0.39</td><td>10</td></tr><tr><td>overall</td><td>0.75</td><td>0.63</td><td>41</td></tr></table>

Table 6: Cohen’s κ per concept. Participant: two extractions of the same participant texts, 157 cells over 32 participants, three unresolved cells excluded. Viewpoint: re-labelling of the 47 target viewpoints against the corpus labels, 235 cells. Neutral: how many of the 47 corpus labels for that concept are neutral. The two concepts the labellers agree on least are the two with the most neutral labels.

Agreement is lowest on the same concept in both comparisons, good, which is also the most abstract of the five. The two weakest concepts, good and outdoor activity, are the two whose corpus labels contain the most neutral values. Counted over participant cells, where each of the 63 participants contributes the five concepts of their own target viewpoint, the two labellers differ on 62% of the cells the corpus labels as neutral (50/81) against 12% of polar cells (27/234). Disagreement is therefore concentrated on neutral stances, which is where participants were also least accurate.

## I Target Specificity

For each participant, g is the share of concepts on which the reconstruction matches the assigned viewpoint, minus the mean share over the four other viewpoints in the session. Because g depends on how similar the five viewpoints are, we subtract its expected value under a reconstructor with no knowledge of the assigned viewpoint, evaluated on the participant’s own five viewpoints. Table 7 reports the condition difference under three such reference reconstructors. Under uniform random answers the expected value is zero, so that row equals the raw difference.

<table><tr><td>Reference</td><td>n</td><td>b (PSV-SCALAR)</td><td>p</td></tr><tr><td>None (raw g)</td><td>63</td><td>+0.055 [-0.07, 0.18]</td><td>.41</td></tr><tr><td>Participant&#x27;s stance</td><td>59</td><td>+0.083[-0.10, 0.27]</td><td>.37</td></tr><tr><td>Session centroid</td><td>63</td><td>+0.043 [-0.09, 0.18]</td><td>.52</td></tr><tr><td>Uniform random</td><td>63</td><td>+0.055[-0.07,0.18]</td><td>.41</td></tr></table>

Table 7: Condition difference in target specificity, before and after adjusting for how far apart a session’s five viewpoints were. OLS with HC3 standard errors, one row per participant.

More participants scored above zero on the target-specificity difference g under PSV than under SCALAR, 24/32 against 13/31. The mean difference in g is small and its confidence interval includes zero, and adjusting for session geometry does not change this. The adjustment is needed because PSV sessions were more spread out, with a mean $L _ { 1 }$ distance of 6.3 from the target to the other four viewpoints against 5.3 under SCALAR, which makes the target easier to tell apart from its neighbours.