# Breakdown of Conversational Involvement: A Joint Multi-Modal Analysis of Conversation Floors and F-formation Detection with Multiple Features

Zonghuan Li, Chirag Raman, Hayley Hung Member, IEEE

Abstract—Interactions happen universally and spontaneously in daily lives. Studying involvement in interactions helps us analyze who can influence who and has a wide potential in applications. Specifically, we consider conversational involvement in free-standing conversations. We study two operationalizations of an FCG: F-formations and conversation floors, and propose that current assumption that F-formations capture single conversations is insufficient. To illustrate how different implementations could impact the detected F-formations, we employ three analyses. First, we propose that speaking status can be used for estimating conversation floors, and leveraged it to analyze floors within F-formations. Second, we analyze spatial characteristics of the detection results by applying different feature sets on chosen baselines, and find that groups resulting from head features are more likely to be subsumed by groups detected using other features. Finally, we do a joint analysis which combines the spatial features and floor patterns in the conversation. We observe that certain pipelines can detect groups with single floors from upper body features, while retaining F-formations with lower body features. We conclude that a more detailed theory and framework is needed to holistically demonstrate conversational involvement within interactions.

Index Terms—Conversations, F-formations, floors, involvement, schisming.

## I. INTRODUCTION

As humans, interactions are an indispensable part of our lives. We interact with the environment, with other people, and with artificial systems/agents. Studying interaction and in particular who can influence who, is not just a conceptual problem, but also motivated by technical applications. In modern conversational agents, this helps us develop personalized supportive technologies (synchronous or asynchronous, embodied or disembodied), and enhance Human-Robot Interaction (HRI) by identifying the influential source or agents. In fields like health or security, a deeper understanding of who has the influence can help supportive surveillance, for instance, identifying gatherings and detecting potential crimes. Another major application of this is to support people in spontaneous conversations in the wild, which has been linked with organizational learning and entrepreneurial success.

It is common to define and analyze involvement to describe who is influencing who. Multiple sources have linked involvement with attention, coordination, and elaboration [1, 2, 3]. Indeed, involvement can largely represent influence in interactions, and provides crucial insights in interpreting intentions and social experience.

A large part of daily interactions are associated with conversations, and we are in particular interested in those where people can freely move around and forming groups. In such scenes, encapsulating involvement requires knowing who is talking with who, and specifically, needs to know free standing conversation groups, or FCGs. By its notion, it is natural to expect that each FCG contain exactly one conversation. However, conversation itself is complicated, especially in the temporal dimension. Indeed, it is hard to delineate the boundary between one conversation and another one, especially with the presence of multiple people joining/leaving, and the change in topics. As we discuss later, this often lead to multiple choices of operationalization of turn, topic, or floor [4], and often needs cognitive analysis.

However, such cognitive analysis is hard to be built upon for automated implementations, and as a result, FCGs are typically operationalized differently. Specifically, spatial factors like positions and orientations, are much easier to obtain and compute. Kendon [5] has proposed the term F-formation to describe spatial structure of people’s interactions, which has then received much attention in the domain of automatic FCG detection. It becomes natural, then, to question whether these FCGs detected as operationalized by F-formations really contain single conversation floors. If the detected F-formations do not align well with the temporal characteristics exhibited by a floor, this implementation is insufficient in terms of accurate locating the conversation.

In this article, we distinguish but also link the two operationalization mentioned: the conversation floor based on topic analysis and F-formation widely used in FCG detections. We explore the influence of spatial factors on the detection of FCGs and link the results back to what we observe by focusing on floor as the temporal factor. We choose speaking as the decisive indicator of a conversation. Concretely, we pose the following research questions:

1) Can we use observed speaker turns to infer the conversation floors within an F-formation?

2) What is the influence of different spatial factors to detected F-formations in terms of conversation floors? Our contributions are as follows:

1) In conceptual perspective, we propose to use speaking turn as an indicator of distinct conversation floors. We also connect this notion to analyzing schisming in conversations [6, 7, 8].

2) In methodological perspective, we provide evidence that multiple conversations can exist in detected Fformations, and show that different spatial factors contribute differently in terms of floors. We also provide two metrics to measure spatial relationship within schisming.

The remaining parts of the paper are structured as following. We describe some of the spatial and temporal notions and factors used to study FCGs in section II. In section III we provide a literature review involving the use of these spatial or temporal notions, motivating the need to consider both of these aspects. In section IV, we first propose to use speaking turns to indicate distinct conversation floors. Then, we propose the Soft Homogeneity (SH) and Split Factor (SF) to analyze spatial relationship between group detection results. Third, we link floor characteristics with different features by performing a joint analysis. In section V we provide details of the experimental settings and analyze the results. Finally, section VI summarizes our findings and concludes the paper.

## II. BACKGROUND

## A. Spatial factors

Originally proposed by Kendon [5], the notion of Fformation has become the embodiment of FCG that is adopted in many studies. An F-formation is defined as follows:

An F-formation arises whenever two or more people sustain a spatial and orientational relationship in which the space between them is one to which they have equal, direct, and exclusive access. [5, p. 150]

Kendon argues that activity is always located, and denotes the space extending in front of a person that is used for the activity as the person’s transactional segment. “When two or more people come together to perform some activity, they are liable to arrange themselves such that their individual transactional segments overlap to create a joint transactional space.” This joint space between the participants is an o-space, and an F-formation arises whenever there is an o-space.

Since this notion depends on physical aspects, there has been many approaches trying to detect F-formations in an automatic way. While at the same time, most of these computational works often assume that the transaction involves a single conversation.

## B. Temporal factors

The conversation of focus in an FCG, however, is dynamic in nature. If conversations change over time, what are the temporal units that describe their underlying structure? Some of the terms used in early literature to organize conversational units include turn, topic, gap, and floor. First proposed by [9], Sacks equaled the term of a turn and a floor. On the other hand, Edelsky [4], while providing an excellent review of these concepts, has proposed that a floor is different from a turn. She states that most of these units were defined previously on the basis of some technical or mechanical structure such as signals of speakers or listeners, ignoring the intention of the participant. In contrast, Edelsky defines floor from a cognitive perspective:

The floor is defined as the acknowledged what’s going-on within a psychological time/space. What’s going on can be the development of a topic or a function (teasing, soliciting a response, etc.) or an interaction of the two. It can be developed or controlled by one person at a time or by several simultaneously or in quick succession.[4, p. 405]

Many later works also adopt to leverage cognitive analysis for defining floors. McNeil [10] used the term “growth point” to be the mental package that carries linguistic components, and sharing the same growth point indicates that they are within the same mental being. Similarly, [11] proposed that a floor to be “a means of communicative attention orientation which exists not at the level of turn and move but at a higher level of conversation structure”, which approaches also from a cognitive perspective.

## III. RELATED WORKS

## A. Estimating Involvement

Involvement, as described by Goffman [12], refers to “sustain some kind of cognitive and affective engrossment in it, some mobilization of one’s psychobiological resources”. Many works have been developed to analyze involvement through verbal cues like turn-taking patterns. Bohus and Horvitz [13] propose a self-supervised method for forecasting disengagement with an interactive robot using a conservative heuristic. Some works also used turn-taking features to estimate some notion of involvement. Pentland, Madan, and Gips [14] measured engagement by the z-scored influence each person has on the other’s turn-taking for a pair of participants. Hung and Gatica-Perez [15] found that the pause duration between an individual’s turns, aggregated at group level, is highly predictive of cohesion in small group meetings.

However, as the floor is typically held by only a single speaker at a time [6] in a conversation, estimating involvement in terms of turns and the floor holder poses potential risk. This is because there is an implicit assumption that involvement is only embodied in speaking behavior. As such, many other studies also tries to detect involvement with silent participants. Oertel, Funes Mora, Gustafson, and Odobez [16] characterize silent participants into multiple categories (attentive listener, side participant, bystander) from audiovisual cues. Oertel and Salvi [17] also show that it is possible to estimate individual engagement and group involvement in a multiparty corpus by analyzing the participants’ eye-gaze patterns.

The following works demonstrate that the task of estimating participant involvement is subjective in nature. Zhang and Hung [18, 19] study the task of detecting associates of an F-formation: members that are attached to an F-formation but do not have full status. They argue that the labeling of conversation groups is not an objective task. By collecting multiple annotations of perceived associates, they demonstrate how detecting them can improve initial estimates of fullmembers of an F-formation.

## B. Detecting FCGs

In most works regarding the detection and locating the conversations, an FCG is operationalized as an F-formation. Early work on the task of detecting FCGs in video data developed concurrently from two perspectives: those that estimate the location of the o-space [20, 21]; or those that view an Fformation as a set with individuals being assigned exclusive membership [22, 23]. There has also been considerable work focused on incorporating temporal information for the same task of detecting conversational groups [24, 25, 26, 27]. Notably, these approaches utilize the head pose as a proxy for the transactional segment (in addition to the body pose) to model F-formation membership. The head pose also acts as a proxy for the Visual Focus of Attention (VFoA) [28] which further perpetuates an implicit assumption that the joint mutual gaze exists and must be singular within an F-formation. Vazquez, Steinfeld, and Hudson [29] also assume a single conversation within an F-formation while developing a policy for a robot to be aware of a single focus of attention of the conversation. Recently, methods leveraging Visual Language Models (VLMs) were proposed to predict conversation groups dynamically [30]. However, it does not explicitly distinct in the notion between F-formations and conversation floors.

## C. Schisming in conversation

For a single conversation, it has been pointed out that it is hard to keep more than 4 participants. In a conversation with at least four participants, sometimes it splits up into two or more conversations. This transformation is referred to as a schism [8] or schisming.

One of the earliest allusions to the phenomenon of schisming based on anecdotal evidence occurs in the work of Goffman, who suggested that a gathering of two participants exhausts an encounter and forms a fully-focused gathering [12]. With more than two participants, there may be persons officially present in the situation who are not themselves so engaged. These bystanders change the gathering into a partlyfocused one. If more than three persons are present, there may be more than one encounter carried on in the same situation, resulting in a multi-focused gathering, which effectively has introduced schisming. In subsequent work, Sacks, Schegloff, and Jefferson [6] and Goodwin [7] both indicated that the co-existence of two turn-taking systems is the most decisive characteristic of schisming. Specifically, Egbert also makes a systematic differentiation between overlap and simultaneous talk during schisming. In overlap, simultaneous speakers compete for the floor, an event usually resolved by returning to one-speaker-at-a-time. In schisming by contrast, simultaneous speakers orient to one of two distinct floors, an event which if resolved successfully, results in the establishment of two floors [8]. Later, Dunbar, Duncan, and Nettle [31] demonstrates “clique size” which describes the number of people in a conversation. It reveals that the frequency of occurrence drops significantly for cliques with size equal or above 4.

Other studies attempts to answer the “why” question to this limitation. More recently, [32] has attributed it to limited cognitive capacity, which could store only 3 to 5 meaningful items for adults. The author further hypothesizes that too many working memory items could prohibit the brain from keeping all of them active, assuming a turn-taking pattern in neuron firing. [33] attempts to answer from a societal perspective, by experimenting with mathematical models that simulate human society and count the pairs of inclusive dyads (including the particular person) and exclusive dyads (excluding the particular person). Since the number of inclusive dyads grow linearly and the number of exclusive dyads grows quadratically, they propose that it is possible that people benefit more from being in the inclusive dyads and not in the exclusive ones, and thus inhibits forming conversations of more than 4 people.

![](images/75b64661606fa6ea3b34ee0e10a7b9cb30e289dec7565e32532a84e3e65befe6.jpg)  
Fig. 1. Depiction of a single F-formation with multiple conversation floors. The darker green regions within dotted lines represent distinct simultaneous conversation floors. Most works representing a conversing group as an Fformation make the simplifying assumption that a single conversation occurs within an F-formation with a joint focus of attention for all members.

## IV. METHOD

In this section we build upon the previously discussed concepts to distinguish as well as link the spatial and temporal factors in F-formation detection. First, we propose to use simultaneous speakers in an F-formation as an indicator of the existence of distinct conversation floors. Second, we relate the detected F-formations with different body clues in spatial aspect with SH and SF metrics. Finally, we propose a joint analysis method to combine the spatial relationship with temporal floor pattern on the detections.

## A. Analyzing temporal characteristic of an F-formation

Conversations are inextricably connected with the content of speech. However, a common concern with observing conversation groups by verbal information is the potential violation of privacy. How can we then investigate the existence of distinct conversations without high-quality audio (verbal data)? We propose the following insights:

1) Inferring schisms without audio data: The relationship between body movements such as gestures and aspects in speech such as emphasis, rhythm, and content has been long established in literature [34, 35, 36]. Some works have shown promising results in estimating the presence of voice activity from automated gestural analysis or accelerometer data [37, 38, 39, 40]. These approaches that refrain from using privacysensitive data suggest that it is feasible to automatically estimate speaking behavior, and in turn, speaking turns, without audio. Additionally, since the co-existence of two turn-taking systems is the most decisive characteristic of schisming, we argue that it is in turn reasonable to explore the inference of schisms without audio data, but through speaking turns.

![](images/cc1a3a5bfd4496a16fe5d26f4bbda080111b33da6c8e2bda33e2ffcd9c147b6a.jpg)  
Fig. 2. Illustration of gaps, within-overlaps, and between-overlaps for two speakers (S1 and S2) within the same floor. The scheme was originally proposed by Heldner and Edlund [41] and adopted by Levinson and Torreira in their analysis [42].

2) Linking schisming to floors and F-formations: Schisming, by its nature, influences floors as Edelsky defined as the “acknowledged what’s-going-on within a psychological time space”. This acknowledgment is essentially a focus, either on topic or some other functions. When schisming occurs, it changes this single-focused interaction into a multi-focused one, and each of the focused gathering is a new floor.

When thinking of the influence of schism to F-formations defined by Kendon, on the other hand, we find that it may not essentially change it. Kendon defines F-formations based on transactional segments, which come from lower-body orientations. If the participants moved their upper-bodies to form multiple floors, while the lower bodies remain configured such that an common o-space is still maintained, then they form multiple conversation floors within the same F-formation. This phenomenon is shown conceptually in figure 1.

Combining the discussions, we argue that it is feasible to detect distinct floors within an F-formation without audio. Specifically, we consider the following metric: for a time window w with duration d, we define a person to be a speaker if they speak for the entire period. For a group with multiple persons, the number of floors within it for w is then the number of simultaneous speakers, since we are inferring floors from speaking turns, and each speaker corresponds to a turn holder.

3) Choice ofspeaking window duration: In order to capture useful dynamics in a conversation about floors, we need to choose proper ds. As in Egbert’s study, short overlapping speech can exist within one floor. Thus, if the duration was too short, it may capture such back-channels of overlaps, and a floor thus detected is not a sensible one. If the duration was too long, the simultaneous speaker number may quickly drop to 0, as having just 2 people speaking for an extremely long period of time in a group is very unlikely. Therefore, setting an upper and lower bound for ds and vary them in the range might be a good choice. In this article, we vary d from 1 second to 20 seconds with an interval of 1 second.

Our choice of the duration comes from related studies. Heldner and Edlund studied gaps and overlaps in conversations, and discovered that on average 40% of the speaker transitions contain overlaps [41]. In Spoken Dutch Corpus they used, their histogram shows that the duration of these overlaps follows has a mean of 610 ms and median of 470 ms, all under one second. Levinson and Torreira differentiate between types of overlaps in a later research: 1) between overlaps, which refer to overlaps where the floor holder was changed, but without a silent gap between speakers; 2) within-overlaps, where the floor holder was unchanged, and overlapping speech occurred within a speaking turn [42]. These two types are illustrated in figure 2. For the duration, between overlaps had a modal duration of 96 ms, a median of 205 ms, a mean of 275 ms. On the other hand, within-overlaps exhibited an estimated modal duration of 350 ms, a median of 389 ms, a mean of 447 ms. Within the same study, they also found that only 3.8% of the time were simultaneous speaking from two speakers observable, by using Switchboard Corpus of English telephone conversations. This aligns with Sacks et al’s conclusion that “overwhelmingly, one party speaks at a time” [6].

These results suggest that although back-channels and overlapping speech is common, the most of them are rather short and do not indicate a change in floors. Therefore, we can choose the lower bound of window duration d such that it filters out those overlaps that do not come with a change in floors. For the upper bound, a reasonable value should be at least greater than the average turn duration of a speaker. As in [42], Levinson and Torreira report that contiguous speech delimited by a silent interval of at least 180 ms had a mean duration of 1680 ms and a median of 1227 ms. Following this, we leverage a 1 second minimal duration for analysis.

## B. Combining turn-taking and spatial relationship within Fformation detections

1) Using different sets offeaturesfor F-formation detection: Although Kendon has described F-formations based on lowerbody orientations, he also mentions that one may move their head or upper body parts to adapt to conversations without moving their lower bodies:

However, if p rotates his head so that a line projected from the midline of his face forms an angle of more than thirty degrees from the midline of his lower body, p may be said to be facing out of his transactional segment. It is not uncommon for someone to look out of his transactional segment in this way – for example when someone looks over his shoulder. Such head orientations are rarely continued for long, however. Any sustained looking in such a direction is usually associated with a reorientation of the lower body, so that the direction of the transactional segment again coincides with the direction in which the face is oriented. [5, p. 212]

Therefore, moving the head away from the transactional segment is not uncommon, although they typically don’t last for very long. Also, this aligns exactly with our previous discussion that one may move their upper body towards conversations while still remaining configured in an F-formations with the lower bodies. This naturally raises the following question: if there exists F-formation detections that leverage the head or body orientation [27, 43], can we detect this upper/lower body spatial distinction with multiple clues from the body? Specifically, with the same detection pipeline, can we observe the pattern that groups resulting from lower body features (LBF) correspond more to F-formations, while those from upper body features (UBF) correspond more to single conversation groups? This enables us to extract the features (position and orientation data) from different body parts and leverage them in existing pipelines. In this paper, we utilize features from 4 body parts: C = {head, shoulder, hip, foot}.

![](images/1ce4ef060518bd1005f6c5629bb6200dbcf7c48ca91ba1ea1467d1e5a61a1977.jpg)  
Fig. 3. Illustration of the experimental design. We start with a dataset with speaking status and F-formation annotations, as well as body spatial information of human bodies. Temporal analysis: we analyze simultaneous speakers within an F-formation to demonstrate there might be more than one speaker; Spatial analysis: We run selected F-formation detection pipelines and analyze spatial patterns between them; Joint analysis: we combine the spatial features with conversation floors to study the relationship between detection results and actual physically embodied floors.

2) Measuring spatial relationships: We argued earlier that simultaneous floors could exist within an F-formation. From the spatial perspective, we operationalize this as a subsumption relationship between groups. To be more specific, given a whole set of people $P ,$ , which represents every participant in a scene or frame, a group $g$ is a set of people. Running an F-formation detection baseline on one image with people $P$ would give a grouping $G = \{ g _ { 1 } , \ldots , g _ { k } \}$ , which is a partition of the entire set P into one or more smaller groups, where

$$
g _ { i } \cap g _ { j } = \emptyset , \quad \forall i \ne j , \quad \mathrm { a n d } \quad \bigcup _ { i = 1 } ^ { k } g _ { i } = P\tag{1}
$$

i.e. any two different subsets are disjoint and the union of all subsets is the whole set P. As previously stated, we are interested in the case where people in the same conversation floor rotate their heads towards each other, while their lower bodies remain configured in the same F-formation. This is embodied in the spatial relationship that groups detected from LBF could split into groups detected from UBF. In other words, given the grouping $G _ { U }$ from UBF and $G _ { L }$ from LBF,

$$
\exists g _ { L } \in G _ { L } , \exists g _ { U } ^ { ( 1 ) } , g _ { U } ^ { ( 2 ) } \in G _ { U } , \quad \mathrm { s . t . } \quad g _ { U } ^ { ( 1 ) } , g _ { U } ^ { ( 2 ) } \subset g _ { L _ { \ O } }\tag{2}
$$

i.e. there exists a group in $G _ { L }$ such that at least two groups in $G _ { U }$ are its proper subset.

However, for a dataset with F-formation annotations, there are several issues with the direct measurement of different detected groups:

• The detections are usually unstable. For many methods that utilize body positions and orientations to infer Fformations, neighboring frames can have different results, although they vary little in terms of the positions and orientations. Our analysis, on the other hand, needs temporal consistency to analyze the conversation floors. The annotations, on the other hand, provides a good anchor for this as they are stable and have a clear, tractable lifetime.

• The annotations may or may not correspond to true F-formations. Many datasets provide “F-formation” or “conversation group” annotations, but since they typically do not distinct one from the other, it is unknown whether the annotations are made by leveraging upper or lower body orientations. In fact, we have observed many instances where the annotation actually aligns more with the conversation groups by looking at upper body orientations. Therefore, anchoring the annotations as the ground truths of F-formations may lose the larger context.

With the above issues, we design the following analysis pipeline. We start from a dataset with F-formation annotations $\{ \bar { A } _ { i } \} _ { i = 1 } ^ { N _ { A } }$ . Each annotated F-formation is a group with its composition $A _ { i }$ and lifetime $\mathcal { T } _ { i } = \{ t _ { 1 } ^ { ( i ) } , \dots , t _ { T _ { i } } ^ { ( i ) } \}$ . For each frame t, we have the groupings detected using different feature sets $\left\{ G _ { \alpha } ^ { t } \right\} _ { t = t _ { 1 } ^ { ( i ) } } ^ { t _ { T _ { i } } ^ { ( i ) } }$ where $\alpha \in { \mathcal { C } }$ . Within each frame, we define the relative detection groups with respect to $A _ { i }$ as:

$$
{ \mathcal { R } } _ { \alpha } ( A _ { i } ) = \{ g | g \in G _ { \alpha } , g \cap A _ { i } \neq \emptyset \} .\tag{3}
$$

Since $A _ { i }$ may align more with F-formations or conversation floors, this allows us to have a full contextual picture of people potentially in the same F-formation, regardless of whether $A _ { i }$ is closer to either distinction.

With each pair of $\alpha , \beta \in { \mathcal { C } }$ and each frame detection $G _ { \alpha } .$ we compute the following asymmetric pairwise metrics:

Soft Homogeneity (SH). Inspired from the entropy-based homogeneity score [44] which evaluates the level of homogeneity over clustering, our proposed SH measures whether one group $g _ { \alpha }$ subsumes other groups in $G _ { \beta } ,$ , or the other way around (being subsumed in one group in $G _ { \beta } )$ . We start with the soft subsumption score $s _ { u }$ and soft partition score $s _ { p } ,$ and take SH as the maximum among them:

![](images/440a8fd19b4a2bbb420487c21a9876caf317b9dc9a9212b381d1f8a4b11726ed.jpg)  
Fig. 4. Illustration of spatial measures $S _ { H }$ and $S _ { F }$ . First, we identify the composition of annotation $A _ { i } ,$ , detection results $G _ { \alpha } , G _ { \beta } ,$ and the relative detection $\mathcal { R } _ { \alpha } ( A _ { i } )$ . For SH, we first calculate $s _ { u }$ and $s _ { p } ,$ then take the maximum to get $S _ { H }$ . For SF, it is 1 when there are at least 2 subsets of $g _ { \alpha }$ in $G _ { \beta }$ with at least 2 members, and 0 otherwise.

$$
s _ { u } ( g _ { \alpha } , G _ { \beta } ) = \operatorname* { m a x } _ { g _ { \beta } \in G _ { \beta } } | g _ { \alpha } \cap g _ { \beta } | / | g _ { \alpha } |\tag{4}
$$

$$
s _ { p } ( g _ { \alpha } , G _ { \beta } ) = \left| \bigcup g _ { \beta } \in G _ { \beta } : g _ { \beta } \subseteq g _ { \alpha } \right| / | g _ { \alpha } |\tag{5}
$$

$$
S _ { H } ( g _ { \alpha } , G _ { \beta } ) = \operatorname* { m a x } [ s _ { u } ( g _ { \alpha } , G _ { \beta } ) , s _ { p } ( g _ { \alpha } , G _ { \beta } ) ]\tag{6}
$$

For one specific group $g _ { \alpha } , \ s _ { u }$ shows the degree that it is subsumed by other groups in $G _ { \beta } \colon$ it is 1 if and only if it is completely subsumed by another group in $G _ { \beta }$ , and decreases if its members are distributed across different groups in $G _ { \beta } .$ $s _ { p }$ shows the degree that smaller groups in $G _ { \beta }$ covers it, and only take to 1 when there it is fully partitioned into smaller groups in $G _ { \beta }$ . Finally, $S _ { H }$ takes the larger number between $s _ { u }$ and $s _ { p } .$ . Intuitively, $S _ { H } ( g _ { \alpha } , G _ { \beta } )$ is high when $g _ { \alpha }$ has an identical group in $G _ { \beta }$ , or splits into multiple groups in $G _ { \beta }$ , or is subsumed by a larger group in $G _ { \beta }$ , and becomes low when members in $g _ { \alpha }$ spreads across multiple groups in $G _ { \beta }$ . When two detection results agree with each other, or a clean split pattern exists, the SH would be high. On the other hand, if the detections from α and $\beta$ are very irrelevant and produce random results, SH would be low.

Split Fraction (SF). Although SH accounts for whether the detections from clue α and β “agree” in the sense of group level splitting, it does not tell which way around. For example, SH could be high because that groups in $G _ { \alpha }$ splits into other groups in $G _ { \beta }$ , or inversely, or because that groups in $G _ { \alpha }$ are nearly identical to those in $G _ { \beta }$ . To address the feature in equation 2, we employ SF as follows.

Given annotation group $A _ { i }$ and frame t, we define the subsumption subset as

$$
\mathcal { U } ( g _ { \alpha } , G _ { \beta } ) : = \{ g _ { \beta } | g _ { \beta } \in G _ { \beta } , | g _ { \beta } | \geq 2 , g _ { \beta } \subsetneq g _ { \alpha } \} ,\tag{7}
$$

which measures when the outer group $g _ { \alpha }$ from body clue α contains at least two non-trivial sub-groups in the inner clue $\beta ^ { \bullet }$ detections. Based on it we compute the SF as

$$
S _ { F } ( g _ { \alpha } , G _ { \beta } ) = \Im \{ | \mathcal { U } ( g _ { \alpha } , G _ { \beta } ) | \ge 2 \}\tag{8}
$$

which counts whether there are at least two proper subsets of $g _ { \alpha }$ existing in $G _ { \beta }$ . This directly address if a splitting pattern of $g _ { \alpha }$ exists within $G _ { \beta }$ , and measures the direction of splitting that is ambiguous in SH. It also ignores identical grouping results, letting us focus on the non-trivial splitting patterns. Figure 4 demonstrates the $S _ { H }$ and $S _ { F }$ scores.

Aggregation. We compute the aggregated score for both SH and SF by averaging over relative detection groups and then over all frames and all annotation F-formations:

$$
\Theta ( \alpha , \beta ; A _ { i } , t ) = \frac { 1 } { | \mathscr { R } _ { \alpha } ( A _ { i } ) | } \sum _ { g _ { \alpha } \in \mathscr { R } _ { \alpha } ( A _ { i } ) } \Theta ( g _ { \alpha } , G _ { \beta } ^ { t } ) ,\tag{9}
$$

$$
\Theta ( \alpha , \beta ) = \sum _ { i = 1 } ^ { N _ { A } } \sum _ { t \in \mathcal { T } _ { i } } \Theta ( \alpha , \beta ; A _ { i } , t ) \Big / \sum _ { i = 1 } ^ { N _ { A } } | \mathcal { T } _ { i } | .
$$

where $\Theta \in \{ S _ { H } , S _ { F } \}$ represents either score.

(10)

3) Joint analysis with spatial-temporal characteristics: SH and SF measures whether a splitting pattern exists in the detection results from two body clues $\alpha , \beta .$ . However, it does directly link back to our observation that more than one floor exist within one F-formation. Concretely speaking, we want to know that if this splitting really corresponds to the Fformation-conversation group distinction illustrated in figure 1. To demonstrate this, we employ the joint analysis with spatialtemporal characteristics.

We continue to use contiguous speaking status within a window as an indication of floor occupation. Within each annotated F-formation window, we partition time into nonoverlapping tiles of duration $t _ { 0 } .$ . For each tile, we determine the speaker set S – the set of people whose speaking status equals 1 for at least $f _ { r }$ of their annotated frames in the tile. Every frame t inherits the $s$ of its enclosing tile. We set $t _ { 0 } = 1 \mathrm { s }$ as it is the shortest window we use for analyzing conversation floors and affords to analyze more F-formations which are short. We choose $f _ { r } = 0 . 9$ to rule out noise in the speaking status annotation.

For each annotated F-formation $A _ { i } ,$ we define frames t that contain a split event as follows:

$$
\Psi _ { 1 } ^ { ( i ) } ( \alpha , \beta ) = \{ t \in \mathcal { T } _ { i } | \exists g _ { \alpha } \in \mathcal { R } _ { \alpha } ( A _ { i } ) , S _ { F } ( g _ { \alpha } , G _ { \beta } ^ { t } ) = 1 \}\tag{11}
$$

and for each $t \in \Psi _ { 1 } ^ { ( i ) } ( \alpha , \beta )$ we compute

$$
M _ { 1 } ( g _ { \alpha } , G _ { \beta } ) = \mathbb { 1 } \{ | g _ { \alpha } \cap { \cal S } | \ge 2 \} ,\tag{12}
$$

where each S is computed based on the tile that the frame t belongs to. $M _ { 1 }$ indicates when a split event happens, whether the larger group has at least 2 speakers.

Built upon $M _ { 1 }$ , we further define all frames with $M _ { 1 } = 1 \colon$

$$
\Psi _ { 2 } ^ { ( i ) } ( \alpha , \beta ) = \{ t \in \Psi _ { 1 } ^ { ( i ) } ( \alpha , \beta ) | \exists g _ { \alpha } \in \mathcal { R } _ { \alpha } ( A _ { i } ) , M _ { 1 } ( g _ { \alpha } , G _ { \beta } ) = 1 \}
$$

and for each $t \in \Psi _ { m } ^ { ( i ) } ( \alpha , \beta )$ we compute metric $M _ { 2 }$

(13)

$$
M _ { 2 } ( g _ { \alpha } , G _ { \beta } ) = \frac { \vert \{ g _ { \beta } \vert g _ { \beta } \in \mathcal { U } ( g _ { \alpha } , G _ { \beta } ) , \vert g _ { \beta } \cap \mathcal { S } \vert = 1 \} \vert } { \vert \mathcal { U } ( g _ { \alpha } , G _ { \beta } ) \vert }\tag{14}
$$

which measures the fraction of detected sub-groups that are occupied by exactly one speaker – the ideal signature of one dedicated floor per spatial sub-group.

![](images/5c9437bf2dfbf12c7367175f0ccb2115675e18eddf6298b07df38895bf66a8d9.jpg)  
Fig. 5. Illustration of joint analysis pipeline. First we split the duration of an annotated F-formation into tiles of length $t _ { \mathrm { 0 } } .$ , then compute speaker within each tile. For each frame in each tile where a split event exists, we compute $M _ { 1 }$ and M . M calculates whether the larger group contain at least 2 speakers, and M<sub>2</sub> calculates the proportion of subgroups that contain exactly one speaker. Together, they measure how much the detected split events align with the characteristic that multiple floors exist within one F-formation.

Aggregation. Since the tiles are of equal lengths and nonoverlapping, aggregating within tiles first then across tiles is equivalent with aggregating directly over frames. Therefore, we first leverage equation 9 to compute per frame metric for $\Theta \in \{ M _ { 1 } , M _ { 2 } \}$ and use

$$
M _ { j } ( \alpha , \beta ) = \sum _ { i = 1 } ^ { N _ { A } } \sum _ { \substack { t \in \Psi _ { j } ^ { ( i ) } ( \alpha , \beta ) } } \Theta ( \alpha , \beta ; A _ { i } , t ) \Big / \sum _ { i = 1 } ^ { N _ { A } } \Big | \Psi _ { j } ^ { ( i ) } ( \alpha , \beta ) \Big | .\tag{15}
$$

to aggregate over all formation, weighted by valid frame numbers where $j \in \{ 1 , 2 \}$

## V. EXPERIMENTS

## A. Dataset and baseline

Our desiderata of the dataset are in two folds. First, it needs to have the body clues from different parts to allow for cluewise F-formation detection. Second, having annotations for Fformations and speaking status is essential as they are needed for anchoring spatial analysis and inferring conversation floors respectively. We therefore choose the ConfLab dataset [45] which contains videos of free-mingling scenes. It has rich scences recorded at high resolution which allows us to extract positions $\mathbf { p } _ { i }$ and orientations $\theta _ { i }$ for each person in each frame. Detailed process of the extraction is provided in the supplementary. Also, it has F-formation and speaking status annotation to facilitate our designed analysis of inferring floors without explicit audio data.

We choose the GTCG [27] and GCFF [43] baselines. The former models the transactional segment by sampling points and calculates pairwise affinity matrix by measuring the distances between distributions. The latter votes for centers of the o-spaces and optimize by assigning and updating people to these centers. Both pipelines provide an explicit and explainable description of the spatial structure of an F-formation by either modeling transactional segments or formation centers, and they provide grouping results $G _ { \alpha }$ with respect to features constructed from feature α for each frame by only taking positions and orientations for each person.

For both pipelines, the parameters are chosen empirically, and later verified that they are close to optimal choice where the detection results $( F _ { 1 } )$ reaches maximum by averaging all 4 feature set detection results. Specifically, for GTCG we use frustum= 110cm aperture= 160 degrees, and for GCFF we use stride= 40cm and MDL= 6000.

## B. Turn-Taking Analysis Results

1) There is indication that multiple simultaneous speakers exist within annotations of F-formations for ConfLab: We apply the sliding window varying from 1 to 20 seconds on annotations of ConfLab. We take each continuous sequence of identical annotation as the lifetime of an F-formation. The simultaneous speaker over window length plots are shown in figure 6. We can observe that simultaneous speaker number decreases as window duration increases, which matches with our expectation that it becomes rarer for cases where people simultaneously speak for longer. For groups of size 4, the simultaneous speaker number reaches around 2 when window size is 1 second, and keeps decreasing but stays above 1 until window duration reaches 6 seconds. For groups of size 5, however, the simultaneous speaker number is generally below 1. For groups of size 6 and 7, we also observe more than one simultaneous speakers in window lengths of below 3 and 10 seconds respectively. The 0 variance in the last 2 plots indicate that there are only one instance for both cases.

We can infer that there is evidence that more than one simultaneous speakers exist within the annotated F-formations in ConfLab. Although for group size of 5 this number is below 1, it could be influenced by the scarcity of samples, and we propose that more cases in group sizes 4, 6, and 7 are persuasive enough to draw our conclusion.

![](images/4998d5e2ed0180382242a36423b15c6212e282c4143e1c655f5353c09300b4bc.jpg)

![](images/f4152af03a2f4e7c41c36897a2b7ce540860257376c9777fe948463283fae6c7.jpg)

![](images/49ee355244eab40a8a38caf44f65db6a628d4fa6dd56081f002c9fdb8b8bc276.jpg)

![](images/e5d763a85d42a0a24151b1b6cc07c0e0cd26595c13cbdb05ca05b40b99e8f461.jpg)  
Fig. 6. Simultaneous speakers per cardinality of detection using annotations. Each bar is average speaker number with standard deviation, with respect to a specific window duration. For groups with cardinality 4, more than 1 speaker is observed for 1-5 seconds of window. For groups with cardinality 5, no average speaker number is above 1. For cardinality 6-7, although observed with more than 1 speaker for windows duration up to 2 and 9 seconds respectively the number of samples also drops substantially, resulting in no standard deviation.

2) Group cardinality has a significant impact on the turn durations of simultaneous speakers: Previously, we discussed that it is difficult for a single conversation to sustain over four participants. In this experiment, we explore this effect and study the following question: for a given speaking turn duration $d ,$ do we observe a decrease in the maximum number of conversation floors observed over an F-formation’s lifetime with an increase in the cardinality of an F-formation? Qualitatively, this corresponds to the steepness of fall-off of the curves in 6. To account for the influence of group cardinality and window duration, we model the simultaneous speaker y<sub>i</sub> as instances of independent Poisson variables: $Y _ { i } \sim \log ( \mu _ { i } )$ and $\mu _ { i }$ is modeled by the following formula:

$$
\log ( \mu _ { i } ) = \beta _ { 0 } + \beta _ { 1 } d _ { i } + \beta _ { 2 } c _ { i } + \beta _ { 3 } d _ { i } c _ { i }\tag{16}
$$

where $d _ { i }$ denotes the duration of the window and $c _ { i }$ denotes the group cardinality. The $\beta _ { i } \mathrm { s }$ are regression coefficients. We treat each possible observation of a window as a data point. For example, an F-formation of cardinality 4 with the lifetime of 20 seconds would result in 20 windows of duration of 1 second. This would add 20 data points with $c _ { i } = 4$ and $d _ { i } = 1 . \ y _ { i } \mathbf { s }$ for these data would correspond to the number of simultaneous speakers of the respective windows.

We fit the data using a Generalized Linear Model (GLM) by the Python statsmodel <sup>1</sup> package. The results are shown in table I. We observe that the turn duration and turnduration:cardinality factors have an impact on the simultaneous speakers at significance level of 0.01, while cardinality alone have an impact with significance level of 0.05.

Although this test tells us that all three factors influence the simultaneous speaker numbers, we still need a post-hoc analysis to confirm each possible pair of cardinality makes difference among them. Therefore, we apply multiple GLMs using the same formula and present the results in table II. In the last column, we observe specifically that cardinality and its interaction between turn duration are significant at level 0.001 for {4/5} pair.

We acknowledge that a possible limitation of this analysis is the difference and scarcity of the number of available Fformations under some circumstances. We present the number of F-formations per cardinality with respect to speaking turn duration in figure 7.

TABLE I  
GENERALIZED MODEL REGRESSION RESULTS
<table><tr><td></td><td>Coeff(β)</td><td>Std Err</td><td>z</td><td> $\overline { { P > z } }$ </td></tr><tr><td>Intercept</td><td>-0.7164</td><td>0.564</td><td>-1.270</td><td>0.204</td></tr><tr><td>Turn-duration</td><td>0.2424</td><td>0.076</td><td>3.173</td><td>0.002</td></tr><tr><td>Cardinality</td><td>0.2726</td><td>0.121</td><td>2.244</td><td>0.025</td></tr><tr><td>Turn-duration:Cardinality</td><td>-0.0754</td><td>0.018</td><td>-4.298</td><td>0.000</td></tr></table>

TABLE II

NOMINAL P-VALUES FOR SIX POST-HOC GLM REGRESSION COMPARISONS
<table><tr><td>Cardinality pairs</td><td>Intercept(β0)</td><td>d(β1)</td><td>c(β2)</td><td>d:c(β3)</td></tr><tr><td>4-5</td><td>0.831</td><td>0.0005</td><td>0.879</td><td>0.0003*</td></tr><tr><td>4-6</td><td>0.426</td><td>0.011</td><td>0.099</td><td>0.002</td></tr><tr><td>4-7</td><td>0.295</td><td>0.592</td><td>0.017</td><td>0.086</td></tr><tr><td>5-6</td><td>0.463</td><td>0.042</td><td>0.285</td><td>0.109</td></tr><tr><td>5-7</td><td>0.387</td><td>0.001</td><td>0.113</td><td>0.006</td></tr><tr><td>6-7</td><td>0.918</td><td>0.048</td><td>0.600</td><td>0.104</td></tr></table>

## C. Spatial Analysis Results

1) Detected groups have high agreement over different clues: The SH and SF scores for both pipelines are shown in figure 8 and our observation is similar for both pipelines: $S _ { H } ( \alpha , \beta )$ is close to 1 for every pair of clues. This indicates that running both pipelines would give us high agreement over the splitting pattern of subgroups: either one detection subsume the other, or they are nearly identical. It reveals that although the detection themselves are noisy across frames, they produce high agreement over groupings of people. Suppose with a frame of person set $P = \{ 1 , 2 , 3 , 4 , 5 \}$ as in figure 3, the differences in detection regarding to different body clues are more of a higher level of organization of smaller subgroups: all clues would mostly agree that [1, 2, 3], and [4, 5] are within the same group, and only vary on whether to merge the two groups together to form a larger, coarser group, instead of a random restructuring on individual level. A complete shuffle, for example, one clue produces [1, 2, 3], [4, 5] while the other results in [1, 3, 5], [2, 4], is very unlikely.

![](images/0585bf7333720e27ba4c914529df673271c780230d8dcc46c6224cb0c56d588d.jpg)  
Fig. 7. Number of F-formations at different speaking turn durations.

![](images/2f92062e27e4d0aeb5e38a22aa008f6bb23e132b2e08479ac8b648d2eb19add4.jpg)

![](images/2d6e963aba8c9383e3e18e13669d7ae6da573433b4f222273da03a714867e73f.jpg)  
(a)

![](images/212c8ac45b634b5ea119238e01ad95b92a0f05889b2e5922f122adff764e95e8.jpg)

![](images/39b8abe38f8d93e74bc492ddd9044b1c0221cdcab5665713c89bf4d0151c0248.jpg)  
(b)  
Fig. 8. Spatial analysis results for both pipelines. All figures are interpreted from row to column. For instance, a cell located at shoulder row and hip row of SH is interpreted as $S _ { H } \rvert$ (shoulder, hip). 8a: SH and SF for GTCG pipeline. SH is at least 0.96 which shows high agreement over groupinf patterns. SF shows that it takes highest for the (shoulder, head) pair and is substantially higher when β takes head features. 8b: SH and SF for GCFF pipeline. SH score is 0.99 for every pair of different $( \alpha , \beta )$ , showing high agreement over groupings. SF shows clear asymmetry: $S _ { F } ( \alpha , \beta )$ is much larger than the opposite when α is physically lower than $\beta$ on the body.

2) LBF detected results split into UBF groups, especially thosefrom headfeatures, more than inversely: The SF scores tell us the direction of the splitting, as illustrated in section IV. In figure 8a and 8b, we observe that SF show clear asymmetrical patterns. Specifically, for GTCG pipeline, $S _ { F } ( \alpha , \beta )$ takes the largest value when $( \alpha , \beta ) =$ (shoulder, head). Also, SF is larger when $\beta$ takes head features. This shows that it happens more frequently when groups detected from other features split into groups detected from head features, indicating that more frequently people formulate smaller, finer-grained groups with positions and orientations of the head, while other parts remain in the larger group, especially with their shoulders. For GCFF pipeline, the observation is even stronger, as the SF is significantly larger in the bottom-left corner than the top-right corner. This means $S _ { F } ( \alpha , \beta )$ is larger when α belongs to LBF like hip or foot, and $\beta$ belongs to UBF like head or shoulder. $S _ { F } ( \alpha , \beta )$ takes the largest value for (foot, head) pair, and it is more than 2 times larger than its symmetrical pair of (head, foot). This means that statistically, groups detected from foot features split into groups resulting from head features more than 2 times than the inverse case. Also, we note that for each pair of $( \alpha , \beta ) , S _ { F } ( \alpha , \beta ) > S _ { F } ( \beta , \alpha )$ if α is physically lower than $\beta ,$ aligning with the the observed pattern. We also notice that the $\mathrm { S F }$ is much smaller for GTCG than GCFF, as the bar scale for the former is $1 0 ^ { - 3 }$ compared to the $1 0 ^ { - 2 }$ for the latter. This indicates that the splitting over two proper groups happens much sparser in GTCG pipeline than GCFF.

Nevertheless, SF from UBF to LBF is still non-zero for both pipelines. This means at least in some cases, groups from UBF would also split into groups from LBF. This may be due to inaccurate keypoint extraction, internal feature of the pipeline itself, or the fact that sometimes people do orient their heads towards each other and formulate larger groups than their lower body configurations. However, our analysis reveals that this case is much rarer than the opposite, and in the following joint analysis we illustrate that in these cases it is less typical for the hypothesis that one F-formation could contain more than one floor.

## D. spatial-temporal joint analysis

In this section we present results for the level of alignment between the subdivided groups and true conversation floors. We show the results for $\bar { M _ { 1 } }$ and $\bar { M _ { 2 } }$ in figure 9a and 9b for the results of GTCG and GCFF respectively.

1) Larger group containing more than one floor is generally more common when LBF groups split into UBF groups: The $M _ { 1 }$ scores are showing similar trending for both pipelines. $M _ { 1 } ( \alpha , \beta )$ is larger when α belongs to LBF and $\beta$ belongs to UBF, except for the case of (shoulder, foot) and (hip,shoulder) in GTCG pipeline. This indicates that when groups from LBF do split into groups from UBF, the larger group contains more than one floor more frequently. Inversely, when UBF groups split into multiple LBF groups, this is less frequent. This indicates that the pattern captured in LBf groups splitting into UBF groups align more with the hypothesis that the larger F-formation contain more than one floor. Specifically for GTCG, the $M _ { 1 }$ score is generally lower than GCFF. There are no instances for the (hip,shoulder) pair, and (shoulder, foot) pair has the highest value. Nevertheless, all other instances of (LBF, UBF) pair are higher than the opposite. For GCFF, the pattern is cleaner: $M _ { 1 } ( \alpha , \beta )$ is approximately 0.12 higher when computed from LBF to UBF than the opposite direction. The $M _ { 1 }$ score within LBF or UBF, is placed in between and in consistency with the observation.

![](images/45c1eeac69c67e472d4aef456c8f23dbe3a3c3c44da882bf5d6286952377532d.jpg)

![](images/22f0cad44bdf90cc421fa862930c1eda2320058cb2158da1900db48b483635e8.jpg)  
(a)

![](images/9b3e9094ff5e12da13d4546e6aaeda35ea7438772659cb48db82530a97381827.jpg)

![](images/228404030d8a71a74def058128dede779966e0201ab38867f85ffebe6fe4db1b.jpg)  
(b)  
Fig. 9. $M _ { 1 }$ and $M _ { 2 }$ for each clue pair for both pipelines. Within the labels, $\alpha  \beta$ refers to $M ( \alpha , \beta ) . M _ { 1 }$ calculates how often there are at least 2 speakers in the larger group when a split event happens, and $M _ { 2 }$ calculates how frequently each subgroup contain exactly one speaker when $M _ { 1 }$ holds. The data bars are grouped in 4 sets: (LBF, UBF), (UBF, LBF), within UBF, and within LBF. 9a: $M _ { 1 }$ and $\boldsymbol { { \bar { M _ { 2 } } } }$ for GTCG. 9b: $M _ { 1 }$ and $M _ { 2 }$ for GCFF.

2) Relationship between smaller groups and conversation floors is pipeline specific: The $M _ { 2 }$ scores evaluates that when such a splitting happens, the proportion of subgroups which contain exactly one speaker. For GTCG, we observe that $M _ { 2 }$ is higher for (UBF,LBF) pairs, and two pairs in (LBF, UBF) are zero, indicating that there is no such case where the subgroups contain exactly one speaker. Although this doesn’t align with our assumptions that groups from UBF features correspond more to conversation floors, it could be influenced by the fact that these splits are sparse anyway in GTCG pipeline, as previously mentioned. GCFF pipeline, however, aligns with our hypotheses much more. We observe that when it is computed from LBf groups splitting into UBF groups, $M _ { 2 }$ is close to 1, indicating that whenever the larger group contain more than one speaker, almost all subgroups contain one speaker each. This is a strong indication that when the splitting pattern from LBF groups to UBF groups is observed, it is almost certain that each of the UBF groups contain exactly one floor. On the other hand, when UBF groups split into LBF groups and the larger group has more than one speaker, it is less certain that the LBF groups contain one speaker each, although the probability is still high. Combined together with the $M _ { 1 }$ measurement, it demonstrates that when such splitting pattern occurs in GCFF, we observe much more instances for the case of LBF splitting into UBF groups that it is actually corresponding to the fact that one F-formation containing more than one floor. When detected UBF groups split into LBF groups, on the other hand, is less likely observed, and less likely resulting from such subsumption relationship.

## VI. CONCLUSION

In the above experiments, we have studied the difference of the F-formation detection results from different feature sets. Summing up the observations, we conclude that

• An F-formation may contain more than one conversation floor in both theoretical and realistic aspects;

• For existing pipelines for detecting F-formations, using different clues from body parts makes a difference. Some resulting groups using LBF split into two or more in groups resulted from UBF and this happens more frequently than the opposite splitting, especially when groups from other features splits into groups from head features. This indicates that head positions and orientations may carry more fine-grained conversational involvement information within a larger interaction context.

• When LBF groups split into UBF groups, the correspondence between larger/smaller groups is pipeline specific. In GCFF pipeline, we observe clear alignment that LBF groups correspond to F-formations and UBF groups correspond to single conversation floors. This indicates that certain methods could be promising in detecting and translating conversational involvement at floor level.

## VII. DISCUSSION AND LIMITATION

Discussion. The results in this article show that current operationalization of FCGs is not enough for capturing single floors. We propose the following points for future interests:

• Develop a multi-tier theory to describe involvement in FCGs. Operationalizing FCGs as F-formations are insufficient to encapsulate single conversation floors. However, it does not mean that F-formations does not carry any socially meaningful information regarding involvement. By defining it with the lower-body orientations, it still suggests some structure in which common activities take place - or common interest/attention in cognitive perspective, but this structure may contain multiple conversation floors. Therefore, we propose that a multi-tier theory may be of interest to future studies, which could describe different categories of interactional involvement on different levels, and F-formations/conversation groups correspond to one particular level/tier.

• A more fine-grained model to capture VFoA of people. With the advance in skeleton annotation in multiperson scenes, we believe that VFoA is not just about the direction of eye gaze or body pose, but a more complicated structure which integrates features from all around the body. Using a single orientation may overlook certain aspects in attention, and describing such attention mechanism in multi-dimensional way may pose further discoveries regarding involvement in an interaction.

Limitation. We used an approximation of using binary speaking status to account for conversation floors. Although this is conceptually feasible as in our arguments, it may not afford identically towards the analysis of floors as real speech data. We made this design choice to be privacy preserving, but future work may reveal different patterns for floors if privacy is not an issue. Also, we leverage keypoint data to infer positions and orientations of people (see supplemental material for details), which may not be as accurate as datasets with full mesh capturing and allows for direct feature extraction.

Our conclusion should not be interpreted as exact correspondence between body feature and F-formation/floor distinction. Rather, we point out the statistical difference in the group patterns that result from different features. The exact profile of correspondence should be figured out by future works.

## ACKNOWLEDGMENTS

We thank Gara Dorta Perez for explorations of feature extraction process in the ConfLab dataset. Plotting of figure 3 and 5 as well as this paper’s code implementation <sup>2</sup> is done with AI assistance.

## REFERENCES

[1] Dan Bohus and Eric Horvitz. “Learning to predict engagement with a spoken dialog system in openworld settings”. In: Proceedings of the SIGDIAL 2009 Conference. 2009, pp. 244–252.

[2] Deborah A Coker and Judeek Burgoon. “The nature of conversational involvement and nonverbal encoding patterns”. In: Human Communication Research 13.4 (1987), pp. 463–494.

[3] Christopher Peters et al. “A model of attention and interest using gaze behavior”. In: International Workshop on Intelligent Virtual Agents. Springer. 2005, pp. 229–240.

[4] Carole Edelsky. “Who’s got the floor?” In: Language in society 10.3 (1981), pp. 383–421.

[5] Adam Kendon. Conducting interaction: Patterns of behavior in focused encounters. Vol. 7. CUP Archive, 1990.

[6] Harvey Sacks, Emanuel A Schegloff, and Gail Jefferson. “A simplest systematics for the organization of turn-taking for conversation”. In: language 50.4 (1974), pp. 696–735.

[7] Charles Goodwin. “Forgetfulness as an interactive resource”. In: Social psychology quarterly (1987), pp. 115–130.

[8] Maria M Egbert. “Schisming: The collaborative transformation from a single conversation to multiple conversations”. In: Research on Language and Social Interaction 30.1 (1997), pp. 1–51.

[9] Harvey Sacks. “On the analyzability of stories by children”. In: Directions in Sociolinguistics: The Ethnography of Communication. Ed. by John J. Gumperz and Dell Hymes. New York: Rinehart & Winston, 1972, pp. 325–345.

[10] David McNeill et al. “Mind merging”. In: Expressing Oneself/Expressing One’s Self. Psychology Press, 2009, pp. 143–164.

[11] Reiko Hayashi. “Floor structure of English and Japanese conversation”. In: Journal of Pragmatics 16.1 (1991), pp. 1–30.

[12] Erving Goffman. Behavior in public places. Simon and Schuster, 2008.

[13] Dan Bohus and Eric Horvitz. “Managing human-robot engagement with forecasts and... um... hesitations”. In: Proceedings of the 16th international conference on multimodal interaction. 2014, pp. 2–9.

[14] Alex Pentland, Anmol Madan, and Jon Gips. “Perception of social interest”. In: Proc. IEEE Int. Conf. on Computer Vision, Workshop on Modeling People and Human Interaction (ICCV-PHI). Citeseer. 2005.

[15] Hayley Hung and Daniel Gatica-Perez. “Estimating cohesion in small groups using audio-visual nonverbal behavior”. In: IEEE Transactions on Multimedia 12.6 (2010), pp. 563–575.

[16] Catharine Oertel et al. “Deciphering the silent participant: On the use of audio-visual cues for the classification of listener categories in group discussions”. In: Proceedings of the 2015 ACM on International Conference on Multimodal Interaction. 2015, pp. 107–114.

[17] Catharine Oertel and Giampiero Salvi. “A gaze-based method for relating group involvement to individual engagement in multimodal multiparty dialogue”. In: Proceedings of the 15th ACM on International conference on multimodal interaction. 2013, pp. 99–106.

[18] Lu Zhang and Hayley Hung. “Beyond f-formations: Determining social involvement in free standing conversing groups from static images”. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 2016, pp. 1086–1095.

[19] Lu Zhang and Hayley Hung. “On social involvement in mingling scenarios: Detecting associates of f-formations in still images”. In: IEEE Transactions on Affective Computing 12.1 (2018), pp. 165–176.

[20] Marco Cristani et al. “Social interaction discovery by statistical analysis of f-formations.” In: BMVC. Vol. 2. 4. 2011, pp. 10–5244.

[21] Francesco Setti et al. “Multi-scale F-formation discovery for group detection”. In: 2013 IEEE International Conference on Image Processing. IEEE. 2013, pp. 3547–3551.

[22] Hayley Hung and Ben Krose. “Detecting f-formations¨ as dominant sets”. In: Proceedings of the 13th international conference on multimodal interfaces. 2011, pp. 231–238.

[23] Francesco Setti et al. “F-formation detection: Individuating free-standing conversational groups in images”. In: PloS one 10.5 (2015), e0123783.

[24] Khai N Tran et al. “Activity analysis in crowded environments using social cues for group discovery and human interaction modeling”. In: Pattern Recognition Letters 44 (2014), pp. 49–57.

[25] Xavier Alameda-Pineda et al. “Analyzing free-standing conversational groups: A multimodal approach”. In: Proceedings of the 23rd ACM international conference on Multimedia. 2015, pp. 5–14.

[26] Elisa Ricci et al. “Uncovering interactions and interactors: Joint estimation of head, body orientation and fformations from surveillance videos”. In: Proceedings of the IEEE International Conference on Computer Vision. 2015, pp. 4660–4668.

[27] Sebastiano Vascon et al. “Detecting conversational groups in images and sequences: A robust gametheoretic approach”. In: Computer Vision and Image Understanding 143 (2016), pp. 11–24.

[28] Loris Bazzani et al. “Social interactions by visual focus of attention in a three-dimensional environment”. In: Expert Systems 30.2 (2013), pp. 115–127.

[29] Marynel Vazquez, Aaron Steinfeld, and Scott E Hud-´ son. “Parallel detection of conversational groups of freestanding people and tracking of their lower-body orientation”. In: 2015 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE. 2015, pp. 3010–3017.

[30] Kaname Yokoyama, Chihiro Nakatani, and Norimichi Ukita. Dynamic Group Detection using VLM-augmented Temporal Groupness Graph. arXiv:2509.04758 [cs]. Sept. 2025. DOI: 10 . 48550 / arXiv.2509.04758. URL: http://arxiv.org/abs/2509.04758 (visited on 09/14/2025).

[31] Robin IM Dunbar, Neill DC Duncan, and Daniel Nettle. “Size and structure of freely forming conversational groups”. In: Human nature 6 (1995), pp. 67–78.

[32] Nelson Cowan. “The magical mystery four: How is working memory capacity limited, and why?” In: Current directions in psychological science 19.1 (2010), pp. 51–57.

[33] Jaimie Arona Krems and Jason Wilkes. “Why are conversations limited to about four people? A theoretical exploration of the conversation size constraint”. In: Evolution and Human Behavior 40.2 (2019), pp. 140–147.

[34] Peter Bull and Gerry Connelly. “Body movement and emphasis in speech”. In: Journal of nonverbal behavior 9.3 (1985), pp. 169–187.

[35] Allen T Dittmann and Lynn G Llewellyn. “Body movement and speech rhythm in social conversation.” In: Journal of personality and social psychology 11.2 (1969), p. 98.

[36] David McNeill. Language and gesture. Vol. 2. Cambridge University Press, 2000.

[37] Hayley Hung and Sileye O Ba. “Speech/Non-Speech Detection in Meetings from Automatically Extracted low Resolution Visual Features.” In: ICASSP. 2010, pp. 830–833.

[38] Ekin Gedik and Hayley Hung. “Speaking status detection from body movements using transductive parameter transfer”. In: Proceedings of the 2016 ACM international joint conference on pervasive and ubiquitous computing: adjunct. 2016, pp. 69–72.

[39] Ekin Gedik and Hayley Hung. “Personalised models for speech detection from body movements using transductive parameter transfer”. In: Personal and Ubiquitous Computing 21 (2017), pp. 723–737.

[40] Ekin Gedik and Hayley Hung. “Detecting conversing groups using social dynamics from wearable acceleration: Group size awareness”. In: Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 2.4 (2018), pp. 1–24.

[41] Mattias Heldner and Jens Edlund. “Pauses, gaps and overlaps in conversations”. In: Journal of Phonetics 38.4 (2010), pp. 555–568.

[42] Stephen C Levinson and Francisco Torreira. “Timing in turn-taking and its implications for processing models of language”. In: Frontiers in psychology 6 (2015), p. 731.

[43] Francesco Setti et al. “F-formation detection: Individuating free-standing conversational groups in images”. In: PloS one 10.5 (2015), e0123783.

[44] Andrew Rosenberg and Julia Hirschberg. “V-measure: A conditional entropy-based external cluster evaluation measure”. In: Proceedings of the 2007 joint conference on empirical methods in natural language processing and computational natural language learning (EMNLP-CoNLL). 2007, pp. 410–420.

[45] Chirag Raman et al. “ConfLab: A Data Collection Concept, Dataset, and Benchmark for Machine Analysis of Free-Standing Social Interactions in the Wild”. In: Advances in Neural Information Processing Systems 35 (2022), pp. 23701–23715.

[46] Nicolas Carion et al. “Sam 3: Segment anything with concepts”. In: arXiv preprint arXiv:2511.16719 (2025).

[47] Yufei Xu et al. “Vitpose: Simple vision transformer baselines for human pose estimation”. In: Advances in neural information processing systems 35 (2022), pp. 38571–38584.

## VIII. BIOGRAPHY SECTION

Zonghuan Li received BEng degree in Electrical Engineering from Tsinghua University in China, 2016, and master’s degree in Electrical Engineering from University of California, San Diego in United States, 2023. He is currently a PhD student under the Socially Perceptive Computing Lab, TU Delft, Netherlands since 2024. His research interests include conversational involvement analysis with multimodal signals, multimodal large language model applications and evaluation.

Chirag Raman received his master’s degree in entertainment technology from Carnegie Mellon University, USA in 2013 and the Ph.D in Computer Science from TU Delft in 2023. He is currently an Assistant Professor at TU Delft where he leads the Tapri Lab within the Department of Intelligent Systems. His research interests include gnerative modeling, socially intelligent embodied agents, with insights from cognitive science and social psychology.

Hayley Hung (Member, IEEE) received the PhD degree in computer vision from the Queen Mary University of London in 2007. She is currently an Associate Professor with the Socially Perceptive Computing Lab, TU Delft, The Netherlands, where she works since 2013. She works towards finding how automated systems best serve people with social awareness. Her research interests include devising multimodel models to automatically interpret faceto-face human social behavior.

![](images/7bfac8c460f50bad30af01c4943d2251371ab08a9452e94222972fcd3737c174.jpg)  
Fig. 10. Illustration of the position and orientation extraction pipeline. Left: we construct 2D keypoints on a frame in COCO format; Middle: we project the keypoint into 3D space by assuming the height of each keypoint; Right: we construct for sets of features (positions and orientations) to leverage in GTCG/GCFF pipelines.

APPENDIX: FEATURE EXTRACTION OF CONFLAB DATASET

Here we introduce the process to obtain positions and orientations from different body parts from the ConfLab [45] dataset. Specifically, for each person in a frame and for each body feature $\alpha \in \mathcal { C } \mathrm { ~ = ~ }$ {head,shoulder,hip,foot}, we extract positions $\mathbf { p } _ { \alpha }$ and orientations $\theta _ { \alpha }$ , which are then processed by GTCG and GCFF pipelines. This comprises three steps.

## A. Extracting 2D keypoints

To infer positions and orientations, we use keypoints as an anchor. ConfLab provides video recording from top-view cameras at 60 fps. We first use SAM3 [46] model to obtain a mask for each person, and then use ViTPose [47] to obtain 2D keypoints for them, by providing the convex hull of each mask as the bounding box. This results in 2D keypoints in COCO format <sup>3</sup> for each person and each frame. It contains head, pose, and shoulder/hip/ankle/toe keypoints on left and right sides.

The 2D keypoints are denoted by $\mathbf { x } \in \mathbb { R } ^ { 2 }$ which lies in the pixel domain. In the following context, subscripts $h ,$ n s, i, a, t stands for head, nose, shoulder, hip, ankle, and toe respectively. We also use l, r subscripts to denote left/right keypoint, which applies to all keypoints other than head and nose. For instance, $\mathbf { x } _ { l , s }$ represents pixel coordinates of the left shoulder keypoint.

## B. 3D Projection of keypoints

2D keypoints on pixel domain is insufficient for GTCG and GCFF pipelines, as these methods assume positions and orientations to be on a plane in 3D space, instead of in 2D pixel space. Therefore, we transfer the keypoints to 3D by leveraging intrinsic matrix K and extrinsic matrix $C$ provided by the ConfLab dataset. We do this by assuming the heights of each keypoint, and calculate its space coordinates by taking the the intersection of the ray represented by the pixel keypoint and the horizontal plane at the height level.

Specifically, since we need head, shoulder, hip, ankle, and toe keypoints, we assume that the height for each keypoint is $h _ { j } = \gamma _ { j } h _ { }$ , where $j$ represents the feature and $j \in \mathcal { K } =$ {head,nose,shoulder,hip,ankle,toe}. h is the assumed common height of a person. $\gamma$ is a height coefficient that indicates the ratio between the respective keypoint and the total body height. In practice we use $h = 1 . 7 \mathrm { m }$ and we show the assumed γ in table III. This process is shown in the left part of figure 10.

After this step we obtain keypoint coordinates in 3D space $\mathbf { k } _ { j }$ where $j \in \mathcal { K }$

## C. Body-part specific feature construction

After transferring the keypoints into 3D, we illustrate how the the input features are constructed. We need to note here that although we have transferred into the 3D space and the coordinates are in $\mathbb { R } ^ { 3 }$ , we omit the third dimension and limit the discussion to 2D in this context, i.e. on the plane that is parallel to the ground, and the third coordinate remain the identical height as in previous step. Since both GTCG and GCFF pipelines take only 2D inputs, this is unharmful. Therefore, all ps and ks in the following context are in $\mathbb { R } ^ { 2 }$

From the previous step we obtain keypoints in space as $\mathbf { k } _ { j } \in$ $\mathbb { R } ^ { 2 }$ . We now construct a set of positions p and orientations $\theta$ as follows:

$$
\begin{array} { r l } & { \mathrm { { p } } _ { h } = \mathbf { k } _ { h } ; } \\ & { \mathrm { { p } } _ { s } = \left( \mathbf { k } _ { i , s } + \mathbf { k } _ { r , s } \right) / 2 ; } \\ & { \mathrm { { p } } _ { i } = \left( \mathbf { k } _ { i , t _ { i } } + \mathbf { k } _ { r , i } \right) / 2 ; } \\ & { \mathrm { { p } } _ { i } = \left( \mathbf { k } _ { i , t _ { i } } + \mathbf { k } _ { r , i } \right) / 2 ; } \\ & { \mathrm { { p } } _ { f } = \left( \mathbf { k } _ { i , \alpha _ { s } } + \mathbf { k } _ { r , i } + \mathbf { k } _ { t , t } + \mathbf { k } _ { r , t } \right) / 4 ; } \\ & { \begin{array} { r l } { \theta _ { h } ^ { \prime } = \mathrm { a r g } ( \mathbf { k } _ { n } - \mathbf { k } _ { h } ) ; } \\ { \theta _ { s } ^ { \prime } = \mathrm { a r g } ( \mathbf { R } _ { z } \left( \mathbf { k } _ { r , s } - \mathbf { k } _ { I , s } \right) ) ; } \\ { \theta _ { l } ^ { \prime } = \mathrm { a r g } ( \mathbf { R } _ { z } \left( \mathbf { k } _ { r , t _ { i } } - \mathbf { k } _ { I , s } \right) ) ; } \\ { \theta _ { f } ^ { \prime } = \left( \theta _ { r } ^ { \prime } + \theta _ { t } ^ { \prime } \right) / 2 , } \\ { \theta _ { s } ^ { \prime } = \mathrm { a r g } ( \mathbf { R } _ { z } \left( \mathbf { k } _ { r , t _ { i } } - \mathbf { k } _ { I , \alpha _ { s } } \right) ) } \\ { \theta _ { l } ^ { \prime } = \mathrm { a r g } ( \mathbf { R } _ { z } \left( \mathbf { k } _ { r , t _ { i } } - \mathbf { k } _ { I , \alpha _ { s } } \right) ) } \end{array} } \end{array}\tag{17}
$$

(18)

For subscripts, $h , s , i , f$ denotes head, shoulder, hip, and foot respectively. Note that the foot feature is an integration of ankle and toe keypoints. For the orientations, assume $\mathbf { v } = ( x , y ) \in \mathbb { R } ^ { 2 }$ , then $\arg ( \mathbf { v } ) = \tan ^ { - 1 } ( y / x )$ calculates the argument of a vector in terms of radians and ensures it to be within [0, 2π), and

TABLE III  
CHOICE OF HEIGHT COEFFICIENT
<table><tr><td rowspan=1 colspan=1>Keypoint</td><td rowspan=1 colspan=1>head</td><td rowspan=1 colspan=1>nose</td><td rowspan=1 colspan=1>shoulder</td><td rowspan=1 colspan=1>hip</td><td rowspan=1 colspan=1>ankle</td><td rowspan=1 colspan=1>toe</td></tr><tr><td rowspan=1 colspan=1>γ</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.02</td></tr></table>

$$
R _ { \phi } = \left[ \begin{array} { c c } { \cos \phi } & { - \sin \phi } \\ { \sin \phi } & { \cos \phi } \end{array} \right] .\tag{19}
$$

is the rotation matrix. For shoulder and hip orientations, the above operation is equivalent to calculating the orientation of the normal vector of the segment which points from the left keypoint to the right one. For foot orientations, it takes the mean value of the orientation obtained by doing the same operation to ankle and toe.

Finally, each set of constructed $( \mathbf { p } , \theta )$ for all person and all frames are then used in the GTCG/GCFF pipelines.