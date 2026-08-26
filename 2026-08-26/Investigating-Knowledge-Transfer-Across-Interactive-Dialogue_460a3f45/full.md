# Investigating Knowledge Transfer Across Interactive Dialogue Games

Filippo Momentè<sup>⊗</sup>\*, Mir Nafis Sharear Shopnil<sup>⋄∗</sup>, Andrea de Varda<sup>§</sup>,

Pavel Merinov<sup>‡</sup>, Raffaella Bernardi<sup>‡</sup>, Oswald Lanz<sup>‡</sup>, Alessandro Suglia<sup>△</sup>,

Alessandro Torcinovich<sup>‡</sup>,

<sup>⊗</sup>University of Trento, <sup>⋄</sup>Technovative Solutions Ltd, <sup>§</sup>Massachusetts Institute of Technology, <sup>‡</sup>Free University of Bozen Bolzano, <sup>△</sup>University of Edinburgh

## Abstract

Dialogue games represent a challenging setting where complex cognitive skills are required to accomplish tasks while coordinating with other players. Considering that language represents an interface for both understanding the game rules and executing actions, it is reasonable to assume that training on a specific language game will enhance specific capabilities that might be relevant for other tasks as well. Motivated by this rationale, in this paper, we investigate how knowledge transfers across different dialogue games. We study transferability by finetuning LLM models on games from the clembench suite (Chalamalasetti et al., 2023) and performing two analyses: i) we derive a task-transferability graph using a binary integer optimization program from Zamir et al. (2018), using task performance as the main metric; and ii) we compute task vectors (Ilharco et al., 2022) for each game to study similarities across finetuned models and their task transferability. In our first analysis, we find that some games benefit more from transfer than finetuning, and that the visuospatial family (e.g., exploration games) transfers best. With our task vector analysis instead, we find that similarity-based approaches capture game-role relationships but almost no transferability patterns, suggesting that more complex metrics are required. Code available at https://anonymous.4open. science/r/yk6ub2dtwl-E76C

## 1 Introduction

Dialogue games have posed themselves as complementary to the traditional, reference-based paradigm (e.g., MMLU from Hendrycks et al., 2020) that has long been central in the evaluation of LLMs (Schlangen, 2023; Bertolazzi et al., 2023; Momentè et al., 2025; Suglia et al., 2024).

Their interactive, multi-turn, goal-oriented nature provides for an interesting environment to evaluate agents’ capability of understanding language and using it productively to accomplish specific tasks, rather than just retrieving knowledge.

In dialogue games, language expresses both the goals and the constraints (i.e., rules) that are required for their completion. The complexity of the game structure requires the exercise of a variety of skills to play and makes its analysis particularly challenging. For instance, understanding the mechanisms employed by a model to reach a game goal implies disentangling reasoning processes from rule-following, identifying the contributions of individual turns towards the objective, and the global strategy applied to solve the task.

Intuitively, we may expect that whenever two games match in any of the aforementioned aspects, training on one game would also help in playing the other. In the literature, there is evidence of such positive knowledge transfer (Liu et al., 2026; Hu et al., 2026; Park et al., 2026). However, experiments have been performed only on a limited number of tasks, and a systematic study of these effects across a variety of dialogue games is currently missing.

In this work, we are therefore interested in further investigating relationships among dialogue games, and studying their transferability, i.e., the degree to which specific game knowledge can be used to also play other games. To this aim, we focus on the popular clembench evaluation framework (Chalamalasetti et al., 2023), and we adapt the Taskonomy framework (Zamir et al., 2018), finetuning LLMs on specific game tasks and obtaining a series of graph representations that capture the underlying structure of knowledge transfer relationships. With our approach, we showcase that it is possible to find relevant relationships between different dialogue games. To our surprise, we observe several cases where finetuning on a given game may not be the best way to improve at it, and that it would be advisable to train our model on another game instead. We also report that the family of visuospatial games, involving reasoning on spatial information (e.g., exploration games), provides transferable knowledge to purely verbal games (e.g., Taboo). Thanks to our analysis, we also identify several situations of asymmetric transfers, where finetuning on a given game A generalizes to another game B, but doing the opposite degrades performance. These results support the fact that dialogue games require complex natural language understanding and reasoning skills that are instantiated in different ways by each game.

We also provide a complementary weight-space analysis of transferability and the connection between transferability and weight updates of finetuned models. To do so, we leverage task vectors (Ilharco et al., 2022) to determine the difference in parameter space between a finetuned model on a specific game and its baseline, i.e., the model used as the starting point. By comparing task vectors associated with each game, we find that weight similarity does not explain transfer relationships but rather offers a fingerprint of the game and the player role.

Taken together, the taxonomy and the weightspace analysis offer two complementary views of the same phenomenon: weight similarity records what was trained, while the direction of transfer must be measured by playing the games. Beyond the specific relationships we map, this gives an interpretability handle on what game-specific finetuning does to a model, a step towards understanding the mechanisms LLMs employ while playing dialogue games.

## 2 Related Work

Dialogue Games for LLM Evaluation. Dialogue games are well-structured activities with a goal state which players have to reach mostly through the production and use of language (Levin and Moore, 1977; Schlangen, 2023). They are typically multi-turn and can involve multiple players with different roles. While prior work has demonstrated their value in revealing capabilities that static benchmarks fail to surface, studies have generally treated game performance as an end in itself (Chalamalasetti et al., 2023; Hakimov et al., 2026; Qiao et al., 2023). Similar to Momentè et al. (2025), we instead use performance across a structured collection of dialogue games as a lens to study the internal organisation of model capabilities and how they transfer across tasks.

Interactive and Collaborative Games as Assessment Probes. Game-based approaches have a well-established history in assessing human cognitive and collaborative skills. Prior work has leveraged gamified environments to measure Collaborative Problem Solving (CPS) competencies such as strategy and perspective taking (Stoeffler et al., 2020), and serious games have been used to evaluate soft skills including communication and teamwork (Marengo et al., 2025). Such studies derive rich behavioural signals from naturalistic interaction traces without disrupting gameplay. Our work shares this methodological intuition — that interactive, collaborative environments make for uniquely informative probes — but shifts the subject of assessment from human participants to language models, employing dialogue games as structured instruments to study LLM capabilities across a diverse range of tasks.

Transfer Learning Across Tasks. The broader question of which tasks benefit from shared training has been extensively studied in NLP. Pruksachatkun et al. (2020) and Vu et al. (2020) systematically characterised intermediate-task transfer, showing that transfer relationships are often asymmetric and difficult to predict from surfacelevel task similarity alone. To organise such relationships structurally, we adapt the Taskonomy framework (Zamir et al., 2018), originally developed for computer vision, to the dialogue game domain. Our work is the first to apply this kind of structured transfer analysis systematically across a broad suite of dialogue games.

Parameter Space Analysis and Mechanistic Interpretability. A growing body of work has attempted to characterise what fine-tuning changes inside a model, both from a parameter-level and a mechanistic perspective. Building on the task vector framework of Ilharco et al. (2022), several studies have examined how task-specific adaptations interact and compose in weight space (Yadav et al., 2023; Wortsman et al., 2022), while Aghajanyan et al. (2021) show that such adaptations tend to occupy low-dimensional subspaces — providing theoretical grounding for interpreting task vector geometry. Complementing this parameterlevel view, mechanistic interpretability work has probed the internal structure of transformer models more directly (Elhage et al., 2021; Dai et al., 2022; Geva et al., 2021). Particularly relevant to our findings is the work of Prakash et al. (2024), who show that fine-tuning tends to enhance preexisting internal mechanisms rather than introduce new ones, offering a potential explanation for why some game-specific adaptations generalise more broadly than others.

## 3 Investigating Knowledge Transfer

## 3.1 Dialogue Games

Terminology. Dialogue games are instantiated through prompts, composed of a fixed template and variable parameters (e.g., the word to guess, the forbidden ones, etc.). Each of these prompts constitute an instance of that game. Players play a game instance by performing, at each turn one among all the possible actions available at that point, until reaching a final or an aborted state. The sequence of the actions taken at any turn is collected in an episode. In dialogue games, actions are expressed through language, making an episode a multi-turn conversation. Episodes are recorded into transcripts, and used to construct a dataset.

clembench (Chalamalasetti et al., 2023) provides a dataset of textual and multimodal dialogue games and a game evaluation framework for language models. In two-player games, the players never communicate directly and a game master mediates their communication, by preprocessing messages according to game logic before sending them. Additionally, the players may be shown different prompt templates reflecting their different roles within a game. This affects their context received at each turn, leading them to develop different perspectives, or points of view, on the episode that is being played. The framework records transcripts for played episodes, taking into account each player’s perspective.

Game Selection and Dataset Preparation. First, we collect game transcripts of the 17 textual dialogue games in the open-source clembench-runs repository (clembench, 2026) (1.6, 2.0, 3.0) obtained by having multiple models play several instances of the clembench dialogue games. We remove duplicate transcripts, those with more than 3000 tokens (as measured by the Qwen3.5-4B model’s tokenizer (Qwen Team, 2026)), those associated with episodes that did not terminate succesfully. In two-player games, transcripts are split into two independent conversations mirroring the players roles (from now on, identified with P1 and P2). We obtain a total of 27 role-specific tasks. To obtain a balanced dataset, we exclude tasks with fewer than 800 samples. Additionally, we exclude two tasks for their similarities with the other considered games, resulting in a final selection of 15 role-specific tasks associated with 9 games.

We finetune our model with a standard autoregressive loss on the transcripts. In particular, we hold out 20 transcripts per task to create the validation set. We do the same for the test set, keeping only the game instances to test the game play of the trained model. With the remaining transcripts, we build the training set, subsampling each task to the smallest task set size, resulting 585 samples per task.

Game Overview. We group the selected games into two families: verbal operating over words; visuospatial operating over grids and spatial configurations. Such distinction coarsely characterize the primary cognitive demands of each game, and appears both in models of working memory (Baddeley and Hitch, 1974), and in large-scale factor analyses of cognitive abilities (Carroll, 1993). In what follows, we provide a brief description of each game and their selected role-specific tasks.

codenames: verbal, two-player. The players compete together against a programmatically defined team. Both teams are provided with the same list of words, where some are associated with one team, some with the other. Only one team member is aware of the distinction. At each turn, this player generates a clue associated to a subset of words in the team list, helping the other into guess. Found words are removed from the list until one team guesses all of them, or a designed “killer word” is guessed.“Distractor words” are also present.

• P1: identifies targets and provides clues.

• P2: makes the guesses.

guesswhat: verbal, two-player. A player is given a list of words and needs to guess a target word by asking questions to the other player who knows the answer and is only allowed to reply with“yes”/“no”.

• P1: asks questions and makes guesses.

• P2: answers “yes” or “no”.

taboo: verbal, two-player. A player helps the other to guess a word by providing clues at each turn. Clues must not contain the target and other related words.

• P1: provides the clues.

• P2: makes a guess.

wordle\_withclue: verbal, single-player. The player tries to guess a 5-letter word based on clues and feedback indicating which guessed letters are present and whether they are in the correct position. wordle\_withcritic (VB): two-player. Identical to wordle\_withclue but a second player provides additional feedback after each attempt.

• P1: makes a guess.

• P2: provides the feedback.

adventuregame: visuospatial, single-player. A player explores and interacts with a simulated environment composed by multiple rooms, takes and replaces objects to reach its goal. imagegame: visuospatial, two-players, collaborative. One player receives a representation of an ASCII grid as input and has to guide the other player to reconstruct it turn by turn starting from an empty one.

• P1: provides the instructions.

• P2: draws a the grid based on the received instructions.

matchit\_ascii: visuospatial, two-player. Each player is provided with an ASCII grid and has to understand whether it is the same the other player sees or not by asking questions. Both players cover the same role, although the input grid may differ. referencegame: visuospatial, twoplayer. One player is provided with an ASCII grid to describe and another one has to exploit the description to pinpoint it among three options.

• P1: describes the target grid.

• P2: makes a guess.

## 3.2 Taxonomy Modeling

We are given a set of tasks T and a training budget. Our objective is to train some models on a subset of tasks $S \subseteq \tau$ constrained by the budget, in order to maximize the overall performance in $\tau$ . Then, from the solution of such problem, structural information on the transferability between tasks emerges, i.e., which task knowledge learned by a model transfers better to other tasks.

More formally, we define a task-transferability graph as a weighted directed hypergraph $G \ =$ $( V , E , w )$ whose vertex set $V = { \mathcal { T } } , | V | = n \operatorname { r e p } -$ resents tasks we want to analyze. Given the set of edges $E = V \times V$ , we define a weighting function $w : V \times V \to \mathbb { R } _ { \geq 0 }$ that associates each directed edge $( s , t ) \in E$ to a transfer performance, i.e., the performance obtained on a target task t by a model trained on the source task s.

Given a set $S \subseteq V$ of source tasks, its transfer set is defined as:

$$
T = \{ t \in V \mid \operatorname* { m a x } _ { s \in S } w ( s , t ) > 0 \} ,\tag{1}
$$

and its collective transfer performance is:

$$
P ( S ) = \sum _ { t \in T } \operatorname* { m a x } _ { v \in V } w ( v , t ) .\tag{2}
$$

Given a budget $\gamma \in \mathbb { N } ^ { + }$ , the collective transferability problem (CTP) requires then finding a set $S ^ { * } \subseteq V , | S ^ { * } |$ maximizing the collective transfer performance, i.e., $P ( S ^ { \ast } ) \geq P ( S ) , \forall S \subseteq V$ . The problem can be defined with the following binary integer program:

$$
\operatorname* { m a x } _ { \mathbf { x } } ~ \mathbf { w } ^ { \top } \mathbf { x }\tag{3}
$$

subject to

$$
\mathbf { A } \mathbf { x } \preceq \mathbf { b } , \ \mathbf { x } \in \{ 0 , 1 \} ^ { | E | + | V | } .\tag{4}
$$

where:

$$
w _ { k } = { \left\{ \begin{array} { l l } { w ( s , t ) } & { { \mathrm { i f ~ } } k \leq | E | \wedge \sigma ( k ) = s \wedge \tau ( k ) = t } \\ { 0 } & { { \mathrm { o t h e r w i s e , } } } \end{array} \right. }\tag{5}
$$

with $\sigma ( k ) = k$ mod n and $\tau ( k ) = k$ div n returning respectively the source and the target tasks of an edge $( \sigma ( k ) , \tau ( k ) )$ .

We have three constraint types.

Constraint I If an edge is selected, $i . e . , x _ { k } = 1$ for any $k \ \leq \ | E |$ , then its corresponding source

![](images/cc7081fbca6b73398b26e8776fcc783b6e7300a1d4b70c2c3d2530455bc7f632.jpg)

<table><tr><td rowspan=1 colspan=2>adventuregame</td><td rowspan=1 colspan=1>0.50</td><td rowspan=1 colspan=1>0.29</td><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1>0.20</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=6></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>codenames_P1</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.27</td><td rowspan=1 colspan=1>0.30</td><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.12</td><td rowspan=1 colspan=1>0.20</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.01</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>codenames_P2</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1>0.22</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=1>0.09</td><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>guesswhat_P1</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.27</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.15</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>0.050.02</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>guesswhat_P2</td><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>0.50</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=3>imagegame_P1</td><td rowspan=1 colspan=1>0.17</td><td rowspan=1 colspan=1>0.29</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.12</td><td rowspan=1 colspan=1>0.19</td><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1>0.20</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=3>imagegame_P2</td><td rowspan=1 colspan=1>0.17</td><td rowspan=1 colspan=1>0.36</td><td rowspan=1 colspan=1>0.47</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.23</td><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1>0.15</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=3>matchit_ascii_P1</td><td rowspan=1 colspan=1>0.27</td><td rowspan=1 colspan=1>0.42</td><td rowspan=1 colspan=1>0.18</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.06</td><td rowspan=1 colspan=1>0.20</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=3>referencegame_P1</td><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>0.30</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.20</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=2>0.01</td><td rowspan=1 colspan=1>0.03</td></tr><tr><td rowspan=1 colspan=3>referencegame_P2</td><td rowspan=1 colspan=1>0.44</td><td rowspan=1 colspan=1>0.36</td><td rowspan=1 colspan=1>0.14</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.20</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.15</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=3>taboo_P1</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.33</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=3>taboo_P2</td><td rowspan=1 colspan=1>0.18</td><td rowspan=1 colspan=1>0.29</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.06</td><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>wordle_withclue</td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1>0.31</td><td rowspan=1 colspan=1>0.14</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.06</td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=3>wordle_withcritic_P1</td><td rowspan=1 colspan=1>0.09</td><td rowspan=1 colspan=1>0.38</td><td rowspan=1 colspan=1>0.23</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1>0.11</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.15</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=3>wordle_withcritic_P2</td><td rowspan=1 colspan=1>0.12</td><td rowspan=1 colspan=1>0.33</td><td rowspan=1 colspan=1>0.22</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1>0.15</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=3></td></tr></table>

Table 1: Filtered transfer modeling matrix. Tasks are divided into visuospatial, and verbal; P1/P2 denotes player roles, while empty cells correspond to specialists underperforming the baseline.

node must be selected too. $\forall i \leq | E | \colon$

$$
a _ { i , k } = \left\{ { \begin{array} { l l } { 1 } & { { \mathrm { i f ~ } } k = i } \\ { - 1 } & { { \mathrm { i f ~ } } k - | E | = \sigma ( i ) } \\ { 0 } & { { \mathrm { o t h e r w i s e } } , } \end{array} } \right.\tag{6}
$$

$$
b _ { i } = 0 .\tag{7}
$$

Constraint II Each target task has at most one in-transfer (i.e., at most one incoming source task). $\forall j \leq | V |$

$$
a _ { | E | + j , k } = { \left\{ \begin{array} { l l } { 1 } & { { \mathrm { i f ~ } } k \leq | E | \wedge \tau ( k ) = j } \\ { 0 } & { { \mathrm { o t h e r w i s e , } } } \end{array} \right. }\tag{8}
$$

$$
b _ { | E | + j } = 1 .\tag{9}
$$

Constraint III The source set must contain at most $\gamma$ source tasks.

$$
a _ { | E | + | V | + 1 , k } = { \left\{ \begin{array} { l l } { 1 } & { { \mathrm { i f ~ } } k > | E | } \\ { 0 } & { { \mathrm { o t h e r w i s e } } , } \end{array} \right. }\tag{10}
$$

$$
\begin{array} { r } { b _ { | E | + | V | + 1 } = \gamma . } \end{array}\tag{11}
$$

The optimum $\mathbf { x } ^ { * }$ entails then the searched graph structure.

Transferability Scores. We now discuss how to obtain the transferability scores required by our CTP. We finetune instances of the same baseline model on each of the tasks in our dataset, obtaining task specialists. Each specialist is then evaluated on all the tasks, computing its mean quality score $( q s )$ , a metric defined in the clembench paper to disentangle game-playing competence from rule-following.

The baseline model scores are then subtracted from each other score, to obtain an estimate of the improvement attributable by the training. Some specialists perform worse than the baseline on certain games, thus, we remove the corresponding entries of the matrix, obtaining a filtered transfer modeling matrix shown in Tab. 1.

## 3.3 Weight Space Analysis

We complement the previous analysis by taking a look at the differences induced by model finetuning on a given task.

For each task and specialist, we compute its task vector (Ilharco et al., 2022) $\tau _ { t } = \theta _ { t } - \theta _ { B a s e } .$ , with $\theta _ { t }$ and $\theta _ { B a s e }$ being respectively the parameters of the specialist and the base model. $\tau _ { t }$ then represents the change induced by the training.<sup>1</sup>

We then compare the task vectors to identify similar patterns. We calculate a global score, $i . e .$ , the cosine similarity $\langle \tau _ { t _ { 1 } } , \tau _ { t _ { 2 } } \rangle$ , and an ad-hoc per-layer metric, called subspace overlap (simply overlap). Let $\tau _ { t } ^ { ( \ell ) }$ denote the weight matrix of $\tau _ { t }$ of the layer/module ℓ. We compute its truncated SVD and retain an orthonormal basis $V _ { t } ^ { ( \ell ) } \in \mathbb { R } ^ { d \times k }$ of its top-k right singular vectors. For two tasks $t _ { i }$ and $t _ { j }$ , the overlap at ℓ is:

![](images/1bddd96e18a2492b18bcd8ec82dd016bfd4124876132317961a01158530e3227.jpg)

![](images/af9b733a4ad0a229b2e837b9eddc6d873c1cbed3fd408ff1631fb87f4709b583.jpg)

![](images/6a67402dad2cc65df0e0cc021dab92b7eb88f5c86b295d39481d9a6a73c6bd4d.jpg)

![](images/9c591d3f03bc314e77b6d34c46a0482f419c30d7a455b92987f591b7c8b4df7b.jpg)  
Figure 1: Taxonomies across different $\gamma$ values. Tasks are divided into visuospatial, and verbal; blue encodes the transfer set. Edges represent transfers, with edge thickness proportional to the improvement. Dashed edged indicate multiple optimal transfers for the target. Edges depart prevalently from vertices of visuospatial tasks, indicating their versatile transferability across all of the considered games.

$$
\mathrm { o v l } _ { \ell } ( t _ { i } , t _ { j } ) = \frac { 1 } { k } \big \| V _ { t _ { i } } ^ { ( \ell ) \top } V _ { t _ { j } } ^ { ( \ell ) } \big \| _ { F } ^ { 2 } ,\tag{12}
$$

with $\| \cdot \| _ { F }$ being the Frobenius norm. In other words, ovl represents the mean squared cosine of the principal angles between the two rank-k update subspaces, ranging from 0 (orthogonal updates) to 1 (coincident subspaces). The metric applies to any pair of models finetuned from a shared base model, following the same setup as the transferability analysis.

## 4 Experimental Evaluation

## 4.1 Training and Optimization Settings

We use Qwen3.5-4B (Qwen Team, 2026) as the baseline model, maintaining the same training environment (transformers 5.7.0, Wolf et al., 2020), (TRL 1.3.0, von Werra et al., 2020) and hyperparameters across the specialists. Specifically, we set the learning rate at 1e−5, 8-bit

AdamW as optimizer and the batch size to 2, with a gradient accumulation of 8. For the evaluation, we use clembench 3.7.2, with temperature set to 0, setting enable\_thinking to False. For evaluating on single roles of two-player games, we let the corresponding specialist play against a Qwen3.5-27B model taking the other role. For the integer optimization, we use the milp function in the scipy library (Virtanen et al., 2020).

## 4.2 Transferability

We report the resulting taxonomies in Fig. 1.

We can first note that visuospatial competence tasks prevalently transfer better in all budget regimes. Additionally, at $\gamma = | \mathcal { T } | = 1 5$ (representing the best transfer performance), the visuospatial tasks are self-sufficient, i.e., they obtain their best performance without transfer (indicated by the looping edges), while verbal tasks benefit more, in general, from visuospatial task transfer (indicated by edges connecting visuospatial to verbal vertices). Taken together, these considerations suggest that the visuospatial family presents more generic challenges which help models develop versatile skills to play a broader range of games. In particular, adventuregame represents a good baseline specialist, due to the planning capabilities learned by the model to solve the task (i.e., navigating rooms and interacting with objects).

![](images/e7495238311061a57dca3d15c4a6049d84fd9500c9cd3b124609dcfbd8600b55.jpg)  
Figure 2: Comparison of the collective transfer performance against random transfer sets.

## 4.3 Robustness of the Identified Taxonomies

In order to assess whether the identified taxonomies contain meaningful information, we have compared the obtained collective transfer performance with that of 1000 taxonomies built by randomly sampling the source tasks and the connecting edges, while maintaining the supervision budget constraint. As can be seen in Fig. 2, the found transfer sets are indeed outperforming random ones by a significant margin.

## 4.4 Weight Analysis

The taxonomies above measure transfer by playing the games. We ask whether the same relationships can be recovered by comparing the specialists’ updates directly. We first assess if two tasks/roles related to the same game have more similarities than tasks belonging to different games. We measure a same-game overlap score by computing the overlap metric as in Eq. (12) and averaging it over every layer-module and every pair of P1/P2 tasks. Similarly, we compute a cross-game overlap score by averaging over pairs of different game tasks. Each of the model’s decoder layers contains an attention block and an MLP block; we analyse only the MLP, taking its three projection matrices (gate, up and down) of the SwiGLU feed-forward block (Shazeer, 2020)<sup>2</sup> for the 96 layer-module locations in total. For each location, we compare the two updated subspaces with Eq. (12), setting $k = 5 ,$ , and average the results. We recorded a same-game overlap of 0.406 against a cross-game overlap of 0.239. We additionally attested that the same-game overlap beats the cross-game one at each of the 96 layer–module locations. These results suggest, as expected, that the task vectors of two roles of the same game are far more aligned than those of two different games. Additionally, the two scores sit two orders of magnitude above the chance overlap baseline of 0.0015 (≈ 0.0020 for the gate and up projections, 0.0005 for down), i.e., the expected overlap of two randomly sampled subspaces.

<table><tr><td>task A</td><td>task B</td><td> $\langle \tau _ { A } , \tau _ { B } \rangle$ </td><td> $\mathrm { o v l } ( \tau _ { A } , \tau _ { B } )$ </td><td> $q s ( A , B )$ </td><td> $q s ( B , A )$ </td></tr><tr><td>img_P1</td><td>mia_P1</td><td>12.9</td><td>44.7</td><td>+10.0</td><td>+6.9</td></tr><tr><td>img_P1</td><td>ref_P1</td><td>2.3</td><td>16.5</td><td>+20.0</td><td>-12.9</td></tr><tr><td> $\mathrm { i m g \_ p 1 }$ </td><td>ref_P2</td><td>2.0</td><td>12.7</td><td>+0.0</td><td>+3.7</td></tr><tr><td>mia_P1</td><td>ref_P2</td><td>2.2</td><td>13.3</td><td>+5.0</td><td>-15.0</td></tr><tr><td> $\boldsymbol { \ t a b \_ P 1 }$ </td><td> $\pm \mathrm { a b \_ p 2 }$ </td><td>7.4</td><td>35.7</td><td>-13.3</td><td>-12.5</td></tr></table>

Table 2: Task-vector comparison computed as cosine $( \langle \cdot , \cdot \rangle )$ and top-5 subspace overlap (ovl) against qs transfer in both directions. Rows are restricted to targets where the share of completed episodes is unchanged from Base, so the quality-score gain is unambiguous. All values are reported $\times 1 0 ^ { 2 }$ . High similarity coincides with mutual harm (row 1) and with mutual help (row 2), so it does not determine transfer. Strong asymmetrical transfers (rows 3–5) imply low similarity.

In Tab. 2, we compare cosine and overlap values with the $q s$ transfer scores, for some representative task pairs. While the most similar pair – imagegame\_P1/P2 – displays good transferability, this is not always the case. Indeed, taboo\_P1/P2, has an overall good similarity but does not transfer in any directions, suggesting that transferability is unrelated to similarity in task vectors. Conversely, the other pairs of games show asymmetric transferability, in one direction only, and score, as a consequence, lower similarities. Part of these results were to be expected: the symmetrical nature of the similarity can only partially capture transferability patterns, which are, by nature, asymmetric.

## 5 Conclusions

In this work, we have conducted an analysis on the transferability between dialogue games, adapting the methodology originally proposed by Zamir et al. (2018). Our experiments showed that visuospatial games display superior generalization performance compared to verbal ones, notably also to games belonging to different families. Additionally, we applied the task-vector approach by Ilharco et al. (2022) to study the relationship between transferability and weight updates induced by finetuning on dialogue games, observing that similarity-based metrics are insufficient for recovering transferability from the weight space.

## 6 Limitations

The nature of this work is mainly exploratory. To confirm our results, we plan to extend our analysis both increasing the number of experiments and testing other LLMs. Additionally, we have limited our analysis to games provided by clembench, which are mostly collaborative. It is relevant to extend the analysis to other games, especially competitive ones. The limitation highlighted by weight-space analysis suggest going beyond symmetrical operators and explore non-symmetrical ones, that induce total order relations between games, attempting to define approaches that predict transferability without requiring evaluation on target games. Furthermore, our work focuses only on positive transfers. Expanding our taxonomy experiments to negative knowledge transfer relationships, i.e., transfers where finetuning on a certain task causes performance degradation on another one, would enable understanding how games affect each other unfavorably. Finally, it would be interesting to investigate whether the identified relationships can be exploited to build more effective curriculum-learning regimes for dialogue game training.

## References

Armen Aghajanyan, Sonal Gupta, and Luke Zettlemoyer. 2021. Intrinsic dimensionality explains the effectiveness of language model fine-tuning. In Proceedings of the 59th annual meeting of the associationfor computational linguistics and the 11th international joint conference on natural language processing (volume 1: long papers), pages 7319–7328.

Alan D. Baddeley and Graham Hitch. 1974. Working memory. volume 8 of Psychology of Learning and Motivation, pages 47–89. Academic Press.

Leonardo Bertolazzi, Davide Mazzaccara, Filippo Merlo, and Raffaella Bernardi. 2023. ChatGPT’s information seeking strategy: Insights from the 20- questions game. In Proceedings of the 16th International Natural Language Generation Conference,

pages 153–162, Prague, Czechia. Association for Computational Linguistics.

John Bissell Carroll. 1993. Human cognitive abilities: A survey of factor-analytic studies. 1. Cambridge university press.

Kranti Chalamalasetti, Jana Götze, Sherzod Hakimov, Brielen Madureira, Philipp Sadler, and David Schlangen. 2023. clembench: Using game play to evaluate chat-optimized language models as conversational agents. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 11174–11219, Singapore. Association for Computational Linguistics.

clembench. 2026. clembench-runs. https://github.com/clembench/ clembench-runs. Accessed 2026-07-12.

Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. 2022. Knowledge neurons in pretrained transformers. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8493–8502, Dublin, Ireland. Association for Computational Linguistics.

Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, and 6 others. 2021. A mathematical framework for transformer circuits. Transformer Circuits Thread.

Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. 2021. Transformer feed-forward layers are key-value memories. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 5484–5495, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Sherzod Hakimov, Roland Bernard, Tim Leiber, Karl Osswald, Kristina Richert, Ruilin Yang, Raffaella Bernardi, and David Schlangen. 2026. The price of thought: A multilingual analysis of reasoning, performance, and cost of negotiation in large language models. In Findings of the Association for Computational Linguistics: EACL 2026, pages 529–570, Rabat, Morocco. Association for Computational Linguistics.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Xiaodong Song, and Jacob Steinhardt. 2020. Measuring massive multitask language understanding. ArXiv, abs/2009.03300.

Lanxiang Hu, Mingjia Huo, Yuxuan Zhang, Haoyang Yu, Eric P. Xing, Ion Stoica, Tajana Rosing, Haojian Jin, and Hao Zhang. 2026. lmgame-bench: How good are LLMs at playing games? In The Thirteenth International Conference on Learning Representations.

Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. 2022. Editing models with task arithmetic. arXiv preprint arXiv:2212.04089.

James A Levin and James A Moore. 1977. Dialoguegames: Metacommunication structures for natural language interaction. Cognitive science, 1(4):395– 420.

Bo Liu, Leon Guertler, Simon Yu, Zichen Liu, Penghui Qi, Daniel Balcells, Mickel Liu, Cheston Tan, Weiyan Shi, Min Lin, and 1 others. 2026. Spiral: Self-play on zero-sum games incentivizes reasoning via multi-agent multi-turn reinforcement learning. In International Conference on Learning Representations (ICLR).

Agostino Marengo, Alessandro Pagano, and Vito Santamato. 2025. A machine learning framework for soft skills assessment: Leveraging serious games in higher education. Computers and Education: Artificial Intelligence, page 100469.

Filippo Momentè, Alessandro Suglia, Mario Giulianelli, Ambra Ferrari, Alexander Koller, Oliver Lemon, David Schlangen, Raquel Fernández, and Raffaella Bernardi. 2025. Triangulating LLM progress through benchmarks, games, and cognitive tests. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 20051– 20072, Suzhou, China. Association for Computational Linguistics.

Dongmin Park, Minkyu Kim, Beongjun Choi, Junhyuck Kim, Keon Lee, Jonghyun Lee, Inkyu Park, Byeong-Uk Lee, Jaeyoung Hwang, Jaewoo Ahn, Ameya S. Mahabaleshwarkar, Bilal Kartal, Pritam Biswas, Yoshi Suhara, Kangwook Lee, and Jaewoong Cho. 2026. Orak: A foundational benchmark for training and evaluating llm agents on diverse video games. Preprint, arXiv:2506.03610.

Nikhil Prakash, Tamar Rott Shaham, Tal Haklay, Yonatan Belinkov, and David Bau. 2024. Finetuning enhances existing mechanisms: A case study on entity tracking. In International Conference on Learning Representations.

Yada Pruksachatkun, Jason Phang, Haokun Liu, Phu Mon Htut, Xiaoyi Zhang, Richard Yuanzhe Pang, Clara Vania, Katharina von der Wense, and Samuel Bowman. 2020. Intermediate-task transfer learning with pretrained language models: When and why does it work? In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 5231–5247.

Dan Qiao, Chenfei Wu, Yaobo Liang, Juntao Li, and Nan Duan. 2023. Gameeval: Evaluating llms on conversational games. arXiv preprint arXiv:2308.10032.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

David Schlangen. 2023. Dialogue games for benchmarking language understanding: Motivation, taxonomy, strategy. Preprint, arXiv:2304.07007.

Noam Shazeer. 2020. Glu variants improve transformer. arXiv preprint arXiv:2002.05202.

Kristin Stoeffler, Yigal Rosen, Maria Bolsinova, and Alina A von Davier. 2020. Gamified performance assessment of collaborative problem solving skills. Computers in Human Behavior, 104:106036.

Alessandro Suglia, Ioannis Konstas, and Oliver Lemon. 2024. Visually grounded language learning: a review of language games, datasets, tasks, and models. Journal of Artificial Intelligence Research, 79:173– 239.

Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, and 16 others. 2020. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17:261– 272.

Leandro von Werra, Younes Belkada, Lewis Tunstall, Edward Beeching, Tristan Thrush, Nathan Lambert, Shengyi Huang, Kashif Rasul, and Quentin Gallouédec. 2020. TRL: Transformers Reinforcement Learning.

Tu Vu, Tong Wang, Tsendsuren Munkhdalai, Alessandro Sordoni, Adam Trischler, Andrew Mattarella-Micke, Subhransu Maji, and Mohit Iyyer. 2020. Exploring and predicting transferability across NLP tasks. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7882–7926, Online. Association for Computational Linguistics.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, and 3 others. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online. Association for Computational Linguistics.

Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and 1 others. 2022. Model soups: averaging weights of multiple finetuned models improves accuracy without increasing inference time. In International conference on machine learning, pages 23965–23998. PMLR.

Prateek Yadav, Derek Tam, Leshem Choshen, Colin A Raffel, and Mohit Bansal. 2023. Ties-merging: Resolving interference when merging models. Advances in neural information processing systems, 36:7093–7115.

Amir R Zamir, Alexander Sax, , William B Shen, Leonidas Guibas, Jitendra Malik, and Silvio Savarese. 2018. Taskonomy: Disentangling task transfer learning. In 2018 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE.