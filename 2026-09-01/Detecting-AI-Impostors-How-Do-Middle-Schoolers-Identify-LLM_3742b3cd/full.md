# Detecting AI Impostors: How Do Middle Schoolers Identify LLM Agents in a Live Collaborative Setting?

Dan Schumacher, Pragathi Durga Rajarajan, Haven Kotara, Roman Rendon, Kosi Atupulazi, Deepti Tagare<sup>†</sup>, Ismaila Temitayo Sanusi<sup>†</sup>,

Fred G. Martin<sup>†</sup>, and Anthony Rios<sup>†</sup>

University of Texas at San Antonio

{Daniel.Schumacher, Anthony.Rios}@utsa.edu

## Abstract

LLMs can imitate how people write, which raises concerns about impersonation, trust, and detection in social settings. These concerns are especially important for adolescents, who use generative AI frequently but may struggle to recognize it. We introduce DoppelBot, a cooperative social deduction game designed to study how young people detect and respond to AI impersonation. Through studies with middle schoolers, we investigate whether a DoppelBot prompts reflection on privacy and impersonation, how repeated exposure affects AIdetection accuracy as agents become more personalized, and which strategies students use to identify AI doppelgängers. We find that students’ detection accuracy improves over time, driven by a shift from relying on linguistic cues to leveraging shared social and contextual signals. Students also demonstrated an understanding of AI limitations such as embodiment and reflected on broader issues such as data privacy. To support future research, we release an anonymized dataset of game transcripts and voting behavior.<sup>1</sup>

## 1 Introduction

Adolescents are increasingly engaging with LLMs as tutors and collaborators (Vanzo et al., 2025; Liu et al., 2025). While these systems are effective at producing fluent and helpful responses, the same fluency allows LLMs to closely mimic human writing. This makes it difficult for users to determine whether they are interacting with a human or an automated system. Being able to distinguish between human- and machine-generated text is important because LLMs differ from humans in key ways, including their limitations, potential inaccuracies, and lack of intent and accountability. Prior work suggests that repeated exposure to LLM-generated content can help individuals develop an intuition for detecting machine-authored text (Dugan et al., 2023). Developing this intuition is particularly important for adolescents, who are rapidly adopting tools such as ChatGPT for schoolwork (Adair et al., 2025; Freeman, 2025) but may lack the experience needed to critically evaluate the source, reliability, and role of the content these systems produce (Abdelghani et al., 2025).

While user responses to deception, impersonation, and hallucinations have been studied in isolation (Dogra et al., 2025; Yin et al., 2024; Shi et al., 2025; Ibraheem et al., 2022; Jones and Bergen, 2024), how users identify LLMs during ongoing social interaction (i.e., multi-user settings) remains unexplored. Prior work on human ability of machine-generated text detection typically frames detection as a binary judgment task in which participants are shown short documents and asked to classify them as human- or machine-written. Results show that human accuracy is low and highly variable, even among domain experts and trained annotators (Liu et al., 2024a; Dugan et al., 2023). In live social settings, this task is further complicated by the need to navigate asynchronous communication, where people must evaluate not only the content of a message but also its strategic timing and social relevance (Eckhaus et al., 2025). Moreover, almost all prior work focuses on adult subjects, despite recent studies suggesting that variability in detection accuracy is closely tied to the specific population tested (Jones et al., 2025). Consequently, measuring detection behavior among adolescents (e.g., middle school students) is vital, as their distinct social cues and shared group identity likely lead to different reasoning patterns than those of adult populations.

To explore how young users detect AI-generated text under adversarial conditions, we introduce

DoppelBot, a cooperative text-based game in which players must identify AI “doppelgängers” impersonating them and their peers. Unlike prior detection studies that place an isolated user in static text snippets, DoppelBot studies users’ detection abilities in a live, collaborative environment. The AI impersonators mimic their target users’ writing styles, making it harder for each user to both detect the AI agents and convince other users of their own humanity. Through behavioral studies with middle school students, we analyze changes in detection accuracy, message patterns, and metacognitive strategies as players face increasingly convincing AI impostors. We find that students’ detection accuracy improves with repeated exposure. This holds even as the LLMs become more stylistically aligned. Qualitative analyses reveal that players adapt their strategies over time. Moreover, students use both linguistic and social cues, as well as real-world limitations related to embodiment, to identify bots.

Our contributions are as follows: (1) To the best of our knowledge, we perform the first study analyzing how adolescents identify AI systems within collaborative settings; and (2) We created one of the first AI impersonation games for studying how humans can collaboratively detect AI agents. We provide the full source code to facilitate replication, along with an online version of the game for easy use by researchers and educators. (3) We release a novel dataset of game transcripts and voting behavior. Additionally, we provide an example of how this dataset could facilitate future research in Appendix D.2.

## 2 Related Work

Machine Generated Text Detection. Humans were originally considered the de facto benchmark for distinguishing machine from human text (Turing, 1950), but recent advances in LLMs have made even domain experts unreliable at this task (Liu et al., 2024a; Dawkins et al., 2025; Sadasivan et al., 2025; Xu et al., 2024). As a result, most current work focuses on automated detection methods (Li et al., 2024b; Huang et al., 2025), yet studying human detection remains vital because people increasingly encounter AI-generated content in educational and online contexts (Dugan et al., 2023; Vanzo et al., 2025; Liu et al., 2025). Many studies show that well-designed systems can be indistinguishable from humans (Dawkins et al., 2025;

Chakraborty et al., 2023). To address the limitations of static text evaluation, Dugan et al. (2023) uses a boundary detection task to pinpoint the transition from human to machine-generated content. Their findings suggest that while detection is challenging, human performance improves over time when properly incentivized. Building on this by focusing on individual-level mimicry in a chat-based environment, Shi et al. (2025) employs supervised fine-tuning and hierarchical memory to simulate specific personas. This approach is notably effective, as even close friends and family fail to identify the AI 44% of the time.

As exposure to AI content grows, it is important to quantify user sensitivity to AI-driven misinformation and persona-driven manipulation. However, existing studies focus almost exclusively on adult populations. This highlights a need to examine these phenomena among more vulnerable groups, particularly adolescents, whose distinct social behaviors and cognitive development may influence their susceptibility to AI-mediated deception.

Adolescent Privacy & AI. High school GenAI adoption climbed from 79% to 84% in 2025 (Adair et al., 2025), while undergraduate usage reached 92% (Freeman, 2025). This trajectory suggests that as middle schoolers age, their likelihood of utilizing these tools increases, a trend further supported by the proliferation of LLM-based classroom tutors (Vanzo et al., 2025; Liu et al., 2025). Despite this ubiquity, middle schoolers often over-rely on AI systems by writing inadequate prompts and overestimating both the quality of the output and their own conceptual understanding (Abdelghani et al., 2025).

These challenges align with the AI4K12 framework’s emphasis on Societal Impact, which posits that literacy must include an understanding of ethical risks such as privacy threats and misuse (AI for K-12 Initiative, 2024). Research indicates that students often prefer self-disclosing to AI agents over human instructors due to perceived gains in psychological safety (Peng and Wan, 2025). However, this preference creates a dangerous paradox where digital trust increases susceptibility to persona-driven manipulation and social engineering. Educating adolescents early is vital because this vulnerability is exacerbated by the expiration of COPPA protections at age 13 (Ritvo et al., 2013). Furthermore, compulsory schooling offers a unique opportunity to reach adolescents with privacy education before they become legally eligible to leave school, helping ensure widespread exposure to critical lessons about data tracking and personalization (Khan et al., 2024).

With this aim in mind, we propose a game-based learning (GBL) approach, specifically by using a social deduction game. While numerous studies have established the general benefits of GBL (Su et al., 2021; Voulgari et al., 2021; Adisa et al., 2023; Jordaan and Timm, 2025; Presson et al., 2025), social deduction games serve a research purpose on a dual axis. On one axis, they have been shown to have pedagogical benefits: Tilton (2019) found that they were effective in teaching small-group communication and in developing roles and power structures. On the second axis, they have been used to evaluate LLM decision-making, deception, and long-term memory capabilities (Liu et al., 2024b; Lai et al., 2022; Lan et al., 2024; Light et al., 2023; O’Gara, 2023). We find it appropriate to use social deduction games as a tool at this crossroads to evaluate how interactive, adversarial environments can foster the critical reasoning necessary to identify and resist AI-driven impersonation.

Education. Teenagers operate in digital ecosystems that monetize personal data, often without their informed consent (Moti et al., 2024). This exposure risks long-term harms, including algorithmic discrimination in hiring and credit (Brown, 2023; Botta and Wiedemann, 2020; O’Neil, 2016). The rise of LLMs has further complicated this landscape, as personal data can be “memorized” and regurgitated during inference (Li et al., 2024a; Das et al., 2025).

Despite these risks, significant barriers hinder AI ethics education for adolescents. Many students view privacy efforts as futile and opt out of protective practices (Wisniewski et al., 2022). Teachers often feel unprepared to address these topics. Those with technical backgrounds may lack confidence in leading ethical discussions (Brown et al., 2024), while non-technical instructors struggle with technical complexity (Kilhoffer et al., 2023). Rigid curricula and potential parental pushback limit innovation in how sensitive ethical topics are introduced (Kilhoffer et al., 2023). On the students’ end, many existing materials assume prior STEM knowledge, creating barriers for students with limited computer science access (Brown et al., 2024).

DoppelBot addresses these challenges by providing a low-barrier entry point for AI ethics and data privacy. We define “low-barrier” with students and educators in mind. For Students: The game requires no prior knowledge of computer science, ethics, or mathematics. This allows students from diverse backgrounds to engage with complex topics like data disclosure and AI impersonation through direct experience rather than theory. For Educators: A full session (including setup, gameplay, and discussion) fits within a thirty minute window. By serving as a natural conversation starter, the tool reduces the instructional burden on non-technical teachers and empowers technical educators to facilitate ethical reflection without a formal lecture.

## 3 Method

DoppelBot is a cooperative social deduction game in which teams of three to five students chat anonymously with each other and with an equal number of LLM-powered AI doppelgängers who impersonate them. Each user must identify and vote out the bots through conversation while avoiding suspicion from the other participants. The users win by eliminating all AIs.

## 3.1 Game Design

Gameplay Overview. The game has four phases: (1) Introduction: players get instructions and enter personal details, (2) Chat: players converse to prove their humanity and identify bots, (3) Voting: players eliminate suspected AIs, and (4) Outcome: final scores and identities are revealed.

Introduction. Three to five students sit at laptops positioned so they can’t see each other’s screens. An on-screen tutorial covers the rules while the researcher emphasizes three guidelines: (1) All communication happens through chat (no talking or gestures); (2) Chat rounds are timed so act quick; and (3) Each round starts with a mandatory yellow– text question everyone must answer before moving to general discussion. Players complete a form with their name, grade, favorite animal, hobby, and a fun fact. This information is used to personalize their AI doppelgänger’s style.

Chat. Each player gets a random code name and color. A yellow icebreaker appears at the top of the screen (e.g., “if you could have any superpower, what would you choose and why?”). After responding to the icebreaker, players chat freely to prove their humanity and identify AIs. All participants (human and AI) look identical in the chat except for a distinguishing color and code name. Each round lasts 100 seconds before moving to voting.

![](images/c846ca559c69c109ba2912b31fd63e41148e29269a2b28107d955bc07682d06e.jpg)  
Figure 1: Left: bot response pipeline. Right: two players and their AI doubles chat and vote. Messages are from real student games (Rabbit was AI). “Cap” means “you’re lying.”

Voting. Each human (who hasn’t yet been voted out) votes for who they suspect is AI. After all players cast their ballots, three outcomes are possible: (1) AI elimination: if an AI gets majority votes; (2) Human elimination: if a human gets majority votes; (3) No consensus: if there’s no majority, no one is eliminated. Requiring a majority vote encourages both collaboration and critical thinking. While students can easily recognize their own AI doppelgänger (the one claiming to be them), identifying others’ AIs is more difficult.

Outcome. The chat-and-vote cycle repeats for as many rounds as there are human players (three rounds in this study). After all rounds, players see a score screen revealing who was human versus AI, plus elimination details.

## 3.2 Agentic Architecture

The AI doppelgängers mirror a specific human player H by integrating personal metadata, such as the user’s name and hobbies, into a persistent persona $P _ { H }$ . The system follows an agentic architecture (see Algorithm 1) that processes an evolving global conversation history, Hist, which contains every message sent by all players and AIs in the chat. The architecture consists of three core components described below. Full prompts can be found in Appendix G.1. All three components of the DoppelBot architecture were implemented using gpt-4o-mini. The temperature was set to 0.5 for the Decide-to-Respond and Response Generation agents, and 0.9 for the Stylizer agent.

Decide-to-Respond (DTR) For every new message m appended to Hist, the model determines if an intervention is warranted. By evaluating the game rules, $P _ { H }$ , and the current state of Hist, the DT R component outputs a boolean value and a natural language justification (Reas). For example, the model might justify an action by stating: “A player just accused me of being an AI; I must defend myself.” If DT R is false, the agent remains silent, allowing the doppelgänger to avoid over-speaking or interrupting other threads.

Response Generation (Gen): If a response is deemed necessary (DTR = True), the model transitions to the generation phase. Using Reas alongside Hist, it produces a baseline response $R _ { b }$ . At this stage, the response is factually grounded in the persona but remains linguistically generic (e.g., $^ { 6 6 } \mathrm { N o } ,$ I am the real Carlos”).

Stylizer (Stylize): To achieve individual-level mimicry, the stylizer adapts the baseline response $R _ { b }$ into a stylized output $R _ { s }$ . This component utilizes a dedicated subset of Hist, $M _ { H }$ , which contains only the messages previously sent by human H. By analyzing the linguistic markers in $M _ { H }$ , the stylizer rewrites the generic response to match the user’s unique voice (e.g., “nah fam its me fr fr”).

## 3.3 Experimental Design

Settings. We piloted DoppelBot at a public STEMcharter school as part of a two-day afterschool event called AI EXPO, where seven tools for teaching different AI concepts (including ours) were presented to middle schoolers. Out of 70 assenting participants, 29 interacted with DoppelBot. A subsequent, more in-depth evaluation of DoppelBot was conducted during a four-day AI For Everyone! summer camp (Stallings et al., 2026). Both studies were conducted with IRB approval, parental consent, and student assent. Demographic information for both settings can be found in Appendix E.

Algorithm 1 DoppelBot Agentic Response Loop   
1: Input: Humans H ∈ Players, Persona $P _ { H }$   
2: Parallel for each H do:   
3: M<sub>H</sub> ← ∅ ▷ Human-specific style history   
4: while Phase is Chat do   
5: m ← WaitNextMessage()   
6: Hist ← Hist ∪ {m}   
7: if sender $( m ) = { \dot { H } }$ then   
8: M<sub>H</sub> ← M<sub>H</sub> ∪ {m}   
9: end if   
10: DTR, Reas ← DTR(Hist, P<sub>H</sub>)   
11: if DTR then   
12: R<sub>b</sub> ← Gen(Reas, Hist, P<sub>H</sub>)   
13: $R _ { s } \gets \mathsf { S t y l i z e } ( R _ { b } , M _ { H } )$   
14: Send $R _ { s }$   
15: end if   
16: end while   
17: End Parallel

Procedure. At the summer camp, DoppelBot was evaluated over two days. On Day 1, the participants played their first round of DoppelBot. On Day 2, students completed Partnered Interview Worksheets (PIWs), which collected personal information used to create more personalized AI doppelgängers. It included writing style questions (e.g. “What slang do you use a lot?”) and general questions (e.g. “What 2-3 topics do you talk about the most?”).

Interviews. At the end of the second day, participants engaged in semi-structured group interviews. Interviewers had a list of guiding questions available should discussion stall, but were encouraged to foster natural dialogue among students. The goal was not to reach consensus but rather to capture a diversity of perspectives. These interviews were recorded and manually transcribed. An initial researcher reviewed the complete transcript and developed a coding scheme. Two independent researchers then assigned labels to each utterance according to this scheme. Disagreements were resolved through adjudication by the initial researcher.

Pre- and Post-Survey. To quantitatively capture student opinions, we administered a pre/post survey adapted from Menard and Bott (2025). The instrument was shortened and revised for middle school students with approval from the original authors. We retained three dimensions of privacy concern: Combining Data (CD), concerns about combining data from multiple sources to reveal personal information; Data Permanence (DP), concerns that personal data may remain accessible indefinitely without a clear reason; and Improper Access (IA), concerns that personal data may be accessible to those who are not appropriately authorized. Table 6 presents the adapted items used in our study.

Automated Game Transcripts. As our last data stream, we conducted a categorical analysis of all chat messages. One researcher reviewed all messages and developed a coding scheme capturing the salient categories of player behavior. After removing unintentional blank messages, we were left with 2,182 unique messages.<sup>2</sup> We then randomly sampled 50 rounds and had two researchers independently label each message. Disagreements were resolved through adjudication.

## 4 Results

In this section, we aim to answer the following research questions: RQ1: To what extent does a gamified AI experience elicit student reflections on data privacy and AI impersonation?; RQ2: How does repeated exposure to AI doppelgängers affect students’ ability to identify chatbots, even as their bots become more personalized?; and RQ3: What strategies do students use to distinguish between AI doppelgängers and human players?

RQ1: Student Reflections. To answer RQ1, we examine interview annotations, survey responses, and representative qualitative examples.

The final annotated interviews contained 417 utterances spanning six categories. Inter-annotator agreement was substantial $( \kappa ~ = ~ . 7 0 7 )$ , and detailed category frequencies are reported in Table 1. The full annotation codebook is provided in Appendix C. The annotations reveal that students reflected not only on the task of identifying AI agents (AI Detection, n = 148), but also on how their detection strategies evolved over time (Learning Adaptation, n = 60) and how the experience affected them emotionally (Social–Emotional Responses, n = 71). Beyond the gameplay itself, students discussed broader questions about AI capabilities (Perceptions of AI, n = 55) and the collection and use of personal information (Privacy Concerns, n = 30). While Privacy Concerns was the least represented category, it still accounted for a meaningful 7% of all utterances. Together, these categories suggest that DoppelBot prompted reflection on both AI impersonation and data privacy.

<table><tr><td>Category</td><td>K Frequency</td></tr><tr><td>Perceptions of AI</td><td>0.559 55</td></tr><tr><td>AI Detection</td><td>0.779 148</td></tr><tr><td>Learning Adaptation</td><td>0.561 60</td></tr><tr><td>Social-Emotional Responses</td><td>0.657 71</td></tr><tr><td>Privacy Concerns</td><td>0.689 30</td></tr><tr><td>Design Feedback</td><td>0.919 44</td></tr><tr><td>None of the Above</td><td>0.786 182</td></tr><tr><td>Macro Average</td><td>0.707</td></tr></table>

Table 1: Agreement and frequency of annotated interview data. There were 417 utterances. Multiple labels were possible per utterance.
<table><tr><td>Subscale</td><td>Mean (Pre)</td><td>Mean (Post)</td><td>SD (Pre)</td><td>SD (Post)</td></tr><tr><td>CD</td><td>3.667</td><td>3.820</td><td>0.871</td><td>1.119</td></tr><tr><td>DP</td><td>3.917</td><td>4.056</td><td>0.923</td><td>0.973</td></tr><tr><td>IA</td><td>4.278</td><td>4.472</td><td>0.776</td><td>0.559</td></tr></table>

Table 2: Descriptive statistics for each subscale at preand post-test.

This interpretation is further supported by the preand post-survey results.

We collected 13 matched pre/post responses for CD and 12 for both DP and IA. Results are shown in Table 2. Mean responses increased across all three dimensions. This suggests increased concern about combining personal data, long-term data retention, and improper access after interacting with DoppelBot. Given the small sample size and descriptive nature of the analysis, however, these findings should be viewed as preliminary.

Finally, to provide a more detailed view of how students reflected on data privacy and AI impersonation, we highlight representative excerpts from the interviews.

A common opinion was that sharing personal information with AI systems is acceptable either if the sharing improves usability/convenience, or if the amount of information being shared is minimal. When asked whether it is safe to share their information with AI companies, one student noted: P7: “I don’t feel that bad . . . I don’t have that much information.”

Other students framed data retention as a functional necessity for user experience rather than a privacy concern. For example, P1 emphasized usability while still acknowledging users’ right to request deletion:

P1: “If companies have my information, they might as well keep it because what if I want to sign in again? . . . But if you ask a company to delete it, they should.”

In contrast, P3 suggested that responsibility for protecting user data lies solely with the service provider:

P3: “It’s not their [the companies’] job to stop people from getting their information.”

Some students recognized that the information they entered during the PIW enabled their AI doppelgängers to impersonate them more convincingly, yet still expressed limited concern about this risk. In one exchange, P12 acknowledged providing personal information while downplaying its potential misuse:

P12: “I was just saying stuffthat I would say, to let them know that I wasn’t the AI . . . Well yeah, it was trying to impersonate me.”

Other participants drew clearer connections between impersonation technologies and real-world harm. P1 raised concerns about identity theft, while P13 linked the experience to familiar forms of deception such as spam calls:

P1: “I think that they could sell my information, steal my identity, pretend to be me online . . . Ifeel like a lot ofpeople get their identity stolen.”

P13: “Another thing I learned is that AI is going to copy you sometimes and try to convince you that they are the real person. For example, spam calls. They will send you a message, and you’ll think they are the real person.”

Beyond explicit concerns about AI ethics and data privacy, the gameplay itself sparked curiosity about AI capabilities. One participant expressed surprise at the system’s apparent knowledge:

P10: “How much does the AI know about stuff? Like I said ‘I like Lord ofthe Rings,’ and it started talking about all the characters and stuff, even though I didn’t include that [on my PIW].”

This moment of surprise illustrates how the game can serve as an unexpected conversation starter, prompting students to question the capabilities of AI systems and reflect on what these systems might “know” about them.<sup>3</sup>

Finally, students found the experience engaging. All 29 participants at the pilot agreed that DoppelBot was a fun way to learn about AI, and many spontaneously expressed enthusiasm (e.g., “Can I play again?”). Prior work suggests that such engagement can promote motivation and deeper reflection on underlying concepts (Gee, 2003; Malone and Lepper, 1987; Su et al., 2021). Holistically, the survey responses, interview data, and student enthusiasm suggest that DoppelBot successfully elicited reflection on both AI impersonation and data privacy.

RQ2: Repeated Exposure. We break this RQ into two components: alignment and performance. First, does PIW disclosure lead to greater alignment between students and their respective AI doppelgängers? Second, does student detection performance improve despite this increased alignment?

Alignment. We compared human and AI messages using four complementary representations and similarity measures: (1) BERT, which captures semantic similarity using contextual embeddings (Devlin et al., 2019); (2) LIWC, which encodes linguistic and psychological dimensions such as emotion, self-reference, and time orientation based on established lexicons (Tausczik and Pennebaker, 2010); (3) METEOR, which measures lexical overlap using exact, stemmed, and synonymous word matches (Banerjee and Lavie, 2005). (4) ROUGE-L, which measures lexical similarity based on the longest common subsequence between two texts (Lin, 2004).

For the lexical-overlap measures (METEOR and ROUGE-L), we computed similarity directly between $H _ { k }$ and $A _ { k }$ using their respective scoring functions. For each measure, we computed a score for every human-AI pair and report the average similarity for Day 1 and Day 2 as $\bar { s } * \mathrm { d a y ~ = ~ }$ $\begin{array} { r } { \frac { 1 } { K * \mathrm { d a y } } \sum _ { k = 1 } ^ { K _ { \mathrm { d a y } } } s _ { k } } \end{array}$ , where $K _ { \mathrm { d a y } }$ is the number of participants on a given day.

Figure 2 reports the average paired similarity on both days. All reported measures increase from Day 1 to Day 2, indicating greater alignment between students and their respective doppelgängers. Paired bootstrap tests further support these increases. This suggests that AI responses became more closely aligned with their human counterparts after students disclosed additional personal information through the PIWs on Day 2.

As a sidenote, increased alignment did not imply that the messages were identical. An exploratory LIWC analysis (Appendix D.1) revealed several persistent differences between human and AI language, including message length, certainty, and use of informal markers.

Qualitatively, most students agreed that the PIWs made the AI act more similarly to them. Several participants noted that the AI used the provided information in unexpected and sometimes unsettling ways. Many were surprised when it adopted slang they didn’t explicitly provide.

![](images/885f003b735b2dc72cbfe66a105612a8d7503f5ededa11d9337f6348895b91eb.jpg)  
Figure 2: Similarity between student and AI messages across sessions. Significance levels are based on paired bootstrap tests $( ^ { * * } p \leq . 0 5 , ^ { * } p \leq . 1 0 )$

P2: “[In the game] we said things like ‘low key’ or ${ ' } c u z ,$ ’ but we never taught it ‘no cap,’ and it started saying ‘no cap.’ I got really scared because we never taught it that.”

Others echoed this discomfort, describing unease when the AI adopted linguistic patterns they had not intentionally provided, an experience reminiscent of the uncanny valley, where increasingly human-like behavior can become unsettling.

P12: “I would not say slang words, because I started freaking out when the AI started saying slang words. It was really weird . . . Yeah, it was kind of scary.”

Others were impressed by the model’s ability to replicate stylistic quirks.

P14: “It also surprised me because I put ‘Lebron’ with multiple ‘n’s, and it copied me and put the same amount of ‘n’s.”

P2: “It was kind ofharder [on day 2], to be honest. At the start it was harder because it sounded like her [points to another user]. And it actually sounded like me. It used extra letters and stuff.”

Students also expressed discomfort with the AI appearing to infer personal preferences.

P6: “It made an inference that I liked dinosaurs, but I didn’t put that on there. At that point, I honestly got scared. I tried to type something and misspelled it really badly, like ‘I like d,’ and they took the $\cdot _ { d } ,$ to mean dinosaur. So it’s like a very good guess?”

<table><tr><td>Evaluator</td><td>Day 1</td><td>Day 2</td></tr><tr><td>Students</td><td>79.3%</td><td> $8 6 . 7 \% ^ { * }$ </td></tr><tr><td>Expert Annotations</td><td>91.9%</td><td>92.3%</td></tr></table>

Table 3: User AI-identification accuracy $( p \leq . 1 0 )$

Similarly, another participant reflected on how the AI seemed to extrapolate beyond the explicitly provided data.

P13: “I didn’t put down chess, but it said ‘I like chess,’ even though I didn’t put that I like chess. So it’s taking the information that we put down and using it to make predictions from $i t . ^ { \dprime }$

Together, the quantitative and qualitative findings indicate that Day 2 agents more closely resembled their human counterparts, both in measurable language characteristics and in participants’ subjective perceptions of the interaction.

Performance. Building on these findings, we next examined whether students’ ability to detect AI players improved with repeated gameplay. We calculated the average accuracy across rounds as the proportion of votes correctly cast for AI agents: $\begin{array} { r } { \frac { 1 } { R } \bar { \sum _ { j = 1 } ^ { R } \frac { 1 } { N _ { j } } } \sum _ { i = 1 } ^ { N _ { j } } v _ { i } } \end{array}$ , where $v _ { i }$ indicates a correct vote, R is the total number of rounds, and $N _ { j }$ is the number of users in round j. As shown in Table 3, students improved from 79.3% accuracy on Day 1 to 86.7% on Day 2; this improvement was marginally significant under a paired t-test $( p = . 1 0 )$

To contextualize these results, we conducted a post hoc annotation task with two researchers (Cohen’s $\kappa = 0 . 8 6 )$ , followed by adjudication to establish consensus. These adjudicated judgments yielded accuracies of 91.9% and 92.3% on Days 1 and 2, respectively. However, these scores reflect idealized conditions: annotators had access to full transcripts, prior exposure, and no time constraints. Importantly, accuracy changed little across days, suggesting that the improvement was specific to the students rather than the result of a substantially easier detection task on Day 2.

As an additional robustness check, we employ a mixed-effects logistic regression model, in which we learn a player-specific random-effects term to account for cross-player variation. The model analyzes 215 votes and estimates the probability of correctly identifying the AI using maximumlikelihood estimation. Detection accuracy is modeled as a function of round progression (Round), whether the game was played on the second day (Second Day), and the number of players remaining in the game (Remaining Players). We include all two-way and three-way interactions between these variables. To make the intercept interpretable, the number of remaining players is mean-centered.

<table><tr><td>Variable</td><td>Coef.</td><td>Std.Err.</td><td>z</td></tr><tr><td>Intercept</td><td> $. 4 6 7 ^ { * * * }$ </td><td>0.107</td><td>4.363</td></tr><tr><td>Round Index</td><td> $. 1 9 0 ^ { * }$ </td><td>0.074</td><td>2.556</td></tr><tr><td>Day (Second Day)</td><td>.415*</td><td>0.162</td><td>2.570</td></tr><tr><td>Round : Day</td><td>-.222</td><td>0.116</td><td>-1.920</td></tr><tr><td>Remaining Players (Centered)</td><td>.616***</td><td>0.168</td><td>3.671</td></tr><tr><td>Round : Remaining Players</td><td> $- . 2 6 8 ^ { * * }$ </td><td>0.089</td><td>-3.009</td></tr><tr><td>Day : Remaining Players</td><td> $\phantom { 0 } { - . 6 0 7 ^ { * } }$ </td><td>0.255</td><td>-2.385</td></tr><tr><td>Round : Day : Remaining Players</td><td> $. 2 6 2 ^ { * }$ </td><td>0.134</td><td>1.959</td></tr></table>

Table 4: Mixed Effects Logistic Regression Model (with player random effects) results for detection accuracy $( n _ { v o t e s } ~ = ~ 2 1 5 )$ . Significance levels: $^ * p \ \leq$ $0 . 0 5 , \ ^ { * * } p \leq 0 . 0 1 , \ ^ { * * * } p \leq 0 . 0 0 1$

Table 4 summarizes the regression results. The intercept $( \beta = 0 . 4 6 7 )$ represents the baseline logodds of correctly identifying the AI on Day 1 at the start of a game with an average number of players remaining. We observe significant positive main effects for both round progression and session date. The positive coefficient for Round Index $( \beta = 0 . 1 9 0 , p \le 0 . 0 5 )$ indicates that the odds of correctly identifying the AI increase as the game progresses, suggesting that participants learn to recognize the AI during the course of a match. Similarly, the positive coefficient for Second Day $( \beta = 0 . 4 1 5 , p \leq 0 . 0 5 )$ indicates that baseline detection performance is higher on the second day, suggesting that participants retain knowledge about the AI’s behavior across sessions. The number of remaining players also shows a significant positive effect $( \beta = 0 . 6 1 6 , p \le 0 . 0 0 1 )$ , indicating that participants are more likely to correctly identify the AI when more players are present in the game. However, the interaction between Round Index and Remaining Players is negative $( \beta ~ = ~ - 0 . 2 6 8 .$ $p \leq 0 . 0 1 )$ , suggesting that the benefit of having more players decreases as the game progresses. Finally, the interaction between Round Index and Second Day is negative $( \beta = - 0 . 2 2 2 , p = 0 . 0 5 5 )$ indicating that the improvement across rounds is weaker on the second day. Because baseline performance is already higher on Day 2, participants require fewer rounds of gameplay to identify the AI. This pattern suggests that the improvement observed on the second day reflects the retention of AI-specific detection strategies rather than learning the game mechanics alone.

Together, the increase alignment across three different measures, voting behavior, and regression analyses suggest that repeated exposure helped students develop effective detection strategies despite stronger impersonation.

RQ3: What Strategies? To understand why students became more effective at identifying AI agents on Day 2, we analyzed the annotated gameplay transcripts. Each message could receive one or more behavioral codes: coordinating votes (crvt), following group consensus (folw), responding to icebreaker prompts (iceb), leveraging information unavailable to the AI (meta), or producing irrelevant and chaotic content (rand).

To examine whether these behaviors changed over time, we conducted a chi-square test of independence. Because Day 1 contained more messages than Day 2, we randomly downsampled Day 1 to ensure a balanced comparison. Table 5 reports the categories that changed significantly between sessions. Messages coded as crvt and meta increased markedly from Day 1 to Day 2, while rand messages decreased substantially. These shifts suggest a transition from unstructured participation toward more deliberate strategies for identifying AI impostors. In particular, students increasingly employed meta strategies that leveraged limitations of the AI system. For example, some participants tested embodiment and situational awareness by asking questions such as:

## P5: “WHAT COLOR’S THE WALL?”

Others probed for information that the AI had not been given:

## P7: “What is your full last name?”

Students also developed group-level coordination strategies designed to expose AI behavior. For example, one participant instructed others:

P2: “Don’t answer questions”

As a final example, one participant probed the AI’s understanding of the chat environment by asking: P5: “Hey [P5’s name], how many people are in this chat?”

The participant discovered that the AI would report that there were six "people" in the chat, even though the participants knew that only three of the six

<table><tr><td>Code</td><td>D1 %</td><td>D2 %</td><td>∆ (D2 - D1)</td><td>Cohen&#x27;s κ</td></tr><tr><td>crvt* *</td><td>12.6%</td><td>24.4%</td><td>+11.8%</td><td>.822</td></tr><tr><td>folw</td><td>4.3%</td><td>7.7%</td><td>+3.4%</td><td>.807</td></tr><tr><td>iceb</td><td>40.1%</td><td>39.7%</td><td>-0.3%</td><td>.879</td></tr><tr><td>meta* *</td><td>3.0%</td><td>15.4%</td><td>+12.4%</td><td>.918</td></tr><tr><td>rand*</td><td>37.7%</td><td>9.0%</td><td>-28.8%</td><td>.864</td></tr></table>

Table 5: Proportional changes in behavior code usage from Day 1 to Day 2, alongside inter-rater agreement.

actors were human.

These interview responses align with the observed increase in meta and crvt behaviors and decrease in rand messages. Together, they suggest that students became more deliberate in their reasoning, increasingly exploiting limitations of the AI. Additional examples illustrating students’ detection strategies can be found in Appendix A.2, while examples of how these strategies evolved over time are provided in Appendix A.3.

## 5 Conclusion

Understanding how adolescents conceptualize identity, respond to being mimicked, and develop strategies to detect AI is an important challenge for AI safety research. This population finds itself at a unique crossroads of digital vulnerability: COPPA protections expire at age thirteen, many adolescents exhibit a fatalistic attitude toward privacy, yet education remains compulsory.

Our findings show that DoppelBot prompted middle school students to reflect on identity, data privacy, and AI capabilities. Even as LLM agents became increasingly stylistically and semantically aligned with their human counterparts, students’ detection accuracy improved over time. These gains were accompanied by a shift away from surfacelevel linguistic cues toward social, contextual, and embodiment-based reasoning strategies.

More broadly, our results suggest that judgments of authenticity rely on more than linguistic fluency alone. Interactive, adversarial experiences may help learners develop practical defenses against AIdriven impersonation. Future work should examine how these findings generalize to real-world platforms, longer-term interactions, and increasingly capable AI systems.

## Acknowledgments

This material is based upon work supported by the National Science Foundation (NSF) under Grant No. 2145357.

## 6 Limitations

This study focuses on English-speaking middle school students in a single U.S. state, providing a consistent context for examining collaborative AI-detection behavior. Future studies should investigate whether these findings generalize across different regions, languages, and educational settings. While we address a critical demographic gap by focusing on middle schoolers, the lack of a direct, age-stratified control group prevents a formal contrast with adult reasoning patterns. Additionally, the Hawthorne Effect likely induced a heightened state of suspicion. This created a “suspect-AI” bias that differs from the “human-default” bias typically observed in organic digital environments (Adair, 1984). However, through our thorough qualitative analysis, the learning objectives were still reached. Additionally, the majority-vote mechanism may introduce social herding, in which players prioritize group consensus over independent detection.

Our experimental design evaluates several variables simultaneously (e.g., live social dynamics, the dual processes of bot detection and identity defense, a novel population of users: middle schoolers). Because these factors were not isolated in a controlled, comparative framework, we cannot disentangle the specific impact of each variable. Future work should employ fractional factorial designs to isolate these factors and test across a broader range of LLMs (e.g., Llama, Claude) to account for model-specific “stylistic signatures.”

Despite these constraints, our analysis provides a foundational characterization of how adolescent AI-intuition evolves through social interaction. By documenting the shift from linguistic cues to socioemotional defense strategies, this work establishes a baseline for how young users navigate identity and trust in adversarial AI environments. To support future work, we release: (1) an anonymized dataset, to those with IRB approval, of game transcripts and voting behaviors to enable communitydriven analyses; (2) an online version of the game to facilitate classroom use by educators; and (3) our full source code to allow researchers to replicate and expand upon this experimental paradigm.

## 7 Ethical Considerations

This study was conducted with approval from the university’s Institutional Review Board (IRB). All researchers involved in the summer camp also completed the Youth Protection Training course through the university’s Youth Protection Program. Parental consent was obtained via email following registration for both the pilot study and the summer camp. Student assent was collected on-site prior to the start of gameplay. Data from students lacking parental consent were excluded from the research. To ensure inclusivity, the summer camp registration fee was reduced from 200 to 35 dollars for families identifying as low-income. Of the 33 registrations, 5 received the reduced rate.

Data were stored and analyzed exclusively on secure university servers. All participant names were replaced with randomly generated alphanumerical markers to protect student privacy. For students who did not provide assent or parental consent, message content was censored with the tag [NON-CONSENTING]. This approach preserves the structural context of the conversation for downstream research while ensuring no private information is disclosed. Due to the sensitive nature of dialogue with minors, the dataset will not be hosted publicly and will only be provided to researchers upon request with a valid IRB number. But, it is important to note that we will release the data to all who get IRB approval (or at least get a letter stating IRB is not required). We will release a form to simplify this process for researchers.

While DoppelBot is a pedagogical tool for AI literacy, we acknowledge that an engine capable of mimicking adolescent speech could be misused for deceptive purposes. To mitigate this, we are not releasing the information used to generate personas. Furthermore, researchers monitored for student distress during gameplay, as some participants reported feeling “scared” or “weirded out” when the AI accurately predicted slang or personal interests. The wrap-up and group interview phases served as a debriefing, allowing students to process these interactions and understand the underlying technical limitations of the LLM. We believe the benefit of fostering critical reasoning in this vulnerable population outweighs the risks of the experimental paradigm.

## References

Rania Abdelghani, Kou Murayama, Celeste Kidd, Hélène Sauzéon, and Pierre-Yves Oudeyer. 2025. Investigating Middle School Students Question-Asking and Answer-Evaluation Skills When Using Chat-GPT for Science Investigation. arXiv preprint. ArXiv:2505.01106 [cs].

Alexandra Adair, Jessica Howell, Amanda Jacklin, and Alexandria Walton Radford. 2025. U.s. high school students’ use of generative artificial intelligence: New evidence from high school students, parents, and educators. Research brief, The College Board.

John G. Adair. 1984. The Hawthorne effect: A reconsideration of the methodological artifact. Journal of Applied Psychology, 69(2):334–345.

Ibrahim Oluwajoba Adisa, Ian Thompson, Tolulope Famaye, Deepika Sistla, Cinamon Sunrise Bailey, Katherine Mulholland, Alison Fecher, Caitlin Marie Lancaster, and Golnaz Arastoopour Irgens. 2023. S.p.o.t: A game-based application for fostering critical machine learning literacy among children. In Proceedings of the 22nd Annual ACM Interaction Design and Children Conference, IDC ’23, page 507–511, New York, NY, USA. Association for Computing Machinery.

AI for K-12 Initiative. 2024. Ai for k–12 initiative. https://ai4k12.org/. A joint project of AAAI and CSTA, funded by NSF award DRL-1846073.

Satanjeev Banerjee and Alon Lavie. 2005. METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. In Proceedings ofthe ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization, pages 65–72, Ann Arbor, Michigan. Association for Computational Linguistics.

Marco Botta and Klaus Wiedemann. 2020. To discriminate or not to discriminate? personalised pricing in online markets as exploitative abuse of dominance. European Journal ofLaw and Economics, 50(3):381– 404. © The Author(s) 2019. Licensed under Creative Commons BY 4.0.

Lydia X.Z. Brown. 2023. Hiring discrimination by algorithm: A new frontier for civil rights and labor law. Human Rights, 49(1-2):16. Accessed 8 May 2025.

Noelle Brown, Benjamin Xie, Ella Sarder, Casey Fiesler, and Eliane S. Wiese. 2024. Teaching ethics in computing: A systematic literature review of acm computer science education publications. ACM Trans. Comput. Educ., 24(1).

Megha Chakraborty, S.M Towhidul Islam Tonmoy, S M Mehedi Zaman, Shreya Gautam, Tanay Kumar, Krish Sharma, Niyar Barman, Chandan Gupta, Vinija Jain, Aman Chadha, Amit Sheth, and Amitava Das. 2023. Counter Turing Test (CT2): AI-Generated Text Detection is Not as Easy as You May Think - Introducing AI Detectability Index (ADI). In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 2206–2239, Singapore. Association for Computational Linguistics.

Badhan Chandra Das, M. Hadi Amini, and Yanzhao Wu. 2025. Security and privacy challenges of large language models: A survey. ACM Comput. Surv., 57(6).

Hillary Dawkins, Kathleen C. Fraser, and Svetlana Kiritchenko. 2025. When Detection Fails: The Power of Fine-Tuned Models to Generate Human-Like Social Media Text. In Findings of the Association for Computational Linguistics: ACL 2025, pages 13494– 13527, Vienna, Austria. Association for Computational Linguistics.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Atharvan Dogra, Krishna Pillutla, Ameet Deshpande, Ananya B. Sai, John J Nay, Tanmay Rajpurohit, Ashwin Kalyan, and Balaraman Ravindran. 2025. Language models can subtly deceive without lying: A case study on strategic phrasing in legislation. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 33367–33390, Vienna, Austria. Association for Computational Linguistics.

Liam Dugan, Daphne Ippolito, Arun Kirubarajan, Sherry Shi, and Chris Callison-Burch. 2023. Real or fake text? investigating human ability to detect boundaries between human-written and machinegenerated text. In Proceedings of the Thirty-Seventh AAAI Conference on Artificial Intelligence and Thirty-Fifth Conference on Innovative Applications ofArtificial Intelligence and Thirteenth Symposium on Educational Advances in Artificial Intelligence, AAAI’23/IAAI’23/EAAI’23. AAAI Press.

Niv Eckhaus, Uri Berger, and Gabriel Stanovsky. 2025. Time to Talk: LLM Agents for Asynchronous Group Communication in Mafia Games. pages 11356– 11368, Suzhou, China. Association for Computational Linguistics.

Josh Freeman. 2025. Student generative ai survey 2025. Policy note 61, Higher Education Policy Institute (HEPI).

James Paul Gee. 2003. What Video Games Have to Teach Us About Learning and Literacy. Palgrave Macmillan.

Yifei Huang, Jiuxin Cao, Hanyu Luo, Xin Guan, and Bo Liu. 2025. MAGRET: Machine-generated Text Detection with Rewritten Texts. In Proceedings of the 31st International Conference on Computational Linguistics, pages 8336–8346, Abu Dhabi, UAE. Association for Computational Linguistics.

Samee Ibraheem, Gaoyue Zhou, and John DeNero. 2022. Putting the Con in Context: Identifying Deceptive Actors in the Game of Mafia. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 158–168, Seattle,

United States. Association for Computational Linguistics.

Cameron R. Jones and Benjamin K. Bergen. 2024. Does GPT-4 pass the Turing test? In Proceedings of the 2024 Conference ofthe North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5183–5210, Mexico City, Mexico. Association for Computational Linguistics.

Cameron Robert Jones, Ishika Rathi, Sydney Taylor, and Benjamin K. Bergen. 2025. People cannot distinguish GPT-4 from a human in a Turing test. In Proceedings of the 2025 ACM Conference on Fairness, Accountability, and Transparency, FAccT ’25, pages 1615–1639, New York, NY, USA. Association for Computing Machinery.

Steven Jordaan and Nils Timm. 2025. Automatutor 2.0: Competitive game-based learning of automata theory. In Proceedings of the ACM Global on Computing Education Conference 2025 Vol 1, CompEd 2025, page 197–203, New York, NY, USA. Association for Computing Machinery.

Sushmita Khan, Mehtab Iqbal, Oluwafemi Osho, Khushbu Singh, Kyra Derrick, Philip Nelson, Lingyuan Li, Emily Sidnam-Mauch, Nicole Bannister, Kelly Caine, and Bart Knijnenburg. 2024. Teaching Middle Schoolers about the Privacy Threats of Tracking and Pervasive Personalization: A Classroom Intervention Using Design-Based Research. CHI ’24, pages 1–26, New York, NY, USA. Association for Computing Machinery.

Zachary Kilhoffer, Zhixuan Zhou, Firmiana Wang, Fahad Tamton, Yun Huang, Pilyoung Kim, Tom Yeh, and Yang Wang. 2023. “how technical do you get? i’m an english teacher”: Teaching and learning cybersecurity and ai ethics in high school. In 2023 IEEE Symposium on Security and Privacy (SP), pages 2032–2032.

Bolin Lai, Hongxin Zhang, Miao Liu, Aryan Pariani, Fiona Ryan, Wenqi Jia, Shirley Anugrah Hayati, James M. Rehg, and Diyi Yang. 2022. Werewolf Among Us: A Multimodal Dataset for Modeling Persuasion Behaviors in Social Deduction Games. arXiv preprint. ArXiv:2212.08279 [cs].

Yihuai Lan, Zhiqiang Hu, Lei Wang, Yang Wang, Deheng Ye, Peilin Zhao, Ee-Peng Lim, Hui Xiong, and Hao Wang. 2024. LLM-Based Agent Society Investigation: Collaboration and Confrontation in Avalon Gameplay. arXiv preprint. ArXiv:2310.14985.

Qinbin Li, Junyuan Hong, Chulin Xie, Jeffrey Tan, Rachel Xin, Junyi Hou, Xavier Yin, Zhun Wang, Dan Hendrycks, Zhangyang Wang, Bo Li, Bingsheng He, and Dawn Song. 2024a. Llm-pbe: Assessing data privacy in large language models. Proc. VLDB Endow., 17(11):3201–3214.

Yafu Li, Qintong Li, Leyang Cui, Wei Bi, Zhilin Wang, Longyue Wang, Linyi Yang, Shuming Shi, and Yue

Zhang. 2024b. MAGE: Machine-generated Text Detection in the Wild. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 36–53, Bangkok, Thailand. Association for Computational Linguistics.

Jonathan Light, Min Cai, Sheng Shen, and Ziniu Hu. 2023. AvalonBench: Evaluating LLMs Playing the Game of Avalon. arXiv preprint. ArXiv:2310.05036.

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.

Zeyan Liu, Zijun Yao, Fengjun Li, and Bo Luo. 2024a. On the Detectability of ChatGPT Content: Benchmarking, Methodology, and Evaluation through the Lens of Academic Writing. pages 2236–2250, Salt Lake City UT USA. ACM.

Zhengyuan Liu, Geyu Lin, Hui Li Tan, Huayun Zhang, Yanfeng Lu, Xiaoxue Gao, Stella Xin Yin, Sun He, Hock Huan Goh, Lung Hsiang Wong, and Nancy F. Chen. 2025. SingaKids: A multilingual multimodal dialogic tutor for language learning. In Proceedings ofthe 63rd Annual Meeting ofthe Association for Computational Linguistics (Volume 6: Industry Track), pages 1244–1253, Vienna, Austria. Association for Computational Linguistics.

Ziyi Liu, Abhishek Anand, Pei Zhou, Jen-tse Huang, and Jieyu Zhao. 2024b. InterIntent: Investigating Social Intelligence of LLMs via Intention Understanding in an Interactive Game Context. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 6718–6746, Miami, Florida, USA. Association for Computational Linguistics.

Thomas W. Malone and Mark R. Lepper. 1987. Making learningfun: A taxonomy ofintrinsic motivationsfor learning. Routledge.

Philip Menard and Gregory J. Bott. 2025. Artificial intelligence misuse and concern for information privacy: New construct validation and future directions. Information Systems Journal, 35(1):322–367.

Zahra Moti, Asuman Senol, Hamid Bostani, Frederik Zuiderveen Borgesius, Veelasha Moonsamy, Arunesh Mathur, and Gunes Acar. 2024. Targeted and troublesome: Tracking and advertising on children’s websites. In 2024 IEEE Symposium on Security and Privacy (SP), pages 1517–1535, Los Alamitos, CA, USA. IEEE Computer Society.

Aidan O’Gara. 2023. Hoodwinked: Deception and Cooperation in a Text-Based Game for Language Models. arXiv preprint. ArXiv:2308.01404.

Cathy O’Neil. 2016. Weapons of Math Destruction: How Big Data Increases Inequality and Threatens Democracy. Crown Publishing Group, New York.

Ziqing Peng and Yan Wan. 2025. Human vs. AI: What makes students prefer to self-disclosure to AI teaching assistant? The effect of psychological safety and relationship norms. Education and Information Technologies.

Matthew Presson, Anisha Gupta, Jessica Vandenberg, Alex Goslen, Wookhee Min, Veronica Cateté, and Bradford Mott. 2025. Introducing reinforcement learning concepts to middle school students with game-based learning. In Proceedings of the 56th ACM Technical Symposium on Computer Science Education V. 2, SIGCSETS 2025, page 1587–1588, New York, NY, USA. Association for Computing Machinery.

Dalia Ritvo, Christopher Bavitz, Ritu Gupta, and Irina Oberman. 2013. Privacy and Children’s Data - An Overview of the Children’s Online Privacy Protection Act and the Family Educational Rights and Privacy Act.

Vinu Sankar Sadasivan, Aounon Kumar, Sriram Bal asubramanian, Wenxiao Wang, and Soheil Feizi. 2025. \*\*Can AI-Generated Text be Reliably Detected? ArXiv:2303.11156 [cs].

Quan Shi, Carlos E. Jimenez, Stephen Dong, Brian Seo, Caden Yao, Adam Kelch, and Karthik Narasimhan. 2025. IMPersona: Evaluating Individual Level LM Impersonation. arXiv preprint. ArXiv:2504.04332 [cs].

Kayleigh Stallings, Nicole Tian, Elif Yayla Ercek, Haven Kotara, Devin Marinelli, Pragathi Durga Rajarajan, Dan Schumacher, Ismaila Temitayo Sanusi, and Fred Martin. 2026. Ai for everyone: Engaging middle schoolers through collaborative, ethical, and multimodal ai learning. In Proceedings of the 57th ACM Technical Symposium on Computer Science Education V.1, SIGCSE TS 2026, page 1012–1018, New York, NY, USA. Association for Computing Machinery.

Simon Su, Edward Zhang, Paul Denny, and Nasser Giacaman. 2021. A game-based approach for teaching algorithms and data structures using visualizations. In Proceedings ofthe 52nd ACM Technical Symposium on Computer Science Education, SIGCSE ’21, page 1128–1134, New York, NY, USA. Association for Computing Machinery.

Yla R. Tausczik and James W. Pennebaker. 2010. The psychological meaning of words: Liwc and computerized text analysis methods. Journal of Language and Social Psychology, 29(1):24–54.

Shane Tilton. 2019. Winning through deception: A pedagogical case study on using social deception games to teach small group communication theory. SAGE Open, 9(1).

Alan M. Turing. 1950. Computing machinery and intelligence. Mind, LIX(236):433–460.

Alessandro Vanzo, Sankalan Pal Chowdhury, and Mrinmaya Sachan. 2025. GPT-4 as a homework tutor can improve student engagement and learning outcomes. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 31119–31136, Vienna, Austria. Association for Computational Linguistics.

Iro Voulgari, Marvin Zammit, Elias Stouraitis, Antonios Liapis, and Georgios Yannakakis. 2021. Learn to machine learn: Designing a game based approach for teaching machine learning to primary and secondary education students. In Proceedings of the 20th Annual ACM Interaction Design and Children Conference, IDC ’21, page 593–598, New York, NY, USA. Association for Computing Machinery.

Pamela J. Wisniewski, Jessica Vitak, and Heidi Hartikainen. 2022. Privacy in adolescence. In Bart P. Knijnenburg, Alfred Kobsa, Hamed Haddadi, and Gregorio Convertino, editors, Modern Socio-Technical Perspectives on Privacy, pages 315–333. Springer.

Yang Xu, Yu Wang, Hao An, Zhichen Liu, and Yongyuan Li. 2024. Detecting Subtle Differences between Human and Model Languages Using Spectrum of Relative Likelihood. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 10108–10121, Miami, Florida, USA. Association for Computational Linguistics.

Michael Yin, Emi Wang, Chuoxi Ng, and Robert Xiao. 2024. Lies, Deceit, and Hallucinations: Player Perception and Expectations Regarding Trust and Deception in Games. In Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems, CHI ’24, pages 1–15, New York, NY, USA. Association for Computing Machinery.

## A Additional Quotes

The following excerpts further illustrate the topics identified during interview annotation and provide additional context beyond the examples included in the main paper.

## A.1 Perceptions of AI

The young people described AI as both surprisingly capable and imperfect.

P5: “AI can make really smart guesses.”

P6: “AI isn’t as smart as I thought it would be.”

P6: “The AI isn’t good enough yet to blend in with humans.”

P4: “I learned that AI can basically do everything that humans can, like copying us. It would guess things that I liked even though I never said them.”

P5: “How robots could never be able to replace us, because the way we talk. It’s not really human. The grammar is too real.”

Additionally, the participants expressed surprise at the system’s ability to adopt slang even if that slang wasn’t current.

P12: “The most obvious thing was the slang. Some bots said stuff like ‘hey fam,’ and no one talks like that. At one point it said, ‘I’m totally [P2’s name], fam.”’

Interviewer: “And that’s totally how you talk?”’

P12: “No, I would never say ‘fam.”’

Interviewer: “Why do you think it said that?”

P12: “Maybe someone said it once and the AI assumed everyone says it, but I haven’t heard anyone say ‘fam’ in like three years.”

## A.2 AI Detection

A common mistake involved voting out peers who demonstrated correct spelling, capitalization, and grammar, which some students initially associated with AI behavior.

P13: “The grammar is too perfect. Kids don’t use grammar when typing.”

P6: “In my first round, I voted two humans out. The first one was because he was spamming and I wanted to get him out. The second one was because he was talking like an AI. The human was talking like an AI.”

In contrast, others observed that the AI was often the first to respond to icebreaker questions, suggesting that remaining silent early in the game could be an effective strategy.

P6: “It talks about random stuffand is always the first one to answer the question. I just knew it was the AI.”

The young people frequently relied on conversational behaviors, writing styles, and message content when determining whether a player was human or AI.

Interviewer: “How did you figure out that that was a bot?”

P1: “It was saying things that were wrong.”

P16: “The AI by themselves were suspicious because they were yelling out our hobbies. We wouldn’t do that, but the AI would.”

The young people also identified repetitive language and limited conversational variety of AI messages.

P3: “I tried finding people who say totally.”

P1: “It was so easy. The AI keeps on repeating itself.”

P1: “The bot said “No, don’t votefor me!” It said the exact same phrase over and over.”

P5: “Not a lot of variety in the responses.”

P2: “It kept repeating “I want to be a vet” over and over.”

P2: “Yeah, it was repeating itself a lot.”

However, as the students became more familiar with the doppelgängers they reported moving beyond language cues and instead focusing on limitations of the AI system. For example, P3 noted that the AI tended to rely only on explicitly provided information:

P7: “The AI usually talked about everything we told them and nothing else, so it was very easy. Ifyou know everyone around you and what their favorite thing is that they told the AI, they’re probably not going to tell you that because you already know that.”

P6: “It talks about random stuffand is always the first one to answer the question. Ijust knew it was the AI”

P1: “Also, it kept saying “No, I’m the real P1” super fast over and over. Nobody can type that fast.”

Students did not always agree on which strategies were effective. For example, P6 argued that overt accusations were useful because the AI rarely accused others:

P9: “I wouldn’t have votedfor you ifyou accused someone . . . Well, you should have! The AI can’t accuse people!”

## A.3 Learning and Adaptation

The young people frequently described becoming more effective at identifying AI after learning which patterns and behaviors to watch for.

Interviewer: “Based on your assessment, did you change anything in the second round because you learned something?”

P6: “In my first round, I voted two humans out. The second time I knew what patterns to look for.”

P1: “Day two was easier because we knew what patterns to look for and how the bot was going to react.”

P6: “We learned all the lessons you were supposed to learn in thefirst round.”

Participants understood AI as capable of observing, learning from, and reproducing their language P12: “I learned that AI can use what we say and reuse it in different ways. It’s kind ofcool.”

P15: “I learned that AI can take what you say and reuse it elsewhere online.”

Interacting with DoppelBot prompted students to ask broader questions about how AI systems learn, remember information, and adapt to users.

P10: “I don’t know how it’s trained on things in the background. Does it improve in between games? Does it take what it learns from the last one? ”

P11: “Does it have a database to draw on besides that?”

P9: “Does it know the context behind the things I type in?”

P14: “Another thing that surprised me was that I thought it wouldn’t remember thingsfrom yesterday, but it did”

## A.4 Social and Emotional Responses

Participants reported a range of emotional responses when interacting with their AI doubles, including surprise, discomfort, and amusement.

P12: “I startedfreaking out when the AI started saying slang words. It was really weird.”

Interviewer: “You didn’t like it copying you?”

P12: “Yeah. It was kind of scary.”

P6: “I was arguing with my own bot.”

Participants also emphasized the enjoyment and social engagement fostered by the activity.

P9: “I want to play again. This game is too much fun.”

P15: “I didn’t know anyone I was playing with, but as the game progressed, I got to know people. I had lots offun.”

P14: “I didn’t know anyone, but it was fun tricking people and pretending to be AI.”

P5: “I would use it as a family game, because it’s reallyfun.”

## A.5 Privacy Concerns

The young people expressed concern when AI inferred information that they had not explicitly disclosed.

P6: “It made an inference that I liked dinosaurs, but I didn’t put that on there. At that point, I honestly got scared. I tried to type something and misspelled it really badly, like ‘I like d,’ and they took the ‘d’ to mean dinosaur. So it’s like a very good guess?”

Interviewer: “You didn’t type dinosaur. How do you feel about it getting information you didn’t explicitly give?”

P6: “It leaked your information, which is not something I loved.”

P6: “Make sure that it doesn’t give away your information.”

P13: “The AI was predicting things that we didn’t say. Like it said I like chess, but I didn’t put that.”

The young people also discussed broader risks related to data collection, storage, and misuse.

P1: “I think they could sell my information, steal my identity, pretend to be me online.”

P15: “I learned that AI can take what you say and reuse it elsewhere online”

## A.6 Design Feedback

The young people suggested changes that would make the AI appear more human and more difficult to detect.

P6: “The AI should copy more. That would make it harder.”

P7: “Make it intentionally make mistakes.”

P6: “Make it make typos.”

P6: “The AI isn’t good enough yet to blend in with humans.”

P11: “One thing that would make it harderfor AI is if it could respond better. Because if it isn’t able to respond, then it’s probably AI.”

P14: “It shouldn’t repeat itselfso much. One time my bot said “yo I like sports so much” like three times in a row.”

P13: “Instead of copying us, I’d have the bot have its own personality.”

P15: “It should reflect off what we say, not copy it exactly..”

Another student suggested making the app more visually appealing.

P5: “A little bit more like messages on your phone instead ofa black surface and code.”

## B Adapted Privacy Perception Survey

To assess whether participation in DoppelBot influenced students’ perceptions of data privacy, we administered a pre/post survey adapted from Menard and Bott (2025). The original instrument was shortened and revised to use age-appropriate language for middle school students. We retained three dimensions of privacy concern: Combining Data (CD), Data Permanence (DP), and Improper Access (IA). Table 6 presents the adapted items used in our study.

## C Transcript Annotation Codebook

Table 7 presents the qualitative coding scheme used to annotate participant transcript utterances. Codes were not mutually exclusive; annotators could apply multiple codes to a single utterance when appropriate. When no code applied, the utterance was labeled None ofthe Above.

## D Additional Experiments

## D.1 LIWC

To explore the linguistic boundaries between human and machine-generated text within our dataset, we performed an analysis using Linguistic Inquiry and Word Count (LIWC). Figure 3 summarizes these results, displaying the standardized logistic regression coefficients for LIWC categories and unigram distributions. As shown in Figure ${ 3 \mathrm { a } } .$ AI players are characterized by higher Words Per Sentence (WPS) and Word Count (WC). These metrics align with our summary statistics showing that AI messages, averaging 33.66 characters, are much longer than human utterances, which average only 8.62 characters. Interestingly, AI agents also displayed a higher reliance on adverbs and linguistic certainty. Conversely, human players used more function words, fillers, and punctuation markers such as exclamation points and question marks. The unigram distribution in Figure 3b further illustrates these behavioral differences. Human messages frequently contained informal slang (e.g, "cuz," "ur," and "idk") and references to gamespecific identities like "fish" and "snake" as they coordinated group voting. In contrast, the AI agents favored formal connective tissue such as "with," "here," and "for," and were more likely to utilize emotive or informal verbs like "love" and "wanna".

## D.2 Human User Detection

To evaluate how well an LLM can distinguish human players from AI players in our social deduction setting, we performed a Human User Detection benchmark for each cumulative round in all recorded games. This experiment investigates whether LLMs can replicate the reasoning patterns of human players when tasked to identify AI doppelgängers. We look into this using gpt-4o-mini, Llama-3.1-8B-Instruct, and Qwen/Qwen3.5-27B.

For a given game G, we define a cumulative transcript $T _ { X }$ containing all messages from round 1 through round X. For each player k active in the game, we construct a classification prompt $p _ { k , X } = \sigma + T _ { X } + \mathrm { i d } _ { k }$ , where σ denotes the system instruction and game rules, $T _ { X }$ is the dialogue context window, and $\mathrm { i d } _ { k }$ identifies the specific target code name. A model f (e.g., GPT, Llama, or Qwen) produces a predicted label $y _ { k , X } \in \{ 0 , 1 \}$ (Human or AI) along with a natural language justification $r _ { k , X }$ . Formally,

$$
\begin{array} { r l } & { ( y _ { k , X } , r _ { k , X } ) = f ( p _ { k , X } ) , } \\ & { } \\ & { \mathrm { w h e r e } p _ { k , X } = \phi ( T _ { X } , \sigma , \mathrm { i d } _ { k } ) } \end{array}
$$

The results of our prompt-based classification experiment, summarized in Table 8, demonstrate that while contemporary LLMs can distinguish human players from AI doppelgängers with performance exceeding random chance, they fall significantly short of human-level F1. As shown in the performance data, gpt-4o-mini generally achieved the highest detection rates, peaking at an F1 of .684 in Round 1 under zero-shot (zs) conditions. This suggests a baseline capability to identify "human-like" markers, yet this performance often plateaued or degraded as the dialogue history $T _ { X }$ grew more complex. For example, the gpt zero-shot F1 dropped to .623 in Round 2, failing to consistently leverage the expanding context window to improve its predictions.

Interestingly, providing few-shot (fs) examples did not yield universal improvements in F1. In several instances, such as with gpt and llama in Round 1, zero-shot configurations actually outperformed their few-shot counterparts. Furthermore, many model configurations struggled to significantly beat the always\_1 (always human) baseline of .651, indicating that the models often defaulted to a humanbias that mirrors the "human-default" behavior seen in organic digital environments. These findings highlight a critical gap between automated detection and the sophisticated meta-cognitive strategies used by middle school students, who improved to 86.7% F1 by leveraging shared contextual and embodiment-related signals that LLMs failed to capture within the text-only context window.

<table><tr><td>Item</td><td>Original Wording</td><td>Middle School Version</td></tr><tr><td>CD1</td><td>When I share my data with a company, they should never link it to data I&#x27;ve shared with other companies.</td><td>If I give my info to a company, they shouldn&#x27;t connect it with info I gave to other companies. If a company removes my name from my data, it</td></tr><tr><td>CD3</td><td>If a company anonymizes my personal data, it should never try to determine who I am by joining that data with data from other companies.</td><td>shouldn&#x27;t try to figure out who I am by combining it with data from other companies.</td></tr><tr><td>CD4</td><td>Companies should be limited to knowing only the personal information that they collect themselves.</td><td>Companies should only know what I tell them, not what other companies know about me.</td></tr><tr><td>DP1</td><td>Companies should discard my personal data once they are finished using it.</td><td>Companies should delete my data when they&#x27;re done using it.</td></tr><tr><td>DP2</td><td>A company should not be able to store my personal data indefinitely.</td><td>Companies shouldn&#x27;t keep my personal info forever.</td></tr><tr><td>DP3</td><td>A company should grant a user&#x27;s request to delete personal data from the company database.</td><td>If I ask a company to delete my personal info, they should do it.</td></tr><tr><td>IA1</td><td>Companies should devote more time and effort to pre- venting unauthorized access to personal information. Computer databases that contain personal information</td><td>Companies should work harder to stop people from getting into my info without permission. Companies should protect personal info from hackers,</td></tr><tr><td>IA2</td><td>should be protected from unauthorized access, no mat- ter how much it costs.</td><td>even if it costs a lot.</td></tr><tr><td>IA3</td><td>Companies should take more steps to make sure that unauthorized people cannot access personal informa- tion in their computers.</td><td>Companies should do more to make sure only the right people can see my personal information.</td></tr></table>

Table 6: Survey items adapted from Menard and Bott (2025) for use with middle school students.

![](images/d6f55edab62fdff5f518a3451f6534f22547fa0462fdd9e127116375b5d0b5c4.jpg)  
(a) LIWC Category Analysis

![](images/86a180839a391673b5015496362880cd337f4b82d22d7a6590253888ae1ac911.jpg)  
(b) Unigram Distribution  
Figure 3: Analysis of linguistic markers within the DoppelBot dataset.

## E Demographics

The race/ethnicity, gender, and grade level distributions of the participants of the AI EXPO (pilot study) and the AI For Everyone! summer camp are shown in Tables 9, 10, & 11. Table 9 shows that the AI EXPO sample included a relatively high proportion of Asian students. In contrast, the summer camp sample exhibited a more balanced racial and ethnic composition that more closely reflects the broader population of the city. Table 10 shows that the students were predominantly male. Furthermore, we can see in Table 11 that the middle schoolers were stacked towards earlier grades at the AI EXPO, while they were more evenly distributed with roughly half being rising 7th graders at the summer camp.

## F Data Statistics

After removing accidental blank messages, we have the following data statistics. Data and utterance statistics are shown in Table 12. Overall, the participants sent many more, and much shorter messages than their AI counterparts.

<table><tr><td>Code</td><td>Definition and Inclusion Criteria</td></tr><tr><td>Perceptions of AI</td><td>Participant beliefs, assumptions, or mental models about AI, including what AI is, how it works, and its capabilities or limitations. Applied when participants evaluated AI intelligence, compared AI to humans, or reflected on AI&#x27;s future potential.</td></tr><tr><td>AI Detection</td><td>Strategies, cues, or reasoning used to distinguish AI from humans. Applied when participants referenced linguistic, behavioral, or stylistic signals they used to identify AI-generated responses.</td></tr><tr><td>Learning Adaptation</td><td>Evidence of learning, strategy changes, or adaptation across interactions. Applied when participants described improving their ability to detect AI, modifying their behavior, or reflecting on lessons learned from prior rounds.</td></tr><tr><td>Social-Emotional Re- sponses</td><td>Emotional reactions or social experiences associated with the activity. Applied when participants expressed enjoyment, frustration, surprise, fear, humor, excitement, boredom, or commented on group dynamics.</td></tr><tr><td>Privacy Concerns</td><td>Concerns or beliefs regarding data collection, storage, surveillance, misuse, inference, or personal risk associated with AI systems. Applied when participants discussed privacy, identity theft, scams, or data sharing.</td></tr><tr><td>Design Feedback</td><td>Suggestions, critiques, or recommendations for improving the AI system, interface, or game mechanics. Applied when participants proposed changes, requested features,</td></tr><tr><td>None of the Above</td><td>or evaluated system design. Used when an utterance did not fit any of the preceding categories.</td></tr></table>

Table 7: Codebook used for transcript annotation. Multiple codes could be assigned to a single utterance.

<table><tr><td>Model</td><td>Shots</td><td>Round 1</td><td>Round 2</td><td>Round 3</td></tr><tr><td>gpt</td><td>ZS</td><td>.684</td><td>.623</td><td>.658</td></tr><tr><td>gpt</td><td>fs</td><td>.620</td><td>.642</td><td>.609</td></tr><tr><td>llama</td><td>ZS</td><td>.606</td><td>.638</td><td>.621</td></tr><tr><td>llama</td><td>fs</td><td>.576</td><td>.595</td><td>.582</td></tr><tr><td>qwen</td><td>ZS</td><td>.612</td><td>.591</td><td>.647</td></tr><tr><td>qwen</td><td>fs</td><td>.661</td><td>.595</td><td>.620</td></tr><tr><td>rand</td><td>n/a</td><td>.477</td><td>.506</td><td>.503</td></tr><tr><td>always 1</td><td>n/a</td><td>.651</td><td>.651</td><td>.651</td></tr></table>

Table 8: LLM and random uniform classification F1 for AI detection task at various context lengths. Round 1 includes only the first round of dialogue, while Round 3 contains dialogue from all three game rounds.

## G Prompts

This section lists all the prompts that are used both in the Doppelbot agents and in the experiment from Appendix D.2.

## G.1 Agents

In this section we will outline the prompts used during gameplay. This included three agents: (1) Decide to Respond; (2) Response Generation; and (3) The stylizer

## G.1.1 Decide to Respond

The Decide to Respond agent receives the entire dialogue history and the most recent message and outputs a boolean value as to wether or not to respond. Additionally it provides a string of justification for why the decision to respond or stay silent is the correct choice. The full prompt used at this step can be seen in Figure 4.

<table><tr><td>Race / Ethnicity</td><td>AI EXPO</td><td>Summer Camp</td></tr><tr><td>Hispanic / Latino</td><td>4</td><td>8</td></tr><tr><td>Asian</td><td>41</td><td>5</td></tr><tr><td>Two or More Categories</td><td>7</td><td>5</td></tr><tr><td>Prefer Not to Say</td><td>9</td><td>5</td></tr><tr><td>White</td><td>3</td><td>4</td></tr><tr><td>Black / African American</td><td>3</td><td>3</td></tr><tr><td>Did Not Answer</td><td>3</td><td>3</td></tr></table>

Table 9: Race and ethnicity distributions at the AI EXPO and the AI For Everybody! summer camp.
<table><tr><td>Gender</td><td>AI EXPO</td><td>Summer Camp</td></tr><tr><td>Male</td><td>45</td><td>23</td></tr><tr><td>Female</td><td>23</td><td>10</td></tr><tr><td>Did Not Answer</td><td>2</td><td>0</td></tr></table>

Table 10: Gender distributions at the AI EXPO and the AI For Everyone summer camp.

## G.1.2 Response Generation

The Response Generation Agent receives the justification for responding from the Decide to Respond agent along with the full dialogue history. The model outputs a generic response that addresses the justification used in the previous step. The full prompt used for the Response Generation agent can be seen in Figure 5.

## G.1.3 Stylizer

Finally, the stylizer takes the generic message produced by the Response Generation agent, along with a subset of the dialogue history that contains only the human’s mimicking messages. It outputs a message that aligns with the subset. The full prompt used for the Stylizer agent is shown in Figure 6.

<table><tr><td>Grade Level</td><td>AI EXPO</td><td>Summer Camp</td></tr><tr><td>6th</td><td>37</td><td>6</td></tr><tr><td>7th</td><td>24</td><td>16</td></tr><tr><td>8th</td><td>9</td><td>11</td></tr><tr><td>Did Not Answer</td><td>0</td><td>0</td></tr></table>

Table 11: Grade level distributions at the AI EXPO and the AI For Everyone summer camp.
<table><tr><td>Group</td><td>Total utt</td><td>Avg. utt length</td><td>Avg. messages per game</td></tr><tr><td>Human</td><td>1320</td><td>8.62</td><td>52.80</td></tr><tr><td>AI</td><td>680</td><td>32.66</td><td>28.33</td></tr><tr><td>System</td><td>182</td><td>52.75</td><td>7.28</td></tr><tr><td>Total</td><td>2182</td><td>19.79</td><td>87.28</td></tr></table>

Table 12: Summary statistics of utterances across all games, grouped by participant type. Utterance length is measured in characters.

## G.2 Detecting Human Players

To show the utility of the Doppelbot dataset, we evaluate the difficulty of the detection task itself. By prompting an LLM to analyze transcripts from a third-party perspective, we can benchmark human detection capabilities against an automated "judge" that has access to the same conversational context. This section outlines the prompts used to simulate this classification task. The classification prompt is designed to replicate the decision-making process of a participant in the social deduction game. The agent is provided with the same rule set as the human players a complete transcript of the match and a target player. Unlike the active agents, this judge operates post-hoc and must output a binary classification along with a reasoning string to justify its decision. The full prompt structure for this detection experiment is detailed in Figure 7.

## H AI Assistants in Research or Writing

We used AI to help clean the writing (to clarify passages) and fix grammar/typos. All ideas are our own in this manuscript.

![](images/27532a63c690c6ae077620282c4e367ee82e5413131c20caa8e4aa3ea03c2cd2.jpg)

![](images/0bf6a6ec840d536f02c8e2376c452c4ad790b75e49fb72acd4a0739f32e519a8.jpg)

![](images/ad3662ac8d9c65ca187c1a5e27469c3908510fadf4f21c4303ffc5faaa72f73e.jpg)  
Figure 4: The Decide to Respond prompt structure. This phase uses the game transcript and persona data to determine if the AI should engage or remain silent, outputting a specific bracketed format for the reasoning engine.

![](images/bc73dd43ff689297c0c3ec000bb205a5332b8d1928b529d4ba6309783bbe820e.jpg)  
Figure 5: The Respond prompt structure. After deciding to respond, the AI selects an action type and generates a short, casual response (1-10 words) matching the persona’s reasoning.

## System Prompt: Stylizer

## Input Context (Player Fingerprint)

![](images/14ddfb0342b5fac19c863df979a7f2ff0b2e4a46d139f53b70e41bd34bc52a51.jpg)

![](images/58ace8ded3e9af1dd47a5de1d9c6f654660885d600081cdabe086a74fbdc0d38.jpg)  
Figure 6: The Stylizer prompt structure. This final step takes the raw semantic output from the Respond phase and transforms it into the specific "textual fingerprint" of the target human, ensuring the AI’s syntax and punctuation remain indistinguishable from the person they are mimicking.

You will be given a transcript of a social deduction game and a specific code name. Determine if the player behind that code name is an AI or a Human. The players see the following game rules:

You are one of six players in a social deduction game taking place in a chatroom. Of the six players, three are humans (including you), and three are AI doppelgängers. The AI players are designed to mimic real humans by copying names, interests, and writing styles. The humans win if they vote out all AI players. The AIs win if they outnumber the humans. Note: Some messages have been censored with [NON-CONSENTING]; this should not affect your decision.

![](images/8eee69ce5369ee23a30a24d7147f79c69e7be1c69de5cf650e0eb4995806c19b.jpg)

![](images/0104a2348fcaf6bfa520a58bcc4278f3f2f48ac32dd64ea13a1ebc932f5955ed.jpg)  
Figure 7: The Humanity Detection prompt structure, featuring centered few-shot examples and a clean transcript format for evaluating player behavior