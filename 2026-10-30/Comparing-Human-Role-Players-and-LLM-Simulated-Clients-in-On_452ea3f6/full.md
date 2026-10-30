# Comparing Human Role-Players and LLM-Simulated Clients in Online Counselling Training: An Analysis of Counselling Patterns

Eric Rudolph, Philipp Steigerwald, Jens Albrecht

Technische Hochschule Nürnberg Georg Simon Ohm

Nürnberg, Germany

{eric.rudolph, philipp.steigerwald, jens.albrecht}@th-nuernberg.de

## ABSTRACT

This study investigates the capabilities of Large Language Models to simulate counselling clients in educational roleplays in comparison to human role-players. Initially, we recorded role-playing sessions, where novice counsellors interacted with human peers acting as clients, followed by role-plays between humans and clients simulated by Mistrals Mixtral 8x7b using 4-bit quantization. These interactions were analysed with a counselling communication pattern system at sentence level. We investigated two key questions: (1) to what extent LLM-generated responses replicate authentic conversational dynamics and (2) whether counsel lors’ communication behaviour difers when interacting with human versus LLM-simulated clients. The findings highlight both similarities and diferences in the application of counselling patterns across scenarios, showing the potential of LLM-based role-playing exercises to enhance counselling competencies and to identify areas for further refinement in virtual client simulations.

## Keywords

Large Language Models, Counselling, Natural Language Processing, Educational Data Mining, AI in Education, Counselling Training, Client Simulation

## 1. INTRODUCTION

Text-based online counselling has gained increasing prominence as a flexible and accessible alternative to traditional face-to-face therapy, especially in reaching diverse populations with unique needs [5, 13]. Within counsellor education, role-playing exercises are central to developing crucial communication competencies [15, 14], ofering trainees controlled, practice-oriented scenarios to explore, reflect on, and refine their skills [18, 20]. However, role-plays with human peers often face logistical constraints such as coordinating schedules and achieving consistent complexity across multiple sessions [22], while also being limited by the role-play partner’s availability, engagement level and ability to maintain character depth and consistency throughout the interactions [12].

With the ongoing advancements in large language models (LLMs), the prospect of simulating client roles using AIdriven conversation agents has become an attractive avenue [9, 7, 10]. In principle, these virtual clients ofer standardization, flexibility, and round-the-clock availability, allowing trainees to practice and experiment in a safe, repeatable environment [2, 19]. Despite the practical advantages, key questions remain regarding the authenticity of the learning experience [21]. Do LLM-generated clients exhibit realistic conversational dynamics, and do they elicit the same communication patterns from counsellors as human interlocutors would?

This paper examines text-based role-play counselling sessions, comparing interactions between human counsellors and human clients with those between human counsellors and LLM-simulated clients. We draw on a specialised categorisation framework for counselling communications, analysing the distribution and temporal evolution of specific interaction patterns in both settings. Our findings highlight significant similarities and diferences in how clients disclose their concerns and how counsellors respond and structure the session. In doing so, this study provides insights into the potential of LLM-based role-plays to augment

— and possibly transform — the way future counsellors are trained, while also identifying areas where further development is needed to narrow the gap between virtual and human clients.

## 2. RELATED WORK

The integration of LLMs into counselling education and training has gained increasing attention. In particular, AI-driven role-playing simulations are being explored as a means to enhance the learning experience of trainee counsellors by providing standardised, risk-free, and adaptive training environments.

A recent study examined the impact of role-play interactions between humans and an LLM-powered agent on educational outcomes [4]. This research compared Human-LLM role-play interactions with traditional Human-Human role-play settings, assessing their efectiveness in improving participants’ communication skills. The findings indicated that while LLMs can facilitate structured and engaging role-play scenarios, there are still diferences in spontaneity and contextual adaptation when compared to human interlocutors. A study leveraged GPT-4 to generate and analyse psychological counselling sessions through role-play interactions between human counsellors and LLM-simulated clients [6]. Professional counsellors evaluated the AI-generated responses against human client responses in identical counselling contexts. The findings demonstrated GPT-4’s capability to efectively replicate conversational patterns, suggesting potential for LLM-based training simulations, though challenges in response consistency and emotional authenticity indicated the need for further refinements. VirCo (Virtual Client for Online Counselling) demonstrates another practical application of LLMs in counsellor training [16, 17]. The system leverages an open-source LLM with persona descriptions and role-play transcripts, showing high coherence in evaluation while addressing privacy concerns through its architecture. A systematic review investigated the broader applications of LLMs in mental health education, particularly in psychotherapy training [3]. The analysis revealed that AI-driven simulations enable efective practice of communication strategies and provide immediate feedback in risk-free environments, though success depends on proper integration with established educational frameworks. Similarly, another study explored the opportunities and risks associated with deploying LLMs in counselling education [8]. While LLMs have the potential to provide consistent and data-driven training experiences, concerns regarding bias, ethical considerations, and misinformation remain. These challenges are particularly critical in educational contexts where reliable learning outcomes must be ensured.

To the best of our knowledge, this is the first study to directly compare human–human and human–LLM counselling conversations using a standardized category system like GeCCo. Prior works have either demonstrated the feasibility of LLM-simulated clients or discussed their general benefits and challenges, but none have quantitatively contrasted the interaction patterns of AI-driven clients with those of real humans in a training context. Our study expands upon the existing literature by bridging that gap: We integrate the advances of an LLM-based virtual client platform (based on [16]) with a fine-grained communication analysis methodology (GeCCo) to evaluate how AI-simulated dialogues difer from and resemble traditional role-plays.

## 3. DATASET

The dataset consists of two types of educational role-plays: Human-Human and Human-LLM interactions. The Human-Human role-plays were conducted as part of a degree program in social work and focused on online counselling funda mentals. A total of 64 chat-based counselling sessions were conducted via a chat platform, with one student acting as the client. Students selected a case study to shape their role. The Human-LLM role-plays were recorded in a learning platform to practice chat counselling skills. The learning platform is based on the methodology described in VirCo [16]. While the prompt stayed the same as described in VirCo the language model has been exchanged from Vicuna 13B to Mixtral 8x7B. Two distinct student groups participated:

one from the online counselling course and another comprising students from a diferent program at a partnering university. As LLM Mistrals Mixtral 8x7B with 4-bit quantization has been chosen because its eficient inference enabled by the mixture-of-experts architecture, ensures fast responses. This is especially important when having multiple requests at the same time. We also selected a single model to maintain comparability between the Human-Human and Human-LLM datasets, ensuring similar sentence counts and minimizing confounds in our analysis. In total, the data set contains 74 completed Human-LLM counselling conversations. We used the same case studies as in the Human-Human role plays by integrating them into the system prompt of the language model.

Table 1: Comparison of Human-Human and Human-LLM Counselling Messages
<table><tr><td>Statistic</td><td>Human-Human</td><td>Human-LLM</td></tr><tr><td>Number of Conversations</td><td>64</td><td>74</td></tr><tr><td>Number of Messages (N)</td><td>2411</td><td>2892</td></tr><tr><td>Total Sentences</td><td>5383</td><td>7040</td></tr><tr><td>Avg. Sentences per Msg</td><td>2.23</td><td>2.43</td></tr><tr><td>Total Words</td><td>59805</td><td>82587</td></tr><tr><td>Avg. Words per Msg</td><td>24.81</td><td>28.56</td></tr></table>

## 4. METHOD

Central to our analytical approach is the GeCCo (German E-Counselling Conversation Dataset) category system [1], a specialised framework designed for analysing online counselling conversations. Structured hierarchically across up to five layers of abstraction, GeCCo enables a highly granular classification of both counsellor and client communications. In its fifth layer, the system details 40 distinct counsellor categories and 28 client categories, capturing both broad dimensions (such as “Formalities” or “Impact factors”) and subtle communicative nuances.

GeCCo was selected because it synthesises several established categorisation frameworks, including Motivational Interviewing, and integrates insights from diverse counselling models. Given that our dataset comprises German-language counselling conversations—and considering that GeCCo is based on a German dataset—the system aligns well with our needs. Moreover, the category system has been successfully trained using a GBERT Large classifier, which was further fine-tuned with synthetic data from GPT-4 to classify our conversations efectively. The analysis was then conducted on layer 3 of the GeCCo hierachy (layer 1: most abstract; layer 5: finest granularity). The following paragraph shows an explanation of the used categories:

## Client Categories:

Analysis & Agreement on Counselling Goals: Statements involving the formulation and mutual agreement on counselling goals.

Analysis & Clarification of Problems: Expressions aimed at exploring and clarifying the client’s issues.

Help, Problem-Solving: Utterances intended to facilitate problem resolution or ofer direct assistance.

Generating Motivation: Strategies that evoke or enhance

the client’s intrinsic motivation for change.

Resource Activation: Identification and mobilization of internal or external resources to support the client.

Other Statements: Miscellaneous expressions that do not fit into the main categories.

Formalities at the Beginning: Opening formalities such as greetings and introductory remarks.

Formalities at the End: Closing formalities including farewells and session summaries.

Counselor Categories:

Other Statements: Counselor’s miscellaneous expressions that do not fit into the main categories.

Conversation Opening: Counselor-led initiations that set the tone and structure for the session.

Moderation: Eforts by the counselor to steer and structure the conversation efectively.

Formalities at the Beginning: Counselor’s introductory remarks and greetings to initiate the session.

Formalities at the End: Counselor’s closing remarks and farewells to conclude the session.

Analysis & Agreement on Counselling Goals: Counselor’s statements involving the formulation and mutual agreement on counselling goals.

Analysis/ Clarification-Reflection (Emotion): Reflective responses aimed at clarifying the emotional content of the dialogue.

Analysis/ Clarification-Reflection (Factual): Reflective responses focused on clarifying factual details shared during the session.

Help, Problem-Solving: Expressions intended to assist in problem resolution or to provide direct help.

Generating Motivation: Strategies to evoke or enhance the client’s intrinsic motivation.

Resource Activation: Identification and mobilization of resources to support the client.

The analysis shows an examination of pattern distributions by computing the frequency of each GeCCo category in both Human-Human and Human-LLM interactions. This calculation identifies diferences in the manifestation of coun selling goals, problem clarification, and other communicative patterns. A temporal analysis segments conversations into deciles, revealing how interaction patterns—such as factual reflection and problem solving—vary across diferent stages of the counselling session.

A semantic analysis employs embedding-based representations, where embeddings for each message are generated using the GBERT classifier and visualised via t-distributed Stochastic Neighbour Embedding (t-SNE) [11]. The resulting projections uncover clusters corresponding to various categories, such as “Analysis & Clarification of Problems” and “Formalities,” thereby elucidating the semantic structure of both human and LLM-generated responses.

## 5. RESULTS

This section presents the findings of a comparative analysis of Human-Human and Human-LLM counselling roleplays. It examines diferences in communication patterns using three complementary approaches: (1) Pattern Distribution Analysis, which quantifies the frequency of diferent counselling behaviours across both interaction types; (2)

Temporal Analysis, which investigates how interaction patterns evolve over the course of a conversation; and (3) Semantic Analysis, which visualises conversational similarities and diferences using embedding-based representations.

## 5.1 Pattern Distribution Analysis

The distribution of the client pattern shows a significant diference between Human-Human and Human-LLM interactions in the category Analysis & Clarification of problems (Figure 1). Human clients use this pattern significantly more often (59.07%) than LLM-simulated clients (45.47%). This discrepancy results from the fact that LLM-simulated clients tend to disclose the problems embedded in the prompt more readily, which enables a faster transition to problem solving.

In addition, category Help, Problem-Solving occurs more frequently with LLM-simulated clients (17.63%) than with human clients (10.67%), suggesting that simulated clients take a more solution-focused approach. Similarly, Formalities at the End are more common in LLM-generated clients (11.95%) than in human clients (5.95%), probably reflecting structured conversation patterns typical of LLM responses. The counsellor distribution shows significant diferences in the category Analysis/Clarification–Reflection (Factual), which is more pronounced in Human-LLM interactions (36.37%) compared to Human-Human conversations (26.07%). This suggests that counsellors interacting with LLM clients focus more on factual reasoning, which may compensate for the LLM’s tendency to disclose rather than explain problems. On the other hand, moderation, which includes facilitation and structuring responses, is significantly higher in Human-Human interactions (23.20%) compared to Human-LLM conversations (12.17%). This may suggest that counsellors in Human-Human interactions take a more active role in leading the discussion, whereas those conversing with LLM clients rely more on reflective techniques.

Other notable diferences include Help, Problem-Solving, which remains relatively stable across conditions (17.62% in Human-Human interactions vs. 18.58% in Human-LLM), and Conversation Opening, which occurs slightly more frequently in Human-Human interactions.

## 5.2 Temporal Analysis

Figures 2 and 3 illustrate the temporal evolution of the predicted interaction labels for counsellors and clients across the relative position in conversations. The x-axis represents the conversation progression in deciles (0-10% to 90-100%), while the y-axis shows the normalised frequency (%) of each pattern. These graphs can be used to examine how the interaction patterns evolve over time and how they difer between Human-Human and Human-LLM counselling sessions.

In the counsellors’ interaction patterns (Figure 2), the most dominant pattern throughout the conversation is Analysis / Clarification - Reflection (factual), which is particularly prevalent at the beginning of the conversation in both Human-Human and Human-LLM conditions. However, in Human-LLM interactions this pattern peaks early (around 10-20%) and then gradually decreases, whereas in Human-Human interactions it remains more stable throughout the session. This is consistent with the results of the general pattern distribution, where factual reflection was more frequent in Human-LLM interactions, probably because LLM-generated clients disclose their problems more directly. Help, Problem Solving and Moderation increase as the conversation progresses, especially in Human-Human interactions. This suggests that in the latter stages, human counsellors focus on leading the discussion and addressing the client’s problems. In contrast, these patterns are less pronounced in Human-LLM interactions, suggesting a diferent conversation dynamic where problem solving does not escalate as much over time. Another noticeable trend is in category Generating Motivation, which remains relatively low throughout the interview but increases slightly towards the end. This could indicate that counsellors are making an efort to introduce motivational interventions before the end of the counselling.

![](images/e7dc77fb87f7483409952f32b7d5f4d7af1ca44b0aa90e91cc5bbe9c2dd6a0bd.jpg)

Figure 1: Pattern Distribution for Client and Counselor Patterns across all conversations  
![](images/46615707776731e9758e42c5e1568c525e1d67857162b67b23dbb0f8096e0fc7.jpg)  
Figure 2: Frequency of Predicted Counsellor Labels along Conversation Progression

In the client interaction patterns (Figure 3), Analysis and

![](images/eba82a3b9588d9aa2786457abaf33a111cdf77947c02beb1281b1410de466947.jpg)  
Figure 3: Frequency of Predicted Client Labels along conversation progression

Clarification of Problems is the predominant category, especially at the beginning of the conversation. This trend is much more pronounced in Human-Human interactions, where problem elaboration starts high (over 8%) and gradually decreases over time. In contrast, LLM-simulated clients disclose their problems more immediately but with less intensity, which is consistent with the results of the overall distribution that showed a lower percentage for this pattern in LLM-generated conversations. A key diference between human and LLM clients is the increase in helping, problem solving, and resource activation in the second half of the conversation, especially in the Human-LLM interactions. This suggests that LLM-generated clients are more inclined to shift to problem-solving and resource-orientated discussions as the conversation progresses, whereas human clients may continue to reflect on their problems. In addition, formality at the beginning is higher of Human-LLM interactions but quickly decreases, while formality at the end increases sharply in the last decile. This reflects the structured nature of LLM-generated responses, which tend to adhere to clear opening and closing conventions.

## 5.3 Semantic Analysis

To further examine diferences in conversational patterns, we applied embedding-based representations using the GBERT-Large Client and Counsellor Classifiers. These models, initially trained for sentence classification, were also used to generate embeddings for client and counsellor messages. The resulting embeddings were visualised using t-SNE to illustrate the clustering of diferent interaction patterns.

The t-SNE projection for client messages (Figure 4) shows that Analysis & Clarification of Problems (orange) is the most dispersed category, indicating that problem elaboration occurs across diverse conversational contexts. In addition to the static t-SNE plot analysis an interactive version has been analysed. It reveals that the orange cluster at the bottom of the client projection corresponds to General Inquiries, which appears more frequently in Human-Human interactions. This suggests that human clients tend to ask general questions more often than LLM clients, possibly because LLM-generated responses are shaped by instruction-following rather than reciprocal questioning. Additionally, Agreement statements appear as a distinct cluster, with LLM-generated agreements tending to be embedded in longer sentences rather than short afirmations. This aligns with our previous findings that human clients engage more extensively in problem clarification, while LLM-simulated clients disclose problems more directly. Another visible cluster in the center of the projection represents negative feedback on prior problem-solving eforts, further suggesting that LLM clients engage in a more structured feedback process compared to human clients. Formalities at the Beginning form a distinct, tightly clustered region, suggesting that these statements play a structurally separate role from the core counselling discussion. Formalities at the End exhibit a similar concentrated pattern, reinforcing the notion that LLM-generated clients tend to follow clear opening and closing conventions, as observed in the pattern distribution.

![](images/4e51dd67a8418a699afa03ad52d693a3fa762fb4c3193f55b3ddd4108bd4d8b3.jpg)  
Figure 4: t-SNE Projection of Client Sentences

The t-SNE projection for counsellor messages (Figure 5) reveals well-defined clusters for Help, Problem-Solving, reflecting the central role of problem resolution in counselling interactions. Analysis/Clarification - Reflection (Factual) and Analysis/Clarification - Reflection (Emotion) are positioned closely, indicating that factual and emotional reflections frequently co-occur within conversations. A key distinction in Human-LLM interactions emerges in the Analysis/Clarification (Factual) category. An interactive plot analysis reveals that when engaging with human clients, counsellors tend to provide more detailed follow-up questions regarding past solution attempts. Examples include inquiries such as ”How did he join your circle of friends?” or ”What was it like when you weren’t arguing so often?” This pattern is evident in the lower-right region of the t-SNE plot. Another significant finding is the clustering of Moderation statements, which appear more frequently in Human-Human conversations. These messages include procedural clarifications such as ”The chat has a time frame of 60 minutes” or ”Let’s take turns writing so there is no confusion.” This suggests that Human-Human counselling requires more explicit structuring, possibly because human clients exhibit less adherence to a predefined conversational flow compared to LLM-simulated clients. Additionally, the use of emojis was not possible on the learning platform, which may have influenced the nature of moderation-related messages.

![](images/99b210b41c30f4928ca1c3117c37a2e758e0e4f1e26ebcab995a26bf0bc24fb5.jpg)  
Figure 5: t-SNE Projection of Counsellor Sentences

## 5.4 Educational Implications

Learning counselling with an LLM has its upsides and downsides. On the upside, LLM-simulated clients ofer a highly structured and consistent environment that can streamline training. Their immediate disclosure of issues and solution-focused responses allow trainees to quickly engage in problem-solving and to improve their factual reflection skills. This consistency can be especially beneficial in early training stages or in scenarios where repeated prac tice is essential for skill development. On the downside, the structured nature of LLM-generated responses may limit exposure to the rich, spontaneous, and emotionally nuanced interactions that characterise human counselling. The reduced emphasis on elaborate problem clarification and emotional attunement may not fully prepare trainees for the unpredictability of real-world counselling sessions. Consequently, while LLM-based role-plays serve as a valuable supplement by enhancing eficiency and standardization, they also risk oversimplifying the complex interpersonal dynamics necessary for comprehensive counselling training. In summary, integrating LLMs into counselling education presents a promising opportunity to expand training accessibility and consistency, but it should be balanced with traditional human interactions to ensure the development of both technical and interpersonal competencies.

## 6. LIMITATIONS

While our study ofers insightful comparisons between Human-Human and Human-LLM counselling interactions, several limitations should be acknowledged. First, the analysis relied on automated classifiers (GBERT Large) rather than manual annotations, which may not capture every subtle nuance in conversational dynamics. Second, we used role-play scenarios rather than real-world counselling sessions, meaning the findings reflect the dynamics of simulated environments. Third, the analysis focused primarily on the rather coarse categorisation level 3 of the GeCCo system; examining more detailed sub-categories might reveal further nuances. Additionally, it is important to note that our study employed only one large language model – Mixtral 8x7B with 4-bit quantisation. This choice ofers the advantage of better comparability across sessions; however, it also means that alternative LLMs might exhibit diferent conversational patterns.

## 7. CONCLUSION AND FUTURE WORK

The present study provides valuable insights into the diferences and similarities between Human-Human and Human-LLM counselling role-plays. Our results indicate that LLMsimulated clients tend to disclose their issues in a more immediate and structured manner, thereby prompting counsellors to rely more heavily on factual reflection rather than on facilitative moderation. In contrast, human clients typically engage in a more elaborative problem clarification, leading counsellors to adopt a more active moderating role. Furthermore, the temporal and semantic analyses reveal distinct conversational dynamics and clustering patterns between the two interaction types, highlighting the potential and current limitations of using LLMs as simulated clients in counselling training.

For future work, several paths could be pursued to bridge the gap between LLM-simulated and human clients. Refinements in LLM training—such as fine-tuning with more representative conversational data or leveraging reinforcement learning approaches like Direct Preference Optimization (DPO)—could enhance the naturalness and emotional authenticity of simulated interactions. Additionally, incorporating manual annotation and establishing a counselling communications gold standard would provide a more robust framework for evaluation. Working with more detailed categories for utterance classification and exploring diverse linguistic and cultural contexts could further validate and generalise these findings.

## 8. REFERENCES

[1] J. Albrecht, R. Lehmann, and A. Poltermann. GeCCo 1.0 - Erstellung eines ¨ofentlichen Datensatzes fur die¨ KI-basierte Inhaltsanalyse in der Online-Beratung. 2024.

[2] E. Baker and A. Jenney. Virtual Simulations to Train Social Workers for Competency-Based Learning: A Scoping Review. Journal of Social Work Education, 59(1):8–31, Jan. 2023. Publisher: Routledge eprint: https://doi.org/10.1080/10437797.2022.2039819.

[3] S. Bhatt. Digital Mental Health: Role of Artificial Intelligence in Psychotherapy. Annals of Neurosciences, page 09727531231221612, Apr. 2024. Publisher: SAGE Publications.

[4] A. Chen, J. Jia, Y. Li, and L. Fu. Investigating the Efect of Role-Play Activity With GenAI Agent on EFL Students’ Speaking Performance. Journal of Educational Computing Research, 63(1):99–125, Mar. 2025. Publisher: SAGE Publications Inc.

[5] A. Dwyer, A. d. A. Neto, D. Estival, W. Li, C. Lam-Cassettari, and M. Antoniou. Suitability of Text-Based Communications for the Delivery of Psychological Therapeutic Services to Rural and Remote Communities: Scoping Review. JMIR Mental Health, 8(2):e19478, Feb. 2021. Company: JMIR Mental Health Distributor: JMIR Mental Health Institution: JMIR Mental Health Label: JMIR Mental

Health Publisher: JMIR Publications Inc., Toronto, Canada.

[6] M. Inaba, M. Ukiyo, and K. Takamizo. Can Large Language Models be Used to Provide Psychological Counselling? An Analysis of GPT-4-Generated Responses Using Role-play Dialogues, Feb. 2024. arXiv:2402.12738 [cs].

[7] A. Kerr, J. Strawbridge, C. Kelleher, J. Barlow, C. Sullivan, and T. Pawlikowska. A realist evaluation exploring simulated patient role-play in pharmacist undergraduate communication training. BMC Medical Education, 21(1):325, June 2021.

[8] H. R. Lawrence, R. A. Schneider, S. B. Rubin, M. J. Matari´c, D. J. McDuf, and M. J. Bell. The Opportunities and Risks of Large Language Models in Mental Health. JMIR Mental Health, 11(1):e59479, July 2024. Company: JMIR Mental Health Distributor: JMIR Mental Health Institution: JMIR Mental Health Label: JMIR Mental Health Publisher: JMIR Publications Inc., Toronto, Canada.

[9] Y. Li, C. Zeng, J. Zhong, R. Zhang, M. Zhang, and L. Zou. Leveraging Large Language Model as Simulated Patients for Clinical Education, Apr. 2024. arXiv:2404.13066 [cs].

[10] R. Louie, A. Nandi, W. Fang, C. Chang, E. Brunskill, and D. Yang. Roleplay-doh: Enabling Domain-Experts to Create LLM-simulated Patients via Eliciting and Adhering to Principles, July 2024. arXiv:2407.00870 [cs].

[11] L. v. d. Maaten and G. Hinton. Visualizing Data using t-SNE. Journal of Machine Learning Research, 9(86):2579–2605, 2008.

[12] R. K. Maurya. Using AI Based Chatbot ChatGPT for Practicing Counseling Skills Through Role-Play. Journal of Creativity in Mental Health, 19(4):513–528, Oct. 2024. Publisher: Routledge eprint: https://doi.org/10.1080/15401383.2023.2297857.

[13] J. Moock. Support from the Internet for Individuals with Mental Disorders: Advantages and Disadvantages of e-Mental Health Service Delivery. Frontiers in Public Health, 2, June 2014. Publisher: Frontiers.

[14] B. T. Nair. Role play – An efective tool to teach communication skills in pediatrics to medical undergraduates. Journal of Education and Health Promotion, 8:18, Jan. 2019.

[15] D. Osborn and L. Costas. Role-Playing in Counselor Student Development. Journal of Creativity in Mental Health, 8(1):92–103, Jan. 2013. Publisher: Routledge eprint: https://doi.org/10.1080/15401383.2013.763689.

[16] E. Rudolph, N. Engert, and J. Albrecht. An AI-Based Virtual Client for Educational Role-Playing in the Training of Online Counselors. In Proceedings of the 16th International Conference on Computer Supported Education - Volume 2, pages 108–117, Feb. 2025.

[17] E. Rudolph, H. Seer, C. Mothes, and J. Albrecht. Automated feedback generation in an intelligent tutoring system for counselor education. pages 501–512, Oct. 2024.

[18] S. B. Rønning and S. Bjørkly. The use of clinical role-play and reflection in learning therapeutic communication skills in mental health education: an

integrative review. Advances in Medical Education and Practice, 10:415, June 2019.

[19] S. Shorey, E. Ang, J. Yap, E. D. Ng, S. T. Lau, and C. K. Chui. A Virtual Counseling Application Using Artificial Intelligence for Communication Skills Training in Nursing Education: Development Study. Journal of Medical Internet Research, 21(10):e14658, Oct. 2019.

[20] W. M. Shurts, C. S. Cashwell, S. L. Spurgeon, S. Degges-White, C. A. Barrio, and K. N. Kardatzke. Preparing Counselors-in-Training to Work With Couples: Using Role-Plays and Reflecting Teams. The Family Journal, 14(2):151–157, Apr. 2006. Publisher: SAGE Publications Inc.

[21] T. Stamer, J. Steinh¨auser, and K. Fl¨agel. Artificial Intelligence Supporting the Training of Communication Skills in the Education of Health Care Professions: Scoping Review. Journal of Medical Internet Research, 25:e43311, June 2023.

[22] M. M. Yee, M. K. Nyunt, A. M. Thidar, M. S. Khine, C. Y. Ong, and O. G. Seong. Challenges and Opportunities of using Role-players in Medical Education: Medical Educator’s Perspective. Medical Research Archives, 12(8), Aug. 2024.