# Robots in the City: Persona-Based Perceptions of Urban Robots

Keonhee Jang<sup>⋆</sup>, Kanghoon Lee<sup>⋆</sup>, Seokwoo Yun, Namwoo Kim, Yoonjin Yoon<sup>†</sup>

Abstract—Pedestrians increasingly encounter robots in urban public spaces without interacting with them, but how robots change the perceived safety of a street remains unclear. We examine how robot presence and type affect perceived safety across urban contexts. We generate 18 images of Seoul streets that vary in time of day, road class, and robot condition (no robot, a delivery robot, or a humanoid). Two multimodal large language models (MLLMs), Qwen3.8-27B and Gemma4-26B, each conditioned on 500 synthetic personas, choose the safer image in each pair, and 25 human respondents complete the same task. Both models prefer streets without robots and favor the delivery robot over the humanoid. Robots are chosen more often at night and less often by older personas, whereas road class has smaller effects. In the model rationales, the delivery robot appears as a physical obstacle, whereas the humanoid appears as either a threat or a source of protection. Like humans, both models chose robots more often at night, but chose the humanoid over the delivery robot far less often, and Gemma showed a night effect for the humanoid about three times that of humans. These results show that the effect of a robot on perceived safety depends on when it appears, and that persona-conditioned MLLMs capture only part of human judgments.

## I. INTRODUCTION

Advances in embodied AI are bringing robots out of laboratories and into everyday urban public spaces. People on the street often pass these robots without interacting with them directly, so robots must also be understood from the perspective of these pedestrians [1], [2]. Even without interaction, a robot becomes part of the street scene [1] and can change how safe the street feels, a long-standing topic in urban perception research [3], [4]. While prior studies has focused on the people who encounter robots without directly interacting with them [2], how robot presence and type shape the perceived safety of the surrounding streetscape remains unclear. Also, urban visual perception researches have rarely included robots.

Understanding this effect requires examining how people perceive a robot in a street environment: for example, an octopus on a farm would not fit its surroundings. People recognize such an object less easily [5] and look at it longer [6], which suggests that it takes more effort to process. Objects that require more effort to process tend to be judged less favorably [7], suggesting that an out-of-place object can make people feel uneasy. A robot on a street is also an object in a scene, and whether it fits can depend on what it does and where and when it appears [8]. The same robot can thus fit one street or time of day but not another, changing how safe the street feels. To investigate this, we use street view images, a common stimulus in urban perception research [3], [4], and generate variants of each scene that differ in the robot and the time of day [9], [10]. This allows us to examine how the urban context of a robot shapes its perceived safety.

We evaluate these images through pairwise comparisons of perceived safety, which allow controlled variants to be compared directly without relying on absolute ratings [11], [12]. We conduct these comparisons at scale using multimodal large language models (MLLMs) conditioned on persona descriptions, building on prior work on simulated responses [13], [14] and visual reasoning in urban environments [15]. This allows us to examine whether the same visual changes are judged differently across personas. We also collect the same judgments from human respondents to assess how closely the models align with human perception. Because persona conditioning does not necessarily reproduce human heterogeneity [16], we use persona variation as an analytical factor rather than a direct proxy for human populations.

In summary, our main contributions are as follows:

• We examine how robot presence and type shape perceived safety across urban contexts.

• We compare human and persona-conditioned MLLM judgments, identifying where the models align with or depart from humans.

• We analyze the rationales behind model choices and how they shift with context.

## II. RELATED WORK

Urban streetscape perception. Street view images have been widely used to study perceived safety [3], broader subjective qualities of urban environments [4], [12], and pedestriancentered perceptions such as visual walkability [17], [18]. Recent work has examined how these judgments vary across observer groups [19] and how visual AI can characterize perceptions of natural elements in urban scenes [20]. Prior studies have relied on comparisons across real-world street scenes, making it difficult to isolate the effect of a single visual feature. More recent work has begun to address the issue by using generative transformations to create paired street view images with controlled changes, such as nighttime appearance [9], and to compare perceived safety between matched daytime and nighttime conditions [10]. Recent work has also evaluated vision-language models (VLMs) for spatial reasoning directly from street view images [15]. We extend this controlled visual comparison setting to examine how robot presence and type shape perceptions of urban pedestrian spaces.

Robots in shared public space. Research on robots in public spaces shows that local context, physical environment, and social activity influence how people perceive robots [8]. These experiences are not limited to direct interaction, as people can perceive robots simply by seeing them in the surrounding environment [2]. Delivery robots are a common example because they operate on sidewalks and other pedestrian spaces alongside people [21]. When robots move through these spaces, pedestrian comfort and social constraints become important considerations for human-aware navigation [22]. For sidewalk delivery robots, narrow paths and path intersections can make interactions more difficult [23]. Environmental conditions can also affect how much distance people prefer to keep from robots [24]. We instead examine how robot presence and type shape judgments of pedestrian space before motion or interaction occurs.

Persona-conditioned models for subjective judgments. Large language models (LLMs) have been increasingly explored for simulating responses in human-subject studies [13] and for generating persona-conditioned responses that reflect variation across individuals [14]. In Human-Robot Interaction (HRI), LLMs have also been used as zero-shot human models, although their predictions can be sensitive to prompting and limited in reasoning [25]. Recent work has also used generative agents to evaluate urban qualities such as safety and liveliness from street view images [26]. However, persona conditioning does not necessarily reproduce human heterogeneity. Its benefits can be limited when persona attributes are weakly related to the target judgments [16], and matching empirical response distributions remains an open challenge [27]. We therefore interpret persona-conditioned MLLM outputs as simulated judgments rather than direct observations of human perception.

## III. EXPERIMENTAL DESIGN

## A. Scene and Condition Generation

We collect three street scenes in Seoul from Mapillary [28], one from each of three levels of the functional road hierarchy in Korean urban planning [29]: primary arterial, secondary arterial, and local roads. Each scene serves as a fixed setting for generating different robot and time of day conditions.

For each scene, we generate six visual conditions using GPT-5.6 Sol [30] by varying time of day and robot condition. We first remove all people and vehicles, then set the time of day to a clear summer day or midnight, and reinsert people to reflect typical activity for each time and road class. Each of these day and night images is then created in all three robot conditions, with no robot, a delivery robot, and a humanoid, yielding 18 images in total (3 road classes × 2 times of day × 3 robot conditions). All 18 images are provided in the Appendix A. These variants preserve the original street layout, so that perceived safety can be compared across times of day, road classes, and robot types.

![](images/174d7696f3e452130f43ac5d5832b39f4270e08ba80f023f089c4bc0231e5fa6.jpg)  
Prompt 1: Prompt for persona-conditioned comparison

## B. Persona-Conditioned MLLM Evaluation

To examine whether evaluations vary across personas, we sample 500 synthetic personas from the Nemotron-Personas-Korea dataset [31], balancing the distributions of gender, age, household income, and residential region. The resulting personas include 250 female and 250 male personas, 250 from the capital region and 250 from other regions, 166–167 in each of three income groups, and 83–84 in each of six age groups, from the 20s or younger to the 70s or older.

We conduct the evaluations using 4-bit quantized versions of two open-weight MLLMs, Qwen3.8-27B [32], [33] and Gemma4-26B [34], [35]. Both models are served locally with vLLM [36] at a temperature of 0.7 for Qwen and 1.0 for Gemma, with thinking mode disabled. In each trial, the model is shown a persona card and a pair of images depicting the same scene at the same time of day but with different robot conditions. The left and right positions of the two images are randomized. The persona card lists gender, age, occupation, household income level, and residence in Korean, followed by a one-line introduction and a short description of household income, as detailed in Appendix B. The system prompt is the same for all inferences and includes a brief description of each robot, as shown in Prompt 1. The model first writes a one- or two-sentence rationale in Korean and then chooses the image that looks safer, without ties.

Each persona evaluates the three robot pairs in each of the six combinations of road class and time of day, yielding 18 trials per persona and 9,000 original responses per model. For consistency assessment, each persona repeats one trial per robot pair with both the original and swapped image positions, yielding 3,000 additional responses per model. Original responses are used for all analyses, and repeated responses are used only to assess consistency.

## C. Rationale Coding

To compare the reasoning expressed in model responses across conditions, we categorize the rationales from the 17,998 original responses (9,000 from Gemma and 8,998 from Qwen, after excluding two invalid Qwen responses). Two authors review 100 randomly sampled rationales and identify five topics: (1) collision or path obstruction, (2) threat or unfamiliarity, (3) openness, visibility, or lighting, (4) surveillance, accompaniment, or protection, and (5) robot function or trust in technology. Claude Opus 5.5 [37] then codes each rationale up to three topics it mentions [38]. To assess reliability, a human annotator codes a random sample of 150 rationales, 25 for each of the six combinations of MLLM and robot pair, without seeing Claude’s labels. The agreement between the human annotator and Claude and further details of the coding procedure are provided in Appendix C.

## D. Human Evaluation

We conduct a parallel human survey to provide a human reference for the model evaluations. We recruit 25 respondents (12 female and 13 male). They participate voluntarily without compensation and give informed consent before completing the survey. First, respondents report their gender and age group and read the same robot descriptions given to the models. Next, they complete the same 18 trials as the models, selecting the image perceived as safer without ties, with randomized trial order and image position. Finally, respondents report their education and household income and can describe what mattered most in their safety judgments.

## E. Analysis

We report the win rate as the share of trials in which the target condition is judged safer [12]. The target is the robot when it is compared with the empty street and the humanoid when the two robots are compared. While we use all 500 personas for the main analyses, we match each human respondent with 10 personas of the same gender and age group for the human comparison. We compute 95% confidence intervals using a cluster bootstrap with 2,000 resamples of respondents or personas. Differences by time of day and gender are expressed as Cohen’s $h ,$ which places the gap between two win rates on a common scale, with 0.2, 0.5, and 0.8 marking small, medium, and large effects [39]. Response consistency is the share of repeated trials in which a model chooses the same condition as before.

![](images/526b9a8e751894e21ac07a75295723bc755814d02b4f7917383f735bb6dccde1.jpg)  
Fig. 1: Win rate by urban contexts. Rows show the two conditions: (a) time of day and (b) road class, and columns show robot pairs.

## IV. RESULTS

## A. Perceived Safety across Urban Contexts

We first examine how perceived safety changes with robot presence and type across urban contexts. As shown in Figure 1, both models chose the no-robot condition over the delivery robot and humanoid in most trials. These results suggest that robot presence can reduce perceived street safety. The delivery robot was chosen in 45% and 41% of trials by Qwen and Gemma, respectively, against the no-robot condition, whereas the humanoid was chosen in 13% and 29%. Against the delivery robot, the humanoid was chosen in only 4% and 5% of trials. Overall, both models ranked the no-robot condition first, the delivery robot second, and the humanoid last.

Figure 1a shows that the win rate of the delivery robot against the no-robot condition increased from 32% to 58% for Qwen and from 31% to 51% for Gemma. The humanoid also gained against the no-robot condition, from 3% to 23% for Qwen and from 5% to 53% for Gemma. Against the delivery robot, however, the humanoid remained rarely chosen at night (5% and 10%). Overall, both models chose robots more often at night, but still favored the delivery robot over the humanoid.

We also examine how judgments vary across road classes, as shown in Figure 1b. For Qwen, win rates decreased from the primary to the local road, from 49% to 41% for the delivery robot and from 23% to 7% for the humanoid. In contrast, Gemma’s win rate for the delivery robot ranged from 25% on the secondary road to 57% on the local road, while that for the humanoid ranged from 26% to 32%. Overall, win rates varied less across road classes than between day and night, with no consistent pattern shared by the two models.

![](images/4bd133a6a1c569845b341b4ebb636da085fc97616f4300951f78d0eccaa6bfd4.jpg)  
Fig. 2: Win rate by persona attribute. Rows show the four attributes: (a) gender, (b) age, (c) income, and (d) region, and columns show robot pairs.

## B. Perceived Safety across Persona Attributes

We next examine how model judgments vary across four persona attributes: gender, age, income, and residential region. Figure 2a shows opposite patterns in humanoid win rates by gender across the two models. Against the no-robot condition, female personas chose the humanoid more often than male personas in Gemma (31% vs. 26%), whereas male personas chose it more often in Qwen (14% vs. 12%). Gender differences were smaller for the delivery robot, with female personas choosing it slightly more often in both models. For income and residential region, Figure 2c and Figure 2d show relatively small differences in win rates across both models.

Age showed a more consistent pattern across both models, as shown in Figure 2b. From the youngest to the oldest age group, the delivery robot’s win rate against the no-robot condition decreased from 54% to 26% for Qwen and from 51% to 28% for Gemma. A similar decline appeared for the humanoid only in Gemma, from 39% to 18%, whereas Qwen’s win rates remained between 9% and 16%. Overall, older personas were less likely to choose the delivery robot over the no-robot condition, while a similar age-related decline for the humanoid appeared only in Gemma.

## C. Rationale Topics across Choices and Conditions

To examine how models explain their choices, we analyze the rationales of Gemma, which showed a particularly large nighttime increase in preference for the humanoid. Figure 3 shows the rationale topic distributions across robot choices, time of day, and persona gender. The rationale topic results for Qwen are detailed in Appendix D.

The two robots were described in different terms. The delivery robot was mainly described as a physical obstacle, with concerns about blocked sidewalks and tripping hazards. Collision or path obstruction accounted for 74% of the rationale topics in this comparison during the day. In contrast, the humanoid was described as either threatening or protective. When the no-robot condition was chosen over the humanoid, threat or unfamiliarity represented 73% of the rationale topics by day and 94% at night. When the humanoid was chosen, surveillance, accompaniment, or protection represented more than 90% of the rationale topics both by day and at night. Therefore, the same humanoid was described as threatening when the no-robot condition was chosen but as protective when the humanoid was chosen.

As shown in Figure 3a, topics describing the robot as a help (surveillance and robot function) were more common at night when the delivery robot was chosen, with surveillance increasing from 7% to 53%. Figure 3b shows differences in rationale topics by persona gender. When the delivery robot was chosen, help topics were more common for female personas, whereas topics describing the robot as a risk (collision and threat) were more common for male personas. For the humanoid, surveillance dominated for both genders. Thus, gender differences in rationale topics were more pronounced for the delivery robot than for the humanoid.

## D. Alignment between Human and Model Judgments

We then examine whether the model judgments agree with those of human respondents. For this comparison, we used personas matched to the respondents by gender and age group. As shown in Figure 4, humans and the models made relatively similar choices when each robot was compared with the norobot condition. Humans chose the delivery robot in 59% of these trials (53% for Qwen and 48% for Gemma) and the humanoid in 28% (13% for Qwen and 37% for Gemma). The models differed from humans when the two robots were compared. Humans chose the humanoid in 24% of these trials, but Qwen and Gemma chose it in only 4% and 7%. Thus, humans and the models differed most when choosing between the two robots.

![](images/7edb1d76651ad2a97aeb3c632b8ddf42fc86d020d547cad8febe7d7f3e04bb96.jpg)  
Fig. 3: Share of rationale topics stated by Gemma4-26B for each choice, by (a) time of day and (b) persona gender, grouped by whether they describe the robot as a risk or a help, or the street. Shares within each bar sum to 100%, and numbers to the right give the number of responses.

![](images/dc1ca55fafe918e421eefc58b6d838eccb0d35676b94d5e6a4d6f03554c8037c.jpg)  
Fig. 4: Win rate of humans and matched personas. Rows show (a) time of day and (b) gender, and columns show robot pairs. Model results use personas matched to the human respondents by gender and age group.

As shown in Figure 4a, all three groups chose robots over the no-robot condition more often at night than during the day. When the delivery robot was compared with the no-robot condition, humans chose the delivery robot in 51% of daytime trials and 68% of nighttime trials, and both models showed a similar rise. When the humanoid was compared with the no-robot condition, humans chose the humanoid in 19% of daytime trials and 37% of nighttime trials, and Qwen showed a similar rise. Gemma showed a much larger rise, from 8% by day to 65% at night. The survey responses help explain why humans chose the humanoid at night less often than Gemma did. Some respondents felt safer with a robot on a dark street, but others expressed concerns about an unfamiliar humanoid or a possible malfunction, as detailed in Appendix E.

Figure 4b compares female and male respondents. Female respondents chose the humanoid slightly less often than male respondents (24% vs. 32% against the no-robot condition and 21% vs. 27% against the delivery robot). Qwen showed the same difference, but female personas in Gemma chose the humanoid more often than male personas. Gender differences were small, as shown by the effect sizes in Appendix F. Overall, both models followed humans in choosing robots more often at night. The models differed mainly when choosing between the two robots. Gemma also showed a much larger nighttime rise for the humanoid than humans did.

## E. Response Consistency

To assess the reliability of the model responses, we compared repeated trials with the original responses. As shown in Table I, when a trial was repeated with the two images in the same positions as before, Qwen and Gemma chose the same condition in 81% and 82% of trials on average. When the two images were swapped between left and right, consistency decreased to 74% and 75%. Across robot pairs, consistency was highest for the delivery robot against the humanoid (92% to 94%) and lowest for the delivery robot against the norobot condition (52% to 71%). This matches the win rates in Figure 1, as choices were more stable when one condition was chosen in most trials (95%) than when both were chosen at similar rates (45% and 41%). Overall, the model responses were stable mainly when one condition was clearly preferred.

TABLE I: Response consistency of the two models. Each value shows the proportion of repeated trials where the model chose the same image again. The two images were presented in either the same or swapped positions.
<table><tr><td rowspan="2">Robot pair</td><td colspan="2">Qwen3.8-27B</td><td colspan="2">Gemma4-26B</td></tr><tr><td>Same</td><td>Swapped</td><td>Same</td><td>Swapped</td></tr><tr><td>No robot vs. Delivery</td><td>65%</td><td>52%</td><td>71%</td><td>61%</td></tr><tr><td>No robot vs. Humanoid</td><td>84%</td><td>77%</td><td>81%</td><td>73%</td></tr><tr><td>Delivery vs. Humanoid</td><td>93%</td><td>93%</td><td>94%</td><td>92%</td></tr><tr><td>All pairs</td><td>81%</td><td>74%</td><td>82%</td><td>75%</td></tr></table>

## V. CONCLUSION

This study examined how robots affect the perceived safety of streets that pedestrians pass without interacting with them. We generated street view images that varied the robot, the time of day, and the road class. We then compared the judgments of two persona-conditioned MLLMs with those of human respondents and analyzed the rationales behind the model choices. The models generally judged streets without robots as safest and the delivery robot as safer than the humanoid, but chose robots more often at night than by day. Their rationales showed that the two robots raised different issues, with the delivery robot seen mainly as a physical obstacle and the humanoid as either a threat or a source of protection. Humans showed the same increase at night but chose the humanoid over the delivery robot far more often than the models did. Our findings support the view that whether a robot fits a place depends on its context. Persona-conditioned MLLMs can help explore such effects of context, but their judgments of a specific robot may still differ from those of people.

This study has three main limitations that point to future work. First, the human sample was small and included only respondents in their 20s, 30s, and 50s, and the images covered three scenes in one city, so the findings should be tested with larger and more diverse samples and with more scenes, cities, and robot types. Second, the images were static and did not show robots performing their tasks or interacting with people, although seeing a robot deliver a parcel or help someone may change how safe a street feels, so future work should examine robots in action through videos or field studies. Third, the model results came from only two open-weight models and were partly affected by image position, so future work should reduce such position effects and compare a wider range of models, including closed models.

## REFERENCES

[1] H. R. Pelikan, S. Reeves, and M. N. Cantarutti, “Encountering autonomous robots on public streets,” in Proceedings of the 2024 ACM/IEEE International Conference on Human-Robot Interaction, 2024, pp. 561–571.

[2] A. Dobrosovestnova, F. Babel, and H. Pelikan, “Beyond the user: Mapping subject positions for robots in public spaces,” in 2025 20th ACM/IEEE International Conference on Human-Robot Interaction (HRI), IEEE, 2025, pp. 163–173.

[3] N. Naik, J. Philipoom, R. Raskar, and C. Hidalgo, “Streetscorepredicting the perceived safety of one million streetscapes,” in Proceedings of the IEEE conference on computer vision and pattern recognition workshops, 2014, pp. 779–785.

[4] A. Dubey, N. Naik, D. Parikh, R. Raskar, and C. A. Hidalgo, “Deep learning the city: Quantifying urban perception at a global scale,” in European conference on computer vision, Springer, 2016, pp. 196–212.

[5] I. Biederman, R. J. Mezzanotte, and J. C. Rabinowitz, “Scene perception: Detecting and judging objects undergoing relational violations,” Cognitive psychology, vol. 14, no. 2, pp. 143–177, 1982.

[6] G. R. Loftus and N. H. Mackworth, “Cognitive determinants of fixation location during picture viewing,” Journal of Experimental Psychology: Human perception and performance, vol. 4, no. 4, pp. 565–572, 1978.

[7] R. Reber, N. Schwarz, and P. Winkielman, “Processing fluency and aesthetic pleasure: Is beauty in the perceiver’s processing experience?” Personality and social psychology review, vol. 8, no. 4, pp. 364–382, 2004.

[8] H. R. Pelikan, B. Mutlu, and S. Reeves, “Making sense of public space for robot design,” in 2025 20th ACM/IEEE International Conference on Human-Robot Interaction (HRI), IEEE, 2025, pp. 152–162.

[9] X. Ye, Y. Wang, J. Dai, and W. Qiu, “Generated nighttime street view image to inform perceived safety divergence between day and night in high density cities: A case study in Hong Kong,” Journal of Urban Management, vol. 14, no. 2, pp. 379–401, 2025.

[10] J. Dai, Y. Wang, W. Qiu, J. Ma, and D. Chen, “The urban greening paradox: Measuring day-night perceived safety divergences from paired street view images using generative AI,” Urban Forestry & Urban Greening, p. 129 350, 2026.

[11] J. Gu, X. Jiang, Z. Shi, H. Tan, X. Zhai, C. Xu, W. Li, Y. Shen, S. Ma, H. Liu, et al., “A survey on LLM-as-a-judge,” The Innovation, vol. 7, no. 6, 2026.

[12] P. Salesses, K. Schechtner, and C. A. Hidalgo, “The collaborative image of the city: Mapping the inequality of urban perception,” PloS one, vol. 8, no. 7, e68400, 2013.

[13] G. V. Aher, R. I. Arriaga, and A. T. Kalai, “Using large language models to simulate multiple humans and replicate human subject studies,” in International conference on machine learning, PMLR, 2023, pp. 337–371.

[14] L. P. Argyle, E. C. Busby, N. Fulda, J. R. Gubler, C. Rytting, and D. Wingate, “Out of one, many: Using language models to simulate human samples,” Political Analysis, vol. 31, no. 3, pp. 337–351, 2023.

[15] J. Ro, N. Kim, and Y. Yoon, “How well do vision-language models understand cities? a comparative study on spatial reasoning from street-view images,” in 2025 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), IEEE, 2025, pp. 6535–6544.

[16] T. Hu and N. Collier, “Quantifying the persona effect in LLM simulations,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024, pp. 10 289–10 307.

[17] Y. Li, N. Yabuki, and T. Fukuda, “Measuring visual walkability perception using panoramic street view images, virtual reality, and deep learning,” Sustainable Cities and Society, vol. 86, p. 104 140, 2022.

[18] Y. Xiao and M. Song, “How are urban design qualities associated with perceived walkability? an AI approach using street view images and deep learning,” International Journal of Urban Sciences, vol. 29, no. sup1, pp. 66–91, 2025.

[19] Q. Cui, Y. Zhang, G. Yang, Y. Huang, and Y. Chen, “Analysing gender differences in the perceived safety from street view imagery,” International journal ofapplied earth observation and geoinformation, vol. 124, p. 103 537, 2023.

[20] D. C. Lefosse, F. Duarte, R. P. Sanatani, Y. Kang, A. van Timmeren, and C. Ratti, “Feeling nature: Measuring perceptions of biophilia across global biomes using visual AI,” npj Urban Sustainability, vol. 5, no. 1, p. 4, 2025.

[21] A. M. Abrams, P. S. Dautzenberg, C. Jakobowsky, S. Ladwig, and A. M. Rosenthal-Von Der Pütten, “A theoretical and empirical reflection on technology acceptance models for autonomous delivery robots,” in Proceedings of the 2021 ACM/IEEE International Conference on Human-Robot Interaction, 2021, pp. 272–280.

[22] T. Kruse, A. K. Pandey, R. Alami, and A. Kirsch, “Human-aware robot navigation: A survey,” Robotics and Autonomous Systems, vol. 61, no. 12, pp. 1726–1743, 2013.

[23] S. R. Gehrke, C. D. Phair, B. J. Russo, and E. J. Smaglik, “Observed sidewalk autonomous delivery robot interactions with pedestrians and bicyclists,” Transportation research interdisciplinary perspectives, vol. 18, p. 100 789, 2023.

[24] E. Nahum, Y. Edan, and T. Oron-Gilad, “Advancing a taxonomy of proxemics for socially aware robot navigation,” Frontiers in Robotics and AI, vol. 13, p. 1 800 762, 2026.

[25] B. Zhang and H. Soh, “Large language models as zero-shot human models for human-robot interaction,” in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), IEEE, 2023, pp. 7961–7968.

[26] D. Verma, O. Mumm, and V. M. Carlow, “Generative agents in the streets: Exploring the use of large language models (LLMs) in collecting urban perceptions,” Computers in Human Behavior: Artificial Humans, p. 100 277, 2026.

[27] N. Meister, C. Guestrin, and T. B. Hashimoto, “Benchmarking distributional alignment of large language models,” in Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 2025, pp. 24–49.

[28] G. Neuhold, T. Ollmann, S. Rota Bulo, and P. Kontschieder, “The Mapillary Vistas dataset for semantic understanding of street scenes,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 4990–4999.

[29] Ministry of Land, Infrastructure and Transport, Rules on the decision, structure, and installation standards of urban and county planning facilities, article 9, item 3, Korean Law Information Center, https : //www.law.go.kr, In Korean; English title translated by the authors. Version in force as of Oct. 14, 2026.

[30] OpenAI, GPT-5.6 Sol, https://developers.openai.com/api/docs/models/ gpt-5.6-sol, Accessed Oct. 14, 2026.

[31] H. Kim, J. Ryu, J. Lee, H. Ryu, K. Praveen, S. Prayaga, K. Thadaka, W. Jennings, B. Sadeghi, A. Sharabiani, Y. Choi, and Y. Meyer, Nemotron-Personas-Korea: Synthetic personas aligned to real-world distributions for Korea, https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea, 2026.

[32] Qwen Team, Qwen3.8-27B, https://huggingface.co/Qwen/Qwen3.8- 27B, Hugging Face model card. Accessed Oct. 16, 2026.

[33] Red Hat AI, Qwen3.8-27B-INT4, https://huggingface.co/RedHatAI/ Qwen3.8-27B-INT4, Hugging Face model card.

[34] Gemma Team, “Gemma 4 technical report,” arXiv preprint arXiv:2607.02770, 2026.

[35] cyankiwi, gemma-4-26B-A4B-it-AWQ-4bit, https : / / huggingface . co / cyankiwi/gemma- 4- 26B- A4B- it- AWQ- 4bit, Hugging Face model card.

[36] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. Gonzalez, H. Zhang, and I. Stoica, “Efficient memory management for large language model serving with PagedAttention,” in Proceedings of the 29th symposium on operating systems principles, 2023, pp. 611–626.

[37] Anthropic, System card: Claude Opus 5.5, https : / / www - cdn . anthropic . com / fc1b44717c85dc068bc6ba5024219938094694bd / Claude%20Opus%205.5%20System%20Card.pdf, Accessed Oct. 16, 2026.

[38] F. Gilardi, M. Alizadeh, and M. Kubli, “ChatGPT outperforms crowd workers for text-annotation tasks,” Proceedings of the National Academy of Sciences, vol. 120, no. 30, e2305016120, 2023.

[39] J. Cohen, Statistical power analysisfor the behavioral sciences, 2nd ed. Hillsdale, NJ: Lawrence Erlbaum Associates, 1988.

## GENERATED STREET VIEW IMAGES

Figure 5 shows all 18 images. Each row shows one road class at one time of day, and each column shows one robot condition. All images were resized to $1 0 2 4 \times 7 6 8$ pixels. Few pedestrians were reinserted in the night images, reflecting typical activity at midnight.

![](images/477c4a0768e7a23945ab99a137067ea0eed8cdb3aa43c8e9ba3fa818d57dd253.jpg)  
Fig. 5: The 18 stimulus images. Rows show the three road classes by day and at night, and columns show robot conditions. Each trial compared two images in the same row.

## PERSONA CARDS

Each persona card lists five attributes, followed by a oneline introduction and a short description of household income. The cards were given to the models in Korean; the two examples below are translated by the authors. Table II shows the composition of all personas and of the personas matched to the human respondents.

Gender: Female   
Age: 25 (20s or younger)   
Occupation: Unemployed   
Household income level: Low   
Residence: Michuhol-gu, Incheon (capital region)   
Ms. Chu Yun-ji is a 25-year-old job seeker in Incheon with   
strong artistic sensitivity and empathy but weak practical self  
management.   
Her household income is low, and because she is preparing for   
employment, housing costs outside her current home, transporta  
tion, and food take up a large share of living expenses. She saves   
in advance for large expenses such as exam fees and interview   
clothes and manages within a fixed budget.

TABLE II: Composition of the personas. All shows the 500 personas used for the model results, and Matched shows the 137 unique personas matched to the human respondents.
<table><tr><td>Attribute</td><td>Level</td><td>All</td><td>Matched</td></tr><tr><td rowspan="2">Gender</td><td>Female</td><td>250</td><td>75</td></tr><tr><td>Male</td><td>250</td><td>62</td></tr><tr><td rowspan="4">Age group</td><td>20s or younger</td><td>84</td><td>74</td></tr><tr><td>30s</td><td>84</td><td>39</td></tr><tr><td>40s</td><td>83</td><td>0</td></tr><tr><td>50s</td><td>83</td><td>24</td></tr><tr><td></td><td>60s</td><td>83</td><td>0</td></tr><tr><td rowspan="3">Household income</td><td>70s or older</td><td>83</td><td>0</td></tr><tr><td>Low</td><td>167</td><td>46</td></tr><tr><td>Middle High</td><td>166 167</td><td>43 48</td></tr><tr><td>Region</td><td>Capital region</td><td></td><td></td></tr><tr><td rowspan="2"></td><td></td><td>250</td><td>77</td></tr><tr><td>Other regions</td><td>250</td><td>60</td></tr></table>

## APPENDIX C RATIONALE CODING

Each rationale received up to three topics, regardless of direction: for example, both a high and a low risk of collision were coded as topic (1). Each of the five topics is defined below and illustrated with one rationale that received only that topic, translated from Korean.

(1) Collision or path obstruction. The robot may collide with pedestrians, cause them to trip, or block the walking path. Example (Gemma, chose no robot, day): “A small delivery robot on the walkway could disrupt the flow of pedestrians or bump into them, so on a crowded sidewalk the right image without the robot feels safer.”

(2) Threat or unfamiliarity. The size, human-like form, or approach of the robot feels threatening, its movement seems unpredictable, or it looks unfamiliar. Example (Gemma, chose no robot, day): “A humanoid robot in human form may feel threatening or unsettling because of its unfamiliar movements, so the right image without the robot feels much calmer and safer.”

(3) Openness, visibility, or lighting. The robot blocks or does not block the view, the path is open, the robot is easy to see and avoid, or its light illuminates the path. Example (Gemma, chose the delivery robot, night): “When passing through a dark alley, the light from the small delivery robot makes me less anxious and makes it safer to check the condition of the path.”

(4) Surveillance, accompaniment, or protection. The presence of the robot is reassuring because it watches or patrols the street, signals a managed area, gives a sense that someone is nearby, or may help in an emergency. Example (Gemma, chose the delivery robot, day): “Walking alone in a narrow, quiet alley can feel a little uneasy, but the delivery robot gives a sense of movement nearby, so it feels psychologically safer.”

(5) Robot function or trust in technology. The robot is harmless because it is a simple machine that only carries items, or it is trustworthy because its technology can be controlled. Example (Qwen, chose the delivery robot, night): “As an electronics engineer, I trust the technical safety of small robots, so a space with a small delivery robot feels much safer and calmer than one with a large humanoid robot.”

Table III shows the agreement between the human annotator and Claude on the 150 validation rationales described in Section III-C. The annotator did not assign topic (5) to any of these rationales, so its κ is not defined.

TABLE III: Agreement between the human annotator and Claude for each topic. Counts give the number of rationales that each coder assigned to the topic and sum to more than 150 because a rationale could receive more than one topic.

<table><tr><td>Topic</td><td>Annotator</td><td>Claude</td><td>κ</td></tr><tr><td>(1) Collision or path obstruction</td><td>70</td><td>71</td><td>0.83</td></tr><tr><td>(2) Threat or unfamiliarity</td><td>100</td><td>104</td><td>0.82</td></tr><tr><td>(3) Openness, visibility, or lighting</td><td>20</td><td>23</td><td>0.81</td></tr><tr><td>(4) Surveillance or protection</td><td>33</td><td>29</td><td>0.92</td></tr><tr><td>(5) Robot function or trust in technology</td><td>0</td><td>5</td><td>一</td></tr><tr><td>All topics</td><td>223</td><td>232</td><td>0.87</td></tr></table>

## APPENDIX D RATIONALE TOPICS OF QWEN

![](images/775cd231228303ddebbb1fdc0f140e2f8cc430a4c045b6b16c7a8ed4e15b39cf.jpg)  
Fig. 6: Share of rationale topics stated by Qwen3.8-27B for each choice, by (a) time of day and (b) persona gender. Shares within each bar sum to 100%, and numbers to the right give the number of responses.

Figure 6 shows that the rationales of Qwen described the two robots in a way similar to those of Gemma in Section IV-C. When Qwen chose the no-robot condition over the delivery robot, 64% of the rationale topics by day concerned collision or path obstruction, showing that the delivery robot was seen mainly as a physical obstacle. The humanoid was seen in two opposite ways. Rationales for avoiding the humanoid mostly cited threat or unfamiliarity (53% by day and 81% at night), whereas rationales for choosing the humanoid mostly cited surveillance (57% by day and 78% at night).

Figure 6a shows that time of day had little effect on the rationales for choosing the delivery robot, as help topics stayed at 19% by day and 20% at night. At night, openness, visibility, or lighting appeared more often instead. Figure 6b shows that persona gender also made little difference. Female and male personas gave similar help and risk topics for the delivery robot, and both mostly cited surveillance for the humanoid.

## APPENDIX E

## FREE RESPONSES OF HUMAN RESPONDENTS

15 of the 25 respondents described what mattered most in their safety judgments. Their responses, translated from Korean by the authors, are listed below in two groups. Gender and age group are given in parentheses.

## Robots as a physical risk or an unfamiliar presence

• “The road was very narrow, so having a robot there felt somewhat burdensome. It felt like it was blocking the way, so it did not feel safe.” (Male, 20s)

• “Because it is a robot, it could malfunction, so I worry about bumping into it up close. For the humanoid in particular, I saw on YouTube that it moves dangerously, like a seizure, when it falls, so that worries me.” (Male, 30s)

• “I judged that the delivery robot was unlikely to injure people, but the humanoid could cause harm through malfunction, so meeting it in the middle of the night would not feel safe.” (Male, 20s)

• “I mainly considered how predictable the robot’s behavior was and how unclear its purpose was. I worried about collisions because it was hard to predict where the humanoid and the delivery robot would move. I felt especially uneasy about the humanoid because it was hard to know its purpose and role.” (Female, 30s)

• “Humanoid robots do not feel familiar yet.” (Female, 50s)

• “It would be very scary to meet a humanoid on a street at night.” (Female, 20s)

• “I chose the one that did not feel unfamiliar, the one that did not feel scary.” (Female, 20s)

• “I judged that it was safe without a robot. If a robot had to be there, I considered whether it was a machine-like robot that did not resemble a human.” (Male, 30s)

## Robots as protection, depending on the context

• “During the day, it seems more natural and safer without a robot, but at night, I think a robot could actually make the street safer.” (Male, 20s)

• “I mainly compared safety based on my first impression of each photo. At night, when the street felt dangerous, a street with a robot felt safe, as if there were something to rely on. During the day, when the street felt safe, the robot felt like a threat instead. In short, my impression of the robot depended on the background.” (Male, 20s)

• “The humanoid was unfamiliar, so it did not feel safe. In the dark, a street with a delivery robot giving off light felt safer than an empty street. In daylight, I did not feel safe because I worried that people walking on the street might bump into the robot.” (Female, 20s)

• “I saw the humanoid as a robot that can actually act, so I mainly considered whether it could help me even a little in a dangerous situation. For the delivery robot, I focused less on security and my own safety and more on whether it would be an obstacle while walking.” (Male, 30s)

• “When I felt that someone was watching over me, a visible robot felt safer.” (Female, 50s)

• “The robot moved freely, and above all, it felt reassuring because it seemed like it would keep people safe.” (Male, 20s)

• “I looked at whether the form and function of the robot fit its surroundings.” (Male, 20s)

## APPENDIX F EFFECT SIZES OF TIME OF DAY AND GENDER

![](images/b1741be9daa3c16e0cd666fce8afaddba847a9a574c01d317003054b93b2a409.jpg)  
Human (N=25) Qwen3.8-27B (matched) Gemma4-26B (matched)  
Fig. 7: Effect sizes for humans and matched personas. Rows show (a) time of day and (b) gender, and columns show robot pairs. Each value is Cohen’s h for the change in the win rate of the underlined target condition, from day to night or from male to female. Error bars show 95% confidence intervals, and the dashed line marks no effect.

Figure 7 expresses the changes in win rate shown in Figure 4 as Cohen’s h. Positive values indicate a higher win rate at night or for female respondents and personas. The confidence intervals of humans are wide because the human sample included only 25 respondents.

For time of day, all three groups showed similar effects for the delivery robot against the no-robot condition (h = 0.35 for humans, 0.47 for Qwen, and 0.50 for Gemma). For the humanoid against the no-robot condition, the effect of Qwen (h = 0.64) stayed within the confidence interval of humans (h = 0.42). The effect of Gemma was large (h = 1.25) and fell outside the interval. Gemma also showed a medium effect for the humanoid against the delivery robot (h = 0.60), whereas humans and Qwen showed almost none.

For gender, all effects were below the threshold for a small effect $( | h | < 0 . 2 )$ . Humans and Qwen showed negative effects for the humanoid, while Gemma showed positive effects.