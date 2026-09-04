# INVESTIGATING THE ABILITY OF LARGE LANGUAGE MODELS TO ANALYZE RECIPES FOR DIABETES

PREPRINT

Revathy Venkataramanan, Aditya Luthra, Venkatesan Nadimuthu, Amit Sheth AI Institute, University of South Carolina 1112 Greene Street, Columbia, SC 29201 USA revathycv24@gmail.com

## ABSTRACT

Several studies have evaluated the ability of Large Language Models (LLMs) for meal planning, yielding positive outcomes. These models can process natural language inputs and leverage learned knowledge from their pretraining to generate meal plans. In this work, we investigate the ability of LLMs to analyze the suitability of given recipes for diabetes. The primary challenge for LLMs is to retrieve relevant dietary guidelines for diabetes, decompose recipes into ingredients and cooking methods, and apply these guidelines to determine the recipe’s suitability. To study these challenges, we employ three kinds of prompts namely, (i) Direct Query Prompt (ii) Context-Guided Prompt, and (iii) Exemplary Context Prompt that incorporate different levels of diabetes dietary guidelines from medical sources. We introduce a benchmark dataset curated for this investigation consisting of 7607 recipes that include 3807 recipes suitable for diabetes and 3800 recipes not suitable for diabetes. Our results demonstrate that most LLMs are cautious in predicting recipes as suitable to prevent detrimental outcomes. Further, the models that can reason using the dietary guidelines performed better in predicting the suitability of recipes for diabetes. Overall, Mistral-7B and Llama 70B showed superior performance to their counterparts.

Keywords Large Language Models · LLM Evaluation · Diabetes · Nutrition · Dietary Guidelines · Health Informatics · Prompt Engineering · Reasoning · Benchmark Dataset · Food Computing

## 1 Introduction

The World Health Organization estimates that 830 million people worldwide have diabetes [1], a condition manageable through dietary adjustments. Comprehensive dietary guidelines have been published by medical sources like MayoClinic and the Center for Disease Control (CDC). However, they are extensive, challenging to remember, and difficult to apply in daily meal planning. While the guidelines include examples, they are not exhaustive. For instance, vegetables and fruits are considered nutritious sources of healthy carbohydrates and rich fiber, suitable for diabetes management. However, certain fruits high in glucose should be avoided. This underscores the complexity of dietary decisions for diabetes. While AI-powered meal-tracking tools [2] have gained popularity over the years, they often lack targeted analysis for diabetes, likely due to liability concerns.

General-purpose LLMs such as ChatGPT, Bard, and Claude have become invaluable tools for the general public. They are frequently employed for everyday activities like meal planning [3], vacation itineraries [4], drafting emails, and providing educational assistance. There have been several studies that showcase the positive aspect of LLMs in meal planning [5]. At the same time, they are suggested to be augmented with the clinician’s advice [6]. On the other hand, studies have also shown the limitations of using LLMs for nutrition management, specifically for diabetes [7, 8].

In this study, we aim to investigate the ability of LLMs to assess the suitability of a given recipe for diabetes. Unlike meal planning, where LLMs can take a cautious approach by recommending meals derived using a subset of diabetes guidelines that they can retrieve or remember, assessing suitability demands a thorough understanding. This process requires comprehensive knowledge of the full scope of dietary guidelines. The core challenges for the LLMs include (i) Medical Knowledge Retrieval: Searching their vast embedding space for relevant, targeted and complete dietary guidelines for diabetes. (ii) Conceptual Understanding: Interpreting diabetes dietary concepts (what constitutes healthy carbohydrate or saturated fat). (iii) Deductive Analysis: Applying general dietary rules to specific cases (e.g., “Carrot is a vegetable, vegetables are healthy carbohydrates”) to analyze suitability. For this investigation, we utilize three types of prompts incorporating different levels of diabetic dietary guidelines: (i) Direct Query Prompt (ii) Context-Guided Prompt, and (iii) Exemplary Context Prompt. We further prompt the LLMs to reason using the keywords part of dietary guidelines such as healthy carbohydrates, trans fat, and so on. Four classes of LLMs namely, Mistral, Gemma2, Llama, and ChatGPT with varying sizes were chosen for this analysis. To conduct this investigation, we introduce a recipe dataset consisting of 7607 recipes with 3807 recipes suitable for diabetes gathered from medical sources and 3800 recipes not suitable for diabetes curated using keywords on Recipe1M dataset [9].

## 2 Related Work

LLMs are being explored for generating personalized diets aligned with nutritional guidelines, focusing on managing conditions like diabetes.

## 2.1 LLMs in Diet Management

Recent advances in LLMs have transformed food recommendation systems [10]. [5] developed a framework that identifies healthier and more sustainable meals by exploiting retrieval techniques and large language models. [11] introduced FoodSky, a food-specific LLM with enhanced reasoning for dietary tasks. [12] developed ChatDiet, a chatbot combining LLMs with causal reasoning for personalized recommendations. [13] used multi-task learning with knowledge graphs to balance user preferences and health needs. [14] developed a template-based GPT-4 nutrition assistant, validated with dietitian input for accuracy and relevance. [15] introduced “Dining on Details,” a framework using LLMs and multi-modal approaches for fine-grained food recognition in chronic disease dietary monitoring. LLMs are increasingly utilized for dietary personalization. [16] combined ChatGPT with a generative model to provide robust meal plans aligned with established nutritional guidelines.

## 2.2 LLMs for Disease Diet Management

LLMs have been employed and evaluated for generating meal plans. [17] developed a knowledge-infused conversational health agent that integrates dietary guidelines for diabetes management, outperforming general LLMs in reliability. [18] focused on diabetic nutrition with DIAFOODS, a tool that personalizes dietary recommendations to optimize glycemic control. Additionally, [19] demonstrated the role of structured nutritional strategies in managing Type 2 Diabetes Mellitus, emphasizing individualization. [20] introduced an AI dietitian leveraging LLMs and image recognition for managing diabetes, which integrates ingredient identification and meal assessments. [21] introduced the RISE framework, which augments LLMs with retrieval systems to provide more accurate and comprehensible responses to diabetes-related queries. LLMs have also been evaluated for meal planning in other conditions such as gastrointestinal [22], Mediterranean diet [23] and non-communicable diseases [24].

## 3 Approach

## 3.1 Dataset Creation

(i) Diabetes Specific Recipes: Several reliable medical websites provide curated lists of recipes designed specifically for individuals managing diabetes. These recipes serve as practical guides to help diabetes patients control their condition through healthy dietary choices. In our work, we utilized MayoClinic [25], Diabetes UK [26] and Diabetes Hub [27] to compile a collection of 3,807 recipes. This approach ensures that the dataset is grounded in medical expertise, supporting its relevance and usefulness for dietary management research.

(ii) Non-Diabetes Specific Recipes: MayoClinic [28] provides comprehensive guidelines for diabetes patients, detailing both recommended and discouraged food items. For example, foods high in refined sugars and saturated fats are advised against, while high-fiber vegetables and whole grains are encouraged. We curated a set of keywords listed by MayoClinic to identify recipes that may not be suitable for diabetes. The keywords include ‘ribs’, ‘pork bacon’, ‘sausage’, ‘baked’, ‘shortening’, ‘pancakes’, ‘cookies’, ‘tart’, ‘pudding’, ‘pickles’, ‘soda’, ‘syrup’, ‘churro’, and a list of canned fruits and vegetables. Leveraging the Recipe1M dataset [9], we conducted keyword searches within recipe title and ingredients. To ensure class balance, we randomly selected 3800 recipes from the result to construct the dataset.

Table 1: Performance results of LLMs in analyzing the suitability of recipes for diabetes
<table><tr><td colspan="4">Prompt-1</td><td colspan="3">Prompt-2</td><td colspan="3">Prompt-3</td></tr><tr><td>Models</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>Accuracy</td><td>Precision</td><td>Recall</td></tr><tr><td>Mistral 7B</td><td>0.84</td><td>0.84</td><td>0.84</td><td>0.77</td><td>0.88</td><td>0.63</td><td>0.79</td><td>0.54</td><td>0.53</td></tr><tr><td>Mistral 12B</td><td>0.60</td><td>0.92</td><td>0.21</td><td>0.62</td><td>0.67</td><td>0.45</td><td>0.63</td><td>0.74</td><td>0.41</td></tr><tr><td>Gemma2 2B</td><td>0.51</td><td>0.47</td><td>0.34</td><td>0.78</td><td>0.86</td><td>0.67</td><td>0.75</td><td>0.89</td><td>0.58</td></tr><tr><td>Gemma2 9B</td><td>0.70</td><td>0.53</td><td>0.47</td><td>0.76</td><td>0.96</td><td>0.54</td><td>0.81</td><td>0.95</td><td>0.65</td></tr><tr><td>Gemma2 27B</td><td>0.83</td><td>0.94</td><td>0.71</td><td>0.78</td><td>0.96</td><td>0.59</td><td>0.79</td><td>0.96</td><td>0.61</td></tr><tr><td>Llama3.1 8B</td><td>0.61</td><td>0.90</td><td>0.25</td><td>0.72</td><td>0.90</td><td>0.50</td><td>0.81</td><td>0.90</td><td>0.70</td></tr><tr><td>Llama3.1 70B</td><td>0.69</td><td>0.96</td><td>0.40</td><td>0.83</td><td>0.92</td><td>0.72</td><td>0.85</td><td>0.91</td><td>0.79</td></tr><tr><td>Llama3.2 2B</td><td>0.56</td><td>0.64</td><td>0.29</td><td>0.51</td><td>0.95</td><td>0.03</td><td>0.54</td><td>0.90</td><td>0.09</td></tr><tr><td>Chatgpt 3.5</td><td>0.74</td><td>0.50</td><td>0.49</td><td>0.54</td><td>0.90</td><td>0.09</td><td>0.74</td><td>0.50</td><td>0.49</td></tr></table>

## 3.2 Model Details

Four classes of LLMs namely, Mistral, Llama, Gemma2, and ChatGPT with small (less than 3 billion), medium (between 3 and 9 billion), and large (greater than 9 billion) sizes were chosen for this evaluation. The models are (i) Mistral 7B and Mistral 12B [29], a model leveraging transformers architecture with sliding window attention and pre-fill chunking, (ii) Gemma2 2B, Gemma2 9B and Gemma2 27B, light-weight models with local-global attentions and group-query attention [30] (iii) Llama 3.1 8B, Llama 3.1 70B, Llama 3.2 3B, models with SwiGLU activation function and Rotary Embeddings [31] (iv) ChatGPT-3.5.

## 3.3 Prompting Techniques

To evaluate the core challenges presented, we employ three types of prompt to investigate the ability of LLMs to analyze the suitability of recipe for diabetes.

## 3.3.1 Direct Query Prompt (Prompt-1):

In this approach, a straightforward, single-line query is presented alongside the recipe. This aims to evaluate whether LLMs can recognize that the query pertains to a diabetes-specific diet and retrieve relevant medical guidelines from their vast embedding space learned during their training to analyze meals by reasoning over recipe contents. This refers to medical knowledge retrieval.

Give one word Answer YES/NO. Is the Given Recipe safe for diabetes? The Title of the Recipe is: {recipe title}. The Ingredients of{recipe title} are: {comma separated list ofingredients}. The Instructions of {recipe title} are: {list of instructions}.

## 3.3.2 Context Guided Prompt (Prompt-2):

This prompt incorporates explicit dietary guidelines. Though models can utilize this knowledge instead of retrieving from its search space, this requires a conceptual understanding what are healthy carbohydrates, saturated fat and other dietary concepts. This is essential to apply this knowledge to individual recipe ingredients to analyze their suitability. The guidelines were curated from MayoClinic [28] and NIDDK [32], which outline the recommended foods and those to avoid. This also evaluates if the LLMs prioritize the external knowledge provided over its internal knowledge.

You will be given a recipe with title, list ofingredients and instruction to make the recipe. You will also be given dietary guidelinesfor diabetes that tells items to recommend and avoid. Now, based on this information, tell me in one word if the recipe given is suitable for diabetes or not using YES or NO options. Then provide 2-3 line reasoning along with YES or NO as why a recipe is suitable or not suitable using those keywords in the given dietary guidelines.   
Recipe Title: {add recipe title}

Recipe Ingredients: {list of ingredients separated by comma} Recipe Instructions: {a paragraph ofinstructions}

Diabetes Dietary Guidelines: For diabetes, the recommended items are (i) Healthy Carbohydrates (ii) Fibre rich foods (iii) Heart Healthy fish (iv) Good Fats (v) Animal Protein (vi) Protein (vii) Non fat, Lowfat. The items that should be avoided are (i) Saturatedfat (ii) High dairyfat (iii) Saturatedfat, animal protein (iv) Trans fat (v) Cholesterol (vi) Added sugar (vii) Refined/processed carbohydrates (viii) Ultra processedfoods (ix) deepfriedfoods

## 3.3.3 Exemplary Context Prompt (Prompt-3):

This prompt is an enhanced version of the Context Guided Prompt, that includes specific examples for dietary concepts. This approach aims to achieve a balance between providing direct and indirect examples. For instance, vegetables are included as examples of healthy carbohydrates within the prompt without explicitly labeling whether an ingredient is a vegetable. In contrast, categories such as saturated fats include direct examples like butter, beef, and sausages. This prompt assesses how effectively LLMs can perform deductive analysis. For example, deducing that carrot is a vegetable and vegetable are healthy carbohydrates. Further, this evaluation seeks to determine whether the explicit inclusion of such comprehensive information enhances the LLMs’ performance.

You will be given a recipe with title, list ofingredients and instruction to make the recipe. You will also be given dietary guidelinesfor diabetes that tells items to recommend and avoid. Now, based on this information, tell me in one word if the recipe given is suitable for diabetes or not using YES or NO options. Then provide 2-3 line reasoning along with YES or NO as why a recipe is suitable or not suitable using those keywords in the given dietary guidelines.

Recipe Title: {add recipe title}

Recipe Ingredients: {list ofingredients separated by comma}

Recipe Instructions: {a paragraph ofinstructions}

Diabetes Disease Context: For diabetes, the recommended items are (i) Healthy Carbohydrates such as fruits, vegetables, whole grains, legumes and non-fat dairy (ii) Fibre-rich foods such as fruits, vegetables, nuts, legumes, and whole grains (iii) Heart Healthy fish such as salmon, tuna, mackerel and sardines (iv) Good Fats such as avocado, nuts, olive oil, canola oil and peanut oil (v) Animal Protein such as lean meat, chicken without skin, turkey without skin, fish and eggs (vi) Protein such as nuts, peanuts, dried beans, chickpeas, split peas and tofu (vii) Non-fat, Lowfat such as milk, lactose-free milk, yogurt, and cheese The items that should be avoided are (i) Saturatedfats such asfatty meat, pork, eggs, palm oil, coconut oil and red meat (ii) High dairyfat such as butter (iii) Saturated fat, animal protein such as beef, hot dogs, sausage and bacon (iv) Trans fat such as processed snacks, baked goods, shortening, stick margarine, refrigerated doughs, pastries, cookies, crackers, pies, and stick butter replacements (v) Cholesterol such as high-fat dairy, high-fat animal protein, egg yolks, liver, organ meat (vi) Added sugar such as soda, sports drinks, energy drinks, candy, cannedfruits, jams, jellies and ice cream (vii) Refined/processed carbohydrates such as white bread, pasta, juice, sweets, candy, cake (viii) Ultra processedfoods such as hot dogs,friedfish sticks and fast food burgers (ix) deep fried foods such as potato fries and chicken nuggets

## 3.4 Experimentation

Each pretrained model was downloaded and run using 8xV100-40GB machines. Each model was prompted with one recipe at a time individually for each prompt. The results were stored along with explanations and reasoning provided by LLMs. For prompt-2 and prompt-3, the models were prompted to provide reasons for their decision using the concepts given in the dietary guidelines. Chatgpt-3.5 was accessed using its API.

## 4 Result and Discussion

Model Performance: Table 1 provides the performance metrics for all models across the three prompt types: Direct Query Prompt (Prompt-1), Context Guided Prompt (Prompt-2), and Exemplary Context Prompt (Prompt-3). Mistral 7B and Llama3.1 70B exhibit consistent performance with minimal variation between precision and recall across all prompts compared to other models. Mistral-7B performed well in Direct Query Prompt, and Llama3.1-70B in Context Guided Prompt and Exemplary Prompt. Llama3.1 8B, a medium-sized model also seems to have benefited from Exemplary Context Prompt. While most Gemma2 class models have high precision, the recall is low. As shown in Figure 1, recall (dotted line) tends to be lower than precision (dashed line) across all prompts for most models. This indicates that the LLMs may be biased toward predicting recipes as “not suitable for diabetes” to avoid false positive errors. Consequently, the models appear overly cautious in classifying recipes as “suitable.” In this context, the cost of false positives, incorrectly identifying a harmful recipe as safe for diabetes, could have adverse outcomes and the models seem to be aware of it.

![](images/bd62759b5cbf160758a1737e1d806502367c3cfccc39b10a2fbd9df1c7c27147.jpg)  
Figure 1: Performance metrics of all the LLMs across all three prompts visualized to study correlations. The number -1, -2 and -3 denote results of prompt-1, prompt-2 and prompt-3

Precision-Recall Stability Metric: Intuitively, significant variation between precision and recall suggests that the model favors one class over the other. To quantify the variation between high precision and low recall, we proposed a stability metric in Equation 1. A higher stability score indicates lower variation, signifying a more balanced and stable model. A score of 1 represents a highly stable model, and a score of 0 indicates instability.

$$
S t a b i l i t y = 1 - \vert P r e c i s i o n - R e c a l l \vert\tag{1}
$$

Impact of Reasoning on Model Performance and Stability: In Context Guided Prompt (prompt-2) and Exemplary Context Prompt (prompt-3), models were instructed to justify their decisions using keywords derived from dietary guidelines, such as Healthy Carbohydrates, Trans Fat, Cholesterol, and others. Since dietary guidelines were not incorporated into the Direct Query Prompt (prompt-1), models were not instructed to provide reasoning. Gemma2 27B, a lightweight faster model, was used to extract keywords from the reasoning paragraph provided by LLMs for each recipes. The results showed that models classified a recipe as suitable for diabetes justifying by leveraging keywords from both the “recommend” and “avoid” sections of the guidelines. For example, a model might justify its prediction by stating that a recipe lacks saturated and trans fats, aligning with the “recommend” criteria. Consequently, when the model predicted a positive outcome (suitable), we utilized keywords from the “recommend” section for validation. Conversely, for negative predictions (not suitable), we extracted relevant keywords from the “avoid” section to evaluate the model’s rationale.

The heatmap presented in Figure 2 shows Mistral 7B produced the highest amount of diverse dietary guideline keywords in its reasoning. This supports the results presented in Table 1 where Mistral 7B showed steady results across all prompts with less variation between precision and recall. Llama3.1 70B returned keywords clustered around good fats, healthy carbohydrates, and fibre-rich foods. To correlate better, we computed the average F1-score, average stability score (Equation 1), and the average count of keywords across all prompts for a given model. Figure 3 demonstrates that the more keywords produced by the model in its reasoning, the better the F1-score and/or better the stability (les variation between precision and recall). It is to be noted that Gemma2-2B is the model with the second-highest stability score that correlates with the second-highest amount of keywords provided in the reasoning. Llama3.1-70B has the third-highest average of keyword count. ChatGPT and Llama3.2-2B had the least amount of keywords in their reasoning which correlates with lower F1-score. If a model confidently reasons with sufficient concepts from dietary guidelines, the model’s stability and performance is better.

Deductive Analysis: It is evident from Figure 3 that if a model learned to apply the deductions and provide reasoning using given dietary guidelines, the better the performance. For example, if a model can deduce carrot → vegetable and vegetable → healthy carbohydrate and healthy carbohydrates → suitable for diabetes, the model analyzes the recipes better. This shows that the ability to reason over its decisions improves the model’s stability and performance. However, we cannot say for certain that the models did such detailed reasoning over individual ingredients, which we aim to evaluate in our future work. The individual F1 scores, stability score, and keyword counts can be found here - https://tinyurl.com/ycxf48bz.<sup>2</sup>

![](images/9961fd068d79da2aea53489022454b5021e26a60fcb9e92f58e00ce4951e118d.jpg)  
Figure 2: Keywords from diabetes dietary guidelines used by the models to reason over their decision making. The values are normalized using min-max scaling. P2 denote prompt-2 and P3 denote prompt-3

Medical Knowledge Retrieval: Prompt-1 requires models to retrieve dietary guidelines from their internal embedding space to analyze the recipe’s suitability. To assess if the retrieved knowledge is comprehensive, we re-ran experiments for prompt-1 instructing the models to reason over their decision-making. Mistral 7B, Gemma 27B, and Llama3.1 70B, one model from each class were chosen to comparatively study the reasoning trends for all three prompts. Results showed prompt-1 produced fewer average keywords (765) than prompt-2 (836) and prompt-3 (789). Models retrieved internal medical knowledge, but external prompts provided easier access resulting in better performance supported by reasoning. However, for ChatGPT, prompt-2’s external knowledge conflicted with internal knowledge, yielding high precision but low recall.

Conceptual Understanding: Figure 2 highlights that models confidently predict concepts like fiber-rich foods, good fats, healthy carbohydrates, refined carbohydrates, and added sugar demonstrating a solid understanding of these areas. Because of this, LLMs may have exhibited positive outcomes in meal planning by suggesting meals that confirm to these concepts. However, they perform less effectively with categories such as non-fat, low-fat, protein, cholesterol, high dairy fat, trans fats, deep-fried foods, and ultra-processed foods. Deep-fried food and trans-fat food may require analyzing cooking methods. This underscores the complexity of recipe evaluation, as it demands assessing both ingredients and cooking methods against dietary guidelines individually which is challenging for general-purpose LLMs [33].

![](images/0fde956c3ac33458e2abc16d5d5a8c444531f9de1e71fe2bb869b9ba54d6dfad.jpg)  
Figure 3: Correlation graph illustrating the relationship between models’ reasoning ability and their performance and stability.

Deep Dive into Mistral-7B: Figure 1 shows that larger models do not necessarily guarantee better performance. Mistral 7B, which generally prioritizes and aligns with given external knowledge [29], demonstrated superior reasoning using dietary guideline keywords. It averaged 1200 keywords for prompt-2 and 1008 for prompt-3, the highest among all models. The decrease in keywords from prompt-2 to prompt-3 corresponds to a drop in performance as well (Table 1), reinforcing the correlation between reasoning using keywords and model performance. The performance decline in prompt-3 may be due to prompt length or insufficient examples of dietary concepts like trans fat, as Mistral 7B heavily relies on external knowledge despite its internal understanding.

Curious Case of ChatGPT: The precision and recall did not vary for prompt-1 and prompt-3 for ChatGPT. In other words, the model was stable even though the accuracy was low. However, for prompt-2, which included only diabetes dietary concepts and not examples, the internal knowledge of ChatGPT seems to have conflicted with the external knowledge given. A well-recognized issue [34] that resulted in a low recall. However, the examples given in prompt-3 improved the recall. Going against the well-known trend that bigger models perform better, ChatGPT showed inconsistencies due to conflicting knowledge. To add, a bigger embedding search space makes it difficult for the models to retrieve targeted medical advice [35].

## 4.1 Technical Challenges

Revisiting the challenges presented in the introduction, (i) Medical Knowledge Retrieval: The results of prompt-1 show that the models trained for general purpose find it challenging to search their vast embedding space for targeted relevant dietary guidelines compared to explicitly providing knowledge via prompts. Complete and comprehensive retrieval of medical knowledge is critical in this domain. (ii) Conceptual Understanding: The models seem to have a good understanding of straightforward concepts such as healthy carbohydrates, fiber-rich foods, added sugar, saturated fats, and refined carbohydrates. However, a thorough understanding of all the concepts is essential to make informed decisions. Deep-fried food requires knowledge about cooking methods as well. (iii) Deductive Analysis: To a certain extent, we can say that the models were able to deduce and reason using dietary concepts. However, it is not certain if the models individually applied reasoning over ingredients and cooking methods to derive conclusions. In our future work, we aim to evaluate the ability of LLMs to decompose a recipe into ingredients and cooking methods, reason over each ingredient and cooking method in a recipe to evaluate if LLMs can perform compositional reasoning [33].

## 4.2 Clinical Challenges

(i) Alignment to Medical Guidelines: While analyzing and reasoning, consistent alignment with medical guidelines is critical. This is a challenge as LLMs are known to hallucinate [36]. (ii) Knowledge Limitation: Diabetes guidelines require knowledge of nutrition, glycemic index, and other factors, making it challenging to integrate this information from vast search spaces. (iii) Expert Validation: The model needs to be validated by domain experts which involves tedious human labor.

## 4.3 Ethical Challenges

(i) Safety: Any incorrect or misleading information generated by LLMs poses significant risks, particularly in a high-stakes domain such as diet and chronic conditions. It could lead to adverse health outcomes if this advice is blindly followed. (ii) Biased Representation: There is an unequal representation of dietary guidelines, as only a few categories are used as reasoning despite having several categories to recommend or avoid.

## 5 Conclusion

In this work, we investigated the ability of LLMs to analyze the suitability of a given recipe for diabetes. Unlike meal planning, where LLMs can take a cautious approach by recommending meals derived using guidelines that they can retrieve or remember, assessing suitability demands a full scope of guidelines. Mistral 7B, Llama3.1-70B showed superior performance overall to its counterparts. The Gemma2 class of models had high precision and low recall. Most models were conservative in predicting the recipes suitable for diabetes due to the high-stakes nature of this domain. Of the core challenges discussed, most models were able to retrieve the medical knowledge from their vast embedding space. However, better performance was shown when external knowledge was given in the prompt. Conversely, this external knowledge also conflicted with the internal knowledge of the model in the case of ChatGPT. The models had a good understanding of concepts such as healthy carbohydrates, good fat, added sugar, and refined carbohydrates. On the other hand, models need to understand and utilize categories such as protein, non-fat, low-fat, deep-fried food, and ultra-processed food in their reasoning. In addition, the models that were able to deductively analyze and reason using dietary concepts showed better performance. In our future work, we aim to evaluate if the LLMs can individually reason over ingredients and cooking methods to drive a conclusion on the suitability of recipes for diabetes.

## References

[1] World Health Organization. Diabetes, 2024. URL https://www.who.int/news-room/fact-sheets/ detail/diabetes. Accessed: 2024-11-26.

[2] Weiqing Min et al. A survey on food computing. ACM Computing Surveys (CSUR), 52(5):1–36, 2019.

[3] Corin Cesaric. My experience using chatgpt for meal prep was more than a moneysaver. CNET, 2024. URL https://www.cnet.com/home/kitchen-and-household/ my-experience-using-chatgpt-for-meal-prep-was-more-than-a-money-saver/. Accessed: 2024-11-26.

[4] Seunghun Shin, Jungkeun Kim, Eunji Lee, Yerin Yhee, and Chulmo Koo. Chatgpt for trip planning: the effect of narrowing down options. Journal ofTravel Research, page 00472875231214196, 2023.

[5] Alessandro Petruzzelli, Cataldo Musto, Michele Ciro Di Carlo, Giovanni Tempesta, and Giovanni Semeraro. Recommending healthy and sustainable meals exploiting food retrieval and large language models. In Proceedings ofthe 18th ACM Conference on Recommender Systems, pages 1057–1061, 2024.

[6] Valentina Ponzo, Ilaria Goitre, Enrica Favaro, Fabio Dario Merlo, Maria Vittoria Mancino, Sergio Riso, and Simona Bo. Is chatgpt an effective tool for providing dietary advice? Nutrients, 16(4):469, 2024.

[7] Medical Express. Much hyped ai products like chatgpt can provide medics with ’harmful’ advice, study says, 2024. URL https://medicalxpress.com/news/2024-09-hyped-ai-products-chatgpt-medics. html#google\_vignette. Accessed: 2024-11-28.

[8] Naja et al. Artificial intelligence chatbots for the nutrition management of diabetes and the metabolic syndrome. European Journal of Clinical Nutrition, 78(10):887–896, 2024.

[9] Amaia Salvador, Nicholas Hynes, Yusuf Aytar, Javier Marin, Ferda Ofli, Ingmar Weber, and Antonio Torralba. Learning cross-modal embeddings for cooking recipes and food images. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3020–3028, 2017.

[10] Howard R Moskowitz, Helena MA Bolini, Stephen D Rappaport, Pedro Pio C Augusto, Taylor Mulvey, and Vanessa Marie B Arcenas. Foods: Trends, fads, and mind-sets as envisioned by ai using llms (large language models). Acta Scientific NUTRITIONAL HEALTH (ISSN: 2582-1423), 8(8), 2024.

[11] Pengfei Zhou, Weiqing Min, Chaoran Fu, Ying Jin, Mingyu Huang, Xiangyang Li, Shuhuan Mei, and Shuqiang Jiang. Foodsky: A food-oriented large language model that passes the chef and dietetic examination. arXiv preprint arXiv:2406.10261, 2024.

[12] Zhongqi Yang, Elahe Khatibi, Nitish Nagesh, Mahyar Abbasian, Iman Azimi, Ramesh Jain, and Amir M. Rahmani. Chatdiet: Empowering personalized nutrition-oriented food recommender chatbots through an llm-augmented framework. Smart Health, 32:100465, June 2024. ISSN 2352-6483. doi: 10.1016/j.smhl.2024.100465. URL http://dx.doi.org/10.1016/j.smhl.2024.100465.

[13] Yi Chen, Yandi Guo, Qiuxu Fan, Qinghui Zhang, and Yu Dong. Health-aware food recommendation based on knowledge graph and multi-task learning. Foods, 12(10), 2023. ISSN 2304-8158. doi: 10.3390/foods12102079. URL https://www.mdpi.com/2304-8158/12/10/2079.

[14] Annalisa Szymanski, Brianna L Wimer, Oghenemaro Anuyah, Heather A Eicher-Miller, and Ronald A Metoyer. Integrating expertise in llms: Crafting a customized nutrition assistant with refined template instructions. In Proceedings ofthe 2024 CHI Conference on Human Factors in Computing Systems, CHI ’24, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400703300. doi: 10.1145/3613904.3641924. URL https://doi.org/10.1145/3613904.3641924.

[15] Jesús M. Rodríguez-de Vera, Pablo Villacorta, Imanol G. Estepa, Marc Bolaños, Ignacio Sarasúa, Bhalaji Nagarajan, and Petia Radeva. Dining on details: Llm-guided expert networks for fine-grained food recognition. In Proceedings ofthe 8th International Workshop on Multimedia Assisted Dietary Management, MADiMa ’23, page 43–52, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9798400702846. doi: 10.1145/3607828.3617797. URL https://doi.org/10.1145/3607828.3617797.

[16] Ilias Papastratis, Dimitrios Konstantinidis, Petros Daras, and Kosmas Dimitropoulos. Ai nutrition recommendation using a deep generative model and chatgpt. Scientific Reports, 14(14620), 2024. doi: 10.1038/s41598-024-65438-x. URL https://doi.org/10.1038/s41598-024-65438-x.

[17] Mahyar Abbasian, Zhongqi Yang, Elahe Khatibi, Pengfei Zhang, Nitish Nagesh, Iman Azimi, Ramesh Jain, and Amir M Rahmani. Knowledge-infused llm-powered conversational health agent: A case study for diabetes patients. arXiv preprint arXiv:2402.10153, 2024.

[18] James Darryl D. Bungay, Sheila Marie M. Matias, Andrew A. Santos, Paolo C. Arceno, Charlene F. Dalipe, and Jenny-Ann B. Lomotan. Diafoods: A content based food recommendation system for diabetic patients. In 2024 IEEE 9th International Conference on Computational Intelligence and Applications (ICCIA), pages 262–267, 2024. doi: 10.1109/ICCIA62557.2024.10719360

[19] Tatiana Palotta Minari, Lúcia Helena Bonalume Tácito, Louise Buonalumi Tácito Yugar, Sílvia Elaine Ferreira-Melo, Carolina Freitas Manzano, Antônio Carlos Pires, Heitor Moreno, José Fernando Vilela-Martin, Luciana Neves Cosenso-Martin, and Juan Carlos Yugar-Toledo. Nutritional strategies for the management of type 2 diabetes mellitus: A narrative review. Nutrients, 15(24), 2023. ISSN 2072-6643. doi: 10.3390/nu15245096. URL https://www.mdpi.com/2072-6643/15/24/5096.

[20] Haonan Sun, Kai Zhang, Wei Lan, Qiufeng Gu, Guangxiang Jiang, Xue Yang, Wanli Qin, and Dongran Han. An ai dietitian for type 2 diabetes mellitus management based on large language and image recognition models: Preclinical concept validation study. J Med Internet Res, 25:e51300, Nov 2023. ISSN 1438-8871. doi: 10.2196/ 51300. URL https://www.jmir.org/2023/1/e51300.

[21] Dingqiao Wang, Jiangbo Liang, Jinguo Ye, Jingni Li, Jingpeng Li, Qikai Zhang, Qiuling Hu, Caineng Pan, Dongliang Wang, Zhong Liu, Wen Shi, Danli Shi, Fei Li, Bo Qu, and Yingfeng Zheng. Enhancement of the performance of large language models in diabetes education through retrieval-augmented generation: Comparative study. J Med Internet Res, 26:e58041, Nov 2024. ISSN 1438-8871. doi: 10.2196/58041. URL https: //www.jmir.org/2024/1/e58041.

[22] Bernadette Lamb, Jonathan Herskovitz, Marta Jonson, Harlan Sayles, and Faruq Pradhan. S2185 use of large language model (llm) chatbots for the generation of specialized gastrointestinal diet meal plans: A pilot study. Official journal ofthe American College ofGastroenterology| ACG, 119(10S):S1562, 2024.

[23] Chao Chen, Xinxin Li, and Hongmiin Luo. Evaluation of accuracies of large language models in answering clinical questions related to mediterranean diet on cardiodiabesity. Interdisciplinary Nursing Research, 3(3): 157–162, 2024.

[24] Ilias Papastratis, Andreas Stergioulas, Dimitrios Konstantinidis, Petros Daras, and Kosmas Dimitropoulos. Can chatgpt provide appropriate meal plans for ncd patients? Nutrition, 121:112291, 2024.

[25] Mayo Clinic. Diabetes meal plan recipes, 2024. URL https://www.mayoclinic.org/healthy-lifestyle/ recipes/diabetes-meal-plan-recipes/rcs-20077150. Accessed: 2024-11-28.

[26] Diabetes UK. Diabetes recipes, 2024. URL https://www.diabetes.org.uk/living-with-diabetes/ eating/recipes. Accessed: 2024-11-28.

[27] Diabetes Food Hub. Recipes | diabetes food hub, 2024. URL https://diabetesfoodhub.org/recipes/. Accessed: 2024-11-28.

[28] Mayo Clinic. Diabetes diet: Create your healthy-eating plan, 2024. URL https://www.mayoclinic.org/ diseases-conditions/diabetes/in-depth/diabetes-diet/art-20044295. Accessed: 2024-11-29.

[29] Jiang et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023. URL https://arxiv.org/pdf/2310.06825.

[30] Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, et al. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024.

[31] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

[32] National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK). Healthy living with diabetes, 2024. URL https://www.niddk.nih.gov/health-information/diabetes/overview/ healthy-living-with-diabetes. Accessed: 2024-11-29.

[33] Dziri et al. Faith and fate: Limits of transformers on compositionality. Advances in Neural Information Processing Systems, 36, 2024.

[34] Hao Zhang, Yuyang Zhang, Xiaoguang Li, Wenxuan Shi, Haonan Xu, Huanshuo Liu, Yasheng Wang, Lifeng Shang, Qun Liu, Yong Liu, et al. Evaluating the external and parametric knowledge fusion of large language models. arXiv preprint arXiv:2405.19010, 2024.

[35] Zhongzhen Huang, Kui Xue, Yongqi Fan, Linjie Mu, Ruoyu Liu, Tong Ruan, Shaoting Zhang, and Xiaofan Zhang. Tool calling: Enhancing medication consultation via retrieval-augmented large language models. arXiv preprint arXiv:2404.17897, 2024.

[36] Vipula Rawte, Amit Sheth, and Amitava Das. A survey of hallucination in large foundation models. arXiv preprint arXiv:2309.05922, 2023.