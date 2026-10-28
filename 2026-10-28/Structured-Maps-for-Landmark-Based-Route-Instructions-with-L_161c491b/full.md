# Structured Maps for Landmark-Based Route Instructions with LLMs

Mifu Matsumoto<sup>1[0009</sup>−<sup>0001</sup>−<sup>5392</sup>−<sup>8393]</sup>, Yuki Nakanishi<sup>1[0009</sup>−<sup>0006</sup>−<sup>3226</sup>−<sup>8564]</sup>, Tomoya Hashiguchi<sup>1[0000</sup>−<sup>0003</sup>−<sup>0424</sup>−<sup>2755]</sup>, and Hiroaki Ohshima<sup>1[0000</sup>−<sup>0002</sup>−<sup>9492</sup>−<sup>2246]</sup>

University of Hyogo, Hyogo, Japan

Abstract. To generate landmark-based route instructions, models must understand both map geometry and human landmark preferences. Although large language models (LLMs) show great abilities in language tasks, the best way to structure map data for these models is not yet established. This paper proposes a structured text representation for landmark selection tasks in route navigation, utilizing SVG format to encode spatial features. The structured text maintains a geometric connection with rendered map images, allowing language models to directly access spatial and semantic data. Using a new dataset created from humanwritten route descriptions and OpenStreetMap data, we evaluate models on two tasks: landmark selection at turns and around destinations. Experiments with five LLMs show that the proposed structured text input performs as well as or better than image-based PNG inputs. In addition, task-specific fine-tuning enables smaller local LLMs to efectively leverage map layouts to achieve high selection accuracy. These results demonstrate that structured text is an eficient and lightweight interface for spatial reasoning in human-oriented navigation.

Keywords: large language models, structured maps, landmark selection, route instructions, OpenStreetMap

## 1 Introduction

Navigation systems usually describe routes using quantitative information, such as directions, street names, or distances like “turn right after 300 meters.” Although these instructions are precise, users often find them dificult to follow because humans cannot accurately estimate distances or find abstract road names in physical environments[2].

In everyday life, people prefer to use recognizable landmarks, saying “turn right at Starbucks” or “the destination is next to McDonald’s.” These landmarkbased instructions are intuitive because they connect navigation actions with visible features, thereby reducing the cognitive load during movement.

This paper addresses the problem of landmark selection from local maps using large language models (LLMs). Figure 1 illustrates the overview of this landmark selection task. While LLMs excel at language tasks, understanding map geometry to select appropriate landmarks remains a major challenge. A straightforward approach is to render a map as an image for vision-language models (VLMs). However, image-based inputs force models to infer spatial relations directly from raw pixels, which is highly challenging when reading small text labels or identifying exact relative positions [24].

![](images/483f71f690c4801eea44572af9e5787325a239a10806e9cefa61554db65fe0d8.jpg)  
Fig. 1. Overview of the landmark selection task. The framework takes local geographic data as input, which is then processed by an LLM to select and output the optimal landmark.

To solve this problem, we propose representing local map information in a structured SVG text format. Because SVG functions as both text and a geometric layout, it provides LLMs with direct access to coordinates and semantic categories without relying on pixel analysis. This text-based representation maintains a direct geometric connection with the rendered map image, enabling a fair and controlled comparison between the two modalities using the exact same underlying data.

To evaluate this approach, we establish two landmark selection tasks: selection at turns and selection around destinations. For our experiments, we use a dataset created from human-written restaurant access descriptions linked to OpenStreetMap(OSM) objects, reflecting real human landmark choices. Using this dataset, we investigate the following two research questions:

– RQ1: Which input representation is more efective for landmark selection, a rendered PNG map or an SVG text map?

– RQ2: Does task-specific fine-tuning on SVG text maps allow smaller local LLMs to improve landmark selection performance?

This study provides three main contributions:

– Dataset: We define two human-centric landmark selection tasks and construct a dataset linking real human descriptions with OSM objects.

– SVG Input: We show that the proposed structured SVG text input performs as well as or better than image-based PNG inputs for landmark selection.

– Fine-Tuning: We demonstrate that task-specific fine-tuning allows smaller local LLMs to understand map layouts and learn human landmark preferences efectively.

## 2 Related Work

This section reviews prior research on landmark-based navigation, map-based text generation, and the spatial capabilities of LLMs.

## 2.1 Landmarks in Spatial Cognition and Landmark Selection

Landmarks are vital to human spatial cognition. Humans organize spatial knowledge around recognizable objects rather than abstract coordinates, using them as cognitive points to orient themselves and remember routes [11, 26].

In route descriptions, people frequently select landmarks near decision points like intersections [3, 17]. A high-quality landmark must be recognizable and memorable [22], and landmark-based instructions significantly reduce cognitive load compared to distance-based methods [15, 4]. Furthermore, empirical evidence shows humans prefer qualitative place names over quantitative coordinates for location expressions [20].

However, automating this process is challenging because an optimal landmark must balance spatial relevance, visual salience, and communicative utility; simply choosing the nearest object is often insuficient. This motivates our work to evaluate whether LLMs can accurately mimic human landmark selection.

## 2.2 Route and Destination Description Generation

Generating route descriptions from geographic data has a long history. Early systems often relied on rule-based methods or templates [6], or focused on finding paths that are easy to describe rather than just the shortest distance [7]. Recent data-driven approaches treat this as a graph-to-text problem, modeling road networks and landmarks as spatial graphs to generate navigation instructions [21]. Rich semantic tags from OSM are also useful for helping models understand building functions and spatial attributes through predictive modeling [1] or LLM embeddings [16].

For descriptions around destinations, hybrid methods have combined geometric scoring with text generation. In these methods, OSM data is first used to score and filter candidate landmarks, and LLMs are then used to produce fluent descriptions based on those pre-selected landmarks [19].

Our work difers from these prior studies in two main ways. First, instead of using multi-stage methods that pre-filter candidates with hand-crafted rules, we require LLMs to select landmarks directly from raw structured map inputs. Second, we evaluate how fine-tuning on human data helps LLMs implicitly learn human landmark preferences without using manual scoring rules.

## 2.3 LLMs and VLMs for Map Understanding

While VLMs can process rendered map images directly, doing so is highly challenging. Recent benchmarks like CartoMapQA show that even strong VLMs struggle with fine-grained map reading, text label recognition, and route tracing [24]. Image encoders are efective for global semantic understanding, but they may abstract away precise spatial information that is required for tasks involving object relations, coordinates, or topology [23]. Furthermore, recent studies highlight a unique gap in text-based LLMs: while they resemble humans in choosing spatial words to describe object relationships, they still have fundamental limitations in maintaining accurate internal 2D spatial representations [25].

To address these limitations, recent studies propose converting spatial or visual data into structured textual or code-like formats before feeding them to LLMs. This approach has improved reasoning performance in robotics [13] and various web or chart domains [14, 12]. Crucially, knowledge infusion studies demonstrate that LLMs can efectively process structured data directly from structured text formats without needing natural language translations [18].

Studies on graph reasoning with LLMs suggest that textual graph representations, such as adjacency lists, can be more efective than visual graph images for certain reasoning tasks [8]. This study builds on this insight by comparing structured SVG text directly with rendered PNG images to identify the optimal representation for human-centric landmark selection.

## 3 Task Definition

This study addresses the problem of selecting landmarks from local maps. Given a local map and a specific navigation context, the model is required to identify the correct landmark that matches real human choices.

We define two specific landmark selection tasks:

– Task 1: Landmark Selection at Turns. The model receives a local map around an intersection and a turning direction, such as turning left or turning right. The model must select the most appropriate landmark at the intersection to represent the turn.

– Task 2: Landmark Selection around Destinations. The model receives a local map around a target destination. The model must select a nearby landmark to describe the relative location of the destination.

In both tasks, the evaluation is based entirely on landmark selection accuracy. The goal is to determine whether the selected landmark matches the exact landmark that a human used in a real-world route description.

## 3.1 Task Settings

We assume that the local navigation context is already given, meaning our task does not find the entire route from a start to a destination. For Task 1, the model receives a local map around an intersection and the turning action, such as turning left or turning right. For Task 2, the model receives the target destination and its surrounding map. Under these settings, the model is provided with a list of available candidate landmarks from the map database, and it must output the correct landmark identifier.

In practical human navigation, landmarks are usually referred to by their proper names, such as Starbucks or McDonald’s, rather than generic categories, such as a cafe or a restaurant. Therefore, our tasks require the model to understand both the spatial coordinates of these features and their semantic name attributes simultaneously to make a human-like selection.

## 3.2 Task 1: Landmark Selection at Turns

The first task focuses on landmark selection at intersections where a user makes a turn. When navigating an intersection, the model must identify a prominent reference point to clarify the turning action. The formal input and output of this task are defined as follows:

– Input: A local map M centered on an intersection, and a turning action a. The map M contains a set of landmark candidates $L = \{ l _ { 1 } , l _ { 2 } , \ldots , l _ { n } \}$ ， where each candidate $l _ { i }$ has attributes including its proper name, semantic category, and spatial coordinates. The turning action a represents the route entering and leaving the intersection, which is either a left turn or a right turn.

– Output: A single landmark name $l ^ { * } \in L$ selected by the model that best represents the turning action a at the intersection.

Under this setting, the LLM processing the text-based map, or the visionlanguage model (VLM) processing the image-based map, is required to evaluate the candidates and output the exact landmark name that matches the human choice in the ground-truth data.

## 3.3 Task 2: Landmark Selection around Destinations

The second task focuses on landmark selection for describing a target destination by its relative location to a nearby landmark. When people identify a destination, they often refer to a more recognizable place in the immediate vicinity. In this study, we consider three specific types of spatial relations that were originally expressed in Japanese:

next to: This relation corresponds to the Japanese phrases “隣” or “とな ${ \mathfrak { h } } ^ { * }$

– across from: This relation corresponds to the Japanese phrases $\sqrt [ 6 ] { 1 1 } \cdot x > 0 \ldots$ $\mathrm { o r } ^ { \cdots \oplus \rho } \mathcal { D } ^ { \mathsf { a } } \mathsf { l } ^ { \mathsf { \Pi } ^ { \mathsf { \Pi } ^ { \mathsf { \prime } \dagger } } }$

– beside: This relation corresponds to the Japanese phrases ${ \mathfrak { s o r } } \sharp { \mathfrak { L } } ^ { \sharp } \otimes { \mathfrak { o r } } \ { \mathfrak { s } } \pounds \geq { \mathfrak { n } } .$

We refer to these three terms collectively as target keywords.

The formal input and output of this task are defined as follows:

– Input: A local map M centered on a target destination d. The map M contains the destination d and a set of surrounding landmark candidates $L = \{ l _ { 1 } , l _ { 2 } , \ldots , l _ { n } \}$ , where each candidate $l _ { i }$ is associated with its proper name, semantic category, and spatial coordinates.

Output: A single landmark name $l ^ { * } \in L$ selected by the model that best describes the relative location of the destination d.

Similar to Task 1, the model must process the given map information and select the single most appropriate landmark name from the candidate set L based on human spatial preferences.

## 3.4 Characteristics of the Tasks

The two tasks difer in their navigation contexts, but they share three critical challenges for spatial models:

– Spatial configuration understanding. The model must accurately interpret the geometric layout of roads, candidate landmarks, and the specific target point or movement direction from the map data.

– Human-centric landmark selection. The model must identify a landmark that is useful for human communication. The geometrically closest landmark is not always the best choice if it is obscure or hidden from the user’s perspective.

Proper-name grounding. The model must process landmarks as named entities rather than generic objects. Because candidates are defined by their real names in the map database, the model must simultaneously reason about both spatial positions and semantic name attributes.

## 4 Dataset Construction

This section describes the construction of datasets for the two landmark selection tasks defined in Section 3. Our pipeline pairs human-written route descriptions with local map information, explicitly linking each textual landmark mention to its exact geometric counterpart in a map database.

## 4.1 Source of Human-Written Route Descriptions

We used restaurant access descriptions as a source of human-written text, as they reflect how people choose practical landmarks when explaining routes or locations to actual visitors. We collected restaurant metadata, including coordinates and Japanese access descriptions, using the Hot Pepper Gourmet Search API<sup>1</sup>. From an initial pool of 136,490 raw descriptions, we filtered out generic expressions that do not explicitly mention landmarks, such as phrases describing only travel time or distance.

![](images/f5d1672fb1205102e168ca4eccec2eca9446a497acac2421fd5d0a28cfbdf094.jpg)  
Fig. 2. Examples of dataset instances constructed from raw access descriptions. The left panel shows an instance for Task 1, where an intersection, an underlined landmark, and a turning direction are extracted. The right panel shows an instance for Task 2, where an underlined landmark around the destination is identified.

## 4.2 Filtering and Extraction via LLMs

We first performed a keyword-based screening to collect candidate descriptions. For the turn-based task, we extracted texts containing explicit Japanese turning expressions, which are either “turn right” or “turn left”. For the destinationbased task, we targeted the three target keywords defined in Section 3.

Because descriptions based solely on distance or road names could still contain these keywords, we subsequently utilized an LLM to analyze the screened text. Specifically, GPT-3.5 Turbo via the Azure OpenAI Service was employed for this verification and extraction step. The model verified the presence of a concrete landmark and extracted the landmark name from the description.

## 4.3 Linking to OpenStreetMap and Manual Verification

The extracted landmark names must be grounded in an actual map database. For each restaurant, we queried OSM objects within a 1,000-meter radius of its coordinates. We aligned candidate landmarks primarily using the OSM name tag, while associated geometry and category tags provided the spatial attributes.

Because landmark names in human descriptions are frequently abbreviated or written diferently, automated string matching is insuficient. Therefore, we conducted a manual inspection to check the data and correct any mismatches. We manually identified the specific intersections and route directions for Task 1, and verified the spatial relationships between the restaurants and nearby matched landmarks for Task 2. When a single human-written description contained multiple sequential turns or intersections, we treated each turning point as a separate, independent evaluation instance linked to its respective spatial coordinates while discarding vague descriptions that could not be verified on the map.

Through this filtering, extraction, and manual verification pipeline, we finally constructed two datasets: 500 instances for Task 1 and 339 instances for Task 2. Figure 2 illustrates examples of the final dataset instances and how the spatial context is identified for each task.

## 5 Structured Map Representation

This section describes how we represent local map information as structured text for LLMs. While a straightforward approach renders a map as a PNG image for VLMs, pixel-based inputs force the network to recognize text labels, road geometry, and directions directly from raw pixels, which often introduces optical character recognition errors and spatial reasoning dificulties.

To overcome this limitation, we convert map database information into a structured textual representation that preserves the spatial relations among roads, landmarks, and navigation targets. By representing a local map as structured text, a model gains direct access to symbolic attributes such as proper names and semantic categories while inferring spatial relations from standardized coordinates. We adopt SVG as our format because it describes geometric elements like points and lines within highly readable text. Furthermore, SVG allows an exact geometric correspondence between the structured-text input and a rendered map image, enabling a fair comparison between text-based and imagebased modalities using the identical underlying map data.

## 5.1 Local Map Extraction

For each instance, we extract a local map centered on the relevant navigation context, which is the target intersection for Task 1 and the destination restaurant for Task 2. The extraction range is determined by a zoom level of 18 and a 256 × 256 pixel rendering size, covering a local geographic area of approximately 124 × 124 meters. This scale captures nearby roads and landmarks visible and useful for human route descriptions.

Road networks are obtained using OSMnx, and landmark candidates are collected from OSM objects within this bounding area. We include named objects belonging to specific functional categories such as amenity, shop, tourism, leisure, historic, and building. Each candidate is represented by its proper name obtained from the OSM name tag and position, along with its semantic category when available in the map database.

## 5.2 Coordinate Normalization

The geographic coordinates of roads and landmarks are converted into normalized two-dimensional coordinates. We define a standardized coordinate system where the upper-left corner of the local map is mapped to (0, 0) and the lowerright corner is mapped to (100, 100), positioning the target intersection or destination precisely at the center (50, 50).

This system eliminates the need for the model to interpret raw latitude and longitude values directly, making relative spatial positions straightforward to infer from textual tokens. For example, a landmark at (51, 32) can be logically deduced as being slightly to the right and above the center point.

![](images/80ad5bac640fe39e7775003eafea0d04c2a31dab7f9122a358e939a7f6f1271c.jpg)

Fig. 3. Example of a local map represented in the normalized (100  100) coordinate space, originally expressed in Japanese.  
```svg
<svg width="256" height="256" viewBox="0 0 100 100">
<g id="road_network">
<line id="e1" x1="22" y1="88" x2="14" y2="95" />
</g>
<g id="navigation_vectors" data-turn_direction="left">
<line id="vector_start" x1="37" y1="61" x2="50" y2="50"
data-type="start" />
<line id="vector_goal" x1="50" y1="50" x2="10" y2="4"
data-type="goal" />
</g>
<g id="pois">
<g id="poi_0" data-category="bicycle" data-name="Ogawa Cycle">
<circle cx="51" cy="32" />
</g>
</g>
</svg>
```  
Fig. 4. Example of the structured SVG representation used as input to the LLMs.

## 5.3 SVG Text Components and Structure

The structured map is encoded as an SVG text snippet containing four primary elements:

– Road network: Roads represented as line segments with explicit start and end coordinates.

– Navigation vectors: Incoming and outgoing route vectors indicating how a user enters and leaves the intersection for Task 1.

– Target point: The fixed center point representing the target intersection or destination.

– Landmark candidates: Group elements containing explicit data-name, data-category, and coordinate attributes.

Figure 3 shows the visualized example of a local map, and Figure 4 shows the corresponding structured SVG text tokens provided to the models.

## 5.4 Input Modalities and Prompting Setup

For the text-based modality, the structured map text is fed to the model along with a task-specific prompt instructing it to analyze the spatial layout and output a JSON object containing a single name field, which must match one of the proper names in the input map attributes. The prompts are tailored to each task’s context, directing the model to analyze routing directions for Task 1 or infer appropriate spatial relations for Task 2. The complete prompt templates for the proposed text-based method are provided in Appendix A.

To establish a controlled evaluation, we prepare a corresponding image-based modality as the comparative condition, using an identical PNG map rendered directly from the same underlying geographic data. This image-based counterpart visually displays the road network, route directions, center point, and text labels. The image is provided to VLMs with an equivalent prompt instructing them to analyze visual pixels, track route paths, and output their decisions in the identical JSON format to ensure a rigorous modality comparison. The corresponding image-input prompts are also detailed in Appendix A.

## 6 Experimental Setup

## 6.1 Experimental Configuration

The evaluation framework operates in two distinct stages. In the first stage for RQ1, we compare modalities by feeding the models the exact same local map information represented either as a rendered PNG image or as structured SVG text. While prompts are customized for modality-specific formats, both configurations employ structurally equivalent task descriptions and identical JSON output rules to ensure a fair comparison.

In the second stage for RQ2, we evaluate the efects of task-specific parameter-eficient adaptation. Based on the modality advantages established in RQ1, we adopt the structured SVG text format as the foundational input for all fine-tuning experiments, conducting controlled performance comparisons for each local model before and after training across both Task 1 and Task 2.

Table 1. Fine-tuning batch sizes and optimal checkpoint epochs for each model and task.
<table><tr><td></td><td>Task 1</td><td></td><td>Task 2</td><td></td></tr><tr><td>Model</td><td>Batch size Epochs Batch size Epochs</td><td></td><td></td><td></td></tr><tr><td>Llama-3-ELYZA-JP-8B</td><td>128</td><td>95</td><td>128</td><td>52</td></tr><tr><td>ELYZA-Thinking-1.0-Qwen-32B</td><td>16</td><td>21</td><td>4</td><td>10</td></tr><tr><td>Gemma3:27B</td><td>4</td><td>6</td><td>2</td><td>7</td></tr></table>

## 6.2 Fine-Tuning Environment and Hyperparameters

For computationally eficient adaptation, we employ Low-Rank Adaptation[10] and Quantized Low-Rank Adaptation[5], implemented via the Unsloth framework[9]. Across all models, we apply a uniform adapter configuration with a LoRA rank $r = 1 6$ and an alpha value of 16. The base learning rate is set to $2 \times 1 0 ^ { - 5 }$ utilizing the AdamW 8-bit optimizer under a constant learning rate scheduler. To prevent overfitting, early stopping is applied based on validation set loss. Table 1 details the customized batch sizes and training epochs of the optimal checkpoints utilized for each respective model and task.

## 6.3 Evaluation Metric and Output Parsing

The definitive quantitative metric is landmark selection accuracy. For each test instance, the model must output a JSON object containing a single name field. Text-based models must select a string matching one of the SVG data-name attributes, while vision-based models read text labels directly from the visual canvas. Invalid JSON formatting or missing target fields are strictly penalized as incorrect predictions.

An output is correct if and only if the extracted landmark string matches the human ground truth. With N test instances, where N = 100 in our setup, let $\hat { l } _ { i }$ and $l _ { i }$ denote the predicted and reference landmarks for instance i, respectively. Selection accuracy is formalized as:

$$
\mathrm { A c c u r a c y } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { 1 } ( \hat { l } _ { i } = l _ { i } ) .
$$

As established in Section 3, the textual phrasing of the generated instruction is not scored by lexical similarity metrics, ensuring the evaluation isolates the core spatial reasoning capacity required to ground human landmark choices.

## 7 Results and Discussion

This section presents empirical results regarding map representation, fine-tuning impacts, and qualitative error patterns.

Table 2. Landmark selection accuracy comparison between PNG image input and structured SVG text input.
<table><tr><td>Model</td><td>Task 1 Task 2 PNG SVG PNG SVG</td></tr><tr><td>GPT-5.2</td><td>34% 34% 40% 55%</td></tr><tr><td>Gemma3:27B</td><td>14% 27% 26% 39%</td></tr><tr><td>Llama4</td><td>21% 25% 24% 28%</td></tr><tr><td>Llama-3-ELYZA-JP-8B</td><td>N/A 25% N/A 35%</td></tr><tr><td>ELYZA-Thinking-1.0-Qwen-32B N/A 24%N/A 39%</td><td></td></tr></table>

Table 3. Landmark selection accuracy before and after fine-tuning across both tasks.
<table><tr><td></td><td colspan="2">Task 1</td><td colspan="2">Task 2</td></tr><tr><td>Model</td><td></td><td></td><td>Base Fine-Tuned Base Fine-Tuned</td><td></td></tr><tr><td>Llama-3-ELYZA-JP-8B</td><td>24%</td><td></td><td>49% 35%</td><td>48%</td></tr><tr><td>ELYZA-Thinking-1.0-Qwen-32B 24%</td><td></td><td></td><td>38%39%</td><td>54%</td></tr><tr><td>Gemma3:27B</td><td>27%</td><td></td><td>53%39%</td><td>36%</td></tr></table>

## 7.1 RQ1: Efect of Map Representation

Table 2 compares landmark selection accuracy between rendered PNG images and structured SVG text inputs. The results demonstrate that the structuredtext representation consistently performs as well as or substantially better than the image-based input across all configurations.

In Task 1, Gemma3:27B doubles its accuracy from 14% for PNG to 27% for SVG. This trend is even more pronounced in Task 2, where Gemma3:27B jumps to 39% and GPT-5.2 achieves its highest performance of 55% using SVG text. These findings support the hypothesis that rich spatial data is more accessible when serialized as structured text rather than flattened into raw pixels.

This distinct advantage stems from two factors. First, the text format provides exact landmark names and normalized coordinates as symbolic tokens, bypassing optical character recognition errors inherent in visual processing. Second, the structured layout makes spatial geometry explicit. While a visual image requires a model to simultaneously resolve visual recognition and spatial relations from raw pixels, the SVG representation presents the map directly as an organized topology of named objects and coordinates, allowing the model to focus entirely on landmark selection. We thus answer RQ1 by concluding that structured-text map representation is consistently more efective than imagebased inputs.

## 7.2 RQ2: Efect of Fine-Tuning

To address RQ2, we evaluate task-specific adaptation via LoRA or QLoRA using the structured SVG text format exclusively. Table 3 compiles the performance of the three open-weights local models before and after adaptation.

In Task 1, fine-tuning yields substantial, uniform performance leaps across all architectures, with Llama-3-ELYZA-JP-8B reaching 49% and Gemma3:27B climbing to 53%. This confirms that targeted adaptation successfully teaches smaller local models to internalize human visual and communicative preferences at intersections. In Task 2, the efects remain highly positive for most architectures, where ELYZA-Thinking-1.0-Qwen-32B rises to 54%, though Gemma3:27B exhibits a slight fluctuation from 39% to 36%. This single variation suggests a delicate optimization trade-of, where aggressive parameter tuning on localized destination contexts can occasionally conflict with pre-existing spatial heuristics. Crucially, the broader results answer RQ2 afirmatively, demonstrating that task-specific fine-tuning unlocks sophisticated human-aligned spatial reasoning within eficient, local models.

## 7.3 Qualitative Error Analysis

We systematically reviewed instances where fine-tuned local models failed to match human ground truth, classifying errors into three distinct categories. The first category is the selection of another plausible landmark, where the model selected a spatially reasonable object that difered from the specific human reference in dense urban layouts. The second category is spatial alignment failures caused by misinterpreting vector coordinates, leading to landmarks on the incorrect side of the street. The third category is output formatting violations such as failing to generate a valid JSON object, though this error was extremely rare due to robust constraint adherence.

The prevalence of the first error type reveals that real-world landmark selection is rarely unique; when multiple salient structures exist, diferent human annotators choose diferent reference points. Therefore, our automated exactmatch evaluation likely understates the practical, real-world utility of the model outputs.

## 7.4 Limitations and Future Directions

While our findings validate the eficacy of structured map representations, this study has three main boundaries. First, our evaluation relies strictly on exactmatch landmark selection accuracy without executing human evaluations to gauge linguistic naturalness or practical clarity. Bridging this gap requires designing multi-reference verification frameworks or field studies. Second, the current framework omits visual salience features such as building volumes, heights, or prominent signage, which heavily dictate human visual attention. Third, the dataset is bounded by its modest overall sample size. Expanding the corpus scale across diverse spatial environments will be essential to foster more robust, generalized models of human landmark selection.

## 8 Conclusion

This paper presented a novel framework for landmark-based route instruction generation by converting local geographic data into a structured SVG text format for LLMs. Our empirical evaluations yielded two pivotal findings. First, the structured SVG representation consistently matched or outperformed the visual PNG baseline, confirming that LLMs benefit more from direct exposure to symbolic and geometric tokens than from raw pixels. Second, parameter-eficient fine-tuning on a modest dataset unleashed substantial performance gains in local open-weights models, demonstrating an eficient pathway for alignment with human spatial communication.

While these results validate our core thesis, they outline essential vectors for future research. Our analysis confirmed that human landmark selection is rarely unique; thus, developing multi-reference evaluation metrics or conducting human utility assessments will be crucial for more accurate measurement. Furthermore, enriching SVG representations with visual salience metadata such as building colors, dimensions, and heights remains a vital next step to ensure model-selected landmarks are both geometrically precise and highly visible in physical environments.

Acknowledgments. This work was supported by JSPS KAKENHI Grant Numbers JP25K03229, JP25K03228, and JP24K03228.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

Generative AI Usage Disclosure. During the preparation of this work, the authors used Gemini 3.5 Flash in order to improve the English phrasing and readability of the manuscript. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the publication.

## References

1. Atwal, K.S., Anderson, T., Pfoser, D., Züfle, A.: Predicting building types using openstreetmap. Scientific Reports 12 (2022)

2. Burnett, G.: ‘Turn right at the trafic lights’: The requirement for landmarks in vehicle navigation systems. Journal of Navigation 53(3), 499–510 (2000)

3. Denis, M.: The description of routes: A cognitive approach to the production of spatial discourse. Current Psychology of Cognition 16, 409–458 (1997)

4. Denis, M., Pazzaglia, F., Cornoldi, C., Bertolo, L.: Spatial discourse and navigation: An analysis of route directions in the city of venice. Applied Cognitive Psychology: The Oficial Journal of the Society for Applied Research in Memory and Cognition 13(2), 145–174 (1999)

5. Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L.: QLoRA: Eficient finetuning of quantized LLMs. In: Proceedings of the 37th Annual Conference on Neural Information Processing Systems (NeurIPS 2023). pp. 10088–10115 (2023)

6. Dräger, M., Koller, A.: Generation of landmark-based navigation instructions from open-source data. In: Proceedings of the 13th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2012). pp. 757–766 (2012)

7. Duckham, M., Winter, S., Robinson, M.: Including landmarks in routing instructions. Journal of Location Based Services 4(1), 28–52 (2010)

8. Fatemi, B., Halcrow, J., Perozzi, B.: Talk like a graph: Encoding graphs for large language models. In: Proceedings of the 12th International Conference on Learning Representations (ICLR 2024) (2024)

9. Han, D., Han, M., team, U.: Unsloth (2023), http://github.com/unslothai/unsloth

10. Hu, E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: LoRA: Low-rank adaptation of large language models. In: Proceedings of the 10th International Conference on Learning Representations (ICLR 2022) (2022)

11. Kevin, L.: The Image of the City, chap. 3, pp. 46–90. MIT Press (1960)

12. Lee, K., Joshi, M., Turc, I., Hu, H., Liu, F., Eisenschlos, J., Khandelwal, U., Shaw, P., Chang, M.W., Toutanova, K.: Pix2Struct: Screenshot parsing as pretraining for visual language understanding. In: Proceedings of the 40th International Con ference on Machine Learning (ICML 2023). pp. 18893–18912 (2023)

13. Liang, J., Huang, W., Xia, F., Xu, P., Hausman, K., Ichter, B., Florence, P., Zeng, A.: Code as Policies: Language model programs for embodied control. In: Proceedings of the 40th IEEE International Conference on Robotics and Automation (ICRA 2023). pp. 9493–9500 (2023)

14. Liu, F., Eisenschlos, J., Piccinno, F., Krichene, S., Pang, C., Lee, K., Joshi, M., Chen, W., Collier, N., Altun, Y.: DePlot: One-shot visual language reasoning by plot-to-table translation. In: Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL 2023). pp. 10381–10399 (2023)

15. Lovelace, K.L., Hegarty, M., Montello, D.R.: Elements of good route directions in familiar and unfamiliar environments. In: Proceedings of the 4th International Conference on Spatial Information Theory (COSIT 1999). pp. 65–82 (1999)

16. Memduhoǧlu, A., Fulman, N., Zipf, A.: Enriching building function classification using large language model embeddings of openstreetmap tags. Earth Science In formatics 17(6), 5403–5418 (2024)

17. Michon, P.E., Denis, M.: When and why are visual landmarks used in giving directions? In: Proceedings of the 5th International Conference on Spatial Information Theory (COSIT 2001). pp. 292–305 (2001)

18. Moiseev, F., Dong, Z., Alfonseca, E., Jaggi, M.: SKILL: Structured knowledge infusion for large language models. In: Proceedings of the 2022 Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL 2022). pp. 1581–1588 (2022)

19. Muroň, M., Dařena, F., Procházka, D., Kern, R.: Automatically generated place descriptions for accurate location identification: a hybrid approach with rule-based methods and LLM. Spatial Cognition & Computation 25(4), 269–311 (2025)

20. Richter, D., Winter, S., Richter, K.F., Stirling, L.: How people describe their place: identifying predominant types of place descriptions. In: Proceedings of the 1st ACM SIGSPATIAL International Workshop on Crowdsourced and Volunteered Geographic Information 2012 (GEOCROWD 2012). pp. 30–37 (2012)

21. Schumann, R., Riezler, S.: Generating landmark navigation instructions from maps as a graph-to-text problem. In: Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP 2021). pp. 489–502 (2021)

22. Sorrows, M.E., Hirtle, S.C.: The nature of landmarks for real and electronic spaces. In: Proceedings of the 4th International Conference on Spatial Information Theory (COSIT 1999). pp. 37–50 (1999)

23. Tong, S., Liu, Z., Zhai, Y., Ma, Y., LeCun, Y., Xie, S.: Eyes wide shut? exploring the visual shortcomings of multimodal LLMs. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2024 (CVPR 2024). pp. 9568–9578 (2024)

24. Ung, H.Q., Habault, G., Nishimura, Y., Niu, H., Legaspi, R., Oya, T., Kojima, R., Taya, M., Ono, C., Minamikawa, A., Liu, Y.: CartoMapQA: A fundamental benchmark dataset evaluating vision-language models on cartographic map understanding. In: Proceedings of the 33rd ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (ACM SIGSPATIAL 2025). pp. 40–453 (2025)

25. Wu, W., Deng, W.: Spatial representation of large language models in 2D scene. In: Proceedings of the 4th Workshop on Generation, Evaluation and Metrics (GEM²). pp. 18–29 (2025)

26. Yesiltepe, D., Conroy Dalton, R., Ozbil Torun, A.: Landmarks in wayfinding: A review of the existing literature. Cognitive Processing 22(3), 369–410 (2021)

## A Experimental Prompts

This appendix presents English translations of the prompt templates used in our experiments. In the actual experiments, these prompts were administered in Japanese and the templates provided below accurately reflect the semantic structure and content of the original Japanese instructions.

![](images/a90cc02cfb61343977662b77a554ff3dd12f52335a8ae8840870450d7f010b1c.jpg)  
Fig. 5. Prompt templates for Task 1.

![](images/43b109261e14c288111c399a83bca34275a0ab58dcf681404d3f2b4d6e0ccc55.jpg)  
Fig. 6. Prompt templates for Task 2.