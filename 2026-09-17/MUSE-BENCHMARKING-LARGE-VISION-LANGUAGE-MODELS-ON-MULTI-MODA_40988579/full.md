# MUSE: BENCHMARKING LARGE VISION-LANGUAGE MODELS ON MULTI-MODAL UNDERSTANDING IN SITUATED EDUCATION

## A PREPRINT

Luyao Zhu1, Xun Wei Yee1, © Wei Li3, Mun Thye Mak1,  Wee Siong Ng2

1 AI Singapore, National University of Singapore, Singapore 2 School of Čomputing, National University of Singapore, Singapore 3 Institute of Advanced Intelligence and Computing, A\*STAR

Luyao Zhu: luyaozhu@outlook.com Wei Li: wei008@e.ntu.edu.sg Wee Siong Ng: Ng\_Wee\_Siong@a-star.edu.sg

September 17, 2026

## ABSTRACT

Large vision-language models have achieved remarkable progress in multi-modal understanding, yet their capabilities in educational settings remain insufficiently evaluated. In AI-assisted language learning, models must interpret artistic imagery, understand its semantic, affective, and cultural content, and reason about visual context to support meaningful interaction. However, existing benchmarks primarily focus on real-world images or domain-specific educational reasoning, providing limited coverage of artistic educational content. To address this gap, we introduce MUSE, a benchmark for evaluating large vision-language models on artistic image understanding in situated educational applications. MUSE decouples image annotation from question generation, enabling diverse tasks with controllable difficulty while reducing annotation effort. It comprises twelve tasks spanning visual perception, semantic and affective interpretation, culture understanding, and compositional reasoning, together with diverse artistic images deliberately curated to center Singaporean and Southeast Asian multicultural contexts alongside Western art traditions, covering multiple themes and difficulty levels. Evaluation of open-source and proprietary models reveals substantial disparities across capability dimensions, particularly in affective interpretation and compositional reasoning. Our analysis further identifies common failure modes and key challenges for developing trustworthy multi-modal models for education. We hope MUSE will serve as a standardized benchmark for advancing multi-modal understanding in situated educational applications.

Keywords Benchmark ·Vision language model · Multi-modal understanding

## Code | Dataset

## 1 Introduction

Large vision-language models (VLMs) have made substantial progress in integrating visual perception with language understanding and generation, enabling tasks such as visual question answering, image description, visual grounding, multi-modal dialogue, and visual reasoning OpenAI [2023], Bai et al. [2023], Chen et al. [2024a]. Their growing capabilities have encouraged applications in situated education, including intelligent tutoring, personalized learning, automated feedback, and multi-modal content interaction Chu et al. [2025]. A central requirement in these settings is the ability to interpret instructional images and connect their visual content with meaningful linguistic representations. This is especially demanding in image-based learning, where artworks prompt vocabulary use, description, narrative construction, emotional expression, and cultural discussion Zhuang et al. [2024], Shimabukuro et al. [2025]. An AI tutor must interpret the same image to formulate questions, assess responses, explain linguistic concepts, and provide appropriate feedback, requiring semantic, affective, spatial, compositional, and cultural understanding beyond object recognition.

<table><tr><td>Dimension</td><td>Capability</td><td>Tasks</td></tr><tr><td>Visual Perception</td><td>Objects and their quantities</td><td>Object Classification, Object Count</td></tr><tr><td>Semantic Understanding</td><td>Scenes, human activities, and events</td><td>Scene Classification, Activity Localization, Activity Description</td></tr><tr><td>Affective Interpretation</td><td>Emotions, their causes, and supporting visual evidence</td><td>Emotion Detection, Emotion Cause Inference, Visual Clue Identification</td></tr><tr><td>Compositional Reasoning</td><td>Spatial and structural composition</td><td>Relative Position, Remote Interaction, Jigsaw Puzzle</td></tr><tr><td>Cultural Understanding</td><td>Cultural-specific visual knowledge</td><td>Cultural Identification</td></tr></table>

Table 1: Capability dimensions in MUSE.

![](images/3448d76e41f031404fa1cd78a3a11643c0639d9ff068605cc12e5815725b25dc.jpg)  
Figure 1: Overview of the 12 MUSE tasks. The cropped image on each task card is for illustration only. The model receives the full image and the corresponding question for all tasks except Jigsaw Puzzle, where it receives only the cropped image.

Artistic imagery, such as paintings, illustrations, and cartoons, further complicates this task. Compared with natural photographs, these images often contain stylized or exaggerated forms, non-photorealistic colors, implicit narratives. and culturally dependent cues. General-purpose VLMs have shown limitations in interpreting such content, motivating dedicated models and benchmarks for artistic understanding Yuan et al. [2023], Alfarano et al. [2025]. Consequently, performance on natural-image benchmarks may not reliably reflect a model's ability to understand artistic imagery in educational settings.

Existing VLM benchmarks evaluate broad perception, knowledge, and reasoning abilities, including general multimodal understanding Liu et al. [2024a], academic problem solving Lu et al. [2022], and scientific or mathematical reasoning Lu et al. [2024], Ying et al. [2024]. However, they are not designed to jointly assess the capabilities required for image-based language learning with artistic content. Even when artistic content is included, the evaluation generally targets disciplinary knowledge or a specific aspect of art understanding Yue et al. [2024] rather than the capability required by educational VLMs. This leaves a gap between general VLM evaluation and the competencies needed to interact reliably with artistic educational imagery.

Benchmark construction also presents practical challenges. VLM benchmarks often construct task-specific questionanswer pairs directly from individual images through manual annotation Zhang et al. [2025]. Extending such pipelines to new tasks requires additional annotation effort, while the resulting data are often difficult to reuse across tasks. Moreover conventional question collection offers limited control over question form and complexity; prior work on controllable question generation shows that difficulty control requires explicit modeling of reasoning structure Cheng et al. [2021]. Independently constructed tasks may also adopt inconsistent semantic representations, hindering comparability. We therefore decouple reusable visual-semantic annotations from task-specific question generation, improving scalability, consistency, and controllability.

To address both the evaluation and construction gaps, we introduce MUSE, a benchmark for Multi-modal Understanding in Situated Education using artistic imagery. MUSE adopts an annotation-first, task-generative design: each artwork is annotated once with a reusable structured representation of its visual and semantic content, after which task-specific questions are instantiated through predefined generation rules. By separating what an image contains from how a capability is queried, this design supports annotation reuse, consistent semantics across tasks, and explicit control over question format and difficulty.

Built on this shared representation, MUSE turns each artwork into a multi-view evaluation instance. Its 12 tasks cover five complementary capability dimensions (Table 1) and combine textual and visual multiple-choice questions with numerical and open-ended responses. Tasks such as visual-clue identification and emotion-cause inference therefore test whether models can ground and articulate their understanding, rather than only recognize a correct option. Figure 1 illustrates how one artwork supports the full task suite.

Evaluation of 30 open-source and proprietary VLMs reveals pronounced task-dependent gaps, particularly in visual grounding, affective interpretation, and compositional reasoning. Correlation and error analyses further show that success on general benchmarks or coarse recognition does not reliably transfer to artistic imagery and fine-grained evidence-based reasoning. Our main contributions are:

• We introduce MUSE, a 12-task benchmark that evaluates five dimensions of multimodal understanding over artistic imagery for image-based language learning and educational interaction.

• We propose an annotation-first, task-generative construction framework that reuses structured image annotations to produce semantically consistent questions with controllable formats and difficulty.

• We evaluate 30 open-source and proprietary VLMs on MUSE, revealing fundamental gaps between recognition, grounding, affective interpretation and compositional reasoning through task, correlation, and error analyses

## 2 Related Work

Multimodal and educational benchmarks General VLM benchmarks evaluate perception, knowledge, and reasoning beyond conventional visual question answering. MMBench uses constructed multiple-choice questions (MCQs) for fine-grained assessment, while SEED-Bench uses human-verified questions to evaluate hierarchical capabilities Liu et al. [2024a], Li et al. [2024]. MMMU targets expert reasoning across disciplines; MMStar uses vision-indispensable samples to measure multimodal gain and leakage; and MMMU-Pro strengthens visual dependency through filtering, expanded options, and vision-only evaluation Yue et al. [2024], Chen et al. [2024b], Yue et al. [2025]. ScienceQA and MathVista focus on scientific and mathematical reasoning Lu et al. [2022, 2024]. These benchmarks primarily use natural images, diagrams, charts, documents, or examination materials, offering limited coverage of stylization, implicit narratives, affective evidence, and culturally situated meanings in artistic content for language learning.

Artistic, affective, and cultural understanding Prior work examines artistic, affective, and culturally grounded image understanding. ArtEmis collects emotion labels and visually grounded explanations for artworks, while ArtELingo adds multilingual annotations for cross-cultural affective responses Achlioptas et al. [2021], Mohamed et al. [2022]. VQArt-Bench evaluates symbolic meaning, narratives, counting, and visual relationships in art, whereas AICA-Bench addresses emotion understanding, reasoning, and generation Alfarano et al. [2025], She et al. [2026]. CVQA evaluates culturally grounded visual question answering across regions and languages with native-speaker and expert data Romero et al. [2024]. These resources advance affective, artistic, or cultural understanding but generally focus on individual domains. MUSE instead jointly evaluates visual perception, activity and scene understanding, affective evidence and causes, spatial and compositional reasoning, and cultural understanding. Its decoupled construction reuses annotations across tasks, reduces annotation effort, and controls question formulation and difficulty.

![](images/25bb9306ddd983f879adce52b88cad2bdedf3bdc2d5e684679499cc8408042d2.jpg)  
Figure 2: Taxonomy of MUSE and statistics.

## 3 MUSE Benchmark

MUSE differs from existing multimodal-understanding benchmarks in three ways: (1) it curates original artworks from artists worldwide to diversify image sources; (2) decouples annotation from question generation to control difficulty systematically; (3) and targets the visual capabilities required for reliable image-captioning-based language education. MUSE contains 2,400 questions over 1,174 images, each with a resolution of 1920 × 1080 pixels, across 12 tasks that test alignment between artistic visual content and linguistic descriptions. Figure 2 shows the tasks span 3 cognitive complexity levels, i.e., low-level pattern recognition, mid-level semantic perception, and high-level reasoning, as well as 3 spatial granularities, i.e., pixel-, region-, and image-level understanding. Most use textual or visual multiple-choice questions; Object Count requires numerical prediction, while Visual Clue Identification and Emotion Cause Inference use open-ended responses evaluated by semantic similarity. We next describe its construction and tasks.

## 3.1 Dataset Annotation and Quality Control

Before annotation, 127 annotators receive a briefing on the study motivation, task definitions, guidelines, representative examples, and ambiguous cases. Using a standardized Label Studio Enterprise interface, they annotate activity, character, and object bounding boxes; emotion, object, and position labels; activity descriptions; scene and cultural labels; object counts; visual clues; and emotion causes. Each sample is independently annotated by one annotator, reviewed by two others, and finalized only after consensus, with disagreements resolved using the established guidelines.

## 3.2 Question Generation

To improve benchmark diversity, we explicitly enforce diversity along three dimensions during problem generation: artistic styles (through diverse artists), scene themes, and question difficulty. Scene theme distribution is in Figure 3. Among these tasks, Object Classification, Emotion Detection, Visual Clue Identification, and Emotion Cause Inference form a four-turn sequence for evaluating affective computing, with questions and answers from earlier turns retained in the dialogue history. All bounding boxes below use normalized COCO format ([xmin, ymin, width, height]).

![](images/4ddcfa2d51f0a9d2190a0256252efff9a9437e2a7f67e312082e32c0ba8e3d52.jpg)  
Figure 3: Scene theme distribution.

![](images/0ac83ed8b699384de21ce98d4cde286d4d78ee4178772f006ad234d77b5e7801.jpg)  
Figure 4: Emotion and culture distribution.

1. OBJECT CLASSIFICATION Given a bounding box, models classify the character as Woman, Man, Girl, Boy, or Baby.   
The options are shuffled for each problem.

2. EMOTION DETECTION Models classify characters' emotion as Anxiety, Sadness, Surprise, Joy, Disgust, Fear, Boredom, Guilt, Neutral, Anger, or Confusion. The categories follow Plutchik's emotion wheel and primary, secondary,

and tertiary dyads [Plutchik, 1980], excluding emotions that are rare or difficult to depict visually. Options are shuffled.   
and the label distribution is in the outer ring of Figure 4.

3. VISUAL CLUE IDENTIFICATION Models provide an open-ended description of the visual evidence supporting their preceding emotion prediction. Responses are compared with human references using semantic similarity.

4. EMOTION CAUSE INFERENCE Models provide an open-ended explanation of the predicted emotion's cause, evaluated using the same metrics.

5. ACTIVITY LOCALIZATION Models select the bounding box corresponding to a described activity. Distractors comprise boxes for: i) another activity; ii) a character or inanimate object; iii) a subregion of the ground-truth box; iv) a random region; or v) "None of the above."

6. ACTIVITY DESCRIPTION This task evaluate the VLMs' capability to understand and describe what is happening within the bounding boxes. 10 methods are employed to compose negative options: i) another activity description in the same image (oa); ii) another inanimate object in the same image (oosi); iii) another inanimate object in a different image (oodi); iv) another identity in the same image (oisi); v) another identity in a different image (oidi); vi) shifted the orders of objects in the original description (so); vii) concatenated i activity descriptions in the same image $( i \in \{ 1 , 2 , 3 \} )$ (ca\_s); viii) concatenated i activity descriptions in a different image (i ∈ {1, 2, 3}) (ca\_d); ix) negative descriptions from annotators (neg); and x) the statement "None of the above" (none).

7. CULTURAL IDENTIFICATION Models identify cultural elements within a given bounding box. We embed all ground-truth labels using OpenAI TEXT-EMBEDDING-3-SMALL and cluster them into 15 categories. Three negative options are sampled from categories other than that of the ground truth. The inner ring of Figure 4 shows the category distribution.

8. JIGSAw PUZZLE Models complete jigsaw puzzles by aligning patches through continuity in shape, color, and texture. We use five segmentation grids: (3,4), (4,4), (3,6), (4,5), and (3,7). Distractors comprise: i) another piece from the same image; ii) the ground-truth piece combined with another piece; iii) a zoomed region around the ground-truth piece; or iv) a piece from another image. Pieces may be stretched, upright, or balanced hexagons; wide or landscape rectangles; thin-tall or portrait rectangles; or squares, with angled, rounded, or sharp edges.

9. OBJECT CoUNT Models numerically predict object counts, testing object recognition and compositional reasoning under occlusion and variations in size and appearance.

10. RELATIVE POsITION Given object descriptions and bounding boxes, models predict three-dimensional spatial relations, particularly from the characters' viewpoints: i) left, none, or right laterally; ii) front, none, or back in depth; and iii) above, none, or under vertically.

11. REMOTE INTERACTION Models reason about non-contact interactions between entities localized by descriptions and bounding boxes. Each query contains two MCQs: one identifies the interacting entity, and the other identifies supporting visual evidence. Distractors comprise: i) entities from other interactions in the same image; ii) evidence from other same-image interactions; iii) mismatched text-bounding-box pairs sampled from these candidates and the ground truth; and iv) cross-image candidates with different descriptions and low overlap with the ground-truth box

12. SCENE CLASSIFICATION Models classify the overall scene by integrating global visual and semantic information. We use OpenAI GPT-3.5 to organize all ground-truth scene labels into 13 categories, then generate three negative options by sampling one label from each of three categories other than the ground-truth category.

## 4 Experiments

We evaluate 30 open-source and proprietary multimodal models spanning architectures, scales, and training paradigms. GPT-5.6-Sol and GPT-4o are accessed through APIs, while open-source models are deployed on AWS instances equipped with NVIDIA T4, A10G, or A100 GPUs. The evaluated families include CogVLM2 Hong et al. [2024], DeepSeek-VL2 Wu et al. [2024], Gemma 3 Gemma Team [2025], GLM-4V Hong et al. [2024], InternVL3 Zhu et al. [2025], LLaVA-NeXT Liu et al. [2024b], MiniCPM-V Yao et al. [2024], MiniCPM-o OpenBMB [2025], Qwen2.5. VL Bai et al. [2025a], Qwen3-VL Bai et al. [2025b], and Yi-VL Young et al. [2024]. All models use temperature 0 and are evaluated once as their outputs are stable. A unified parser handles free-form, option-based, and JSON responses; tasks are scored by accuracy or semantic similarity (i.e., cosine similarity between TEXT-EMBEDDING-3-LARGE embeddings).

<table><tr><td rowspan=1 colspan=15>Visual Perception   Semantic Understanding          Affective Interpretation            Compositional Reasoning     CulturalObject Object Activity Activity Scene Emotion  Visual Clue Emotion Cause  Relative   RemoteModel   Jigsaw  CulturalPositionInteractionPuzzleIdent.Cls.   Count   Loc.    Desc.   Cls.   Det.     Ident.       Infer.</td></tr><tr><td rowspan=1 colspan=1>GPT-5.6-Sol</td><td rowspan=1 colspan=2>76.0   71.5</td><td rowspan=1 colspan=1>69.5</td><td rowspan=1 colspan=1>34.5</td><td rowspan=1 colspan=1>86.5</td><td rowspan=1 colspan=1>39.5</td><td rowspan=1 colspan=2>50.90</td><td rowspan=1 colspan=1>49.18</td><td rowspan=1 colspan=1>4.0</td><td rowspan=1 colspan=1>86.5</td><td rowspan=1 colspan=1>35.5</td><td rowspan=1 colspan=2>76.5</td></tr><tr><td rowspan=1 colspan=1>Qwen3-VL-32b</td><td rowspan=1 colspan=2>54.0   59.0</td><td rowspan=1 colspan=1>72.5</td><td rowspan=1 colspan=1>50.0</td><td rowspan=1 colspan=1>87.0</td><td rowspan=1 colspan=1>29.5</td><td rowspan=1 colspan=2>44.34</td><td rowspan=1 colspan=1>40.23</td><td rowspan=1 colspan=1>5.0</td><td rowspan=1 colspan=1>52.0</td><td rowspan=1 colspan=1>28.5</td><td rowspan=1 colspan=2>60.0</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=2>51.0</td></tr><tr><td rowspan=1 colspan=1>Qwen2.5-VL-72b</td><td rowspan=1 colspan=2>52.0   51.0</td><td rowspan=1 colspan=1>56.0</td><td rowspan=1 colspan=1>47.0</td><td rowspan=1 colspan=1>86.0</td><td rowspan=1 colspan=1>23.5</td><td rowspan=1 colspan=2>40.26</td><td rowspan=1 colspan=1>33.63</td><td rowspan=1 colspan=1>8.5</td><td rowspan=1 colspan=1>46.0</td><td rowspan=1 colspan=1>21.5</td><td rowspan=1 colspan=2>50.5</td></tr><tr><td rowspan=1 colspan=1>Qwen3-VL-8b</td><td rowspan=1 colspan=2>46.5   48.5</td><td rowspan=1 colspan=1>58.0</td><td rowspan=1 colspan=1>42.5</td><td rowspan=1 colspan=1>84.5</td><td rowspan=1 colspan=1>25.0</td><td rowspan=1 colspan=2>44.06</td><td rowspan=1 colspan=1>38.47</td><td rowspan=1 colspan=1>2.0</td><td rowspan=1 colspan=1>40.5</td><td rowspan=1 colspan=1>16.5</td><td rowspan=1 colspan=2>58.5</td></tr><tr><td rowspan=1 colspan=1>InternVL3-14b</td><td rowspan=1 colspan=1>48.0</td><td rowspan=1 colspan=1>44.5</td><td rowspan=1 colspan=1>55.0</td><td rowspan=1 colspan=1>44.5</td><td rowspan=1 colspan=1>81.5</td><td rowspan=1 colspan=1>15.0</td><td rowspan=1 colspan=2>34.25</td><td rowspan=1 colspan=1>27.59</td><td rowspan=1 colspan=1>10.0</td><td rowspan=1 colspan=1>39.0</td><td rowspan=1 colspan=1>28.0</td><td rowspan=1 colspan=2>46.5</td></tr><tr><td rowspan=1 colspan=1>GPT-40</td><td rowspan=1 colspan=1>24.5</td><td rowspan=1 colspan=1>49.0</td><td rowspan=1 colspan=1>51.5</td><td rowspan=1 colspan=1>59.5</td><td rowspan=1 colspan=1>86.0</td><td rowspan=1 colspan=1>22.0</td><td rowspan=1 colspan=2>34.84</td><td rowspan=1 colspan=1>27.97</td><td rowspan=1 colspan=1>7.5</td><td rowspan=1 colspan=1>30.0</td><td rowspan=1 colspan=1>28.0</td><td rowspan=1 colspan=2>39.5</td></tr><tr><td rowspan=1 colspan=1>Qwen2.5-VL-32b</td><td rowspan=1 colspan=1>44.5</td><td rowspan=1 colspan=1>50.0</td><td rowspan=1 colspan=1>45.0</td><td rowspan=1 colspan=1>34.5</td><td rowspan=1 colspan=1>83.0</td><td rowspan=1 colspan=1>21.5</td><td rowspan=1 colspan=2>39.65</td><td rowspan=1 colspan=1>34.27</td><td rowspan=1 colspan=1>7.5</td><td rowspan=1 colspan=1>44.0</td><td rowspan=1 colspan=1>20.5</td><td rowspan=1 colspan=2>51.5</td></tr><tr><td rowspan=1 colspan=1>Qwen2.5-VL-7b</td><td rowspan=1 colspan=1>26.0</td><td rowspan=1 colspan=1>42.0</td><td rowspan=1 colspan=1>50.0</td><td rowspan=1 colspan=1>36.5</td><td rowspan=1 colspan=1>83.0</td><td rowspan=1 colspan=1>15.5</td><td rowspan=1 colspan=2>37.99</td><td rowspan=1 colspan=1>28.77</td><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>32.0</td><td rowspan=1 colspan=1>19.0</td><td rowspan=1 colspan=2>54.5</td></tr><tr><td rowspan=1 colspan=1>Gemma3-12b-it</td><td rowspan=1 colspan=1>24.0</td><td rowspan=1 colspan=1>40.0</td><td rowspan=1 colspan=1>39.5</td><td rowspan=1 colspan=1>35.0</td><td rowspan=1 colspan=1>81.0</td><td rowspan=1 colspan=1>17.5</td><td rowspan=1 colspan=2>41.58</td><td rowspan=1 colspan=1>33.66</td><td rowspan=1 colspan=1>1.5</td><td rowspan=1 colspan=1>26.5</td><td rowspan=1 colspan=1>24.5</td><td rowspan=1 colspan=2>55.0</td></tr><tr><td rowspan=1 colspan=1>Gemma3-27b-it</td><td rowspan=1 colspan=1>32.0</td><td rowspan=1 colspan=1>42.0</td><td rowspan=1 colspan=1>47.0</td><td rowspan=1 colspan=1>30.5</td><td rowspan=1 colspan=1>81.0</td><td rowspan=1 colspan=1>18.0</td><td rowspan=1 colspan=2>42.21</td><td rowspan=1 colspan=1>35.74</td><td rowspan=1 colspan=1>4.5</td><td rowspan=1 colspan=1>24.5</td><td rowspan=1 colspan=1>19.5</td><td rowspan=1 colspan=2>48.0</td></tr><tr><td rowspan=1 colspan=1>InternVL3-9b</td><td rowspan=1 colspan=1>20.0</td><td rowspan=1 colspan=1>44.0</td><td rowspan=1 colspan=1>46.0</td><td rowspan=1 colspan=1>35.5</td><td rowspan=1 colspan=1>81.5</td><td rowspan=1 colspan=1>9.0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>MiniCPM-V-2.6</td><td rowspan=1 colspan=1>30.5</td><td rowspan=1 colspan=1>46.0</td><td rowspan=1 colspan=1>43.0</td><td rowspan=1 colspan=1>41.0</td><td rowspan=1 colspan=1>84.5</td><td rowspan=1 colspan=1>16.0</td><td rowspan=1 colspan=2>40.08</td><td rowspan=1 colspan=1>28.79</td><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>9.0</td><td rowspan=1 colspan=1>12.0</td><td rowspan=1 colspan=1>43.0</td><td rowspan=1 colspan=1></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>InternVL3-8b</td><td rowspan=1 colspan=1>33.0</td><td rowspan=1 colspan=1>41.0</td><td rowspan=1 colspan=1>44.0</td><td rowspan=1 colspan=1>21.0</td><td rowspan=1 colspan=1>82.5</td><td rowspan=1 colspan=1>13.0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>34.93</td><td rowspan=1 colspan=1>27.98</td><td rowspan=1 colspan=1>1.0</td><td rowspan=1 colspan=1>23.5</td><td rowspan=1 colspan=1>17.5</td><td rowspan=1 colspan=1>49.0</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>MiniCPM-Llama3-V-2.5</td><td rowspan=1 colspan=1>28.5</td><td rowspan=1 colspan=1>35.0</td><td rowspan=1 colspan=1>42.5</td><td rowspan=1 colspan=1>36.5</td><td rowspan=1 colspan=1>75.5</td><td rowspan=1 colspan=1>11.0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>36.53</td><td rowspan=1 colspan=1>24.67</td><td rowspan=1 colspan=1>3.0</td><td rowspan=1 colspan=1>16.5</td><td rowspan=1 colspan=1>28.5</td><td rowspan=1 colspan=1>47.5</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>MiniCPM-O-2.6</td><td rowspan=1 colspan=1>34.5</td><td rowspan=1 colspan=1>47.0</td><td rowspan=1 colspan=1>47.5</td><td rowspan=1 colspan=1>26.0</td><td rowspan=1 colspan=1>81.0</td><td rowspan=1 colspan=1>19.5</td><td rowspan=1 colspan=2>36.26</td><td rowspan=1 colspan=1>27.09</td><td rowspan=1 colspan=1>3.0</td><td rowspan=1 colspan=1>8.5</td><td rowspan=1 colspan=1>17.0</td><td rowspan=1 colspan=2>47.0</td></tr><tr><td rowspan=1 colspan=1>GLM-4V-9b</td><td rowspan=1 colspan=1>26.0</td><td rowspan=1 colspan=1>29.0</td><td rowspan=1 colspan=1>45.0</td><td rowspan=1 colspan=1>25.0</td><td rowspan=1 colspan=1>64.5</td><td rowspan=1 colspan=1>14.0</td><td rowspan=1 colspan=2>37.07</td><td rowspan=1 colspan=1>24.60</td><td rowspan=1 colspan=1>2.0</td><td rowspan=1 colspan=1>27.5</td><td rowspan=1 colspan=1>42.0</td><td rowspan=1 colspan=2>51.5</td></tr><tr><td rowspan=1 colspan=1>Qwen2.5-VL-3b</td><td rowspan=1 colspan=1>13.0</td><td rowspan=1 colspan=1>45.0</td><td rowspan=1 colspan=1>36.0</td><td rowspan=1 colspan=1>37.0</td><td rowspan=1 colspan=1>77.5</td><td rowspan=1 colspan=1>12.5</td><td rowspan=1 colspan=2>33.50</td><td rowspan=1 colspan=1>20.78</td><td rowspan=1 colspan=1>6.0</td><td rowspan=1 colspan=1>9.5</td><td rowspan=1 colspan=1>14.5</td><td rowspan=1 colspan=2>45.0</td></tr><tr><td rowspan=1 colspan=1>LLaVA-Next-8b</td><td rowspan=1 colspan=1>38.0</td><td rowspan=1 colspan=1>31.5</td><td rowspan=1 colspan=1>44.5</td><td rowspan=1 colspan=1>22.5</td><td rowspan=1 colspan=1>61.0</td><td rowspan=1 colspan=1>12.0</td><td rowspan=1 colspan=2>33.34</td><td rowspan=1 colspan=1>27.63</td><td rowspan=1 colspan=1>1.5</td><td rowspan=1 colspan=1>8.5</td><td rowspan=1 colspan=1>21.5</td><td rowspan=1 colspan=2>41.5</td></tr><tr><td rowspan=1 colspan=1>DeepSeek-VL2-Small</td><td rowspan=1 colspan=1>18.0</td><td rowspan=1 colspan=1>38.0</td><td rowspan=1 colspan=1>31.0</td><td rowspan=1 colspan=1>23.0</td><td rowspan=1 colspan=1>75.5</td><td rowspan=1 colspan=1>15.5</td><td rowspan=1 colspan=2>41.05</td><td rowspan=1 colspan=1>31.63</td><td rowspan=1 colspan=1>2.0</td><td rowspan=1 colspan=1>12.0</td><td rowspan=1 colspan=1>21.5</td><td rowspan=1 colspan=2>44.0</td></tr><tr><td rowspan=1 colspan=1>Gemma3-4b-it</td><td rowspan=1 colspan=1>24.5</td><td rowspan=1 colspan=1>28.0</td><td rowspan=1 colspan=1>32.0</td><td rowspan=1 colspan=1>21.0</td><td rowspan=1 colspan=1>81.0</td><td rowspan=1 colspan=1>14.5</td><td rowspan=1 colspan=2>39.02</td><td rowspan=1 colspan=1>29.36</td><td rowspan=1 colspan=1>2.0</td><td rowspan=1 colspan=1>11.0</td><td rowspan=1 colspan=1>21.0</td><td rowspan=1 colspan=2>43.5</td></tr><tr><td rowspan=1 colspan=1>InternVL3-2b</td><td rowspan=1 colspan=1>32.5</td><td rowspan=1 colspan=1>32.5</td><td rowspan=1 colspan=1>26.0</td><td rowspan=1 colspan=1>23.5</td><td rowspan=1 colspan=1>78.5</td><td rowspan=1 colspan=1>11.0</td><td rowspan=1 colspan=2>34.15</td><td rowspan=1 colspan=1>27.66</td><td rowspan=1 colspan=1>3.0</td><td rowspan=1 colspan=1>9.0</td><td rowspan=1 colspan=1>24.5</td><td rowspan=1 colspan=2>37.0</td></tr><tr><td rowspan=1 colspan=1>Yi-VL-6b</td><td rowspan=1 colspan=1>18.5</td><td rowspan=1 colspan=1>16.5</td><td rowspan=1 colspan=1>35.5</td><td rowspan=1 colspan=1>45.0</td><td rowspan=1 colspan=1>77.0</td><td rowspan=1 colspan=1>11.5</td><td rowspan=1 colspan=2>25.51</td><td rowspan=1 colspan=1>24.98</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>3.5</td><td rowspan=1 colspan=1>21.5</td><td rowspan=1 colspan=2>34.5</td></tr><tr><td rowspan=1 colspan=1>InternVL3-1b</td><td rowspan=1 colspan=1>27.5</td><td rowspan=1 colspan=1>33.5</td><td rowspan=1 colspan=1>35.5</td><td rowspan=1 colspan=1>19.5</td><td rowspan=1 colspan=1>71.5</td><td rowspan=1 colspan=1>11.5</td><td rowspan=1 colspan=2>32.16</td><td rowspan=1 colspan=1>21.35</td><td rowspan=1 colspan=1>7.5</td><td rowspan=1 colspan=1>7.0</td><td rowspan=1 colspan=1>25.5</td><td rowspan=1 colspan=2>31.0</td></tr><tr><td rowspan=1 colspan=1>Yi-VL-34b</td><td rowspan=1 colspan=1>27.0</td><td rowspan=1 colspan=1>24.5</td><td rowspan=1 colspan=1>23.5</td><td rowspan=1 colspan=1>28.5</td><td rowspan=1 colspan=1>63.0</td><td rowspan=1 colspan=1>9.0</td><td rowspan=1 colspan=2>33.31</td><td rowspan=1 colspan=1>25.00</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>20.0</td><td rowspan=1 colspan=1>22.0</td><td rowspan=1 colspan=2>29.5</td></tr><tr><td rowspan=1 colspan=1>LLaVA-Next-34b</td><td rowspan=1 colspan=1>23.5</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>40.0</td><td rowspan=1 colspan=1>36.5</td><td rowspan=1 colspan=1>53.5</td><td rowspan=1 colspan=1>9.0</td><td rowspan=1 colspan=2>36.06</td><td rowspan=1 colspan=1>17.67</td><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>12.0</td><td rowspan=1 colspan=1>21.5</td><td rowspan=1 colspan=2>19.0</td></tr><tr><td rowspan=1 colspan=1>LLaVA-Next-72b</td><td rowspan=1 colspan=1>23.0</td><td rowspan=1 colspan=1>29.5</td><td rowspan=1 colspan=1>54.5</td><td rowspan=1 colspan=1>24.5</td><td rowspan=1 colspan=1>30.5</td><td rowspan=1 colspan=1>9.0</td><td rowspan=1 colspan=2>20.87</td><td rowspan=1 colspan=1>9.31</td><td rowspan=1 colspan=1>3.0</td><td rowspan=1 colspan=1>17.0</td><td rowspan=1 colspan=1>27.0</td><td rowspan=1 colspan=2>19.0</td></tr><tr><td rowspan=1 colspan=1>DeepSeek-VL2-Tiny</td><td rowspan=1 colspan=1>17.5</td><td rowspan=1 colspan=1>31.0</td><td rowspan=1 colspan=1>14.5</td><td rowspan=1 colspan=1>25.5</td><td rowspan=1 colspan=1>53.0</td><td rowspan=1 colspan=1>14.0</td><td rowspan=1 colspan=2>37.11</td><td rowspan=1 colspan=1>17.59</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>1.0</td><td rowspan=1 colspan=1>21.5</td><td rowspan=1 colspan=2>34.0</td></tr></table>

Note: Object Cls.: Object Classification; Activity Loc.: Activity Localization; Activity Desc.: Activity Description; Scene Cls.: Scene Classification; Emotion Det.: Emotion Detection; Visual Clue Ident.: Visual Clue Identification; Emotion Cause Infer.: Emotion Cause Inference. Visual Clue Identification and Emotion Cause Inference are evaluated using semantic similarity scores.

Table 2: Performance on the 12 MUSE tasks, with models ordered by average performance. All results are reported as percentages. The best and second-best results in each column are highlighted in bold and underlined, respectively.  
![](images/88264b6c05dec148ef2897fd264d5e7b49b5229e77746a0f2bf73b203beae3bf.jpg)  
Figure 5: Top representatives from 8 model families; “Best Available" shows the per-task maximum across models.

## 4.1 Main Results

Table 2 shows that performance remains highly task-dependent, with no model dominating across all capabilities. GPT-5.6-Sol achieves the strongest overall results and surpasses GPT-4o on 10 of 12 tasks, yet GPT-4o remains superior on Activity Description and Relative Position. Open-source models also retain task-specific advantages: Qwen3-VL-32b leads Activity Localization and Scene Classification, while InternVL3-14b and GLM-4V-9b perform best on Relative Position and Jigsaw Puzzle, respectively. These results indicate that progress is uneven and does not translate uniformly across capability dimensions.

![](images/db90710b07a610244f6f450f0b6f1ba104b5fe3442ead78c3548a6037610bc1e.jpg)  
Figure 6: Spearman rank correlations among 12 MUSE tasks.

A clear divide emerges between recognition and integrative reasoning. Scene Classification is comparatively mature, with 23 of 30 models exceeding 75.0 and a median score of 81.0. In contrast, Emotion Detection, Relative Position, Remote Interaction, and Jigsaw Puzzle exhibit substantially lower medians, revealing persistent limitations in affective interpretation, and compositional reasoning.

Figure 5 further shows that model families share similar strengths in scene and activity recognition but diverge sharply on Activity Description, Remote Interaction, and Jigsaw Puzzle, suggesting that architecture and training remain important determinants of capability-specific performance. Scaling is also non-monotonic: although InternVL3-38b outperforms InternVL3-14b on most tasks, it performs worse on Activity Description and Relative Position. Overall, current VLMs are more reliable at recognizing visible content than at grounding predictions in visual evidence, explaining affective causes, or reasoning over perspective-dependent and non-contact relations.

## 4.2 Inter-Task Correlation Analysis

We compute pairwise Spearman's ρ across 30 models to examine relationships among tasks. Figure 6 reveals several coherent capability groups: Object Count, Emotion Detection, and Scene Classification are strongly correlated, while Visual Clue Identification closely tracks Emotion Cause Inference, linking visual evidence grounding with affective reasoning. Activity Localization, Activity Description, and Remote Interaction form a moderately correlated group centered on entity-activity and cross-region reasoning. In contrast, Jigsaw Puzzle correlates weakly with most tasks, indicating a distinct compositional capability. Overall, MUSE captures related but non-redundant dimensions of multimodal understanding rather than a single underlying competence.

![](images/96c9892c5ff650dec20ed66ce15a4c0b4f5d928b39fdbd65425cca2b9cdb5725.jpg)  
Figure 7: Spearman rank correlations between 6 existing benchmarks and 12 MUSE tasks.

## 4.3 Complementarity to Existing Benchmarks

Using pairwise-available model scores, we compute Spearman's ρ between the 12 MUSE tasks and 6 external benchmarks.

## 4.4 Cross-Task Error Analysis

Figure 7 shows that performance on existing benchmarks transfers unevenly to artistic educational imagery. MM-Bench Liu et al. [2024a] and AI2D Kembhavi et al. [2016] correlate strongly with several MUSE tasks, indicating partial overlap in perceptual and semantic capabilities, whereas BLINK Fu et al. [2024] exhibits inconsistent correlations across tasks. Activity Description and Relative Position assess capabilities underrepresented in existing benchmarks. Notably, AICA-Bench Emotion Reasoning She et al. [2026] aligns moderately with MUSE's affective tasks, suggesting that emotion reasoning on conventional visual content only partially transfers to stylized expressions and implicit narratives in artworks. Overall, existing benchmarks explain only part of the model variation on MUSE, supporting its complementary coverage of artistic, affective, compositional, and cultural understanding.

Figure 8 reveals a common grounding failure across the three tasks. In Activity Localization, models usually select semantically relevant people or activities rather than random regions, but fail to identify the complete target extent. Activity Description (abbreviated legend labels are detailed in § QUESTION GENERATION) errors similarly favor co-occurring or concatenated activities, indicating weak separation of the queried event from nearby visual semantics. In Remote Interaction, mismatched region-text pairs dominate, showing that models often accept plausible relations without verifying whether entities, regions, and evidence are jointly aligned. Overall, current VLMs capture coarse semantic relevance but struggle with precise region-activity binding and image-specific relational grounding.

## 4.5 Affective Computing Analysis

Figure 9 shows task-dependent ranking shifts across affective tasks, revealing that affective understanding is not a unified capability. Performance in character recognition or emotion classification does not reliably transfer to visualevidence grounding or emotion-cause inference. Emotion Detection is the clearest bottleneck, reflecting the difficulty of interpreting stylized facial, bodily, and contextual cues. Despite differing metrics, within-task rankings indicate that current VLMs lack integrated affective reasoning from recognition to evidence and causal explanation.

## 4.6 Visual Grounding is the Prerequisite of Accurate Affective Interpretation

Figure 10(a) reveals a cascading failure across target grounding, affect recognition, and causal explanation. GPT-5.6-Sol correctly identifies the target man and attends to relevant cues, but misreads his stylized expression as Surprise, indicating an affect-interpretation error rather than a grounding failure. Other models often shift attention to a salient child and predict Joy, then justify the prediction using butterflies, birds, or nearby interactions. This suggests that errors in coordinate grounding and depth assignment leads models to construct a coherent explanation for the wrong character. More broadly, flattened perspective and ambiguous occlusion in artistic images make affective reasoning depend on jointly resolving target identity, spatial structure, body posture, interactions, and scene context.

![](images/75ef8f5411ca802c5d06cb1f286f32aac47be6f8392d2fadbac42c525847c5bc.jpg)  
(a) Activity Localization

![](images/a0be104f45c8e380880ed7b8cf0239500653439621ace81ee1870f2585a4ef49.jpg)  
(b) Activity Description (abbreviated legend)

![](images/25305808773f9a50f32e839ee83cdc9bb636dd6692d7afad3dcf443883819e99.jpg)  
(c) Remote Interaction  
Figure 8: Distributions of incorrect option or evidence-selection types across models for three tasks.

## 4.7 Viewpoint-Aware Spatial Reasoning is a Persistent Bottleneck

Figure 10(b) exposes a strong forced-relation bias in spatial reasoning. Although the ground truth specifies no definite lateral or vertical relation, 90.0% and 73.3% of models, respectively, predict one; depth reasoning is also unreliable, with only 43.3% correctly identifying the girl as in front of the boy. No model resolves all three dimensions correctly. Current VLMs therefore oscillate between two failure modes: asserting definite relations under ambiguous evidence or predicting None across all dimensions and missing valid depth cues. This reveals weak viewpoint-aware spatial reasoning and poor calibration of spatial uncertainty.

## 5 Conclusion

We introduced MUSE, a benchmark for evaluating multimodal understanding of artistic imagery in image-based language learning and educational interaction. Its construction framework decouples reusable visual-semantic annotations from task-specific question generation, enabling 12 tasks across five capability dimensions with control over question format and difficulty. Evaluation of 30 open-source and proprietary VLMs reveals task-dependent performance: models are reliable at scene and activity recognition but remain limited in visual grounding, affective interpretation, and compositional reasoning. Correlation analyses show that MUSE measures related yet non-redundant capabilities and complements general-purpose and emotion-reasoning benchmarks. Our error analyses identify recurring failures in precise region-activity binding, entity-evidence alignment, target grounding, and calibration under ambiguous spatial relations; these errors can propagate into coherent explanations for incorrectly grounded characters. Results indicate that scaling or stronger coarse recognition alone is insufficient. Reliable educational VLMs require region-aware grounding, integrated reasoning from perception to evidence and causes, and viewpoint-aware modeling of spatial uncertainty. MUSE provides a foundation for measuring progress toward these capabilities on artistic and culturally situated imagery.

![](images/cb85a6d96e9db9e667bbe2497db8863835ba9863ed3ee56ca5d4e32697207f35.jpg)  
Figure 9: Five top large VLMs on affective computing.

![](images/5e2e31a7b5e16b0799b15c0f880fd20020548a467d6ea7541916d1fc3358679b.jpg)  
(a) Localization and emotion

![](images/4163f38367cd6a63015e5bafeb518a0434c342fbe555d9bc060c3628e8f546d3.jpg)  
(b) Relative Position  
Figure 10: Failure cases in Affective Computing and Relative Position.

## References

OpenAI. Gpt-4v(ision) system card. Technical Report, 2023.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond, 2023. URL https://arxiv.org/abs/2308.12966.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Intern vl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24185–24198. IEEE Computer Society, 2024a.

Zhendong Chu, Jian Xie, Shen Wang, Zichao Wang, and Qingsong Wen. UniEDU: Toward unified and efficient large multimodal models for educational tasks. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 1007–1016, Suzhou, China, 2025. Association for Computational Linguistics. doi:10.18653/v1/2025.emnlp-industry.68.

Chengxu Zhuang, Evelina Fedorenko, and Jacob Andreas. Visual grounding helps learn word meanings in low-data regimes. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 1311–1329, 2024.

Mariana Shimabukuro, Deval Panchal, and Christopher Collins. LangEye: Toward anytime' learner-driven vocabulary learning from real-world objects. In Ekaterina Kochmar, Bashar Alhafni, Marie Bexte, Jill Burstein, Andrea Horbach, Ronja Laarmann-Quante, Anaïs Tack, Victoria Yaneva, and Zheng Yuan, editors, Proceedings of the 20th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2025), pages 446–459, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-270-1. doi:10.18653/v1/2025.bea-1.33. URL https://aclanthology.org/2025.bea-1.33/.

Zhengqing Yuan, Yunhong He, Kun Wang, Yanfang Ye, and Lichao Sun. Artgpt-4: Towards artistic-understanding large vision-language models with enhanced adapter. arXiv preprint arXiv:2305.07490, 2023.

Andrea Alfarano, Lorenzo Venturoli, and Dario Negueruela Del Castillo. VQArt-Bench: A semantically rich VQA benchmark for art and cultural heritage. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops, pages 396–406, 2025.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. MMBench: Is your multi-modal model an all-around player? In Computer Vision – ECCV 2024, pages 216–233. Springer, 2024a.

Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In Advances in Neural Information Processing Systems, volume 35, pages 2507–2521, 2022.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. MathVista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations, 2024.

Kaining Ying, Fanqing Meng, Jin Wang, Zhiqian Li, Han Lin, Yue Yang, Hao Zhang, Wenbo Zhang, Yuqi Lin, Shuo Liu, et al. Mmt-bench: A comprehensive multimodal benchmark for evaluating large vision-language models towards multitask agi. In International Conference on Machine Learning, pages 57116–57198. PMLR, 2024.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024.

YiFan Zhang, Huanyu Zhang, Haochen Tian, Chaoyou Fu, Shuangqing Zhang, Junfei Wu, Feng Li, Kun Wang, Qingsong Wen, Zhang Zhang, et al. Mme-realworld: Could your multimodal llm challenge high-resolution realworld scenarios that are difficult for humans? In International Conference on Learning Representations, volume 2025, pages 89655–89701, 2025.

Yi Cheng, Siyao Li, Bang Liu, Ruihui Zhao, Sujian Li, Chenghua Lin, and Yefeng Zheng. Guiding the growth: Difficulty-controllable question generation through step-by-step rewriting. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 5968–5978, 2021.

Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. SEED-Bench: Benchmarking multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13299–13308, 2024.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and Feng Zhao. Are we on the right way for evaluating large vision-language models? In Advances in Neural Information Processing Systems, volume 37, 2024b.

Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang Huan Sun, Yu Su, Wenhu Chen, and Graham Neubig. MMMU-pro: A more robust multi-discipline multimodal understanding benchmark. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15134–15186, Vienna, Austria, 2025. Association for Computational Linguistics. doi:10.18653/v1/2025.acl-long.736.

Panos Achlioptas, Maks Ovsjanikov, Kilichbek Haydarov, Mohamed Elhoseiny, and Leonidas J. Guibas. ArtEmis: Affective language for visual art. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11569–11579, 2021.

Youssef Mohamed, Mohamed Abdelfattah, Shyma Alhuwaider, Feifan Li, Xiangliang Zhang, Kenneth Church, and Mohamed Elhoseiny. ArtELingo: A million emotion annotations of WikiArt with emphasis on diversity over language and culture. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 8770–8785, Abu Dhabi, United Arab Emirates, 2022. Association for Computational Linguistics. doi:10.18653/v1/2022.emnlp-main.600.

Dong She, Xianrong Yao, Liqun Chen, Jinghe Yu, Yang Gao, and Zhanpeng Jin. AICA-bench: Holistically examining the capabilities of VLMs in affective image content analysis. In Findings of the Association for Computational Linguistics: ACL 2026, pages 13501–13528, San Diego, California, United States, 2026. Association for Computational Linguistics. doi:10.18653/v1/2026.findings-acl.661.

David Romero, Chenyang Lyu, Haryo Akbarianto Wibowo, Teresa Lynn, Injy Hamed, Aditya Nanda Kishore, Aishik Mandal, Alina Dragonetti, Artem Abzaliev, Atnafu Lambebo Tonja, et al. CVQA: Culturally-diverse multilingual visual question answering benchmark. In Advances in Neural Information Processing Systems, volume 37, 2024. doi:10.52202/079017-0366.

Robert Plutchik. A general psychoevolutionary theory of emotion. In Theories of emotion, pages 3–33. Elsevier, 1980.

Wenyi Hong, Weihan Wang, Ming Ding, Wenmeng Yu, Qingsong Lv, Yan Wang, Yean Cheng, Shiyu Huang, et al. CogVLM2: Visual language models for image and video understanding. arXiv preprint arXiv:2408.16500, 2024.

Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai, Huazuo Gao, et al. DeepSeek-VL2: Mixture-of-experts vision-language models for advanced multimodal understanding. arXiv preprint arXiv:2412.10302, 2024.

Gemma Team. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen Duan, et al. InternVL3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Lee, et al. LLaVA-NeXT: Improved reasoning, ocr, and world knowledge. LLaVA project technical blog, 2024b. Released January 2024.

Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, et al. MiniCPM-V: A GPT-4V level multimodal large language model on your phone. arXiv preprint arXiv:2408.01800, 2024.

OpenBMB. MiniCPM-o 2.6: A GPT-4o-level multimodal large language model on end devices. Model card and technical documentation, 2025. OpenBMB MiniCPM-o 2.6.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, et al. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025a.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, et al. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025b.

Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, et al. Yi: Open foundation models by 01.AI. arXiv preprint arXiv:2403.04652, 2024.

Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Computer Vision – ECCV 2016, pages 235–251. Springer, 2016. doi:10.1007/978-3-319- 46493-015.

Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A. Smith, Wei-Chiu Ma, and Ranjay Krishna. BLINK: Multimodal large language models can see but not perceive. In Computer Vision – ECCV 2024, pages 148–166. Springer, 2024. doi:10.1007/978-3-031-73337-6\_9.

## A Additional Analyses

## A.1 Human-Model Comparison

![](images/7ddba4b758b879b32a0b97a1ef7dfdafff35e9a4cffe3cf6cf6f037631e66bf0.jpg)  
Figure 11: Comparison of human performance with top representatives from eight model families across the 12 MUSE tasks.

To analyze the performance gap between humans and large VLMs across the 12 MUSE tasks, we randomly sampled 20 questions from each task and asked two annotators to answer them. We also collected the corresponding responses generated by different models for the same set of sampled questions. Figure 11 shows that the human average forms the outer performance envelope on nearly all tasks, demonstrating a substantial gap between current VLMs and human multimodal understanding. The largest deficits occur in Relative Position, Emotion Detection, and Jigsaw Puzzle, where even the strongest models remain far below human performance. The gap is narrower for Activity Localization, Object Count, Remote Interaction, and Scene Classification, indicating stronger progress in visible-content recognition and selected relational tasks. Model profiles nevertheless vary considerably: GPT-5.6-Sol is strongest on Object Count and Remote Interaction, while GLM-4V-9b performs particularly well on Jigsaw Puzzle. These differences reinforce that no model family consistently approaches human performance across all capabilities.

## A.2 Performance across Taxonomy Dimensions

Figure 12 reveals consistent performance imbalances across the MUSE taxonomy. GPT-5.6-Sol has the strongest and most balanced overall profile, although other models retain dimension-specific advantages. Across capabilities, semantic and cultural understanding are generally stronger than affective interpretation and compositional visual reasoning. Performance also tends to decrease from low- and mid-level tasks to high-level reasoning, showing that success on recognition and semantic perception does not reliably extend to more complex inference. Across spatial granularities, image-level understanding is consistently strongest, whereas pixel-level understanding is weakest and crop-level performance remains intermediate. This pattern indicates that global scene interpretation is more mature than precise local grounding.

![](images/49b2eb675180119867d969a4936940bc0ca997039f44a4d5d2bb7b94434ac24c.jpg)

(a) Capability  
![](images/4c4d90b282fe2e964454a7e3bdba89250aa7146e9b222d3d139000cf65ced58c.jpg)  
(b) Difficulty

![](images/c04a7c5bda0e451dec2f7272ffb9464a2ade8c0c97f274cfbc1cb2707905fd67.jpg)  
(c) Granularity  
Figure 12: Top representatives from eight model families across capability, difficulty, and granularity dimensions.

![](images/5275fb5b9ec2de490a2ec041f7b527159ff62a41e45e7ac967e9a4a918c4999b.jpg)  
Figure 13: Mean invalid response rate for each VLM.

## A.3 Invalid Response Analysis

Figure 13 shows a highly skewed distribution of invalid responses. Most models have invalid rates below 1%, and several produce no invalid responses. In contrast, Yi-VL-6b and DeepSeek-VL2-Tiny exceed 12%, while Yi-VL-34b, LLaVA-NeXT-72b, Qwen2.5-VL-3b, and smaller InternVL3 variants also exhibit elevated rates. Invalid responses are not determined solely by model scale: models within the same family vary substantially, and GPT-5.6-Sol retains a 2.62% invalid rate despite its strong task performance. Thus, response-format reliability constitutes a distinct evaluation concern alongside answer correctness.

## B Examples for Selected Tasks

## B.1 Cultural Identity

The following is an example Cultural Identification question and its corresponding image Figure 14. The red bounding box is included solely to facilitate interpretation and is not shown to the VLMs during inference.

## Question

Given an image, identify the culture that is most relevant to the content within the bounding box [1.0945860806163514e-17, 0.19206680584551108, 0.15845070422535204, 0.4906054279749479]. The bounding box coordinates are in COCO-format [xmin, ymin, width, height]. All the coordinates are in percentages between 0 to 1. Please select the most appropriate culture option from the following options.

options:

![](images/965bf6df0bbff6fce0263d6b14890b0a1e75820d30a0dc35e6967ee8f5160776.jpg)  
Figure 14: Example image for Cultural Identitfication.

A. Western B. China C. Europe D. Muslim

The response should be in the following format. The answer should be A / B / C / D only.

\- Do not include any additional text or explanation.

## B.2 Jigsaw Puzzle

The following is an example Jigsaw Puzzle question and its corresponding image Figure 15.

![](images/c13595b896aad82ecf5c925ce929c4ea64ce05e60ee374ee0ca92b4363ac234b.jpg)  
Choose the missing jigsaw piece.

![](images/2a25c824152fa1422a88583198eac166efc7656798b674b8565e5f780be6362e.jpg)  
Figure 15: Example image for Jigsaw Puzzle.

## Question

Given an image with a missing region, select the one candidate image piece that best completes the image.

Options: A B C D

Instructions:

1. Exactly one option is correct.

2. Answer using only a single uppercase letter: A, B, C, or D.

3. Do not output any explanation, reasoning, punctuation, or additional text.

## B.3 Affective Computing

The following shows a four-turn-sequence questions for Object Classificaiton, Emotion Detection, Visual Cause Indentification, and Emotion Cause Inference.

Question - Object Classification

![](images/618f20b5ea0c1f4fe1cc1ced3ca60e30af742498d2e5f24d1845c8c1c120e1d3.jpg)  
Figure 16: Example image for affective computing.

Given an image and a bounding box, identify the object category corresponding to the bounding box. The bounding box coordinates are in COCO-format [xmin, ymin, width, height], with all values between 0 and 1

[bounding box] [0.685, 0.679, 0.086, 0.295]

[object options] A. Boy B. Woman C. Baby D. Girl E. Man

Return exactly one line in this format:

[option] <selected object option letter>

Constraints:

\- Output must start with [option]

\- Followed by a space and a single uppercase letter (A-Z)

\- Do not include any additional text or explanation.

Question - Emotion Detection

Given the same image and bounding box, identify the emotion of the person inside the bounding box.

[bounding box] [0.685, 0.679, 0.086, 0.295]

[emotion options]

A. Guilt B. Confusion C. Sadness D. Neutral E. Boredom F. Disgust G. Surprise H. Joy I. Anger J. Anxiety K. Fear Return exactly one line in this format:

[emotion] <selected emotion option letter>

Constraints:

\- Output must start with [emotion]

\- Followed by a space and a single uppercase letter (A-Z)

\- Do not include any additional text or explanation.

## Question - Visual Clue Identification

Based on the image and bounding box below, describe the observable visual clues that support the previously identified emotion.

[bounding box] [0.685, 0.679, 0.086, 0.295]

[emotion] {identified\_emotion}

Return the result in the following format.

[visual clues] <identified visual clues>

Question - Emotion Cause

Based on the image, the bounding box, and the visual clues above, infer the most likely cause of the identified emotion.   
Return the result in the following format.

[emotion cause] <inferred emotion cause>

## C Model Hyperparameters

Table 3 summarizes the computation dtypes used during inference. Most evaluated model families use BF16, while the LLaVA-NeXT models use FP16. CogVLM2 uses BF16 when supported by the hardware and otherwise falls back to FP16. We retain the default dtypes specified by the corresponding inference scripts to reflect standard deployment settings and apply the same numerical configuration across all MUSE tasks for each model.

Table 4 summarizes the default generation configuration used in our inference pipeline. We disable sampling to obtain deterministic outputs and set max\_new\_tokens to 1024 to accommodate both short structured answers and open-ended responses. Consequently, temperature, top-k, and top-p do not affect decoding. All other unspecified parameters inherit the corresponding model or library defaults, preserving each model's native beam-search, repetition-control, and caching behavior.

## D Annotation Process

Annotator Recruitment and Preparation. We recruited 127 undergraduate and postgraduate student annotators. Before annotation, they completed a 0.5-hour training session based on written guidelines specifying annotation categories, bounding-box conventions, and procedures for resolving ambiguous artistic content. Feedback from a pilot annotation stage was incorporated to further clarify the guidelines.

Time, Compensation, and Cost. The average cost of commissioning each image from freelance artists was approximately USD 36. Annotators spent approximately 3-4 minutes per image, which varies based on task categories, corresponding to 4 hours of annotation. They were compensated at USD 16 per hour. Additional costs included platform fees . Compensation was set with reference to local institutional policy.

Ethical and Data-Handling Considerations. Annotators were informed about the purpose and intended use of the dataset. We collected no personal information beyond what was necessary for compensation and quality control. Potentially sensitive cultural or affective annotations were reviewed carefully.

<table><tr><td>Family</td><td>Model</td><td>Dtype</td></tr><tr><td>CogVLM2</td><td>cogvlm2-1lama3-chat-19b</td><td>BF16†</td></tr><tr><td rowspan="3">DeepSeek-VL2</td><td>deepseek-vl2-tiny</td><td>BF16</td></tr><tr><td>deepseek-vl2-small</td><td>BF16</td></tr><tr><td>deepseek-v12</td><td>BF16</td></tr><tr><td rowspan="3">Gemma-3</td><td>gemma-3-4b-it</td><td>BF16</td></tr><tr><td>gemma-3-12b-it</td><td>BF16</td></tr><tr><td>gemma-3-27b-it</td><td>BF16</td></tr><tr><td>GLM-4V</td><td>glm-4v-9b</td><td>BF16</td></tr><tr><td rowspan="5">InternVL3</td><td>internvl3-1b</td><td>BF16</td></tr><tr><td>internvl3-2b</td><td>BF16</td></tr><tr><td>internv13-8b</td><td>BF16</td></tr><tr><td>internvl3-9b internvl3-14b</td><td>BF16</td></tr><tr><td>internv13-38b</td><td>BF16 BF16</td></tr><tr><td rowspan="6">LLaVA-NeXT</td><td>1lama3-1lava-next-8b</td><td>FP16</td></tr><tr><td>1lava-next-72b-hf</td><td>FP16</td></tr><tr><td>1lava-v1.6-34b-hf</td><td>FP16</td></tr><tr><td>1lava-v1.6-mistral-7b-hf</td><td>FP16</td></tr><tr><td>1lava-v1.6-vicuna-7b-hf</td><td>FP16</td></tr><tr><td>1lava-v1.6-vicuna-13b-hf</td><td>FP16</td></tr><tr><td rowspan="3">MiniCPM</td><td>minicpm-1lama3-v-2_5</td><td>BF16</td></tr><tr><td>minicpm-o-2_6</td><td>BF16</td></tr><tr><td>minicpm-v-2_6</td><td>BF16</td></tr><tr><td rowspan="4">Qwen2.5-VL</td><td>qwen2_5_vl_3b</td><td>BF16</td></tr><tr><td>qwen2_5_vl_7b</td><td>BF16</td></tr><tr><td>qwen2_5_vl_32b</td><td>BF16</td></tr><tr><td>qwen2_5_vl_72b</td><td>BF16</td></tr><tr><td rowspan="2">Qwen3-VL</td><td>qwen3_vl_8b-instruct</td><td>BF16</td></tr><tr><td>qwen3_vl_32b-instruct</td><td>BF16</td></tr><tr><td rowspan="2">Yi-VL</td><td>yi-vl-6b</td><td>BF16</td></tr><tr><td>yi-vl-34b</td><td>BF16</td></tr></table>

†The CogVLM2 script uses BF16 when supported by the hardware and otherwise falls back to FP16.  
Table 3: Default computation dtypes used by the inference scripts.

<table><tr><td>Parameter</td><td>Default</td><td>Effect under Default Setting</td></tr><tr><td>max_new_tokens</td><td>1024</td><td>Maximum generated length</td></tr><tr><td>do_sample</td><td>False</td><td>Deterministic decoding</td></tr><tr><td>temperature</td><td>None</td><td>Inactive without sampling</td></tr><tr><td>top_k</td><td>None</td><td>Inactive without sampling</td></tr><tr><td>top-p</td><td>None</td><td>Inactive without sampling</td></tr><tr><td>num_beams</td><td>None</td><td>Uses the library default</td></tr><tr><td>repetition_penalty</td><td>None</td><td>Uses the library default</td></tr><tr><td>num_return_sequences</td><td>None</td><td>Uses the library default</td></tr><tr><td>use_cache</td><td>None</td><td>Uses the model default</td></tr><tr><td>cache_implementation</td><td>None</td><td>Uses the model default</td></tr></table>

Table 4: Default generation settings used in our inference pipeline. Parameters set to None use the underlying model or library defaults. Since sampling is disabled, sampling-specific parameters such as temperature, top-p, and top-k are inactive under the default configuration.