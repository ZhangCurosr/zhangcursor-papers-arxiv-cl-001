# 4MT-VLM: How Coarse Is a VLMs Cognitive Map?

Markus Frey Lamarr Institute for Machine Learning and Artificial Intelligence Fraunhofer IAIS, University of Bonn markus.frey@iais.fraunhofer.de

## Abstract

An agent that moves must recognise a place from a viewpoint it has never seen. We introduce 4MT-VLM, a dataset of procedurally generated landscapes, each rendered across five stimulus modes that remove appearance cues while holding layout fixed: shape and colour, shape only, colour only, bare terrain peaks with no objects, and a valley viewpoint that puts the peaks on the horizon. The last condition is commonly used in clinics to probe hippocampal function in human patients. We test this benchmark across sixteen different open and closed-source models and report 4AFC performance, a measure which is also used to grade human participants. We observe that models identify a place from the studied viewpoint but lose it once the camera moves, dropping below the 25% chance level at 135° where a human observer scores 85%. Frontier models (Gemini 3.8 Flash, GPT-5.6) answer only 39% and 31% of rotated trials correctly, recovering to 85% and 55% only when distractors are moved more than 30 meters apart. Our benchmark demonstrates that while current VLMs possess rudimentary cognitive maps, their spatial resolution remains fundamentally too coarse to maintain a stable, 3D understanding of the world once the viewpoint changes.

## 1 Introduction

An agent that moves must recognise a place from a viewpoint it has never seen. Doing so requires an allocentric representation, i.e. a cognitive map of where things are that does not depend on where the observer stands [8, 11]. As vision–language models (VLMs) are increasingly deployed as the visual reasoning engines for embodied agents, they are expected to maintain a stable, 3D understanding of the world as they navigate and their camera view changes.

How well they actually do this is hard to read off current spatial benchmarks. Most multimodal evaluations score perception, language, and geometry together, returning a single accuracy number. A low score indicates that something is missing without identifying what, and a high score often means the model found a visual shortcut, such as matching textures or background colours. Furthermore, difficulty in these benchmarks is arbitrary. Because embodied agents operate in physical space, their spatial reasoning failures should be measurable in physical units, not just percentage points on a static dataset.

To build a more diagnostic evaluation, we adapt The Four Mountains Test (4MT) [6], which was designed specifically to isolate allocentric spatial memory. A participant studies a computer-generated landscape, then must identify it among four candidates rendered from a new viewpoint, with all colours and textures resampled. Scores fall with hippocampal damage [1, 6] and in pre-dementia Alzheimer's disease [2, 12], which attacks the exact brain regions responsible for allocentric navigation.

For benchmarking allocentric perception, we build a novel test with two primary constraints:

1. Viewpoint change $\Delta \in \{ 0 ^ { \circ } , 4 5 ^ { \circ } , 9 0 ^ { \circ } , 1 3 5 ^ { \circ } , 1 8 0 ^ { \circ } \}$ . Appearance is resampled at every ∆, so ∆ = 0 is not an image match but a check that the place can be identified at all. We call accuracy at $\Delta = 0$ the appearance gate, and treat it as the denominator for any claim about viewpoint invariance.

2. Distractor similarity, in metres. For every pair of scenes we compute the rotation-optimal layout distance $\begin{array} { r } { D ( i , j ) = \operatorname* { m i n } _ { \phi } \frac { 1 } { K } \sum _ { k } \| R \dot { ( } \bar { \phi } ) p _ { i k } - p _ { j k } \| } \end{array}$ , which is what is left after the best rigid rotation, and draw distractors from a chosen percentile band of that distribution.

Crossing these two axes separates three distinct failures that standard benchmarks run together: failing to identify the scene visually, failing to compensate for camera rotation, and failing to distinguish physical layouts that are too close together. Our contributions are:

• 4MT-VLM: A diagnostic benchmark for embodied AI, adapted from a clinical test of hippocampal function. It contains 500 trials over 100 procedurally generated landscapes, systematically stripping away appearance cues to isolate layout across graded viewpoint shifts.

• Evaluation in physical units. We generate metric “hard negatives" by drawing distractors from percentile bands of layout distance in metres. Because we redraw only the distractors while leaving the target and azimuths identical, differences in performance represent a measurable limit on the model's spatial resolution.

• A physical diagnosis of scaling failure. We show that scaling open-weight models from 1B to 235B parameters drastically improves scene identification (the appearance gate) but does nothing for rotational invariance, leaving performance at or below chance. Distance is the only manipulation that recovers performance: Gemini 3.8 Flash jumps from 39% to 85% accuracy only when distractors are moved 31 m apart. The information is there, but its resolution is fundamentally too coarse for precise allocentric navigation.

## 2 Related work

Spatial memory in humans. The 4MT turns the cognitive map account of hippocampal function [8] into a four-alternative forced choice, and its diagnostic value comes from the viewpoint shift specifically [6]. Separately, Shepard and Metzler [10] showed that the time to match two objects across a rotation grows linearly with angular disparity, which indicates a transformation applied to a representation rather than a lookup of a stored view. Both paradigms fix the scene, move the vantage point, and measure what that costs. Frey et al. [3] take the same task to artificial networks, training them on a scene-perception problem while reading out what the learned representations encode.

Spatial benchmarks for vision-language models. Recent evaluations keep finding the same split between egocentric and allocentric questions. Fu et al. [4] turn fourteen classic vision tasks into multiple choice and report frontier accuracy near 50% where people score above 95%. Yang et al. [13] measure spatial memory from video and find that chain-of-thought, self-consistency and tree-of-thoughts all fail to help. Li et al. [7] find competent reasoning in the camera frame and failure in another entity's frame. Zhang et al. [15] isolate perspective taking and rotation across 43 models and report a strong egocentric bias against 91% human accuracy, and Zhang et al. [14] find that models handle 2D relations within one image but not the combination of several views into one global frame. These benchmarks establish that the capability is weak. Difficulty in them comes from the choice of question, not from a measured property of the individual item, so they can report that a model fails without locating the point where it starts to.

Building allocentric representations. A parallel line of work supplies the missing representation instead of measuring it. Gu et al. [5] build a voxelised cognitive map from video and fuse it back into the visual features. Ruan et al. [9] build a top-down landmark tree using reconstruction and segmentation tools, and report that a text-only model reading that tree comes close to multimodal performance. Both assume the representation is missing and add it from outside. We ask instead what the representation already inside the model supports, and where it runs out of resolution.

shape + colour option 1

One place, five stimulus modes cO shape + colour c1 shape only

![](images/b637ab6dd38531f7ae7f6b918eeb3092e59d2c6bacef5ad57c1d41f58198dd74.jpg)

![](images/517afbadc64d511239bf063c59838a090f0fb09e47401d9a4700e9cd92ccdc00.jpg)  
c2 colour only

![](images/29820dcc895f7aa959399801f2186b35475bce14a019fea603c50b89e0d97df9.jpg)  
c3 bare peaks

![](images/50971589e90b2979c6381370f29fbdd3245fcec82a187087d04dedd7497d161d.jpg)  
c4 valley

![](images/479fa4238863f318aab81576060ff82db4083ac9c1d3fa7e0ac3255f63adf427.jpg)  
One trial in c0 study (90°)

![](images/cb69fd3a3d00816706a888bb2ca2e1916cf08db321103975dcffcf267916e841.jpg)  
this place

![](images/7f366ddabcb267c4602bed890a1664bfafa19cb7b267c6f24963e75c6b883932.jpg)  
7.2 m away  
option 2

![](images/e795200640c745cddf7462f60aa785d3b8a5c147151ecd4cde67dc98dd100d0e.jpg)  
3.7 m away

![](images/5da966664c2449b290e5bec8391b785c80c75ea34003ca74ce7a45715f5d66a7.jpg)  
same place  
option 4

![](images/23b51b5084dcf636d10d7600700658d9b93a2765e31691ca4788730e3873d507.jpg)  
11.4 m away  
One trial in c4 valley study (270°)

![](images/620e6109267f4eb6dc2c72e8014f174ae72e8e2e34ad09e3cf63eea6d4ae847b.jpg)  
this place

![](images/4a3b30fe4f8ada8cd895093d6bed199bf9d38d4fdcb94a2d1579f7264eea3ac6.jpg)  
6.8 m away

![](images/053495bb3b6e76dc72c973b4be9edd9ced41a5cc2b05d46862dd00c194b97e39.jpg)  
same place

![](images/d69093e2ab622bc13124c1b51a5d77d3a5fcc2f1568b15d3023f23a2b514687d.jpg)  
4.6 m away  
option 4

![](images/b9f90531b71b96846e12f97d969255dc8b7886fd1e0ac3ab8c8674c204f0c4fb.jpg)  
11.2 m away

Figure 1: Top: one place in the five stimulus modes, at one viewpoint. Colour is removed, then shape, then the objects themselves, and c4 moves the camera into the valley so the peaks sit on the horizon. Layout is identical across the five. Middle and bottom: one trial in c0 and one in c4. The study view, then four candidates rendered 135°away with appearance resampled, labelled with their layout distance to the target.

## 3 Methods

Stimuli. We rendered 100 procedurally generated landscapes of four peaks each in Blender, from eight azimuths under two appearance samples. Five stimulus modes strip appearance cues away while holding layout fixed: shape and colour (c0), shape only (c1), colour only (c2), bare terrain peaks with no objects (c3), and a valley viewpoint with the peaks on the horizon (c4) (see Figure 1). All modes draw from a shared inventory of objects, so that no mode carries landmark identity that another lacks.

Task. Each trial shows a study image and four candidates rendered at a common test azimuth, one of which is the study scene. Study and test images always use different appearance samples, at every ∆. The full set is 500 trials, balanced over 5 modes × 5 values of ∆ × 20 items, with the correct answer spread evenly over the four response positions.

Distractor sets. Distractors come from a percentile band of D(i, j), computed separately within each mode because the modes differ in absolute scale. The main set uses the 0–10th percentile, which gives a median nearest-distractor distance of 6.8 m. Two matched sets redraw only the distractors, from the 40–60th percentile (31.4 m) and the 90–100th (42.8 m).

Landmark count. The distractor sets vary how far the alternatives sit from the target while holding the scenes fixed. A second manipulation varies the number of objects in the scenes. We rendered three further banks of 100 landscapes containing one, two and six landmarks, identical to the main bank in every render setting (eight azimuths, 640 × 440, 24 samples, same seed), and built a matched four-landmark bank from the main scenes so that all four sizes are constructed the same way. These use the object mode (c0) only, because every landmark must be unique in both shape and colour and the inventory holds six of each.

<table><tr><td>Observer</td><td>Params</td><td>Accuracy</td><td>95% CI</td><td> $\Delta { = } 0 \ \mathbf { g a t e }$ </td><td> $\Delta \geq 4 5$ </td></tr><tr><td>Human</td><td>一</td><td>86%</td><td>[78%, 91%]</td><td>100%</td><td>82%</td></tr><tr><td>InternVL3.5-4B</td><td>4B</td><td>17%</td><td>[11%, 26%]</td><td>35%</td><td>12%</td></tr><tr><td>InternVL3.5-2B</td><td>2B</td><td>21%</td><td>[14%, 30%]</td><td>25%</td><td>20%</td></tr><tr><td>InternVL3.5-1B</td><td>1B</td><td>23%</td><td>[16%, 32%]</td><td>35%</td><td>20%</td></tr><tr><td>InternVL3.5-38B</td><td>38B</td><td>28%</td><td>[20%, 37%]</td><td>65%</td><td>19%</td></tr><tr><td>InternVL3.5-8B</td><td>8B</td><td>28%</td><td>[20%, 37%]</td><td>55%</td><td>21%</td></tr><tr><td>InternVL3.5-30B-A3B</td><td>30B</td><td>29%</td><td>[21%, 39%]</td><td>55%</td><td>22%</td></tr><tr><td>Qwen2.5-VL-32B</td><td>32B</td><td>29%</td><td>[21%, 39%]</td><td>75%</td><td>18%</td></tr><tr><td>Qwen2.5-VL-3B</td><td>3B</td><td>29%</td><td>[21%, 39%]</td><td>30%</td><td>29%</td></tr><tr><td>Qwen3-VL-235B (t)</td><td>235B</td><td>30%</td><td>[22%, 40%]</td><td>75%</td><td>19%</td></tr><tr><td>InternVL3.5-14B</td><td>14B</td><td>31%</td><td>[23%, 41%]</td><td>50%</td><td>26%</td></tr><tr><td>Qwen2.5-VL-72B</td><td>72B</td><td>31%</td><td>[23%, 41%]</td><td>75%</td><td>20%</td></tr><tr><td>Qwen3-VL-235B (i)</td><td>235B</td><td>33%</td><td>[25%, 43%]</td><td>90%</td><td>19%</td></tr><tr><td>InternVL3-38B</td><td>38B</td><td>34%</td><td>[25%, 44%]</td><td>85%</td><td>21%</td></tr><tr><td>Qwen2.5-VL-7B</td><td>7B</td><td>36%</td><td>[27%, 46%]</td><td>55%</td><td>31%</td></tr><tr><td>Gemini 3.8 Flash</td><td>一</td><td>44%</td><td>[35%, 54%]</td><td>65%</td><td>39%</td></tr><tr><td>GPT-5.6 Luna</td><td>一</td><td>45%</td><td>[36%, 55%]</td><td>100%</td><td>31%</td></tr></table>

Table 1: Four-alternative forced choice with hard foils (100 trials, chance 25%: 20 unrotated and 80 rotated). The $\Delta = 0$ column is the appearance gate. Qwen3-VL-235B (i) is the instruct variant, (t) the thinking one. Rows are ordered by overall accuracy. Bold marks the human row and the best model in each column. Accuracy by mode is reported in Appendix Table 3.

Procedure. Every observer answered the same 100-trial subset, stratified over mode and ∆. Models got a chain-of-thought instruction, a token budget of 8000 and, where the provider exposes it, medium reasoning effort. Every instruction is reproduced verbatim in the appendix.

Analysis. Chance level for the four-alternative forced choice (4AFC) task is 25%. Confidence intervals are Wilson intervals, paired comparisons on identical trials use the McNemar test, and distributional comparisons use permutation tests with $2 \times 1 0 ^ { 4 }$ resamples. We checked that no observer's answer distribution alone gives an advantage: the accuracy expected from each observer's own answer frequencies, ignoring the images, is between 19% and 27% for all seventeen observers.

## 4 Results

We report two accuracies throughout. The appearance gate is the accuracy at $\Delta = 0$ , where the place is shown from the studied viewpoint under a fresh appearance sample, e.g. the scene did not change but the position of the sun did. The rotated accuracy is the performance of the model for all other images where the viewpoint change is $\Delta \geq 4 5 ^ { \circ }$

Models identify the scene and lose it under rotation. Table 1 reports the two accuracies separately. GPT-5.6 Luna answers 100% of unrotated trials and 31% of rotated trials. Qwen3-VL-235B answers 90% and 19%, where a human observer answers 100% and 82%. On the identical 100 trials, paired exact McNemar puts every model below the human baseline $( p = 1 . 3 \times 1 0 ^ { - 1 0 }$ for Gemini 3.8 Flash, $1 . 3 \times 1 0 ^ { - 8 }$ for GPT-5.6 Luna, $2 . 3 \times 1 0 ^ { - 1 4 }$ for Qwen3-VL-235B), and the gap holds within rotated trials alone $( p \leq 1 . 3 \times 1 0 ^ { - 8 }$ for all sixteen). The task is passable, and the viewpoint change is the part the models fail.

Scale buys scene identification, not viewpoint invariance. Across Qwen2.5-VL at 3B, 7B, 32B and 72B, with prompt, stimuli and decoding fixed, unrotated accuracy rises $3 0 \to 5 5 \to 7 5 \to 7 5 \%$ while rotated accuracy falls $2 9 \to 3 1 \to 1 \bar { 8 } \to 2 0 \%$ and crosses below chance (Figure 2). To rule out a floor effect from distractor difficulty, we repeated the series with distractors 4.6 to 6.3× further away. Rotated accuracy stays flat at $2 4 \overset { \cdot } {  } 2 7 \overset { \cdot } {  } 2 6 \overset { \cdot } {  } 2 8 \%$ . On the same trials the identical change takes Gemini 3.8 Flash from 39% to 85% and GPT-5.6 Luna from 31% to 55%. Another generation at three times the size does not change the picture. Qwen3-VL-235B-A22B identifies 90% of unrotated scenes and answers 19% of rotated ones; widening the distractors takes its gate to 100% and its rotated accuracy to 25%, which is chance. More inference-time computation does not help either: the thinking variant reaches 30% overall against 33% for instruct, with the same 19% rotated.

unrotated (∆=0)rotated, simple trials rotated, hard trials - —· 25% chance  
![](images/b93999ab96fd1d27edc2fb23d415b0ea9ebd46bbfade99c77f4727e1fda5ab79.jpg)

![](images/32a70e5f2d6af0aa55a1a7b463f27c38bd088a127ab84339705f776f74202ef8.jpg)  
Figure 2: Left: unrotated against rotated accuracy. An observer working from appearance sits in the lower right; one that cannot identify the scene at all sits in the lower left. Grey points are the models tested but without labels in this figure. Right: the same two quantities against parameter count, on the identical trials, for Qwen2.5-VL at 3B, 7B, 32B and 72B and Qwen3-VL-235B in instruct mode. Grey is unrotated accuracy, which rises with scale while rotated accuracy does not, on hard trials or on simple ones. InternVL3.5 runs the same ladder and lands on top of the Qwen line and is reported in Table 1.

To test the scaling beyond the Qwen family, we ran InternVL3.5 at 1B, 2B, 4B, 8B, 14B and 38B on the same trials with the same prompt and token ceiling. Unrotated accuracy rises from 25% to 65% between 2B and 38B, while rotated accuracy reads 20, 20, 12, 21, 26, 19% across the whole 38× range and never clears the 25% chance level. Its mixture-of-experts member (30B total, 3B active) behaves like its dense neighbours, at 55% and 22%. Across all fourteen open-weight models in the set, spanning 1B to 235B parameters, no rotated accuracy passes 31% (chance at 25%).

The worst viewpoint change across model is 135°. Pooled over the sixteen models, accuracy at 135° is 48/320 = 15.0%, clearly below chance (one-sided binomial $p = 9 \times 1 0 ^ { - 6 } )$ , and recovers to 23% at 180° (Table 2). Taking each model's own worst viewpoint, seven bottom out at 135°, three at 90°, one at 45° and one at 180°, and four tie across two viewpoints. A 180° turn is the largest change and the condition that should be hardest if accuracy simply decayed with angle. It suggests the models are applying an image heuristic, such as matching against a reflection of the original scene, which partially succeeds at a direct reversal but misleads the model at intermediate angles.

Distractor separation separates frontier from open models. Raising the median nearest-distractor distance from 6.8 m to 31.4 m on identical trials takes Gemini 3.8 Flash from 39% to 85% (exact McNemar $p = 1 . 5 \times 1 0 ^ { - 1 0 } )$ and GPT-5.6 Luna from 31% to 55% $( p = 2 \times 1 0 ^ { - 3 } ;$ Figure 2). The same change moves no open model significantly at either wider separation. From 6.8 m to 31.4 m, Qwen2.5-VL at 3B, 7B, 32B and 72B moves by $- 4 , - 6 ,$ +10 and +4 points $( p = 0 . 6 5 ,$ 0.36, 0.13, 0.63), and InternVL3-38B by —3 (p = 0.66). From 6.8 m to 42.8 m the largest shift in the open set is +11 points, for Qwen2.5-VL-72B (p = 0.08).

Making the task easier raises the gate and leaves rotation alone. Varying the number of landmarks gives a second route to the same conclusion, with different scenes rather than different distractors (Figure 3). Qwen2.5-VL-72B goes from identifying 30% of one-landmark scenes to 100% of six-landmark scenes, and its rotated accuracy over the same range moves from 12% to 19%. InternVL3.5-38B behaves the same way, 55% to 75% unrotated against 15% to 21% rotated. GPT-5.6 Luna identifies every scene at every size, while its rotated accuracy rises from 15% to 75%.

<table><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Observer</td><td>Δ=0°</td><td>∆=45°</td><td>∆=90°</td><td> $\Delta { = } 1 3 5 ^ { \circ }$ </td><td> $\Delta = 1 8 0 ^ { \circ }$ </td></tr><tr><td>Human</td><td>100%</td><td>90%</td><td>80%</td><td>85%</td><td>75%</td></tr><tr><td>InternVL3.5-4B</td><td>35%</td><td>20%</td><td>20%</td><td>5%</td><td>5%</td></tr><tr><td>InternVL3.5-2B</td><td>25%</td><td>10%</td><td>25%</td><td>20%</td><td>25%</td></tr><tr><td>InternVL3.5-1B</td><td>35%</td><td>15%</td><td>30%</td><td>25%</td><td>10%</td></tr><tr><td>InternVL3.5-38B</td><td>65%</td><td>15%</td><td>15%</td><td>25%</td><td>20%</td></tr><tr><td>InternVL3.5-8B</td><td>55%</td><td>40%</td><td>20%</td><td>5%</td><td>20%</td></tr><tr><td>InternVL3.5-30B-A3B</td><td>55%</td><td>35%</td><td>15%</td><td>25%</td><td>15%</td></tr><tr><td>Qwen2.5-VL-32B</td><td>75%</td><td>30%</td><td>15%</td><td>0%</td><td>25%</td></tr><tr><td>Qwen2.5-VL-3B</td><td>30% 75%</td><td>50% 30%</td><td>15% 25%</td><td>25%</td><td>25%</td></tr><tr><td>Qwen3-VL-235B (t) InternVL3.5-14B</td><td>50%</td><td>30%</td><td>25%</td><td>10% 20%</td><td>10%</td></tr><tr><td></td><td>75%</td><td>40%</td><td>15%</td><td>10%</td><td>30% 15%</td></tr><tr><td>Qwen2.5-VL-72B Qwen3-VL-235B (i)</td><td>90%</td><td>40%</td><td>15%</td><td>5%</td><td>15%</td></tr><tr><td>InternVL3-38B</td><td>85%</td><td>40%</td><td>5%</td><td>15%</td><td>25%</td></tr><tr><td>Qwen2.5-VL-7B</td><td>55%</td><td>45%</td><td>15%</td><td>25%</td><td>40%</td></tr><tr><td>Gemini 3.8 Flash</td><td>65%</td><td>45%</td><td>50%</td><td>15%</td><td>45%</td></tr><tr><td>GPT-5.6 Luna</td><td>100%</td><td>35%</td><td>35%</td><td>10%</td><td>45%</td></tr><tr><td>All models pooled</td><td>61%</td><td>32%</td><td>21%</td><td>15%</td><td>23%</td></tr></table>

Table 2: Accuracy by viewpoint change. The images above the columns are one c0 scene at each viewpoint change, from the same study view. Pooled over the models the floor is $\Delta = 1 3 5 ^ { \circ }$ , below chance, and accuracy recovers at a half turn. Bold marks each model's own worst viewpoint; rows that tie for worst are left unmarked.

No instruction style recovers the missing operation. Qwen2.5-VL-32B answered the same 80 rotated trials under six instructions, with stimuli and decoding held fixed (see Appendix A.1 for exact prompts). cot, used for every other run in the paper, says the matching option may be viewed from the study direction or from a different one, while cot + view states that it is viewed from a different viewpoint. The other four keep that framing and add an explicit strategy: estimate the rotation angle and mentally rotate the peaks, checking whether their clockwise order is preserved (mental\_rotation); take the most distinctive peak as an origin and trace the bearings and distances of the other three from it (anchor); imagine the layout from directly above and work out where the camera would stand on that map (birdseye); or look for a geometric contradiction in each candidate and eliminate it (elimination). Accuracy is 23.8% for birdseye, 22.5% for mental\_rotation and anchor, 18.8% for elimination, 17.5% for cot and 13.8% for cot + view. None reaches chance (see Figure 3).

Models fail on the same trials, and their errors ignore layout. If the models were failing for unrelated reasons they would fail on unrelated trials, which they do not. Cohen's κ on trial-level correctness, over all 100 trials, averages 0.18 across the 120 model pairs and 0.05 across the 16 human-model pairs (permutation $p = \mathsf { \bar { 5 } } \times 1 0 ^ { - 4 }$ ; Appendix Figure 4), so the models resemble each other more than any of them resembles a human. Where the errors land is also informative. Each wrong answer picks one of three foils, and we rank those by layout distance to the target: nearest, middle, farthest. An observer with a graded sense of layout should confuse a place with its nearest neighbour most often. Pooled over 1092 model errors the split is 30/37/33, against the 33/33/33 expected from choices made without regard to geometry, and no individual model departs from a third (Appendix Figure 4), while a human picks the nearest foil 10 out of 14 times.

## 5 Discussion

Every model we tested identifies a place from the studied viewpoint and then loses it once the camera moves. Perception is not the bottleneck as we show that with six landmarks, Qwen2.5-VL-72B identifies 100% of unrotated scenes and answers 19% of the rotated ones, and InternVL3.5-38B behaves the same way. A model that resolves every object in the scene well enough to recognise it from the studied viewpoint has demonstrably seen what it needs to see. Scene complexity is not the bottleneck either: the landmark-count banks vary the scenes instead of the distractors and produce the same split, with rotated accuracy tracking how far apart the layouts are rather than how much is in them.

![](images/7a024351f9ec5794050fdad00d9c23e58a4cda7dbc0b1bad2b8489c07246ff4e.jpg)

![](images/7dee45073582bb7303ee4c0f30dfca819303183516e5b82943210f7d83ddeff3.jpg)  
Figure 3: Left: Qwen2.5-VL-32B under six instruction styles, rotated trials only, with the instruction used everywhere else highlighted. The styles are, from the top, "imagine a plan view", "pick an anchor landmark", "rotate the scene mentally", "eliminate alternatives", plain chain of thought, and chain of thought with the viewpoint change asserted. None reaches chance. Right: accuracy against the number of landmarks in a scene, one panel per model. The shaded band is the gap between identifying a place and recognising it after a viewpoint change. Adding landmarks raises the unrotated line in all three models and closes the gap in only one. Landmark count and layout separation cannot be varied independently: the median distractor sits at 0.9, 7.6, 23.2 and 28.6 m for one, two, four and six landmarks

The trials are answerable. A human observer, given the same 100 items and the distractors at their closest spacing, answers 82% of the rotated ones. The information needed is present in the images, and enough of it survives the viewpoint change for at least one observer to use.

Scale does not supply the missing operation. Between 3B and 72B, Qwen2.5-VL gains 45 points on the appearance gate and loses 9 on rotated trials, and InternVL3.5 repeats the pattern from 1B to 38B without ever clearing chance. Neither does inference-time computation: the thinking variant of Qwen3-VL-235B reaches the same 19% rotated accuracy as instruct. We also tested instruction tuning using six different styles, including two that hand the model an explicit procedure for rotating the scene or anchoring on a landmark.

What is left is the resolution of the representation itself, and the distractor manipulation measures it. Moving the alternatives from 6.8 m to 31.4 m apart, on the same questions with the same target and the same answer positions, takes Gemini 3.8 Flash from 39% to 85% and GPT-5.6 Luna from 31% to 55%. A model holding no layout information could not gain 46 points from a change that only moves the distractors, and a model holding layout at human resolution would not need 30 m of separation before it could use it. The allocentric map exists but its coarse.

Limitations. The scaling series are restricted to open-weight models, as frontier parameter counts remain undisclosed. In the landmark-count banks, the number of landmarks and the layout separation co-vary, meaning the axis measures layout separation rather than set size in isolation. Finally, as the stimuli are synthetic, exposure to similar rendered environments during pre-training could provide advantages. The human baseline is based on one participant which establishes that the task is solvable. More participants are currently being evaluated.

## References

[1] Chris M. Bird, Dennis Chan, Tom Hartley, Yolande A. Pijnenburg, Martin N. Rossor, and Neil Burgess. Topographical short-term memory differentiates Alzheimer's disease from frontotemporal lobar degeneration. Hippocampus, 20(10):1154–1169, 2010.

[2] Dennis Chan, Laura M. Gallaher, Kuven Moodley, Ludovico Minati, Neil Burgess, and Tom Hartley. The 4 Mountains Test: a short test of spatial memory with high sensitivity for the diagnosis of pre-dementia Alzheimer's disease. Journal of Visualized Experiments, (116): e54454, 2016.

[3] Markus Frey, Christian F Doeller, and Caswell Barry. Probing neural representations of scene perception in a hippocampally dependent task using artificial neural networks. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2113–2121. IEEE, 2023.

[4] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A. Smith, Wei-Chiu Ma, and Ranjay Krishna. BLINK: multimodal large language models can see but not perceive. In European Conference on Computer Vision (ECCV), 2024.

[5] Bo Gu, Zhikang Zhang, Zizhuang Wei, Zhenyuan Chen, Lingyun Li, and Zhuoyi Song. Space-Mind++: toward allocentric cognitive maps for spatially grounded video MLLMs. arXiv preprint arXiv:2605.09449, 2026.

[6] Tom Hartley, Chris M. Bird, Dennis Chan, Lisa Cipolotti, Masud Husain, Faraneh Vargha-Khadem, and Neil Burgess. The hippocampus is required for short-term topographical memory in humans. Hippocampus, 17(1):34–48, 2007.

[7] Dingming Li, Hongxing Li, Zixuan Wang, Yuchen Yan, Hang Zhang, Siqi Chen, Guiyang Hou, Shengpei Jiang, Wenqi Zhang, Yongliang Shen, Weiming Lu, and Yueting Zhuang. ViewSpatial-Bench: evaluating multi-perspective spatial localization in vision-language models. arXiv preprint arXiv:2505.21500, 2025.

[8] John O'Keefe and Lynn Nadel. The Hippocampus as a Cognitive Map. Oxford University Press, 1978.

[9] Shouwei Ruan, Bin Wang, Zhenyu Wu, Qihui Zhu, Yuxiang Zhang, Hang Su, and Yubin Wang. World2Mind: cognition toolkit for allocentric spatial reasoning in foundation models. arXiv preprint arXiv:2603.09774, 2026.

[10] Roger N. Shepard and Jacqueline Metzler. Mental rotation of three-dimensional objects. Science, 171(3972):701–703, 1971.

[11] Edward C. Tolman. Cognitive maps in rats and men. Psychological Review, 55(4):189–208, 1948.

[12] Ruth A. Wood, Kuven K. Moodley, Colin Lever, Ludovico Minati, and Dennis Chan. Allocentric spatial memory testing predicts conversion from mild cognitive impairment to dementia: an initial proof-of-concept study. Frontiers in Neurology, 7:215, 2016.

[13] Jihan Yang, Shusheng Yang, Anjali W. Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: how multimodal large language models see, remember, and recall spaces. arXiv preprint arXiv:2412.14171, 2024.

[14] Hantao Zhang, Jinru Sui, Ed Li, Dirk Bergemann, and Zhuoran Yang. MultiView-Bench: a diagnostic benchmark for world-centric multi-view integration in VLMs. arXiv preprint arXiv:2607.08970, 2026.

[15] Yuyou Zhang, Radu Corcodel, Chiori Hori, Anoop Cherian, and Ding Zhao. SpinBench: perspective and rotation as a lens on spatial reasoning in VLMs. In International Conference on Learning Representations (ICLR), 2026.

<table><tr><td></td><td></td><td></td><td></td><td></td><td><img src="images/fc73f8d719852777d5a9578fb890218cff42b552257fcb1bdf6b07be9b0c63bb.jpg"/></td></tr><tr><td>Observer</td><td>c0</td><td>cl</td><td>c2</td><td>c3</td><td>c4</td></tr><tr><td>Human</td><td>94%</td><td>75%</td><td>81%</td><td>81%</td><td>81%</td></tr><tr><td>InternVL3.5-4B</td><td>19%</td><td>6%</td><td>25%</td><td>0%</td><td>12%</td></tr><tr><td>InternVL3.5-2B</td><td>12%</td><td>12%</td><td>19%</td><td>25%</td><td>31%</td></tr><tr><td>InternVL3.5-1B</td><td>12%</td><td>19%</td><td>25%</td><td>25%</td><td>19%</td></tr><tr><td>InternVL3.5-38B</td><td>19%</td><td>6%</td><td>19%</td><td>19%</td><td>31%</td></tr><tr><td>InternVL3.5-8B</td><td>19%</td><td>6%</td><td>19%</td><td>44%</td><td>19%</td></tr><tr><td>InternVL3.5-30B-A3B</td><td>6%</td><td>19% 0%</td><td>19% 31%</td><td>31% 6%</td><td>38%</td></tr><tr><td>Qwen2.5-VL-32B</td><td>12%</td><td>12%</td><td>31%</td><td>38%</td><td>38%</td></tr><tr><td>Qwen2.5-VL-3B</td><td>38% 12%</td><td>12%</td><td>19%</td><td>19%</td><td>25%</td></tr><tr><td>Qwen3-VL-235B (t) InternVL3.5-14B</td><td>25%</td><td>6%</td><td>31%</td><td>38%</td><td>31% 31%</td></tr><tr><td>Qwen2.5-VL-72B</td><td>19%</td><td>12%</td><td>12%</td><td>31%</td><td>25%</td></tr><tr><td>Qwen3-VL-235B (i)</td><td>12%</td><td>6%</td><td>19%</td><td>25%</td><td>31%</td></tr><tr><td></td><td>25%</td><td>12%</td><td>12%</td><td>38%</td><td>19%</td></tr><tr><td>InternVL3-38B</td><td>50%</td><td>19%</td><td>25%</td><td>19%</td><td>44%</td></tr><tr><td>Qwen2.5-VL-7B</td><td>38%</td><td>38%</td><td>44%</td><td>44%</td><td>31%</td></tr><tr><td>Gemini 3.8 Flash GPT-5.6 Luna</td><td>38%</td><td>19%</td><td>50%</td><td>31%</td><td>19%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>All 16 models pooled</td><td>22%</td><td>13%</td><td>25%</td><td>27%</td><td>28%</td></tr></table>

Table 3: Rotated-trial accuracy by stimulus mode, 16 trials per cell, with one place shown in each mode above its column. c0 shape and colour, c1 shape only, c2 colour only, c3 bare peaks, c4 valley. Rows are ordered by overall accuracy. Bold marks the human row and, for each model, its weakest mode

## A Additional results

Table 3 splits the rotated trials by stimulus mode. Four of the five modes sit within each other's confidence intervals. The exception is c1, shape without colour, where the models answer 33 of 256 trials correctly (12.9%, Wilson [9.3, 17.6]) against 25.5% [22.9, 28.2] pooled over the other four (two-proportion $z = 4 . 3 , p = 2 \times 1 0 ^ { - 5 } )$ . It is the weakest cell for six of the sixteen models outright and joint-weakest for six more. Colour alone (c2) costs nothing by comparison, at 25.0%. It is consistent with colour carrying more identity information than shape in these renders, which would make c1 the mode with the least appearance to work from. It does not bear on the viewpoint result: every mode is far below the human, at every distractor separation.

![](images/456d46dd0a2467adfe2995e894b988f3e4d29c0192d70c02ab016a0e4572a9c0.jpg)  
19 pairs are slightly negative (min -0.11), floored to 0 in colour only

![](images/8c5f2f7e62332ae91934e118307adc83478a5f17fb5bfa485bbc56da65d09902.jpg)

Figure 4: Left: do a pair of observers agree on which trials are solvable? Cohen's κ on trial-level correctness; higher values mean a pair succeeds and fails on the same trials. The human observer is in the first row, with models ordered by mutual similarity. Right: when an observer is wrong, which distractor does it pick? Distractors are ranked by layout distance to the target, and the dashed lines mark thirds, which is what errors independent of layout geometry would give.

## A.1 The instructions, verbatim

Every model saw one of these, followed by the study image and the four options. The sweep in Figure 3 runs the first six on Qwen2.5-VL-32B; every other run in the paper uses cot.

cot chain of thought, viewpoint unstated.

You are taking the Four Mountains Test of spatial allocentric perception.

Image 1 is the STUDY view of a landscape with four mountain peaks.   
The subsequent 4 images are OPTION 1 to OPTION 4.   
Exactly ONE option shows the EXACT SAME mountain landscape (the same four peaks in   
the same relative spatial arrangement). It may be viewed from the same direction   
as the study image or from a different one, and the lighting/weather may differ.   
The other options show different mountain landscapes.

In 2-4 sentences, compare the 3D spatial layout of the peaks (e.g. relative positions such as in front, behind, left, right) between the study view and the options, accounting for any camera rotation. Avoid lengthy itemized lists.

cot + view chain of thought, viewpoint asserted.

You are taking the Four Mountains Test of spatial allocentric perception.

Image 1 is the STUDY view of a landscape with four mountain peaks.

Exactly ONE option shows the EXACT SAME mountain landscape (the same four peaks in the same relative spatial arrangement), simply viewed from a different viewpoint and under different lighting/weather.

In 2-4 sentences, compare the 3D spatial layout of the peaks (e.g. relative positions such as in front, behind, left, right) between the study view and the options, accounting for camera rotation. Avoid lengthy itemized lists.

State your final decision on the last line as:

Final Answer: Option X

mental\_rotation instructed to mentally rotate.

You are taking the Four Mountains Test of allocentric spatial perception.

Image 1 is the STUDY view of a landscape with four distinct mountain peaks.

The subsequent 4 images are OPTION 1 to OPTION 4.

Exactly ONE option shows the EXACT SAME mountain landscape (the same four peaks in the same relative geometric layout), viewed from a shifted camera viewpoint and under different weather/lighting.

The other options are distractors where the relative 3D spatial arrangement of the peaks has been altered.

To determine the matching scene, perform mental rotation:

1. Estimate the camera perspective shift (rotation angle) between the study image and candidate options.

2. Mentally rotate the four peaks to verify if topological handedness (e.g. clockwise/counterclockwise order of peaks, which peak is opposite or between others) matches the study scene.

3. Ignore superficial differences in sunlight, shadows, fog, and seasonal color.

In 2-4 sentences, explain your mental rotation reasoning, then conclude on the last line:

Final Answer: Option X

anchor instructed to pick an anchor landmark.

You are taking the Four Mountains Test of allocentric spatial perception.

Image 1 is the STUDY view of a landscape with four distinct mountain peaks.

The subsequent 4 images are OPTION 1 to OPTION 4.

Exactly ONE option shows the EXACT SAME mountain landscape under a camera viewpoint rotation and weather change.

The remaining options are distractors.

Spatial Strategy:

1. Select the single most prominent or unique landmark peak as an anchor (origin).

2. Trace the relative bearings and distances of the other 3 peaks surrounding this anchor.

3. Identify which option preserves this exact 3D spatial configuration around the anchor under the new camera angle.

In 2-4 concise sentences, explain your reasoning and conclude on the last line: Final Answer: Option X

birdseye instructed to imagine a plan view.

You are taking the Four Mountains Test of allocentric spatial perception.

Image 1 is the STUDY view of a landscape with four mountain peaks.

The subsequent 4 images are OPTION 1 to OPTION 4.

Exactly ONE option shows the EXACT SAME mountain landscape viewed from a different camera viewpoint and lighting.

The other options are distractors with altered 3D mountain configurations.

Top-Down Cognitive Mapping Strategy:

1. Imagine looking down at the four peaks from directly above (a 2D bird's-eye map). Note their relative positions (which forms a triangle, which is isolated, which is tallest).

2. For each candidate option, determine where the camera would be standing on that

same bird's-eye map.

3. Verify which option is geometrically consistent with the study scene's top-down layout under the new camera angle.

In 2-4 sentences, describe the bird's-eye spatial layout and conclude on the last line:

Final Answer: Option X

## elimination instructed to eliminate alternatives.

You are taking the Four Mountains Test of allocentric spatial perception.

Image 1 is the STUDY view of a landscape with four mountain peaks.

The subsequent 4 images are OPTION 1 to OPTION 4.

Exactly ONE option shows the EXACT SAME mountain landscape viewed from a different camera angle and lighting.

The other options are geometric distractors.

Falsification Strategy:

1. Inspect each candidate option one by one to find geometric contradictions with the study scene (e.g. impossible relative peak heights, wrong peak ordering, or missing ridges).

2. Eliminate the distractor options that cannot possibly match the study landscape under any viewpoint rotation.

3. Select the remaining single candidate that has no geometric contradictions.

Briefly eliminate the distractors and conclude on the last line:

Final Answer: Option X

## neutral\_anyview the human arm, and the only wording a person saw.

You will see a STUDY image of a place, then 4 options.

The place contains several landmarks. Exactly ONE option shows the SAME place as the study image. It may be photographed from the same direction as the study image or from a different one, and the lighting may differ.

The other options show different places, each containing the same landmarks arranged differently, photographed from the same direction as the correct option.

Which option shows the same place as the study image?

Answer on the last line as:

Final Answer: Option X