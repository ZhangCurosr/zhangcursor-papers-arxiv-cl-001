# Reading a Legal Question Word by Word: Embedding Trajectories of 2,144 Vietnamese Legal Headlines

Tran Minh Quan

![](images/6b7083f22a47fae5e6443893f3c7b55c3ee5a7cb44fa56c02f9935c919868b2a.jpg)  
Figure 1: Three headlines walking to their answers, one word at a time. Each panel is the plane of the gold article (star), its twenty strongest competitors (grey) and the word‑prefix path of a headline in Nemotron‑3‑Embed‑8B (green, one dot per word; squares mark the second sub‑question). Every word is written where the prefix ending in it is encoded; bold words are the ones needed to bring the gold article to rank 1 (red ring: the insight point), grey words—the interrogative frame, greeting and attribution—move the point but not the ranking. Thin blue lines are the second sub‑question encoded on its own; they land elsewhere. (a) Nguyên tắc kỷ luật học sinh … (Principles for disciplining pupils …): rank 1 after four content words. (b) Biển số xe 38 ở đâu? (Where is licence plate 38 from?): four words, the number does the work. (c) Hệ số lương của biên tập viên … (Salary coeficient of editors …): six words.

## ABSTRACT

A dense retriever encodes a question as one vector, but the question arrives one word at a time. We read every one of 2,144 held‑out headlines from the Vietnamese legal library Thư Viện Pháp Luật (Legal Library) word by word through four decoder embedders —NVIDIA’s Nemotron‑3‑Embed‑8B and Nemotron‑3‑Embed‑1B, Alibaba’s Qwen3‑Embedding‑8B and Qwen3‑Embedding‑0.6B— encoding all 65,444 word prefixes, ranking each against the 20,034‑article corpus, and keeping the full vector of every prefix. We then split the 1,112 multi‑question headlines into their 3,438 sub‑questions, encode every prefix of every sub‑question on its own, and do the same for 168 answers read word by word. The trajectories show one behaviour with a few well‑defined variants. (i) The gold article becomes rank 1 after a median of 6–7 content words in every encoder (quartiles 4–10), before the interrogative frame is read, and stays there to the end of the headline in 78–85% of cases. (ii) In a multi‑question headline the lock happens inside the first sub‑question 94–98% of the time; the second sub‑question leaves the rank unchanged in 89–95% of headlines, and its own words encoded alone reach rank 1 only 42–58% of the time against 91–96% for the first. The word at which the first sub‑question locks is the same whether it is read alone or inside the headline (95– 97% identical). (iii) Numbers, dates and instrument identifiers move the embedding twice as far as ordinary content words and four times as far as interrogative words at equal position; 72–78% of all steps move toward the gold article, and the interrogative frame that closes a question displaces the vector against the direction the content words built in 95–99% of headlines. (iv) Clustering the rank and cosine curves yields six archetypes—instant, typical, unstable, late and never‑locking paths—whose mix difers by legal area, question form and sub‑question count (all $\chi ^ { 2 } \ : \mathrm { p } < 1 0 ^ { - 8 } ) :$ : real‑estate and litigation headlines never lock on a number, environmental and accounting headlines do so a third of the time. (v) Read word by word, an answer retrieves its own article after 8–16 words and addresses the sub‑questions in the order they were asked in 83–89% of cases. (vi) The walk is neither an independent word‑embedding walk nor a wholesale rewrite at every word: the step a word contributes keeps a consistent direction across headlines (cosine 0.25–0.33 between occurrences, 0.44–0.60 for numbers, 0.00 for unrelated words), a preceding question rotates that step by about 60° and a greeting by about 30°, steps shrink as �<sup>−0.8</sup> under mean pooling and last‑token pooling alike, and the vector of a two‑question headline is reproduced to within 12–17° by a linear mix of its two questions’ standalone vectors. We call this a context‑modulated additive walk. Trajectories are presented as a token‑granularity visual idiom whose ground truth is the rank in the full index rather than proximity in a projection; we show galleries of annotated trajectories for every question form and every one of the 27 legal areas (Appendix A).

## 1 INTRODUCTION

Retrieval‑augmented systems treat a query as an atom: the whole string is encoded, one vector is compared with an index, the best passages are returned. Yet a query is a sequence, and a decoder‑based embedder reads it left to right. Somewhere along the sequence the vector crosses from the region of the index where it is nobody’s neighbour into the region where the right passage is its nearest neighbour. Where that crossing happens, whether it happens once, and which words cause it are empirical questions with practical consequences: they decide how much of a question one needs before retrieval can start, how much of it can be dropped, and whether a compound question is better sent whole or in pieces. Vietnamese legal question answering is a good place to ask these questions. The editorial headlines of Thư Viện Pháp Luật (Legal Library) [30, 38] are long (a mean of 32 words), 52% of them contain two or more questions separated by question marks, 19% open with a greeting (Cho tôi hỏi (Let me ask)) and 22% close with an attribution (Câu hỏi của anh Sơn (Hà Giang) (Question from Mr Sơn (Hà Giang))). They therefore contain, in one string, exactly the ingredients whose contribution we want to separate: a topic noun phrase, an interrogative frame, a second question, and words that carry no information at all. Decoder embedders fine‑tuned for retrieval [2, 15, 26, 44] place the gold article at rank 1 for 93–96% of these headlines; the interesting question is not whether they work but how the answer is reached.

Text visualization has moved, over the last decade, from mapping explicit lexical statistics—tag clouds, word trees, term matrices [14]

—to interpreting the latent spaces of neural language models [18]: attention routes [1, 41], layer‑by‑layer hidden‑state geometry [3], prompt perturbation [20], and corpus cartography by non‑linear projection [8]. Almost all of these idioms treat a text as a point (or a sequence of points across layers); the object we study is the path a single text traces as it is read, and its coordinate system is not a projection but the retrieval index itself: the quantity attached to every word is the rank of the correct article among 20,034, which no two‑dimensional layout can distort.

This paper is about that path. Its contribution is a complete word‑by‑word account of one retrieval benchmark in four encoders: the embedding of every prefix of every headline (65,444 vectors per encoder), of every prefix of every sub‑question read alone (61,568), and of stride‑8 prefixes of 168 answers (15,839), with the rank of the gold article, its cosine, the two strongest competitors and the full vector of each prefix retained. From this we derive the insight point of every headline, test whether the point survives when a headline is split into its sub‑questions, measure which word classes move the vector, cluster the trajectories into archetypes, map the archetypes onto the site’s 27 editorial legal areas, and— because we keep every prefix vector—ask what kind of process the walk is: whether each word adds a step of its own or rewrites the vector in the light of everything before it. Every claim is illustrated with annotated trajectories drawn in the plane of the gold article and its competitors; more than seventy such panels appear in the paper, at least one for each legal area and each question form.

## 2 RELATED WORK

Dense retrievers descend from DPR [10] and Sentence‑BERT [32]; the current generation fine‑tunes decoder LLMs with contrastive objectives [2, 15, 22, 35] and is ranked on MTEB [5, 21] and RTEB [17], where the Nemotron‑3 Embed family [26–28] and Qwen3‑Embedding [44] sit near the top. Work on embedding geometry has described anisotropy and rogue dimensions [6, 7, 40], the modality gap [16] and cross‑model similarity [9, 13], but treats each text as a point; the trajectory of a growing prefix has, to our knowledge, not been measured at corpus scale. Vietnamese legal retrieval has been organised around the ALQAC and Zalo challenges [4, 23, 37, 43] and studied with attentive and multi‑stage models [11, 25, 29, 39]; the syllable‑spaced orthography of Vietnamese [24, 42] makes “one word” a well‑defined unit, which we exploit.

Text visualization. The taxonomy of Kucher and Kerren [14] organised the field around surface lexical features; the task‑driven survey of Liu et al. [18] documents its turn toward bidirectional, model‑steering analytics over dense representations. At token granularity the dominant idiom is the attribution heatmap and the attention graph [1, 36, 41]; at document granularity, narrative trajectories through a low‑dimensional state space [31]; at corpus granularity, projected landscapes of sentence embeddings [8]. Hidden‑state trajectories have been drawn across layers for a fixed input [3]; we draw them across words for a fixed model, and replace projected proximity by rank in the index as the primary channel. In the legal domain, visual analytics has concentrated on citation and precedent networks [12, 33], ontology‑grounded norm graphs and semantic substrates that preserve institutional hierarchy [34], as surveyed by Mentzingen et al. [19]; the word‑level behaviour of the neural retrievers that increasingly front such systems has not been visualised, and the present paper is a first such account.

## 3 DATA, MODELS AND THE PREFIX EXPERI-MENT

## 3.1 Headlines, sub‑questions and answers

The corpus is the Q&A section of Thư Viện Pháp Luật (Legal Library) [30]: 20,034 articles, each a headline paired with an editorial answer that cites, quotes and glosses the governing instruments. We use the 2,144 held‑out headlines of the published split as queries and the whole corpus as the index; the gold passage of a headline is its own article. A rule‑based splitter cuts each headline at every question mark, strips the greeting (Cho tôi hỏi, Xin hỏi, Nhờ anh chị giải đáp (Let me ask, May I ask, Please advise)) and the trailing attribution, and merges fragments shorter than three words into their predecessor. This yields 3,438 sub‑questions: 1,032 headlines carry one question, 946 carry two, 150 three and 16 four. We validated the splitter on 60 random headlines. Each sub‑question receives one of nine interrogative forms by rule (definition, yes/ no, amount/time, procedure, document, sanction, authority, what/ content, other); the headline itself keeps the seven‑class form label of the earlier geometry study, used here only as a covariate.

## 3.2 Encoders

We study four decoder embedders released with open weights: Nemotron‑3‑Embed‑8B and Nemotron‑3‑Embed‑1B (NVIDIA, mean pooling, query: / passage: prefixes, 4096‑ and 2048‑d) [26, 27], and Qwen3‑Embedding‑8B and Qwen3‑Embedding‑0.6B (Alibaba, last‑token pooling, instruction prefix, 4096‑ and 1024‑d) [44]. All run in bfloat16 with HuggingFace Transformers on one NVIDIA GB10; passages are truncated at 1,024 tokens. Colours throughout: greens for Nemotron, purples for Qwen.

## 3.3 The prefix experiment

For every headline we form the word prefixes $w _ { 1 } , w _ { 1 } w _ { 2 } , . . . , w _ { 1 } . . . w _ { n }$ (capped at 64 words; 65,444 prefixes), encode each with the query prompt, and rank it against the 20,034 passage vectors. We record the rank of the gold article, its cosine, the two strongest competitors, the cosine to the previous prefix (the step) and to the final prefix, and keep the full fp16 vector. We do the same for every prefix of every sub‑question encoded on its own (61,568 prefixes). For 168 answers stratified by form and by single/ multi‑question status we encode every eighth word prefix up to 800 words (15,839 prefixes) with the passage prompt and record the rank of the article’s own vector, the cosine to the whole headline and to each sub‑question. In total 127,012 question prefixes and 15,839 answer prefixes were encoded per encoder, 571k encodings in all.

We call the first prefix at which the gold article is rank 1 the insight point and report it in content words, i.e. after removing the greeting. A headline holds if it stays at rank 1 from the insight point to its last word; an exit is a return from rank 1 to a lower rank. Trajectories are drawn in the plane of the first two principal components of the gold vector, the twenty passages nearest the final prefix, and the path itself.

a  
![](images/8266a604f202818ba1bf5d96fed2c667433dcf72ec8c0c9bc64301ce49305dd3.jpg)

b  
![](images/c10fd3321c178e60deec367e2cabe7a0dc486deda7e5a21b1b65f328828ba9bc.jpg)  
C

![](images/cfe72b68135b6a5e478f073549cebba29846eb5737e1102583c17c152ea83ce8.jpg)  
Figure 2: The insight point. (a) Cumulative share of headlines whose gold article is rank 1 after � content words, by interrogative form (Nemotron‑3‑Embed‑8B). (b) The same curve for the four encoders. (c) Share of headlines that hold rank 1 from the insight point to the last word, and share that never reach rank 1 (scaled ×5).

## 4 WHERE A QUESTION LOCKS ON ITS ANSWER

Figure  2 shows the distribution of the insight point. Half of all headlines have their gold article at rank 1 after six content words in Nemotron‑3‑Embed‑8B and seven in the other three encoders; the quartiles are 4–9 words for Nemotron‑3‑Embed‑8B, 5–11 for Nemotron‑3‑Embed‑1B, 4–10 for Qwen3‑Embedding‑8B and 5–10 for Qwen3‑Embedding‑0.6B, and the 90th percentile lies at 13–16 words. Only 2.0% (Nemotron‑3‑Embed‑8B), 3.4% (Qwen3‑Embedding‑8B), 3.5% (Qwen3‑Embedding‑0.6B) and 4.3% (Nemotron‑3‑Embed‑1B) of headlines never reach rank 1 at any prefix. The forms difer less than one might expect: what/ content headlines lock after a median of five content words in Nemotron‑3‑Embed‑8B and six elsewhere, amount/time after 6– 7, and the other five forms after 7–9; the interrogative frame that gives the form its name arrives after the lock and does not move it. Once locked, most headlines stay locked: 84.5% of Nemotron‑3‑Embed‑8B paths hold rank 1 to the last word (81.7% for Qwen3‑Embedding‑0.6B, 79.2% for Qwen3‑Embedding‑8B, 78.4% for Nemotron‑3‑Embed‑1B), with 0.16–0.22 exits per headline on average, and 92.5–96.3% of headlines finish at rank 1. Sanction headlines are the most stable (91% hold, 0.08 exits in Nemotron‑3‑Embed‑8B), non‑legal utility headlines the least (81%, 0.20).

The greeting contributes nothing. In the 406 headlines that open with Cho tôi hỏi (Let me ask) or a variant, the vector after the greeting alone has median rank 1,792 for the gold article and a cosine of only 0.14 to the headline’s final vector; the path starts over at the first content word (grey dots at the start of the paths in Figure 14 e and Figure 15 b, Appendix A).

## 5 SPLITTING THE HEADLINE: DOES THE IN-SIGHT SURVIVE?

If the first six words decide the retrieval, the second question in a compound headline should be nearly inert, and reading the first question alone should reproduce the lock. Both predictions hold (Figure 3). The insight point lies inside the first sub‑question for 96.6–97.9% of multi‑question headlines in Nemotron‑3‑Embed‑8B and 94–97% in the other encoders. Between the end of the first sub‑question and the end of the headline the gold rank does not change for 95.1% of Nemotron‑3‑Embed‑8B headlines (92.5% Qwen3‑Embedding‑0.6B, 91.0% Qwen3‑Embedding‑8B, 89.1% Nemotron‑3‑Embed‑1B); the second question helps in 3.3–8.3% and hurts in 1.6–3.9%. Encoded alone, first sub‑questions reach rank 1 for 95.9% (Nemotron‑3‑Embed‑8B), 94.1% (Qwen3‑Embedding‑0.6B), 93.3% (Qwen3‑Embedding‑8B) and 90.5% (Nemotron‑3‑Embed‑1B) of headlines, second sub‑questions for 57.9%, 41.7%, 46.1% and 50.7%, third for 25–40%. Among two‑question headlines in Nemotron‑3‑Embed‑8B, both halves retrieve the article alone in 54.9%, only the first in 41.5%, only the second in 1.8% and neither in 1.8%; the Qwen models tilt further toward “first only” (50.6% and 55.1%). A failing second sub‑question is not far of—its median rank is 4 (Nemotron‑3‑Embed‑8B), 5 (Nemotron‑3‑Embed‑1B), 7 (Qwen3‑Embedding‑8B) and 15 (Qwen3‑Embedding‑0.6B) and 67–83% of them are within the top ten—but it is the first question that names the topic and the second that presupposes it. The failure rate of second sub‑questions is flat across their forms (50–66% rank 1 in Nemotron‑3‑Embed‑8B), so it is position, not form, that matters.

Context does not move the insight word. For the 2,084 first sub‑questions with a lock in both readings, the content word at which the sub‑question locks alone is identical to the one at which the full headline locks in 95.6% of cases (Nemotron‑3‑Embed‑8B; 96.6% Nemotron‑3‑Embed‑1B, 95.2% Qwen3‑Embedding‑8B, 94.5% Qwen3‑Embedding‑0.6B) and within one word in 98–99%. The prefix vector is a function of the prefix, not of what follows, and the greeting that precedes the first content word is, as shown above, forgotten by the second one. Figure 4 shows nine compound headlines with the isolated sub‑question paths overlaid: the first sub‑question’s path (thin green) retraces the headline’s path; the second sub‑question’s path (thin blue) starts from a diferent corner of the plane and, more often than not, ends in the competitor cloud.

## 6 WHAT MOVES THE VECTOR

Every word shifts the vector, but not equally (Figure 5). Controlling for position (words 5 onward), a word containing a digit—a date, an amount, an instrument number such as 29/2024/TT‑BCT—moves the Nemotron‑3‑Embed‑8B vector by 0.465 on the unit sphere, an ordinary content word by 0.245, an interrogative or function word (như thế nào, gì, có, không (how, what, yes, no)) by 0.115 and an attribution word by 0.113; the ratios are the same in the other encoders (0.394/0.219/0.109/0.096 for Qwen3‑Embedding‑8B). The words with the largest mean steps in the corpus are dates (1/7/2026: 0.77; 01/01/2026: 0.62) and price‑of‑petrol vocabulary (Petrolimex, xăng); the smallest are mong, thế, như and câu (0.06–0.08), the words of the frame and the attribution. Steps shrink along the headline for all classes (Figure 5 b): the first word moves the vector by 0.9, the tenth by 0.35, the thirtieth by 0.15, so the space becomes progressively harder to leave.

b  
![](images/a9d0b05ceb997b8425b6924ac10d1617e59368eed1f17b4a984f344dace4634a.jpg)

![](images/6fdca39901b75d9ffac43d514b2d8221acc3753bc0ac87215da470624dc9cd35.jpg)

![](images/e4332821a3ef11dde909e0c677c3ec1b72aba532b1e19cb12ce2e601fb40b1bf.jpg)

d  
![](images/38511a9085e232782166217e49576f6174b94dde8cd3cc64508f6f6c25482327.jpg)  
Figure 3: Sub‑questions alone and inside the headline. (a) Recall@1 of each sub‑question encoded alone, by its position in the headline. (b) For two‑question headlines: whether the first, the second, both or neither sub‑question alone reaches rank 1. (c) Change of the gold rank between the end of the first sub‑question and the end of the headline. (d) Diference between the insight word of the first sub‑question read alone and inside the headline.

Three vector‑level facts hold in every encoder. First, 72–78% of all steps have a positive component along the direction to the gold article (77.9% Nemotron‑3‑Embed‑8B, 73.5% Nemotron‑3‑Embed‑1B, 74.8% Qwen3‑Embedding‑8B, 72.1% Qwen3‑Embedding‑0.6B): the path is a noisy but directed walk. Second, the paths are tortuous— their length is 5.3–6.0 times the net displacement from first to last prefix—but Nemotron‑3‑Embed‑8B turns least, with 1.3 direction reversals per headline against 3.0 (Qwen3‑Embedding‑0.6B), 3.6 (Nemotron‑3‑Embed‑1B) and 4.5 (Qwen3‑Embedding‑8B). Third, the interrogative frame makes a U‑turn: the displacement accumulated after the insight point up to the end of the first sub‑question has a negative cosine with the displacement from the first content word to the insight point in 98.7% of Nemotron‑3‑Embed‑8B headlines (mean −0.23; 95.2% for Nemotron‑3‑Embed‑1B, 97.6% for Qwen3‑Embedding‑8B, 97.1% for Qwen3‑Embedding‑0.6B, means −0.19 to −0.20). The frame words pull the vector partly back toward where the question started, which is why grey words in the galleries so often double back along the green path.

## 7 ADDITIVE OR AUTOREGRESSIVE? WHAT KIND OF WALK THIS IS

The galleries invite a question that the rank curves cannot answer: is the path an independent walk, in which every word adds a step of its own that is the same wherever the word occurs, or an autoregressive process, in which each new word rewrites the vector in the light of everything before it? The two encoder families make the question sharp. Both are causal decoders, so a token’s hidden state depends on the tokens before it and on nothing after. Nemotron pools by mean, so before normalisation the prefix vector is literally a running sum of the token states and the step of word � is that word’s own contextual state divided by �: additive by construction, with all the context dependence hidden inside the token state. Qwen pools by last token, so the whole vector is rewritten at every word and nothing forces adjacent prefixes to be related at all. Four tests on the stored prefix vectors (Figure 6) give the same answer for both families.

A word’s step keeps part of its direction, not all of it (Figure  6 a). For the 399 words that occur at least thirty times at position five or later (43,937 occurrences in Nemotron‑3‑Embed‑8B), the cosine between the steps of the same word in two diferent headlines averages 0.25 (Nemotron‑3‑Embed‑8B), 0.32 (Nemotron‑3‑Embed‑1B), 0.27 (Qwen3‑Embedding‑8B) and 0.33 (Qwen3‑Embedding‑0.6B); between steps of diferent words it is 0.00. So a word does carry a direction of its own—an angle of 71–76° between two occurrences, against 90° for chance—but that direction accounts for less than a tenth of the step’s variance. The share is a property of the word class: interrogative and function words keep almost nothing (0.13–0.20; là, và, nào, của (is, and, which, of) are at 0.04–0.05), content words keep a quarter to a third, and numbers, dates and instrument identifiers keep half to two thirds (0.44–0.60; the date 1/7/2025 reaches 0.75). The word ranking is shared across families (Spearman 0.87 between Nemotron‑3‑Embed‑8B and Qwen3‑Embedding‑8B): the same words are context‑free in every encoder. A preceding question rotates every step by about 60°; a greeting by about 30° (Figure 6 b, Figure 7). The second sub‑question of a compound headline is encoded twice: alone, and inside the headline after the first question. The words are identical; only the left con text difers. Step by step, the two paths agree with a mean cosine of 0.50 (Nemotron‑3‑Embed‑8B), 0.62 (Nemotron‑3‑Embed‑1B), 0.45 (Qwen3‑Embedding‑8B) and 0.54 (Qwen3‑Embedding‑0.6B)— a rotation of 52–64° per step—and the in‑context steps are a third as long (median ratio 0.33–0.38) because they arrive later in the sequence. The control is the first sub‑question, whose only extra context inside the headline is a greeting: there the cosine is 0.85– 0.92 (24–31°), and when there is no greeting the two texts are identical and the cosine is 1.00 in Nemotron and 0.97–0.99 in Qwen, the latter being the bfloat16 batch‑composition noise floor of last‑token pooling. Rotation grows with the length of the context that precedes: 0.59 when the first question is at most six words, 0.47 when it exceeds fifteen (Nemotron‑3‑Embed‑8B). It is smallest for yes/no second questions (0.54) and largest for definitions (0.43), whose là gì frame is the most context‑bound. The shape of the whole path changes accordingly: after translation and scaling, the distance between the alone and in‑context paths is 0.90–1.03 for second questions against 0.40–0.50 for the greeting control (0 identical, 1.41 unrelated). Figure 7 shows three such pairs.

b  
![](images/24ecd7164d64f6e24e672a753996722704c2b140283309318dbe81299d7eaa94.jpg)

![](images/2dc4922793ebc358d728f5026c0db077dcb28306400e148ccde54ab4afae2f25.jpg)

![](images/89beba3715141763c79ad02c1e3248fa4933b515db4833ada0b04ae6099aa431.jpg)

![](images/72b3a8f77e1244330aa95d9d04594917103b7fff0fc2df9564f1ede64b437db7.jpg)

![](images/e9d27d6bd713f665cd23f91edf23f684a15d5624fceec28ce205e2ceb1718439.jpg)

![](images/458e32168c4a958b390321d93cfd5940be35fb829559238b519b53eac97fce13.jpg)

![](images/93f2ab3b6b87e57fbd337873a532dd12c769e0fa7fb1ef7a09915bfc0d334b73.jpg)

![](images/f192c612d1294991c9959fde352f2df15618968c2b3107362f4947c4ba1cf396.jpg)

![](images/13c64a69956dbee67c2c6dac5f770f2d800d2a99171de238bd827121af488e3f.jpg)  
Figure 4: Nine multi‑question headlines with their sub‑questions read alone (Nemotron‑3‑Embed‑8B). Circles: first sub‑question inside the headline; squares: second sub‑question inside the headline; thin green and blue lines: the first and second sub‑questions encoded on their own, ending at diamond. The second question alone lands away from the gold article in most panels.

a  
![](images/b516719d9370a794a2827132ae68c58e9360d11a7db5c118600b05f548a29aae.jpg)

![](images/c7fdfe87d4de94bbab2a3415dd5a3e0f6a9d9c4b338e32d458c3ba379810bb71.jpg)  
Figure 5: Word‑level steps. (a) Mean step length $\| e ( w _ { 1 } . . w _ { i } ) - e ( w _ { 1 } . . w _ { i - 1 } ) \|$ ‖ by class of the word $w _ { i } ,$ for words at position 5 or later so that the early‑position efect is removed. (b) Mean step by position in the headline for three word classes (Nemotron‑3‑Embed‑8B).

a  
![](images/360dc03f274d8b1edb38432bd645b9bb1ff526d87a094227fa5106b3da469ab1.jpg)

b  
![](images/8f525f9409285c8a852f146d09eae696e14e4a2f86af621d565cdca08fb4fc01.jpg)

![](images/6a3a672ba71ef5ad51bd61796945cd69e0a6cafd4ae19f82a1ec6e496c4dd4d3.jpg)  
d

![](images/47b6ed1e652b70eb69225a00e7994fcbc6fc0e79ecbfaec411bf45dd08691954.jpg)  
Figure 6: Four tests of what kind of process the walk is. (a) Cosine between the step vectors $e ( w _ { 1 } . . w _ { i } ) - e ( w _ { 1 } . . w _ { i - 1 } )$ of the same word � in diferent headlines (399 words with ≥30 occurrences at position ≥5; one dot per word, bars: medians); the dashed line is the cosine between steps of diferent words. (b) For every sub‑question, the mean cosine between its steps read alone and the same steps inside the headline: solid, second sub‑questions (a first question precedes; �=1,262); dashed, first sub‑questions preceded only by a greeting (�=406). (c) Mean step length of content word against word position, log–log, with the fitted slope and a 1/� reference. (d) For 922 two‑question headlines, cosine between the full‑headline vector and the first question’s vector (dashed) or the best linear mix of the first question’s vector and the second question’s standalone vector (solid).

Steps shrink as a power law whatever the pooling (Figure 6 c). Mean pooling predicts that the step of word � scales as 1/�; last‑token pooling predicts nothing. Measured on content words, the mean step falls from 0.73–0.89 at the second word to 0.28–0.35 at the tenth and 0.09–0.13 at the thirtieth, with log–log slopes of −0.81 (Nemotron‑3‑Embed‑8B), −0.72 (Nemotron‑3‑Embed‑1B), −0.83 (Qwen3‑Embedding‑8B) and −0.85 (Qwen3‑Embedding‑0.6B). The two Qwen models, which have no averaging in their read‑out, shrink at least as fast as the two Nemotron models that do. The shrinkage is therefore not an artefact of pooling; it is a property of the contextual token states themselves, whose sensitivity to one more word decays as the context lengthens. The exponent being shallower than −1 says that later words are, per token, slightly more influential than a plain average would make them.

The whole is nearly a mix of the parts (Figure  6 d). For the 922 two‑question headlines whose full text fits the 64‑word cap, we ask whether the headline vector can be written as � ⋅ �(question ) + $\beta \cdot e ( \mathrm { q u e s t i o n _ { 2 } }$ alone). The first question’s vector alone has cosine 0.84–0.92 with the headline (24–33°); the best two‑term mix reaches 0.96–0.98 (12–17°), with coeficients of about 0.65 on the first question and 0.40–0.50 on the second in every encoder. A vector for a compound question can thus be assembled, to within a small angle, from the vectors of its questions encoded separately—even in the Qwen models, whose read‑out contains no sum.

Taken together: the walk is a context‑modulated additive walk. Each word contributes a step whose direction is anchored to the word (strongly for numbers and topical nouns, weakly for the frame), rotated by the words that precede it, and shortened by their number; consecutive steps are uncorrelated in Nemotron‑3‑Embed‑8B (mean cosine 0.00) and mildly anti‑correlated in the Qwen models (−0.05 to −0.09; 63–68% of turns exceed 90°), which is the zigzag that makes their paths more tortuous. The result is the same in the family whose architecture makes additivity exact and in the family whose architecture makes no such promise, so the additivity is learned, not built in.

## 8 SIX ARCHETYPES, AND WHERE THEY LIVE

Clustering the resampled rank, cosine and step curves of the 2,144 Nemotron‑3‑Embed‑8B trajectories gives six archetypes (Figure 8). Instant lock (636 headlines, 30%): the first content word already places the gold article near rank 300 and the lock comes at content word 4; 91% of these are multi‑question headlines whose first word is the topic noun. Two typical clusters (649 and 427 headlines) lock at 7–8 content words from a start near rank 3,000–5,000 and hold in 87–88% of cases; the smaller one is almost entirely single‑question (94%) and greeting‑rich (29%). Unstable lock (288, 13%): the lock comes at word 6 but holds only 67% of the time, with 0.35 exits per headline and 12% of paths finishing below rank 1—the paths that wander back into the competitor cloud when the frame or a second question is read. Late lock (122, 6%): the first word ranks the gold near 8,600 and rank 1 arrives only after 13 content words, two‑thirds of the way through the content span; these are single questions whose topic is generic until a qualifier arrives. Never (22, 1%): 82% never reach rank 1.

The archetype mix is not uniform. It depends on the number of sub‑questions $( \chi ^ { 2 } \mathrm { ~ p ~ } \approx 1 0 ^ { - 2 0 4 } )$ , on the question form $\left( \mathrm { p } \ : = \ : 8 \times 1 0 ^ { - 9 } \right)$ and on the legal area $( \mathrm { p } = 1 . 3 { \times } 1 0 ^ { - 1 0 } )$ . Transport (Giao thông – Vận tải (transport), 47% instant), culture and society (44%) and public administration (40%) are dominated by instant locks; securities (12%), import–export (15%) and litigation procedure (15%) are not. Figure 9 maps four statistics over the 27 areas. The median insight point varies only between 5 and 8 content words, and the hold rate between 75% (Xây dựng – Đô thị (construction and urban planning)) and 95% (Tài chính nhà nước (public finance)), but the share of headlines whose insight word is a number difers by an order of magnitude: 0% in real estate and litigation procedure,

![](images/7f50fd5dbaa45f888b9d643eb1c2e8f5ee3af6bef07f159a5567127b97fce1d1.jpg)

![](images/c7c63271151818cf16e8b8d47323f0859dd4ef758ac9adc8e7dcf19e008fae37.jpg)

![](images/c54e6c9bcbdb276acf7a54a057821c3cf443c48c71d69b7372b23843f361ca0f.jpg)  
Figure 7: The same words, two contexts (Nemotron‑3‑Embed‑8B). Three second sub‑questions read alone (blue) and inside their headline after the first question (green), drawn from a common origin in the plane of the two paths after scaling each to unit length; every word is written at its own step. The title gives the first question that precedes and the two statistics of Figure 6 b. The words are identical; the shape is not.

![](images/586f412e599338efd1935b4d244ce3927468da4ff08e3f02287c50a95fcdd9e7.jpg)

2–3% in intellectual property, administrative violations and legal services, against 21% in criminal liability, 29% in accounting and audit and 32% in natural resources and environment (Tài nguyên

– Môi trường (natural resources and environment)). In the number‑locking areas the discriminating information is an instrument or a date (Thông tư 29/2024/TT‑BCT, Nghị định 70 (Circular 29/2024, Decree 70)); in real estate it is the noun phrase. Number locks are

![](images/f98e64b25bd7d6aef593d8b37373b44823bc3e0a5d49da8d670a9d240c1d03df.jpg)

![](images/03f0bd751f23b12762b28c9dd3c2980564c91bba4343a8bf1f4779b8b59e52de.jpg)

![](images/19fadf5b690aa87b046de9a35c3c428df71e1f7c32c7112940cbb3e610e7187c.jpg)

![](images/cf85ba77a0bcc2785499bfd1e6e46dde7e2abbf5dccfa4d6632806227c989527.jpg)

![](images/c6b7e1ee878e60591730ef0e802e350cb462d9b362b8cda9211c27d661ee2035.jpg)

![](images/b2730b24c0983bee9c22df88a1f46455e53ff181f7d822a45543fd726cec698f.jpg)

![](images/3f09458dc7f6f8fc309b90690d9756f0be4e92d30b5ec3a59cea3c432826a58e.jpg)  
Figure 8: Trajectory archetypes (Nemotron‑3‑Embed‑8B). Each headline’s rank curve, cosine‑to‑gold curve and step curve over its content span were resampled to 20 points and clustered (k‑means, k = 6). (a) Median log‑rank curve of each cluster with interquartile band; (b) median cosine to the gold article. (c–h) The medoid headline of each cluster, word‑annotated.

![](images/83a0a78a55a9125b27a4da8fb23bcab2cdd0b66d71e3d062d035758d07795ff0.jpg)  
Figure 9: Trajectory statistics by editorial legal area (Nemotron‑3‑Embed‑8B, 27 areas ordered by size). (a) Median insight point in content words; (b) share of headlines holding rank 1 to the end; (c) share whose insight word contains a digit (a date, amount or instrument number); (d) median tortuosity.

as stable as word locks (85% hold in both groups) but arrive one word earlier (median 5 against 6).

## 9 THE ANSWER READ WORD BY WORD

Read the same way, an answer behaves like its question (Figure  10). Used as a query, its own first 8 words (median; 16 for Nemotron‑3‑Embed‑1B) already retrieve the article at rank 1—the restated heading is the whole signal. Its similarity to the headline peaks where the form predicts: at 14–27% of the encoded text for document answers, whose first lines name the template, and at 63– 75% for yes/no and 51–71% for sanction answers, whose conclusion (Như vậy (Thus)) states the verdict or the fine. Compound answers address the questions in the order they were asked: the prefix most similar to the first sub‑question precedes the one most similar to the second in 89.3% of multi‑question answers for both Nemotron models, 85.7% for Qwen3‑Embedding‑8B and 83.3% for Qwen3‑Embedding‑0.6B, with a median gap of 35–54% of the answer. The encoders difer in how they converge: the Qwen prefix reaches 90% cosine to the full‑answer vector after 112 (Qwen3‑Embedding‑8B) and 48 (Qwen3‑Embedding‑0.6B) words, the Nemotron prefix after 232 and 304, because last‑token pooling settles once the topic is stated while mean pooling keeps averaging in every new word; the Qwen answer paths are correspondingly rougher (tortuosity 10.1 for Qwen3‑Embedding‑8B against 5.5–6.6).

a  
![](images/702124c4eddb4a4a6658173f66f9f551acdf953b88471c02268f12f167c48a9c.jpg)

b  
![](images/db7c191ecfd7310dcfa221a220e6c7e305766f3702942c38328a37d9d7322527.jpg)  
Qwen converges at once, Nemotron gradually

![](images/eb2808c058f56279333c60c97f34026c567bcca497e80f70ecd0ff87e4dd11eb.jpg)  
Figure 10: Answer walks (168 answers, stride 8 words, up to 800 words). (a) Position, as a share of the encoded answer, of the prefix most similar to the headline, by form (Nemotron‑3‑Embed‑8B; bars: medians). (b) For multi‑question answers, the position of peak similarity to the first sub‑question against the second; points above the diagonal address the questions in order. (c) Words read until the prefix reaches 90% cosine to the full answer vector.

## 10 DO THE ENCODERS AGREE?

The distributions agree; the individual insight words agree only partly. Where both encoders find a lock, Nemotron‑3‑Embed‑8B and Qwen3‑Embedding‑8B lock on the same word for 42.9% of headlines and within one word for 65.6%; Nemotron‑3‑Embed‑8B and Nemotron‑3‑Embed‑1B for 39.7% and 63.7%; Qwen3‑Embedding‑0.6B and Qwen3‑Embedding‑8B for 42.5% and 66.5%; the cross‑family small‑model pairs for 34% and 58%. All four lock on the same word in 16% of headlines and within one word in 37%. The insight point is thus a property of the headline to within a word or two, not to the word; the diferences are in how many qualifiers an encoder needs before the topic noun phrase separates the article from its siblings, and Nemotron‑3‑Embed‑8B needs the fewest.

## 11 IMPLICATIONS

Retrieval can start early. Half of all headlines are resolved after six content words and 90% after 13–16; a streaming or speculative retriever can issue its first index probe as soon as the topic noun phrase is complete and treat the interrogative frame as confirmation, not information. Truncate from the end, and drop the greeting. The greeting is forgotten by the second content word; the frame moves the vector against the direction it came from; the attribution moves it hardly at all. A query budget should spend its tokens on the first sub‑question. Split compound questions for the second question, not the first. The first sub‑question alone reproduces the headline’s retrieval in 91–96% of cases; the second alone reaches rank 1 in fewer than 60% because it presupposes the first. A RAG system that answers a compound question should retrieve for the first sub‑question and re‑query for the second with the first’s topic prepended. Numbers are heavy. A date or instrument number moves the vector twice as far as any other word and, in a third of environmental and accounting headlines, is the word that decides retrieval; systems that normalise or strip numerals before embedding remove the lock. Answers are ordered. Because an answer addresses sub‑questions in order, chunking by sub‑question boundary is sound, and the chunk that answers the first question is usually in the first half. Prefix vectors compose. Because the walk is additive to within a small angle, the vector of a compound query can be approximated from cached vectors of its parts (0.96–0.98 cosine), and because both families are causal decoders, extending a query by one word reuses the key–value cache of the prefix: a streaming retriever that re‑probes the index after every word pays one token of compute per word, not one query. Because the steps are context‑modulated, however, a cache keyed on words rather than prefixes would not work—the step of mẫu (template) after one question is not its step after another.

## 12 LIMITATIONS

Our gold label is the headline→article pair of one publisher, so rank measures self‑retrieval of an editorial headline, not general question answering. Word boundaries are syllable spaces, so a Vietnamese compound is read as two or three steps. The splitter and the sub‑question forms are rule‑based; the interrogative forms were validated on 60 headlines and are covariates, not labels. Answer walks use a stride of eight words and 168 answers. The 2‑D planes are per‑panel PCA projections chosen to show the gold, its competitors and the path; distances between panels are not comparable, and a star that appears far from the path’s end is a projection artefact when the final rank is 1. Prefixes were encoded with the query prompt; a passage prompt would trace a diferent path. The additivity tests compare vectors of prefixes and of sub‑questions encoded separately; they measure how far the whole departs from a linear mix of its parts, not the mechanism inside the network that produces the departure.

## 13 CONCLUSION

Reading 2,144 legal headlines one word at a time through four decoder embedders shows a single behaviour with a few variants: the vector walks, mostly toward the answer, and locks on it after the topic noun phrase—six or seven content words—before the question has been asked. The frame turns the vector back a little but does not unlock it; a second question is read but rarely matters; a first question read alone locks on the same word it locks on inside the headline. Numbers and instrument identifiers are the heaviest words and, in some areas of law, the deciding ones. The answers, read the same way, retrieve themselves after a dozen words and address the questions in order. The walk itself is a context‑modulated additive process: a word’s step keeps a direction of its own, is rotated by about 60° by a preceding question, shrinks as �<sup>−0.8</sup> whatever the pooling, and sums, to within 12–17°, to the vector of the whole—in the family whose read‑out is a sum and in the family whose read‑out is not. The full prefix vectors, ranks and tables for all 127,012 question prefixes and 15,839 answer prefixes in the four encoders accompany the paper.

## BIBLIOGRAPHY

[1] Samira Abnar and Willem Zuidema. 2020. Quantifying attention flow in transformers. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 2020. 4190–4197. https://doi.org/10.18653/v1/2020.acl-main.385

[2] Parishad BehnamGhader, Vaibhav Adlakha, Marius Mosbach, Dzmitry Bahdanau, Nicolas Chapados, and Siva Reddy. 2024. LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders. Retrieved from https://arxiv.org/abs/2404. 05961

[3] Andy Coenen, Emily Reif, Ann Yuan, Been Kim, Adam Pearce, Fernanda Viégas, and Martin Wattenberg. 2019. Visualizing and measuring the geometry of BERT. In Advances in Neural Information Processing Systems 32, 2019. Retrieved from https://arxiv.org/abs/1906.02715

[4] Dinh-Truong Do, Son T. Luu, Trang Pham, Trung Vo, Nguyen-Hoang Chu, Quang-Huy Chu, Cuong Nguyen, Minh Nguyen, An Trieu, Dat Nguyen, Thanh Tran, Cong Nguyen, Hiep Nguyen, Chau Nguyen, Nguyen-Khang Le, Dieu-Hien

Nguyen, Binh Dang, Phuong Nguyen, Ha-Thanh Nguyen, Vu Tran, and Le-Minh Nguyen. 2024. A Summary of the ALQAC 2024 Competition. In 2024 16th International Conference on Knowledge and Systems Engineering (KSE), 2024. IEEE. https:// doi.org/10.1109/kse63888.2024.11063484

[5] Kenneth Enevoldsen, Isaac Chung, Imene Kerboua, Márton Kardos, Ashwin Mathur, David Stap, Jay Gala, Wissam Siblini, Dominik Krzemiński, Genta Indra Winata, Saba Sturua, Saiteja Utpala, Mathieu Ciancone, Marion Schaefer, Gabriel Sequeira, Diganta Misra, Shreeya Dhakal, Jonathan Rystrøm, Roman Solomatin, Ömer Çağatan, Akash Kundu, Martin Bernstorf, Shitao Xiao, Akshita Sukhlecha, Bhavish Pahwa, Rafał Poświata, Kranthi Kiran GV, Shawon Ashraf, Daniel Auras, Björn Plüster, Jan Philipp Harries, Loïc Magne, Isabelle Mohr, Mariya Hendriksen, Dawei Zhu, Hippolyte Gisserot-Boukhlef, Tom Aarsen, Jan Kostkan, Konrad Wojtasik, Taemin Lee, Marek Šuppa, Crystina Zhang, Roberta Rocca, Mohammed Hamdy, Andrianos Michail, John Yang, Manuel Faysse, Aleksei Vatolin, Nandan Thakur, Manan Dey, Dipam Vasani, Pranjal Chitale, Simone Tedeschi, Nguyen Tai, Artem Snegirev, Michael Günther, Mengzhou Xia, Weijia Shi, Xing Han Lù, Jordan Clive, Gayatri Krishnakumar, Anna Maksimova, Silvan Wehrli, Maria Tikhonova, Henil Panchal, Aleksandr Abramov, Malte Ostendorf, Zheng Liu, Simon Clematide, Lester James Miranda, Alena Fenogenova, Guangyu Song, Ruqiya Bin Safi, Wen-Ding Li, Alessia Borghini, Federico Cassano, Hongjin Su, Jimmy Lin, Howard Yen, Lasse Hansen, Sara Hooker, Chenghao Xiao, Vaibhav Adlakha, Orion Weller, Siva Reddy, and Niklas Muennighof. 2025. MMTEB: Massive Multilingual Text Embedding Benchmark. Retrieved from https://arxiv.org/abs/2502.13595

[6] Kawin Ethayarajh. 2019. How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 2019. Association for Computational Linguistics. https://doi.org/10.18653/v1/d19- 1006

[7] Jun Gao, Di He, Xu Tan, Tao Qin, Liwei Wang, and Tie-Yan Liu. 2019. Representation Degeneration Problem in Training Natural Language Generation Models. Retrieved from https://arxiv.org/abs/1907.12009

[8] Maarten Grootendorst. 2022. BERTopic: Neural topic modeling with a class-based TF-IDF procedure. Retrieved from https://arxiv.org/abs/2203.05794

[9] Minyoung Huh, Brian Cheung, Tongzhou Wang, and Phillip Isola. 2024. The Platonic Representation Hypothesis. Retrieved from https://arxiv.org/abs/2405.07987

[10] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-Domain Question Answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020. Association for Computational Linguistics. https://doi.org/ 10.18653/v1/2020.emnlp-main.550

[11] Phi Manh Kien, Ha-Thanh Nguyen, Ngo Xuan Bach, Vu Tran, Minh Le Nguyen, and Tu Minh Phuong. 2020. Answering Legal Questions by Learning Neural Attentive Text Representation. In Proceedings of the 28th International Conference

on Computational Linguistics (COLING), 2020. International Committee on Computational Linguistics. https://doi.org/10. 18653/v1/2020.coling-main.86

[12] Marios Koniaris, Ioannis Anagnostopoulos, and Yannis Vassiliou. 2018. Network analysis in the legal domain: a complex model for European Union legal sources. Journal of Complex Networks 6, 2 (2018), 243–268. https://doi.org/10.1093/ comnet/cnx029

[13] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geofrey Hinton. 2019. Similarity of Neural Network Representations Revisited. Retrieved from https://arxiv.org/abs/ 1905.00414

[14] Kostiantyn Kucher and Andreas Kerren. 2015. Text visualization techniques: Taxonomy, visual survey, and community insights. In 2015 IEEE Pacific Visualization Symposium (PacificVis), 2015. 117–121. https://doi.org/10.1109/PACIFICVIS. 2015.7156366

[15] Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. 2024. NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models. Retrieved from https://arxiv.org/ abs/2405.17428

[16] Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, and James Zou. 2022. Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning. Retrieved from https://arxiv.org/abs/2203.02053

[17] Frank Liu, Kenneth C. Enevoldsen, Roman Solomatin, Isaac Chung, Tom Aarsen, and Zoltán Fődi. 2025. Introducing RTEB: A New Standard for Retrieval Evaluation. Retrieved September 4, 2026 from https://huggingface.co/blog/rteb

[18] Shixia Liu, Xiting Wang, Christopher Collins, Wenwen Dou, Fangxin Ouyang, Mennatallah El-Assady, Liu Jiang, and Daniel A. Keim. 2019. Bridging text visualization and mining: A task-driven survey. IEEE Transactions on Visualization and Computer Graphics 25, 7 (2019), 2482–2504. https://doi.org/10. 1109/TVCG.2018.2834341

[19] Hugo Mentzingen, Nuno António, and Fernando Bacao. 2025. Unveiling legal complexity: a systematic review on the visual analytics of legal corpora. International Review of Law, Computers & Technology 40, 1 (2025), 120–155. https://doi.org/10. 1080/13600869.2025.2497630

[20] Aditi Mishra, Utkarsh Danzy, Utkarsh Soni, Anjana Arunkumar, Jinbin Huang, Bum Chul Kwon, and Chris Bryan. 2025. PromptAid: Visual prompt exploration, perturbation, testing and iteration for large language models. IEEE Transactions on Visualization and Computer Graphics 31, 10 (2025), 6946– 6962. https://doi.org/10.1109/TVCG.2025.3535332

[21] Niklas Muennighof, Nouamane Tazi, Loïc Magne, and Nils Reimers. 2023. MTEB: Massive Text Embedding Benchmark. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics (EACL), 2023. Association for Computational Linguistics. https://doi.org/ 10.18653/v1/2023.eacl-main.148

[22] Niklas Muennighof. 2022. SGPT: GPT Sentence Embeddings for Semantic Search. Retrieved from https://arxiv.org/abs/ 2202.08904

[23] Chau Nguyen, Son T. Luu, Thanh Tran, An Trieu, Anh Dang, Dat Nguyen, Hiep Nguyen, Tin Pham, Trang Pham, Thien-Trung Vo, Dinh-Truong Dol, Nguyen-Khang Le, Dieu-Hien

Nguyen, Ngoc-Cam Le, Thi-Thuy Le, Quan Bui, Phuong Nguyen, Ha-Thanh Nguyen, Vu Tran, and Le-Minh Nguyen. 2023. A Summary of the ALQAC 2023 Competition. In 2023 15th International Conference on Knowledge and Systems Engineering (KSE), 2023. IEEE. https://doi.org/10.1109/kse59128. 2023.10299527

[24] Dat Quoc Nguyen, Dai Quoc Nguyen, Thanh Vu, Mark Dras, and Mark Johnson. 2017. A Fast and Accurate Vietnamese Word Segmenter. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), 2017. European Language Resources Association. https://doi.org/10.63317/32m47vrkp2wj

[25] Ha-Thanh Nguyen, Manh-Kien Phi, Xuan-Bach Ngo, Vu Tran, Le-Minh Nguyen, and Minh-Phuong Tu. 2022. Attentive deep neural networks for legal document retrieval. Artificial Intelligence and Law (2022). https://doi.org/10.1007/ s10506-022-09341-8

[26] NVIDIA. 2026. Nemotron-3-Embed-8B-BF16: Multilingual text embedding model (model card). Retrieved September 4, 2026 from https://huggingface.co/nvidia/Nemotron-3- Embed-8B-BF16

[27] NVIDIA. 2026. Nemotron-3-Embed-1B-BF16: Multilingual text embedding model (model card). Retrieved September 4, 2026 from https://huggingface.co/nvidia/Nemotron-3- Embed-1B-BF16

[28] NVIDIA. 2026. NVIDIA Nemotron 3 Embed Ranks #1 Overall on RTEB, Advancing Agentic Retrieval. Retrieved September 4, 2026 from https://huggingface.co/blog/nvidia/nemotron-3- embed-wins-rteb

[29] Nhat-Minh Pham, Ha-Thanh Nguyen, and Trong-Hop Do. 2022. Multi-stage Information Retrieval for Vietnamese Legal Texts. Retrieved from https://arxiv.org/abs/2209.14494

[30] Tran Minh Quan. 2026. Hỏi đáp pháp luật Việt Nam — Vietnamese Legal Q&A (thuvienphapluat.vn). Retrieved September 4, 2026 from https://huggingface.co/datasets/tmquan/ thuvienphapluat-vn-hdpl

[31] Andrew J. Reagan, Lewis Mitchell, Dilan Kiley, Christopher M. Danforth, and Peter Sheridan Dodds. 2016. The emotional arcs of stories are dominated by six basic shapes. EPJ Data Science 5, 1 (2016), 31. https://doi.org/10.1140/epjds/s13688- 016-0093-1

[32] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 2019. Association for Computational Linguistics. https://doi. org/10.18653/v1/d19-1410

[33] Lucas E. Resck, Jean R. Ponciano, Luis Gustavo Nonato, and Jorge Poco. 2023. LegalVis: Exploring and inferring precedent citations in legal documents. IEEE Transactions on Visualization and Computer Graphics 29, 6 (2023), 3105–3120. https:// doi.org/10.1109/TVCG.2022.3152450

[34] Ben Shneiderman and Aleks Aris. 2006. Network visualization by semantic substrates. IEEE Transactions on Visualization and Computer Graphics 12, 5 (2006), 733–740. https://doi. org/10.1109/TVCG.2006.166

[35] Gabriel de Souza P. Moreira, Radek Osmulski, Mengyao Xu, Ronay Ak, Benedikt Schiferer, and Even Oldridge. 2024. NV-

Retriever: Improving text embedding models with efective hard-negative mining. Retrieved from https://arxiv.org/abs/ 2407.15831

[36] Ian Tenney, James Wexler, Jasmijn Bastings, Tolga Bolukbasi, Andy Coenen, Sebastian Gehrmann, Ellen Jiang, Mahima Pushkarna, Carey Radebaugh, Emily Reif, and Ann Yuan. 2020. The Language Interpretability Tool: Extensible, interactive visualizations and analysis for NLP models. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 2020. 107–118. https://doi.org/10.18653/v1/2020.emnlp-demos.15

[37] Nguyen Ha Thanh, Bui Minh Quan, Chau Nguyen, Tung Le, Nguyen Minh Phuong, Dang Tran Binh, Vuong Thi Hai Yen, Teeradaj Racharak, Nguyen Le Minh, Tran Duc Vu, Phan Viet Anh, Nguyen Truong Son, Huy Tien Nguyen, Bhumindr Butr-indr, Peerapon Vateekul, and Prachya Boonkwan. 2021. A Summary of the ALQAC 2021 Competition. In 2021 13th International Conference on Knowledge and Systems Engineering (KSE), 2021. IEEE. https://doi.org/10.1109/kse53942.2021. 9648724

[38] THƯ VIỆN PHÁP LUẬT. 2026. Thư Viện Pháp Luật: Vietnamese legal document library and legal Q&A portal. Retrieved September 4, 2026 from https://thuvienphapluat .vn/

[39] Son Pham Tien, Hieu Nguyen Doan, An Nguyen Dai, and Sang Dinh Viet. 2024. Improving Vietnamese Legal Document Retrieval using Synthetic Data. Retrieved from https:// arxiv.org/abs/2412.00657

[40] William Timkey and Marten van Schijndel. 2021. All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2021. Association for Computational Linguistics. https://doi.org/10.18653/v1/2021.emnlp-main.372

[41] Jesse Vig. 2019. A multiscale visualization of attention in the Transformer model. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics: System Demonstrations, 2019. 37–42. https://doi.org/10.18653/v1/ P19-3007

[42] Thanh Vu, Dat Quoc Nguyen, Dai Quoc Nguyen, Mark Dras, and Mark Johnson. 2018. VnCoreNLP: A Vietnamese Natural Language Processing Toolkit. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Demonstrations, 2018. Association for Computational Linguistics. https://doi.org/10.18653/v1/n 18-5012

[43] Zalo AI. 2021. Zalo AI Challenge 2021: Legal Text Retrieval. Retrieved September 4, 2026 from https://www.kaggle.com/ datasets/hariwh0/zaloai2021-legal-text-retrieval

[44] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. 2025. Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models. Retrieved from https://arxiv. org/abs/2506.05176

## A GALLERIES

These galleries show trajectories for every question form (six headlines each, three single‑ and three multi‑question where available) and every legal area, drawn from the full vectors of all 2,144 headlines; no headline appears twice in the paper. In each panel the gold article is the star, the twenty passages nearest the final prefix are grey, the headline path is green with one dot per word (squares mark a second sub‑question), the red ring is the insight point, and thin blue lines are second sub‑questions encoded alone. Titles give the forms of the sub‑questions, the insight point in content words and the final rank.

![](images/62e0c407dd7ec3643695140540b580453ee952cee4d001396e737d123762fddf.jpg)

![](images/6bfe0f6035f33e37efa49db7a0c20f20d139c965a7c2d7ab424df82c825272d7.jpg)

![](images/847f68a189a48293b194f441a73ad354e8d0d2fe5f591d7e6462150d12f36360.jpg)

![](images/7363b9489c3000c47d473fdeb4ef53ffdfd752b9c9fd77964abc93a4777d085d.jpg)

![](images/b29285e4ec490e3d7652ab0ce7b2ee63bf67a64d91049c0ce2cc46adeb14e22e.jpg)

![](images/604062719ecfd94ed5b31e6a1a85680dc98a52e2f78a4b79e1a5fe2419f2ef7b.jpg)  
Figure 11: What / content headlines (Nemotron‑3‑Embed‑8B).

![](images/4f67be1c1c596aa4fbd854b0290c2dad26f482546e56e269aeb8cb01b46ba122.jpg)

![](images/950a768007db4398c766e25fbd672e00252c2d359e585e2de17213a1474f56ba.jpg)

![](images/ca8581ee36c5e7aa732b0d318e75f742efde5be1fde1dec09544cae57e25af4a.jpg)

![](images/9592cecbc4f9676d43886d4e284cbf56a2e1f8cc856aae67e226e46260dff80b.jpg)

![](images/d62151ea7a847ca364b82a291aa005ae8ff4221b0eb27b8337c74f2fe19fee82.jpg)

![](images/248994489794e9effdfad7a56fc7539ad0668f55841f561ebd8039125f2e1d0a.jpg)  
Figure 12: Document‑request headlines (Nemotron‑3‑Embed‑8B): the template name locks the article; mẫu, tải (template, download) add nothing

![](images/0a77d4d60e8dcce0804b44194860493849589dea31f7a556e6390d6ad5f5a29a.jpg)

![](images/3a3e462073a7115b74c936d0a3aee80cc079437cd143ee29e33365311f0f04e2.jpg)

![](images/18ba2de8939bfe3627cafc8f4cb32b8fe3d8617f6ac38616e9356bb95736d907.jpg)

![](images/9ce19bfc469229b459130f4e2fa7b99e678f566d3d4821237d5e661c74d3eb55.jpg)

![](images/000a3d87df8f472c04b98a7a4b45efe239c9201e968154b752563d648ea0c5b2.jpg)

![](images/ea4341e9532b40f2ed3ad58edba33ca02411b7cb4ce33a872c31360bfd00f63b.jpg)

Figure 13: Amount / time headlines (Nemotron‑3‑Embed‑8B): the quantity asked for is in the frame, the lock is on the subject.  
![](images/d26e3bf478a73d27c57fbc2ce5e497a7a35fd69f61f323f50870979f38daf7fa.jpg)

![](images/75660ef3ff3666e9083d8e22008c3bb425a22c517c8b967c905f5010b35a9bdf.jpg)

![](images/5e32fd80d3194ff13bf8bcab537ffbacccf206619bd2acd5b5c7e4c3163c312b.jpg)

![](images/180c9399e6ec92c87130f6f5a2b42d891d94642aa26009466c166b29194c9733.jpg)

![](images/cda535ae0bcddda30e314c98a25dd41fe53f094c46a535860258e50e6cd11920.jpg)  
Figure 14: Yes / no headlines (Nemotron‑3‑Embed‑8B): (e) opens with a greeting (grey) that the path leaves at once.

![](images/3a38e8331592af3e18a31c14706da1746daf6e8dcd46e40240b7aa83d74f1908.jpg)

![](images/1df7ef4497d1c6209d85d65f788b26f712016b30182372f423f9421614115319.jpg)

![](images/a2e457e619fb8b737985ebe417f91c00d56387192b3c1ef0af6879cdfb0884dc.jpg)

![](images/fbe45d0fa6896805dc94145c926cd15b777cdade5a41980ab39b744bd1ca8e8a.jpg)

![](images/17a8a6dfea522d213256b3c135bb7158d7bc39e1eedb1c558bc375ce491bc082.jpg)

![](images/49ad3f519eb56418466161cf5ad466e39e43541f6fbe13ed985046a433dfea68.jpg)  
fwhat/content + amount/time insight: content word 7 · final rank 1

![](images/d6cb0773628f9aee561774c809b2b8f7aa839752bb6398db32e4ea9fb5b778ed.jpg)  
Figure 15: Sanction headlines (Nemotron‑3‑Embed‑8B): the most stable form; (b) opens with Cho tôi hỏi (Let me ask).

![](images/5b96e7ebc9ee547e4b30044076fdc8f442396c1750d4c5678191903bf61e4e1f.jpg)

![](images/a89c97bbb25d29ddd6a51390079df5251737b9f9b9bb25074baada6e3737d5bd.jpg)  
C what/content insight: content word 5 · final rank 1

![](images/f4284d0bcc374f440b6ff9c083b6a7fcd88b3a9a7a7c6aa968bb68fda430d14a.jpg)

![](images/807dfcd10f743378ea5c75df5b614e261f0f95654df8cc0e7ec281eb951cc061.jpg)

![](images/c42fdd917f109ad5c65b08bb58af10851579a6968eb21067a624fc0b0d45fbe0.jpg)

![](images/4d53b870afbdafafc49521e38230041398a6e629b759e1ae9c1f995f898a0864.jpg)  
Figure 16: Procedure headlines (Nemotron‑3‑Embed‑8B): (b) is a late lock—twelve content words before Trình tự kiểm tra, giám sát … (Procedure for inspection and supervision …) becomes specific enough.

fdocument insight: content word 8 · final rank 1

d Lao đng - Tin lương (labour & wages)

e document + what/content insight: content word 7 · final rank 1 a what/content insight: content word 4 · final rank 1

![](images/719becb07cc0ee260e5c486f990ec768bbe7b0e3c90ed2e0424a23aac9ab8626.jpg)

![](images/2c36da8967c76d2765fdaa8abe125f8501fa29822de658a7623e6f8aa6c81605.jpg)  
C amount/time insight: content word 11· final rank 1

![](images/86cc586b130a1b6d3d4ea7cb82757ded93d8cf45c0271c2a64afcc0268b5ab60.jpg)  
d procedure + what/content insight: content word 2 · final rank 1

![](images/f0f97f99e41bd4a98aef6a8b7a9726f5cfca28c43e10e671ff1cb6248c172993.jpg)

![](images/7708be35b53d77d209323e6edb44918bd53e58dc0ec94d9746e54a3526ecece0.jpg)

![](images/058956be95fffd571ba76f079f575d787e0549763e2aaf15dd32db67844c5124.jpg)  
Figure 17: Non‑legal utility headlines (Nemotron‑3‑Embed‑8B): dates and lookup tables; the number is the lock.  
a B máy hành chính (public administration) b Giáo dc (education) document + amount/time · insight: content word 5 · rank dther + authority · insight: content word 8 · rank 1  
C Thu - Phí - L phí (tax & fees)

![](images/91ef259ac53dd7f93171f3177298e3edb1c256b555231bd974805976063a2086.jpg)

![](images/9230add2625d3cdf34f0d675881ede712a9fb5fa8a2d38a2df53cb3b556e372c.jpg)

![](images/ce5bd352438c919a5aa12fa811e3dfc6e437a4a6853ca28f8da6fa6118404652.jpg)

![](images/0b89c814596ddb2864bf199a04d6d6648f53eaba2dca7fc7905a258427b75728.jpg)

![](images/b5bab02a8fe07057c96e82fce8b70a28db308c45a6470ffdb9752f407b209f81.jpg)

![](images/659fd8630e2bdd03d998e6b8b478dc7dc90b9c7bf4376f883af6e9c847daeac0.jpg)

![](images/0a0684c690e866f80b62976f9fa4cd1304c1a707f922d6ebd647509b5f2cbd38.jpg)

![](images/d17b4eca2716466ce37720775fa5e8d104af374b31c1767f0d9356d9d6ce89c1.jpg)

![](images/cfb517a4ec74fbee8a8375b20a0b0884d120751f8b04e0f76a2caa481faff74e.jpg)  
Figure 18: One typical trajectory per legal area, areas 1–9 by size (Nemotron‑3‑Embed‑8B). For each area the headline whose insight point is closest to the area’s median was chosen among 12–26‑word headlines.

Th thao - Y t (sport & health)  
![](images/8cc3bc4f6633613f4edc0d2d942604bb8762245f23ab6490ce1f1a8240d1a236.jpg)

![](images/2e63a3462fbf0dfef6fefc08e4eb71fa9253cded60f330d7d12aec080fdeac96.jpg)

![](images/5ab0d74657f25fa0ca6d12691d46b3693ec761359cc5acbe2be78f1eacb68cfd.jpg)

![](images/ebd50e5f44f2caf49125cc59a524198758b1e4a0d2bbb34be9d457e2d5eda86f.jpg)

![](images/907325c4dd3d01c3e638ce5384ed9d16dd6112c9a5c26c811b78e0d3d6edf6ca.jpg)

![](images/50a0571a95c0ab50c04eba4b0ca324fe7b75ec3046ddaadd2c79f2c3ae61216b.jpg)

![](images/803a0375d313fdc0dcf1e03d14b1933ad425cb93b8070126fc1417c3f3726e99.jpg)

![](images/f64d5841b2f2044be3b092933c05c3e62e300933ed9d6dcf343afa80b48ff43a.jpg)

![](images/2c04ef4952933fa0ecee48ca1278a6ec07c0b761ab42ab87e999688c603192f4.jpg)  
Figure 19: One typical trajectory per legal area, areas 10–18.

![](images/af6b78f66315a629feb647c0f0fd58c5cbebec4792e01528018caf74536d57e3.jpg)

![](images/da8fd97cf7721c83556f1f2e01cc00e01e5cb6b77ba5a63afd54e5d8f375fa4b.jpg)

![](images/5d356813ffe184ee54524758c49e92f9ffc693e8f9f4e0578f2bb08531b22881.jpg)

![](images/7711954bfed91dfde1869070e5d16d2f27362f8c8beca75cb9bd1889713698ef.jpg)

![](images/264db237a4458092117a45957932456383496743864a0f4ec89b9509e2cb6648.jpg)

![](images/2959ee5f1ea0b7326cfcc844d27892475175508cdf09b23fb3d6487015cfafbd.jpg)

![](images/2d7feff883f814a104852711e5264394aa6ec35ac1786b56ad85877ae552fd14.jpg)

![](images/19b5d25e2130f550edab593e1011098e16ba5f6f6382a64551856de7ce03024d.jpg)  
Figure 20: One typical trajectory per legal area, areas 19–27.

![](images/6c1f4d273be87c1e30b154b462e0ebbbe69918fc61542c1e527c4f2d30026627.jpg)