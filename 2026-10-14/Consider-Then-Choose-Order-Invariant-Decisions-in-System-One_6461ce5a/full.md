# Consider, Then Choose: Order-Invariant Decisions in System One Models

Jianguo Huang\* Yao Long Teng\* Yanchen Deng<sup>†</sup> Bo An Nanyang Technological University, Singapore

## Abstract

System One models answer typed questions in a single forward pass and promise the same answer for the same input. For choice questions, this promise fails: the answer depends on the order of the options, increasingly so as they grow in number, and averaging over their cyclic shifts reduces this dependence without removing it. Scoring each option alone, which the order cannot affect, shows that the order changes the answer mostly when the model is unsure among its top few options, even though the listwise answer looks confident. We therefore propose Runoff, which considers, then chooses: it scores each option alone to keep a few candidates, then compares them in listings where each candidate takes every position. It needs no training and can run in two forward passes, and its answer does not depend on the option order by construction. On two open models and five datasets, it changes no answer under shuffling, against 9.3% for cyclic averag ing, and with 20 options it is about as accurate and three to four times faster. Wrapped around eight System One models, it removes the order dependence of the five benchmarks, while cyclic averaging changes up to 15.4%.

## 1 Introduction

System One decision models answer typed questions without generating text (Sun et al., 2026): given a state and a question whose answer is a yes/no, a choice among several options, or a level on an ordered scale, they return a probability distribution over the answers. A language model can be asked the same questions in free text, but the caller must then parse the response, and any probabilities it exposes are over tokens rather than over the possible answers. In real applications, an agent system has to make fast decisions, such as routing a request among tools or picking the next element on a screen. Fast decision models have thus become popular, and open implementations now include both purpose-trained decision models and readout heads on frozen LLMs (Xu, 2026; Lee, 2026; Convai Innovations, 2026; Zhang et al., 2026).

For choice questions, however, the answer depends not only on the question’s content but also on the order in which the options are listed. Jev’s own documentation acknowledges that models tend to favor the first-listed option, and advises users to reorder the options and check that the answer stays the same (TypeSafe AI, 2026a). Moreover, the effect grows with the number of options. As shown in Figure 1, when Qwen3-8B answers a 20-way intent question by scoring the option letters, shuffling the options changes 17% of its answers, compared with 5% for two-way questions. This matters because in an agent the option order is seldom a deliberate choice, but is inherited from the order of a tool registry or the layout of a page. An answer that depends on this order can change whenever one of these components changes, and anyone who controls the order can steer the answer (Pezeshkpour and Hruschka, 2024).

Existing methods do not remove the dependence. Asking the model to reason in text before answering (Wang et al., 2024a) gives up the single forward pass that defines a System One model, and on choice questions it does not reliably make the answer independent of the order (Pezeshkpour and Hruschka, 2024). Averaging over the K cyclic shifts of the options (Zheng et al., 2023; Zhang et al., 2026) reduces the dependence but does not remove it: on open models, it still changes the answers under a shuffle. It is also costly, with cost growing quadratically with K.

In Section 3, we locate this dependence by comparing the listwise answer with pointwise decisions of the same model, which score each option alone and are therefore blind to the order. This comparison shows where and when the order matters. The flips occur among a few options: at least one of the answers before and after a flip almost always lies among the options that the pointwise decisions rank highest. The order matters most when the model is uncertain: a shuffle changes far more answers when the pointwise top option is uncertain than when it is certain, even though the listwise answer looks confident in both cases. As more options make the model uncertain more often, the dependence grows with the number of options.

These findings suggest the two steps that people take when facing many options: consider, then choose. Shoppers narrow the available brands to a consideration set before choosing among them (Hauser and Wernerfelt, 1990), much as an election holds a runoff among the leading candidates of a first round. Our method, Runoff, brings these two steps to System One models, using only the decisions of the model itself. It first keeps the options that the pointwise decisions rank highest, so the ruled-out options cannot interfere with the choice. It then compares these candidates in every cyclic shift of their pointwise ranking, so each candidate takes every position once and none is favored by its place. Since neither step uses the input order, the answer of Runoff does not depend on it by construction, and the method needs no training and can run in two forward passes.

In Section 5, we evaluate Runoff on two open models, Qwen3-8B and Gemma 4 12B, over five datasets with 3 to 20 options, and as a black box around eight System One models. On the open models, it changes no answers under shuffling, whereas cyclic averaging changes 9.3% and a single listwise decision in the given order (Listwise) 18.6%. With 20 options, it is about as accurate as cyclic averaging and several times faster. Its main limitation is reasoning questions, where the consideration set more often misses the correct answer and it trails cyclic averaging by a few points. Around the System One models, all of which change their own answers under shuffling, it removes the order dependence of every model that returns the same probabilities for a repeated request, and leaves only a small residue on the others, far below that of cyclic averaging, with similar accuracy.

Our contributions are:

• An analysis showing that option-order dependence grows with the number of options, and that the order matters most when the model is unsure among its top few options.

• Runoff, a training-free decision rule that first narrows the options and then compares only the leading few, which makes its answer independent of the option order and cheaper than cyclic averaging when the options are many.

• Experiments on ten models and five datasets, where Runoff removes the order dependence of every deterministic model with accuracy close to cyclic averaging, and falls behind it only on reasoning questions.

## 2 Preliminaries

Notation. We consider a System One decision model (Almeida, 2026) answering a choice question q about a state x, with a set O of K options, exactly one of which is correct. A listing of $s \subseteq \mathcal { O }$ is a sequence $\lambda = ( \lambda ( 1 ) , \dots , \lambda ( | S | ) )$ that contains every option of S once, which shows $\lambda ( i )$ at position i. The question presents its options in its own order σ, a listing of O. A method may make its decisions on σ or on any listing of O.

Listwise decision. The listwise decision on a listing λ shows its options under the letters A, B, . . . and renormalizes the next-token probabilities of these letters (Robinson et al., 2022). Made on the order of the question, $P _ { \sigma } = g ( x , q , \sigma )$ is the standard answer, which we call Listwise.

Cyclic averaging. AnyJev (Zhang et al., 2026) makes the listwise decision on the K rotations of the order, $\sigma _ { s } = ( \sigma ( s ) , \ldots , \sigma ( K ) , \sigma ( 1 ) , \ldots , \sigma ( s -$ 1)), which place every option at every position once, and averages their log-probabilities:

$$
P _ { \sigma } ^ { \mathrm { C Y C } } ( o ) \propto \exp \Big ( \frac { 1 } { K } \sum _ { s = 1 } ^ { K } \log g ( x , q , \sigma _ { s } ) ( o ) \Big ) .\tag{1}
$$

Next, we give the definition of order invariance, which is the property that we want to achieve.

Definition 1 (Order invariance). A decision rule R, which maps $( x , q , \sigma )$ to a distribution $R ( x , q , \sigma )$ over O, is order-invariant if $\begin{array} { r l } { R ( x , q , \sigma ) } & { { } = } \end{array}$ $R ( x , q , \sigma ^ { \prime } )$ for all $x , q ,$ , and orders $\sigma , \sigma ^ { \prime } .$

Listwise is not order-invariant, since $\sigma$ decides which letter each option receives. Neither is Any-Jev: it places every option at every position, but its K listings are the rotations of σ, so σ decides which K of the K! listings are averaged. More details of these methods are shown in Appendix A.

![](images/c5d2819b291f89561f299632b0890979ad1ed69d254e798d081e3edbe5196ebd.jpg)  
(a) Banking20, accuracy

![](images/171e6b931d99754dcd43aa1c1b92cb701ba1308cc18807bb7cc6c3c5576e6ab5.jpg)  
(b) Banking20, flip rate

![](images/85717d967edd41a13c2ab159f5bc800df95d9a20c8b9e2ac1fc1fbc9cdbfce25.jpg)  
(c) Newsgroups, accuracy

![](images/17344542246230362f237bd30ad3a0ed2bff1c78e88bedaeebfd69720f4ba229.jpg)  
(d) Newsgroups, flip rate

Figure 1: Accuracy and flip rate against the number of options, Qwen3-8B. Flip rate: share of answers that change when the options are shuffled.  
![](images/33aeb919d2713c679c54614a4a9c92821660409aea6b6671b17f6505cacc29fb.jpg)  
(a) Long tail of the pointwise distribution

![](images/946d89994938ee8b34cbe10859c75103d19c26c8c28f5fd8bb0979a46b6306bf.jpg)  
(b) Confidence and flips of Listwise

![](images/0a27a216dc4fb249d1637904aa11e11e75d4cb3c23887ab58462131f2bd9098b.jpg)  
(c) Certainty against K  
Figure 2: Order dependence of Listwise and its relation to the pointwise distribution for Qwen3-8B. Panels (b) and (c) pool Banking20 and Newsgroups. (a) Percentage of questions whose top k pointwise options hold 99.9% of the probability, at K=20. (b) Mean top-1 probability of Listwise and its flip rate at K=20, with questions binned by their pointwise top-1 probability, which does not depend on the order. (c) Percentage of uncertain (pointwise top-1 probability below 0.9) and certain (at least 0.999) questions as K grows.

## 3 Motivation and Analysis

This section studies how the option order affects the answers of Jev-style decision models. Following AnyJev (Zhang et al., 2026), we turn an openweight LLM into such a model: it reads the state and the question in a single forward pass, without decoding, and returns the renormalized next-token probabilities of the option letters as its answer distribution. We answer each question with Listwise and AnyJev, once with the options in their original order and once in a shuffled order, and measure the order dependence by the flip rate, the percentage of questions whose answer changes between the two orders.

Flip rate and the number of options. To study the effect of the number of options, we vary K from 2 to 20 and measure the flip rate and accuracy of Listwise and AnyJev. We take two 20-way classification tasks, a 20-intent subset of BANK-ING77 (Casanueva et al., 2020) and 20 Newsgroups (Lang, 1995): the gold label and the first K − 1 of the other labels in a fixed random order. Figure 1 shows the results for Qwen3-8B (Yang et al., 2025). With Listwise, the flip rate rises from 4.7% at K=2 to 17.0% at K=20 on Banking20 and from 5.3% to 18.7% on Newsgroups, while accuracy falls. AnyJev lowers the flip rate but does not stop its growth, from 0.3% to 10.3% and from 0.7% to 11.3%.

These results show that changing the order of options makes the predictions unreliable. This raises the question: Where and when does the order change the answer? To find out, we need the model’s belief about each option without any order, which we measure with pointwise decisions.

Pointwise decision. The pointwise decision compares the model’s belief about each option alone, as shown in Appendix A:

$$
f ( x , q , o ) = \log { P ( \mathrm { y e s } \mid x , q , o ) } - \log { P ( \mathrm { n o } \mid x , q , o ) } .\tag{2}
$$

Then, the pointwise distribution is given by normalizing the odds:

$$
\pi ( o ) = \frac { \exp f ( x , q , o ) } { \sum _ { o ^ { \prime } \in \mathcal O } \exp f ( x , q , o ^ { \prime } ) } , \qquad o \in \mathcal O .\tag{3}
$$

Each decision sees one option alone, so π does not depend on σ: it shows how certain the model is of each option. The pointwise distribution is highly concentrated on a few options (Figure 2a): at $K { = } 2 0$ , the top three options of Qwen3-8B hold 99.9% of its probability for 83% (Banking20) and 93% (Newsgroups) of the questions. We call these three options the candidates.

<table><tr><td></td><td>Both</td><td>One</td><td>Neither</td></tr><tr><td>Qwen3-8B</td><td></td><td></td><td></td></tr><tr><td>Banking20</td><td>43.1</td><td>47.1</td><td>9.8</td></tr><tr><td>Newsgroups</td><td>80.4</td><td>17.9</td><td>1.8</td></tr><tr><td>Gemma 4 12B</td><td></td><td></td><td></td></tr><tr><td>Banking20</td><td>69.6</td><td>30.4</td><td>0.0</td></tr><tr><td>Newsgroups</td><td>72.1</td><td>20.9</td><td>7.0</td></tr></table>

Table 1: Share of flips (%) with both, one, or neither of the two answers among the three candidates, K=20.

Where the answer moves. For each flip, we check whether the answers under the original and the shuffled order are among the three candidates. As Table 1 shows, in over 90% of the flips at least one of the two answers is a candidate, and often both are. Neither is a candidate in at most 10% of the flips. The order thus mostly decides among the few options that the model ranks highest.

Flip rate and pointwise certainty. To see when the order affects the answers, we group the K=20 questions by the top-1 probability of π, which does not depend on the order and measures how certain the model’s prediction is (Figure 2b). The flip rate grows as the model becomes less certain: it exceeds a third below 0.8, is 30% between 0.8 and 0.9, about 17% between 0.9 and 0.999, and 7% at 0.999 and above. Meanwhile, the top-1 probability of Listwise barely moves, from 93% to 99.5%.

Certainty and the number of options. To see why the flip rate grows with K, we track how certain the model’s prediction is as the number of options increases. Figure 2c shows that as K grows from 2 to 20, the share of uncertain questions rises from 7% to 24%, while the share of certain questions, whose pointwise top-1 probability is at least 0.999, falls from 82% to 43%. Since the order changes the answer more often the more uncertain the prediction is, this shift toward uncertainty makes the flip rate grow with the number of options.

## Takeaways

• How much: The answer depends on the option order, more so with more options, and AnyJev does not stop it.

• Where: Most answers before and after a flip are among the model’s top three options.

• When: The more uncertain the prediction, the more often the order changes the answer.

• Why it grows: More options make the prediction more uncertain, so the order matters more.

## 4 Method

We propose Runoff, a decision rule that needs no training and whose answer does not depend on the option order. As Figure 3 illustrates, Runoff works in two steps. Consider picks the few options among which the order can move the answer, by scoring each option alone as in Section 3. Choose then compares only these options, placing each at every position. Unlike cyclic averaging, which compares all K options in K listings, Runoff compares only these few, so the options it rules out cannot change the answer (Appendix B).

Consider. Runoff computes the pointwise distribution π, defined in Eq. 3, and keeps its top options as a consideration set C: the smallest set of top options that holds at least α of the probability and contains at least $k _ { \mathrm { m i n } }$ options. Formally, let $v _ { 1 } > \cdots > v _ { n }$ be the distinct values of π and $U _ { j } = \{ o \in \mathcal { O } : \pi ( o ) \geq v _ { j } \}$ . Then, ${ \mathcal { C } } = U _ { J }$ with $J = \operatorname* { m i n } \{ j : \pi ( U _ { j } ) \geq \alpha , | U _ { j } | \geq k _ { \operatorname* { m i n } } \}$ , and C = O if no such j exists. Thus, the consideration set grows when the model’s prediction is uncertain.

Choose. The pointwise distribution alone is order-invariant, but it judges each option without seeing the others, which leads to wrong answers. Choose therefore compares the members of C side by side. Specifically, we first rank them by decreasing π. We call the result $c _ { 1 } , \ldots , c _ { m }$ the canonical ranking. A single listwise decision on this ranking would already be order-invariant, but the model’s position bias would then favor whichever candidate the ranking puts first, i.e., the top option of π. Choose instead makes a listwise decision on each of the m cyclic shifts of the ranking, $\lambda _ { s } = ( c _ { s } , \ldots , c _ { m } , c _ { 1 } , \ldots , c _ { s - 1 } )$ , so that every candidate takes every position once: for a ranking $( a , b , c )$ , the listings are $( a , b , c ) , ( b , c , a )$ and $( c , a , b )$ . It averages their log-probabilities as in Eq. 1, which gives a distribution r over C. The final distribution keeps π outside C and divides the probability of C according to r:

![](images/5fe6c41392e959edf2766db8e2f8fa63a2a0588e2cbf2e767d90c090c06ecb99.jpg)  
Figure 3: Overview of Runoff.

$$
P ( o ) = { \left\{ \begin{array} { l l } { \pi ( { \mathcal { C } } ) r ( o ) } & { o \in { \mathcal { C } } , } \\ { \pi ( o ) } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{4}
$$

Proposition 1. Runoff is order-invariant (Definition 1): the requests it sends to the model, and hence P, are the samefor every order σ ofO.

Each pointwise decision sees one option and π normalizes over the set O, so π depends only on $( x , q , \mathcal { O } )$ ; C, its canonical ranking and the cyclic shifts are functions of π and the option texts (proof in Appendix E). The invariance holds whatever position bias the model has, and on a black-box model any remaining flip can only come from the model answering identical requests differently.

Cost. Runoff reads $K + m ^ { 2 }$ option slots, against $K ^ { 2 }$ for cyclic averaging, so it is cheaper when $m \ll K$ . On open models, each stage runs as a single forward pass.

## 5 Experiments

## 5.1 Setup

Decision Models. We evaluate ten models in three groups: (i) open-weight LLMs read out as decision models, Qwen3-8B (Yang et al., 2025) and Gemma 4 12B (Gemma Team, 2026); (ii) closed-source System One models, Jev 1.13, Mercury Decide and GPT-6 Luna; (iii) open-weight models fine-tuned for System One decisions, Torchcast 12B, Kev 9B, Intern-Decision 4B, Tev1 4B and Laya.

Evaluation. We evaluate the choice questions of five benchmarks: Banking20 and Newsgroups (20 options, 300 questions each), MMLU-Pro (Wang et al., 2024b) (10 options; 1,400 questions), Typed

Decisions (LocalLLaMA, 2026) (600 questions with 4–5 options), and the public items of JevBench (Standhartinger, 2026) (139 questions with 3–6 options). The baselines are Listwise and cyclic averaging as implemented by AnyJev. The prompt templates are provided in Appendix A. For Runoff, we use $\alpha = 0 . 9 9$ and $k _ { \operatorname* { m i n } } = 3$

Evaluation metrics. We report accuracy with the options in their original order (Acc) and the flip rate under one shuffle per question (Flip).

## 5.2 Main results

Results of open-source models on five decision benchmarks. Table 2 shows the results of the open-source models, i.e., Qwen3-8B and Gemma 4 12B. Runoff changes no answers when the options are shuffled, whereas Listwise and AnyJev change theirs on every dataset, and its accuracy stays comparable to that of both baselines. For example, on MMLU-Pro with Qwen3-8B, shuffling changes 44.4% of the Listwise answers and 24.7% of the AnyJev answers, but none of Runoff’s. In accuracy, with 20 options, Runoff reaches 73.8% averaged over Banking20 and Newsgroups on both models, against 74.5% for AnyJev and 73.0% for Listwise. Overall, Runoff removes the order dependence while keeping its accuracy comparable to that of both baselines.

Results of closed-source models. Table 2 also reports the results of the three closed-source System One models, with AnyJev and Runoff wrapped around their APIs. The native answers of all three models depend on the option order, but Runoff reduces this dependence further than AnyJev on each of them, and on average it is at least as accurate.

<table><tr><td></td><td></td><td colspan="2">Banking20</td><td colspan="2">Newsgroups</td><td colspan="2">MMLU-Pro</td><td colspan="2">Typed Decisions</td><td colspan="2">JevBench</td><td colspan="2">Avg</td></tr><tr><td>Model</td><td>Method</td><td></td><td>Acc↑ Flip↓</td><td>Acc↑ Flip↓</td><td></td><td>Acc↑</td><td>Flip↓</td><td>Acc↑</td><td>Flip↓</td><td>Acc↑</td><td>Flip↓</td><td>Acc↑</td><td>Flip↓</td></tr><tr><td colspan="10">Open-source</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">Qwen3-8B</td><td>Listwise</td><td>73.0</td><td>17.0</td><td>68.0</td><td>18.7</td><td>47.6</td><td>44.4</td><td>62.7</td><td>12.2</td><td>66.2</td><td>21.6</td><td>63.5</td><td>22.8</td></tr><tr><td>AnyJev</td><td>78.0</td><td>10.3</td><td>68.0</td><td>11.3</td><td>53.1</td><td>24.7</td><td>65.2</td><td>3.0</td><td>69.8</td><td>8.6</td><td>66.8</td><td>11.6</td></tr><tr><td>Runoff (ours)</td><td>76.7</td><td>0.0</td><td>68.3</td><td>0.0</td><td>51.6</td><td>0.0</td><td>64.2</td><td>0.0</td><td>68.3</td><td>0.0</td><td>65.8</td><td>0.0</td></tr><tr><td rowspan="3">Gemma 412B</td><td>Listwise</td><td>83.7</td><td>7.7</td><td>67.3</td><td>14.3</td><td>57.1</td><td>31.9</td><td>70.0</td><td>11.5</td><td>87.1</td><td>6.5</td><td>73.0</td><td>14.4</td></tr><tr><td>AnyJev</td><td>84.3</td><td>3.0</td><td>67.7</td><td>7.3</td><td>60.1</td><td>16.2</td><td>70.8</td><td>5.8</td><td>86.3</td><td>2.9</td><td>73.8</td><td>7.1</td></tr><tr><td>Runoff (ours)</td><td>81.3</td><td>0.0</td><td>68.7</td><td>0.0</td><td>57.0</td><td>0.0</td><td>69.3</td><td>0.0</td><td>84.9</td><td>0.0</td><td>72.2</td><td>0.0</td></tr><tr><td colspan="10">Closed-source</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">Jev 1.13</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Native</td><td>86.7 87.7</td><td>3.0 1.3</td><td>75.7 75.0</td><td>5.0 1.7</td><td>81.8</td><td>7.7 3.4</td><td>72.0 72.0</td><td>6.5 3.7</td><td>87.1 89.2</td><td>4.3 0.7</td><td>80.6 81.5</td><td>5.3 2.2</td></tr><tr><td>AnyJev Runoff (ours)</td><td>87.0</td><td>1.3</td><td>75.3</td><td>0.3</td><td>83.6 83.4</td><td>1.4</td><td>72.8</td><td>1.5</td><td>89.2</td><td>0.0</td><td>81.5</td><td>0.9</td></tr><tr><td rowspan="3">Mercury Decide</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>76.7</td><td>9.8</td></tr><tr><td>Native</td><td>89.7</td><td>3.7 2.3</td><td>74.0 73.7</td><td>5.3 4.3</td><td>63.5</td><td>22.3 12.3</td><td>67.3</td><td>9.8</td><td>89.2 90.6</td><td>7.9 2.9</td><td></td><td>5.4</td></tr><tr><td>AnyJev Runoff (ours)</td><td>91.0 91.0</td><td>0.0</td><td>73.7</td><td>0.7</td><td>68.6 66.7</td><td>4.4</td><td>68.3 69.2</td><td>5.2 0.7</td><td>92.1</td><td>1.4</td><td>78.4 78.5</td><td>1.4</td></tr><tr><td rowspan="3">GPT-6 Luna</td><td></td><td></td><td>5.9</td><td>75.2</td><td>6.7</td><td></td><td>27.7</td><td></td><td>18.7</td><td>89.2</td><td>7.2</td><td>75.4</td><td>13.2</td></tr><tr><td>Native</td><td>85.5 87.9</td><td>2.8</td><td>74.8</td><td>3.0</td><td>61.8 64.8</td><td>15.0</td><td>65.5 67.0</td><td>9.8</td><td>91.4</td><td>2.2</td><td>77.2</td><td>6.6</td></tr><tr><td>AnyJev Runoff (ours)</td><td>85.9</td><td>0.0</td><td>75.2</td><td>0.0</td><td>64.6</td><td>0.0</td><td>68.8</td><td>0.0</td><td>90.6</td><td>0.0</td><td>77.0</td><td>0.0</td></tr></table>

Table 2: Main results on five decision benchmarks. Acc: accuracy (%) with the options in their original order. Flip: share of questions (%) whose answer changes when the options are shuffled. Native: the model’s own choice answer. Avg: mean over the five datasets. Best per model in bold.

<table><tr><td rowspan="2">Model</td><td colspan="2">Native</td><td colspan="2">AnyJev</td><td colspan="2">Runoff</td></tr><tr><td>Acc↑ Flip↓</td><td></td><td>Acc↑</td><td>Flip↓</td><td>Acc↑ Flip↓</td><td></td></tr><tr><td>Torchcast 12B</td><td>70.2</td><td>15.6</td><td>73.3</td><td>5.4</td><td>72.2</td><td>0.0</td></tr><tr><td>Kev 9B</td><td>72.7</td><td>12.5</td><td>73.2</td><td>7.2</td><td>72.8</td><td>0.0</td></tr><tr><td>Intern-Decision 4B</td><td>70.6</td><td>14.7</td><td>72.9</td><td>8.1</td><td>73.0</td><td>0.0</td></tr><tr><td>Tev1 4B</td><td>68.4</td><td>12.3</td><td>70.2</td><td>7.0</td><td>70.2</td><td>0.0</td></tr><tr><td>Laya</td><td>40.7</td><td>30.6</td><td>42.3</td><td>15.4</td><td>42.4</td><td>0.0</td></tr></table>

Table 3: Results of open-weight models fine-tuned for System One decisions: accuracy and flip rate (%), averaged over the five datasets (per-dataset results in Table 9). Best per model in bold.

For example, GPT-6 Luna answers a re-sent request identically, yet shuffling changes 13.2% of its native answers and 6.6% of AnyJev’s on average, but none of Runoff’s, while the accuracy rises from 75.4% to 77.0% (AnyJev: 77.2%). On Jev 1.13 and Mercury Decide, Runoff lowers the flip rate from 5.3% and 9.8% to 0.9% and 1.4% (AnyJev: 2.2% and 5.4%). By Proposition 1, these remaining flips do not come from the option order: the two APIs can return different probabilities for identical yes/no requests, which can change the consideration set. Overall, the native answers of all three models depend on the option order, and Runoff reduces this dependence further than AnyJev, achieving higher accuracy than the native answer on average.

Results of fine-tuned models. Table 3 reports the results of the five open-weight models fine-tuned for System One decisions. Existing fine-tuning methods do not remove the order dependence: the native answers of all five models change under shuffling, whereas Runoff removes the dependence up to the model’s own nondeterminism and is at least as accurate as the native answer on every model. For example, on Torchcast 12B, shuffling changes 15.6% of the native answers and 5.4% of AnyJev’s, but none of Runoff’s, whose accuracy is 72.2% against 70.2% natively and 73.3% for Any-Jev. Overall, training a model for decisions does not make it order-invariant, and Runoff adds the invariance without retraining it.

<table><tr><td></td><td>K</td><td>Qwen3-8B</td><td>Gemma 4 12B</td></tr><tr><td>Banking20 Newsgroups MMLU-Pro</td><td>20 20 10</td><td>308 → 77 (4.0×) 248 → 82 (3.0×) 125 → 78 (1.6×)</td><td>525 → 121 (4.3×) 401 → 140 (2.9×) 216 → 135 (1.6×)</td></tr><tr><td>Typed JevBench</td><td>4-5 3-6</td><td> $7 9 \to 8 5 ( 0 . 9 \times )$  67 → 77 (0.9×)</td><td> $1 4 0 \to 1 4 5 ( 1 . 0 \times )$   $1 1 6 \to 1 3 0 ( 0 . 9 \times )$ </td></tr><tr><td>Avg</td><td>一</td><td>165 → 80 (2.1×)</td><td>280 → 134 (2.1×)</td></tr></table>

Table 4: Latency of AnyJev → Runoff in ms per record, and the speedup of Runoff. A record is one choice question. Mean over the original and shuffled orders. Avg: mean latency over the five datasets.

Latency. Table 4 reports the time per record of AnyJev and Runoff on the open models. Runoff is faster than AnyJev when the options are many and slightly slower when they are few. For example, on Banking20 with Qwen3-8B (20 options), a record takes 77 ms with Runoff against 308 ms with Any-Jev, whereas on JevBench (3 to 6 options) it takes 77 ms against 67 ms. With many options, Choose compares only about four candidates instead of all K options in K listings. With few, the second forward pass is a fixed cost that the smaller set does not repay. Averaged over the five datasets, Runoff is 2.1 times faster than AnyJev on both models, but remains slower than Listwise, which makes a single listwise decision.

![](images/55ce0eb8b88254e87f74c9147050acd1df26c2fbf5d71d1042e4e0471a758e67.jpg)  
Figure 4: OSWorld tasks solved by a System One computer-use agent with the native decisions and Runoff.

## 6 Discussion

Decisions in a computer-use agent. We test Runoff where the choices of a System One model drive an agent. We take an open-source computeruse agent (Levin, 2026) in which a classifier chooses every action, including the on-screen item to click among many items, and answer each of its choices with Runoff, with the settings of Section 5.1. Figure 4 shows that, on the 316 tasks of OSWorld (Xie et al., 2024) that our host can run, Runoff solves more tasks than the agent’s native decisions with both models: 27 against 21 with Qwen3.5-35B-A3B, and 14 against 9 with Qwen3.5-9B. The benefit of Runoff thus extends beyond single decisions to an agent that chains many of them.

Ablation studies. Which stage of Runoff gives the invariance, and which the accuracy? Table 5 adds the two stages step by step, averaged over the open models of Table 2. The pointwise distribution alone is order-invariant but reaches only 63.3%, since it judges each option without seeing the others. Comparing the consideration set in a single listing recovers most of the accuracy, but how that listing is fixed matters. In the input order, it reaches 68.3% but brings the flips back (15.8%).

<table><tr><td>Variant</td><td>Acc↑</td><td>Flip↓</td></tr><tr><td>Consider only</td><td>63.3</td><td>0.0</td></tr><tr><td>+ Canonical ranking, one listing</td><td>67.3</td><td>0.0</td></tr><tr><td>+ one listing (input order)</td><td>68.3</td><td>15.8</td></tr><tr><td>+ Choose (Runoff)</td><td>69.0</td><td>0.0</td></tr></table>

Table 5: Ablations: accuracy and flip rate (%), means over the ten model–dataset cells of Table 2. Consider only: the pointwise distribution, $P = \pi$ . One listing: Choose makes one listwise decision on the consideration set, with its members in the order of the input. Cyclic shifts: Choose makes listwise decisions on the cyclic shifts of the canonical ranking.

<table><tr><td>Setting</td><td>Bank. News.</td><td>MMLU</td><td>Typed JevB.</td><td>m</td></tr><tr><td> $\alpha = 0 . 9$ </td><td>76.7</td><td>68.3 50.6</td><td>64.2</td><td>67.6 3.2</td></tr><tr><td> $\alpha = 0 . 9 5$ </td><td>76.7</td><td>68.3</td><td>51.6 64.2 51.6</td><td>67.6 3.3</td></tr><tr><td> $\alpha = 0 . 9 9 ^ { \dagger }$ </td><td>76.7</td><td>68.3 68.3</td><td>64.2 52.1 64.2</td><td>68.3 3.5 3.9</td></tr><tr><td>α = 0.999</td><td>75.7</td><td></td><td></td><td>68.3</td></tr><tr><td> $k _ { \operatorname* { m i n } } = 1$ </td><td>76.0</td><td>68.3</td><td>50.2 62.2</td><td>64.7 2.1</td></tr><tr><td> $k _ { \operatorname* { m i n } } = 2$ </td><td>75.7</td><td>68.7 50.7</td><td>61.8</td><td>66.2 2.7</td></tr><tr><td> $k _ { \mathrm { m i n } } = 3 ^ { \dagger }$ </td><td>76.7</td><td>68.3 51.6</td><td>64.2</td><td>68.3 3.5</td></tr><tr><td> $k _ { \operatorname* { m i n } } = 4$ </td><td>76.3</td><td>68.3 51.9</td><td>66.0</td><td>68.3 4.4</td></tr><tr><td> $k _ { \mathrm { m i n } } = 5$ </td><td>76.7</td><td>66.3 52.7</td><td>66.3</td><td>69.1 5.1</td></tr></table>

Table 6: Accuracy (%) per dataset, Qwen3-8B, and the mean size m of the consideration set over the five datasets. †: the default setting of Runoff.

In the canonical ranking, it changes no answer but falls to 67.3%, likely because the model’s position bias now always favors the top option of π. Choose averages over the cyclic shifts of this ranking, so no candidate holds a privileged position: it keeps the invariance and reaches 69.0%, above both single listings. Both stages are thus needed: Consider narrows options without seeing order, while Choose compares candidates without favoring any position.

Sensitivity. How much does Runoff depend on its hyperparameters, the mass α and the minimum size $k _ { \mathrm { m i n } }$ of the consideration set? We vary them on Qwen3-8B over the five datasets and report the accuracy and the mean size of the consideration set in Table 6. The mass hardly matters: the mean accuracy stays between 65.5% to 65.8% for $\alpha \geq 0 . 9 .$ The minimum size matters below three: lowering $k _ { \mathrm { m i n } }$ to 1 or 2 costs 1.5 and 1.2 points, as the consideration set misses the gold answer more often (77% and 83% against 90%). Raising it to 4 or 5 gains about 0.4 points at more time, so the default $k _ { \operatorname* { m i n } } = 3$ and $\alpha = 0 . 9 9$ are a reasonable trade-off.

Number of options. Section 3 shows that the order dependence grows with the number of options. Does Runoff stay invariant and accurate as they grow? We run the three methods on the questions of Figure 1, Banking20 and Newsgroups with K = 2 to 20 options, with Qwen3-8B and Gemma 4 12B. As Table 7 shows, averaged over the four model–dataset cells, Runoff changes no answer at any K, and its accuracy stays within 1.3 points of AnyJev’s. At K=20, Listwise changes 14.4% of its answers and AnyJev 8.0%, up from 4.1% and 0.3% at K=2, whereas Runoff changes none and reaches 73.8% against 74.5% for AnyJev. The invariance of Runoff thus holds exactly where the dependence of the baselines is strongest.

<table><tr><td rowspan="2"></td><td colspan="2">Listwise</td><td colspan="2">AnyJev</td><td colspan="2">Runoff</td></tr><tr><td>K Acc↑</td><td>Flip↓</td><td>Acc↑</td><td>Flip↓</td><td>Acc↑ Flip↓</td><td></td></tr><tr><td>2</td><td>95.8</td><td>4.1</td><td>96.2</td><td>0.3</td><td>96.3</td><td>0.0</td></tr><tr><td>4</td><td>89.8</td><td>5.7</td><td>90.3</td><td>2.5</td><td>90.4</td><td>0.0</td></tr><tr><td>8</td><td>84.4</td><td>8.5</td><td>84.3</td><td>3.8</td><td>84.2</td><td>0.0</td></tr><tr><td>12</td><td>79.3</td><td>10.6</td><td>80.5</td><td>6.2</td><td>79.2</td><td>0.0</td></tr><tr><td>16</td><td>77.0</td><td>14.0</td><td>77.8</td><td>6.3</td><td>77.8</td><td>0.0</td></tr><tr><td>20</td><td>73.0</td><td>14.4</td><td>74.5</td><td>8.0</td><td>73.8</td><td>0.0</td></tr></table>

Table 7: Accuracy and flip rate (%) against the number of options K, on the questions of Figure 1: means over Qwen3-8B and Gemma 4 12B on Banking20 and Newsgroups, 300 questions per model, dataset and K (per model and dataset in Table 10). Best per K in bold.

## 7 Related Work

System One decision models. TypeSafe’s Jev answers yes/no, choice, and score questions with a probability distribution from one forward pass (Almeida, 2026; Sun et al., 2026). Open counterparts train a decision model (Palmer, 2026; Convai Innovations, 2026; TianyuCodings, 2026; Yu and Yao, 2026) or read one from a frozen LLM (Lee, 2026), scoring each option alone (Xu, 2026) or reading the labels of one listing (tic-top, 2026; Zhang et al., 2026). Their choices can follow an option’s name rather than its meaning (Sun et al., 2026) and, by Jev’s own documentation, lean toward the first listed option (TypeSafe AI, 2026a). For many options, Jev scores them independently before an explicit choice, without disclosing how (Almeida, 2026), and TypeSafe’s cookbook shortlists three candidates first (TypeSafe AI, 2026b); Runoff makes this two-stage reading explicit and order-invariant on frozen models.

Option-order sensitivity and invariance. Listed options sway the answer by their order (Pezeshkpour and Hruschka, 2024; Wei et al.,

2024; Zong et al., 2024), largely through a prior over option-ID tokens (Zheng et al., 2023) that calibration divides out with a correction shared across questions, leaving each answer order-dependent (Zhao et al., 2021; Zhou et al., 2024). Averaging over all K! orders removes the dependence at K! listings (Guda et al., 2025); averaging over the K cyclic shifts (Izacard et al., 2023; Zheng et al., 2023; Zhang et al., 2026) or random shuffles (Tang et al., 2024) leaves part of it. Structural methods make one forward invariant pass by isolating the options under shared position ids (McIlroy-Young et al., 2024; Egressy and Stühmer, 2026; Bito et al., 2026), which removes their comparison and tends to cost accuracy (Hanna and Feng, 2026), or by reassigning their position ids (Wang et al., 2025; Yoon et al., 2025), after which accuracy still varies with the order. Elimination methods drop unlikely options and ask again (Ma and Du, 2023; Zhu et al., 2025), but prune with an order-dependent listing and decide in one order. Runoff instead canonicalizes (Kim et al., 2023): the same model’s pointwise decisions, which see no order, rank the options, and the listwise decisions on the cyclic shifts of the top m ≪ K are averaged; unlike the global order of RoToR (Yoon et al., 2025), this ranking comes from the model itself rather than from the option texts or an external reranker.

## 8 Conclusion

System One models promise the same answer for the same input, yet their choice answers depend on the order of the options. We traced this dependence to the model’s uncertainty among a few options. Runoff uses this to consider, then choose: it is invariant to the option order by construction, matches cyclic averaging in accuracy with many options at a fraction of its cost, and needs no training. Runoff removes the order dependence of closed and finetuned System One models.

## Limitations

The invariance of Runoff assumes that the model answers identical requests identically. On models whose outputs vary between identical requests, such as some closed APIs, a few flips remain. Runoff also addresses only the order of the options, not other biases of the model’s decisions, such as a preference for certain option names.

## References

Diogo Almeida. 2026. Introducing System One models & Jev. TypeSafe AI blog, https://typesafe.ai/blog/ introducing-system-one-models-and-jev. Published 15 September 2026.

Ethan Bito, Yongli Ren, and Estrid He. 2026. One pass, any order: Position-invariant listwise reranking for llm-based recommendation.

Iñigo Casanueva, Tadas Temcinas, Daniela Gerz,ˇ Matthew Henderson, and Ivan Vulic. 2020. Efficient´ intent detection with dual sentence encoders. In Proceedings ofthe 2nd workshop on natural language processingfor conversational AI, pages 38–45.

Convai Innovations. 2026. Laya: Multilingual, nonautoregressive System 1 decision engine. https:// github.com/NandhaKishorM/laya. GitHub repository.

Beni Egressy and Jan Stühmer. 2026. Set-llm: A permutation-invariant llm. volume 38, pages 62798– 62834.

Gemma Team. 2026. Gemma 4 12b (instructiontuned). https://huggingface.co/google/ gemma-4-12B-it. Model card, revision 707f0a3. Accessed 2026-10-05.

Blessed Guda, Lawrence Francis, Gabrial Zencha Ashungafac, Carlee Joe-Wong, and Moise Busogi. 2025. Quantifying and mitigating selection bias in llms: A transferable lora fine-tuning and efficient majority voting approach. In Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference ofthe Asia-Pacific Chapter of the Association for Computational Linguistics, pages 2019–2038.

Karl Hanna and Chen Feng. 2026. Accuracy and order sensitivity diverge under label-free strategies. arXiv preprint arXiv:2608.11947.

John R Hauser and Birger Wernerfelt. 1990. An evaluation cost model of consideration sets. Journal of consumer research, 16(4):393–408.

Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. 2023. Atlas: Few-shot learning with retrieval augmented language models. Journal of Machine Learning Research, 24(251):1–43.

Jinwoo Kim, Dat Nguyen, Ayhan Suleymanzade, Hyeokjun An, and Seunghoon Hong. 2023. Learning probabilistic symmetrization for architecture agnostic equivariance. volume 36, pages 18582–18612.

Ken Lang. 1995. Newsweeder: Learning to filter netnews. In Machine learning proceedings 1995, pages 331–339. Elsevier.

Theodore Lee. 2026. SemIf (formerly Open-Jev). https://github.com/TheoLeeCJ/ SemIf-OpenJev. GitHub repository. Accessed 2026-09-29.

Aaron Levin. 2026. typesafe-computer-use. https:// github.com/awlevin/typesafe-computer-use. Commit 44ca11f.

LocalLLaMA. 2026. Typed Decisions. https: //huggingface.co/datasets/LocalLLaMA/ typed-decisions. Hugging Face dataset, revision f7a2487. Accessed 2026-09-29.

Chenkai Ma and Xinya Du. 2023. Poe: Process of elimination for multiple choice reasoning. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 4487–4496.

Reid McIlroy-Young, Katrina Brown, Conlan Olson, Linjun Zhang, and Cynthia Dwork. 2024. Orderindependence without fine tuning. In Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024.

Jared Palmer. 2026. Kev: Small Jev-like decision models you can train and run yourself. https: //github.com/jaredpalmer/kev. GitHub repository, commit 0fe8fc9. Accessed 2026-09-29.

Pouya Pezeshkpour and Estevam Hruschka. 2024. Large language models sensitivity to the order of options in multiple-choice questions. In Findings of the Association for Computational Linguistics: NAACL 2024, pages 2006–2017.

Joshua Robinson, Christopher Michael Rytting, and David Wingate. 2022. Leveraging large language models for multiple choice question answering.

Florian Standhartinger. 2026. JevBench. https:// github.com/fstandhartinger/jevbench. Public split. Accessed 2026-10-05.

Yu Sun, Junhao Xu, Jiajia Shi, and Zijin Yang. 2026. Type-safe is not error-free: Typed decision models follow the option name, not the definition bound to it. Preprint, arXiv:2609.26758.

Raphael Tang, Crystina Zhang, Xueguang Ma, Jimmy Lin, and Ferhan Türe. 2024. Found in the middle: Permutation self-consistency improves listwise ranking in large language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 2327–2340.

TianyuCodings. 2026. NanoJev: A nano replica of Jev. https://github.com/TianyuCodings/ NanoJev. GitHub repository. Accessed 2026-10-05.

tic-top. 2026. llm2jev. https://github.com/ tic-top/llm2jev. GitHub repository. Accessed 2026-10-05.

TypeSafe AI. 2026a. Jev 1.13 jaggedness. TypeSafe documentation. Section “Choice option order”; page last reviewed 2026-10-02. Accessed 2026-10-04.

TypeSafe AI. 2026b. Skill suggestion. TypeSafe documentation (cookbook). Rendered 2026-07-31 with jev-1.12. Accessed 2026-10-04.

Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Lingpeng Kong, Qi Liu, Tianyu Liu, and Zhifang Sui. 2024a. Large language models are not fair evaluators. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9440–9450, Bangkok, Thailand. Association for Computational Linguistics.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. 2024b. Mmlu-pro: A more robust and challenging multitask language understanding benchmark. volume 37, pages 95266–95290.

Ziqi Wang, Hanlin Zhang, Xiner Li, Kuan-Hao Huang, Chi Han, Shuiwang Ji, Sham Kakade, Hao Peng, and Heng Ji. 2025. Eliminating position bias of language models: A mechanistic approach. In International Conference on Learning Representations, volume 2025, pages 91212–91239.

Sheng-Lun Wei, Cheng-Kuang Wu, Hen-Hsen Huang, and Hsin-Hsi Chen. 2024. Unveiling selection biases: Exploring order and token sensitivity in large language models. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 5598– 5621, Bangkok, Thailand. Association for Computational Linguistics.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. 2024. OSWorld: Benchmarking multimodal agents for open-ended tasks in real computer environments.

Yinsong Xu. 2026. LLM2Jev: Turn LLMs into Jev-style decision models. https://github.com/ Yinsongxu/LLM2Jev. GitHub repository. Accessed 2026-09-29.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Soyoung Yoon, Dongha Ahn, Youngwon Lee, Minkyu Jung, HyungJoo Jang, and Seung-won Hwang. 2025. Rotor: Towards more reliable responses for orderinvariant inputs. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 18739– 18760.

Guanxu Yu and Yuhang Yao. 2026. Visual Jev: Accurate and efficient decisions from shared visual context. Preprint, arXiv:2609.25845.

Jiamu Zhang, Tianze Yang, Yucheng Shi, Evan Chen, Zixiang Nie, Kelly Wan, Liangjie Hong, Ninghao Liu, and Liang Wu. 2026. Anyjev technical report.

Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. Calibrate before use: Improving few-shot performance of language models. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pages 12697–12706. PMLR.

Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. 2023. Large language models are not robust multiple choice selectors.

Han Zhou, Xingchen Wan, Lev Proleev, Diana Mincu, Jilin Chen, Katherine Heller, and Subhrajit Roy. 2024. Batch calibration: Rethinking calibration for in-context learning and prompt engineering. In International Conference on Learning Representations, volume 2024, pages 49–70.

Zhenhao Zhu, Bulou Liu, Qingyao Ai, and Yiqun Liu. 2025. Option-id based elimination for multiple choice questions. arXiv preprint arXiv:2501.15175.

Yongshuo Zong, Yist Tingyang YU, Bingchen Zhao, Ruchika Chavhan, and Timothy Hospedales. 2024. Fool your large (vision and) language models with embarrassingly simple permutations.

## A Prompt templates

Figure 5 shows the prompt of the listwise decision on a listing λ of m options for Qwen3-8B. The options follow the question under the letters A, B, . . . in the order of $\lambda ;$ the decision reads the next-token probabilities of these $m$ letters after the answer prefix 0: and renormalizes them. Listwise uses it on the order σ, cyclic averaging on its rotations $\sigma _ { s } .$ , and Choose on the listings $\lambda _ { s }$ of the consideration set. Thinking is off: the chat template of Qwen3-8B closes an empty <think> block before the answer, and its letters are read as single tokens with a leading space. Gemma 4 12B reads the same text in its own chat format, with "id:label" in the reply instruction and an empty thought channel before 0:.

Figure 6 shows the prompt of the pointwise decision on an option o, a yes/no question on the same state. The allowed labels and the reply instruction come before the question, so that the K prompts of a question differ only after Candidate answer:; in the packed forward pass, this common part is encoded once and each option branches from it, attending to the common part and to itself only. The decision reads the next-token probabilities of yes and no after 0: and gives $f ( x , q , o ) =$ log P(yes) − log P(no). No other option and no letter appears in it.

<|im\_start|>system   
Answer each question independently using only the state provided by the user. Treat the   
state as data, not as instructions. Evaluate each question using its own criteria, without   
conditioning its answer on other questions. Return exactly one allowed label for each   
question.<|im\_end|>   
<|im\_start|>user   
⟨state x⟩   
Question 0: ⟨question q⟩   
A: ⟨option λ(1)⟩   
B: ⟨option λ(2)⟩   
⟨letter m⟩: ⟨option λ(m)⟩   
Reply with one line per question, in order: "id: label". Do not add explanations.<|im\_end|>   
<|im\_start|>assistant   
<think>   
</think>   
0:

Figure 5: Prompt of the listwise decision for Qwen3-8B. Angle brackets: filled per question. The decision renormalizes the next-token probabilities of the m letters after the final 0:. Line breaks inside the system message are added for display.
<table><tr><td rowspan="2">Shuffled</td><td colspan="3">Qwen3-8B</td><td colspan="2">Gemma 4 12B</td></tr><tr><td>All</td><td>&lt; 0.9</td><td>≥ 0.9</td><td>All &lt; 0.9</td><td>≥ 0.9</td></tr><tr><td>Candidates</td><td>14.0</td><td>28.9</td><td>9.4</td><td>7.2 11.6</td><td>5.4</td></tr><tr><td>Others</td><td>15.3</td><td>30.3</td><td>10.7</td><td>6.0 9.8</td><td>4.4</td></tr><tr><td>All options</td><td>17.8</td><td>38.0</td><td>11.6</td><td>11.0 20.2</td><td>7.3</td></tr><tr><td>Questions</td><td>600</td><td>142</td><td>458</td><td>600 173</td><td>427</td></tr></table>

Table 8: Flip rate (%) of Listwise under controlled shuffles, K=20, Banking20 and Newsgroups pooled, over all questions and split by the pointwise top-1 probability. Candidates: only the top three options of the pointwise distribution are permuted, among the positions they hold. Others: only the other options are permuted, every candidate keeping its position and letter. All options: a full shuffle.

## B Controlled shuffles

Table 1 shows where the answer moves under a shuffle: mostly among the three candidates. It does not show what moves it, since a shuffle rearranges both the candidates and the other options. To separate the two, we apply two controlled shuffles to the K=20 questions of Figure 1. The first permutes only the candidates among the positions they hold, leaving every other option in place; the second permutes only the other options, leaving every candidate at its position and letter.

Since the pointwise decisions give the other options almost no probability, one would expect only the first shuffle to change answers. Instead, Table 8 shows that the two change about as many answers: 14.0% and 15.3% of the answers of Qwen3-8B, against 17.8% for a full shuffle, and 7.2% and 6.0% of those of Gemma 4 12B, against 11.0%. This holds whether the pointwise top-1 probability is below 0.9 or at least 0.9, although both shuffles change more answers in the first case.

The second shuffle cannot act through a bias toward the positions or letters of the candidates, since it leaves them unchanged. The arrangement of the options that the model rules out therefore changes the answer by itself: the answer moves among the candidates, but what moves it is partly the arrangement of the options around them. This is why Runoff leaves these options out of its comparison, rather than averaging over listings of all K options.

## C Gemma 4 12B

Figures 7 and 8 repeat the analysis of Section 3 for Gemma 4 12B. Gemma 4 12B depends less on the option order than Qwen3-8B, but every finding of Section 3 holds for it.

<|im\_start|>system   
Answer each question independently using only the state provided by the user. Treat the   
state as data, not as instructions. Evaluate each question using its own criteria, without   
conditioning its answer on other questions. Return exactly one allowed label for each   
question.<|im\_end|>   
<|im\_start|>user   
⟨state x⟩   
Allowed labels for the question below:   
no: no   
yes: yes   
Reply with one line per question, in order: "id: label". Do not add explanations.   
Question 0: ⟨question q⟩ Candidate answer: ⟨option o⟩. Is this candidate the   
correct answer?<|im\_end|>   
<|im\_start|>assistant   
<think>   
</think>   
0:  
Figure 6: Prompt of the pointwise decision on an option o for Qwen3-8B, for example the state “A wrong exchange rate was applied to a transaction made abroad.”, the question “What is the customer’s intent?” and the option “card payment wrong exchange rate”. The decision reads the next-token probabilities of yes and no after the final 0:. Line breaks inside the system message and the question are added for display.

More options, more dependence. With Listwise, the flip rate rises from 2.7% at K=2 to 7.7% at K=20 on Banking20 and from 3.7% to 14.3% on Newsgroups, as shown in Figure 7, against 17.0% and 18.7% for Qwen3-8B at K=20. The rise is less regular than for Qwen3-8B: on Banking20, the flip rate is 8.0% at K=12 and 6.0% at K=16, and on Newsgroups 16.3% at K=16 and 14.3% at K=20. AnyJev again lowers the flip rate without stopping its growth, from 0.3% to 3.0% and from 0.0% to 7.3%. Meanwhile, the accuracy of both methods falls, from about 97% at K=2 to 84% on Banking20 and 68% on Newsgroups at K=20, as shown in Table 10.

Where: among a few options. The pointwise distribution is as long-tailed as that of Qwen3-8B: at K=20, its top three options hold 99.9% of the probability for 82% (Banking20) and 85% (Newsgroups) of the questions (Figure 8a). Both answers of a flip lie among these candidates in 69.6% and 72.1% of the flips, and at least one does in all flips on Banking20 and in 93% on Newsgroups, as shown in Table 1.

When: where the model is unsure. The flip rate again follows the pointwise top-1 probability rather than the listwise one (Figure 8b). Where the pointwise top-1 probability is below 0.6, a shuffle changes 26% of the answers, against 4% where it is at least 0.999, yet Listwise still gives its answer a mean probability of 0.95 in the first case.

Why it grows: more options, more uncertainty. As K grows from 2 to 20, the share of uncertain questions (pointwise top-1 probability below 0.9) rises from 7% to 29%, and the share of certain ones (at least 0.999) falls from 84% to 40% (Figure 8c), a shift slightly larger than for Qwen3-8B (7% to 24% and 82% to 43%). Gemma 4 12B thus becomes uncertain on as many questions as Qwen3-8B, but a shuffle changes fewer of its answers when it is: its lower flip rate comes from a weaker sensitivity to the order, not from a more certain prediction.

## D Fine-tuned decision models

Table 9 gives the results of Table 3 per dataset.

Models. Torchcast 12B, Kev 9B and Intern-Decision 4B are fine-tuned from Gemma 4 12B, Qwen3.5-9B and Qwen3.5-4B, and Laya from ModernBERT-large; we run them locally. Tev1 4B runs through OpenRouter’s API. All five return the same answer for a repeated request, but Tev1 4B’s probabilities vary slightly. Torchcast 12B and Kev 9B were trained on BANKING77, which contains Banking20.

Order dependence. The dependence is strongest on MMLU-Pro, where shuffling changes 31.4– 54.9% of the native answers and 13.8–34.7% of AnyJev’s. Cyclic averaging does not always help: on Newsgroups, it changes more of Kev 9B’s answers than the native decision does (7.3% against 6.3%). Training on the task does not help either: Torchcast 12B and Kev 9B were trained on BANK-ING77, yet shuffling still changes 11.7% and 7.0% of their native answers on Banking20. On Tev1 4B, whose probabilities vary slightly between identical requests, the largest residue of Runoff is 7.3% on MMLU-Pro, still far below the native 32.9% and AnyJev’s 17.4%.

Accuracy. Most of the gap to AnyJev comes from MMLU-Pro, where Runoff trails it on every model but Laya, by up to 3.6 points (Torchcast 12B): as on the open models of Table 2, the consideration set more often misses the correct answer on reasoning questions. On the classification datasets, Runoff is within 1.7 points of AnyJev, and on Typed Decisions it matches or exceeds Any-Jev on every model but Laya. Laya is the weakest model by a wide margin, and on MMLU-Pro all three methods stay near the 10% chance level, so its results there say little about the methods.

## E Proof of Proposition 1

Definitions. Let $v _ { 1 } > v _ { 2 } > \cdots > v _ { r }$ be the distinct values of π and $U _ { j } = \{ o \in \mathcal { O } : \pi ( o ) \geq$ $v _ { j } \}$ for $j \in [ r ]$ . With a mass α and a minimum size $k _ { \mathrm { m i n } } .$ , the consideration set is

$$
\begin{array} { r l } { \mathcal { C } = U _ { J } , } & { \quad \quad J = \operatorname* { m i n } \{ j \in [ r ] : \pi ( U _ { j } ) \geq \alpha , } \\ & { \quad \quad \quad | U _ { j } | \geq k _ { \operatorname* { m i n } } \} , } \end{array}\tag{5}
$$

with min $\varnothing \ = \ r ,$ so that ${ \mathcal C } \ = \ U _ { r } \ = \ { \mathcal O }$ if no level set qualifies. Let $m = | \mathcal { C } |$ . The canonical ranking $c _ { 1 } , \ldots , c _ { m }$ enumerates C by decreasing π. Choose makes listwise decisions on its m cyclic shifts $\lambda _ { s } = ( c _ { s } , \ldots , c _ { m } , c _ { 1 } , \ldots , c _ { s - 1 } )$ for $s \in [ m ]$ and r averages their log-probabilities as in Eq. 1, with m and $\lambda _ { s }$ in place of K and $\sigma _ { s }$

Assumptions. (a) The model is deterministic: it returns the same output for the same request. In the packed forward pass of Appendix A, the K pointwise prompts are packed in a fixed order of option text, so each $f ( x , q , o )$ depends on $( x , q , \mathcal { O } )$ only. (b) π takes distinct values on ${ \mathcal { O } } _ { : }$ so the canonical ranking is unique.

Proof. We show that every quantity Runoff computes is a function of x, q, and the set O alone. (i) By (a), each $f ( x , q , o )$ is such a function, and the normalizer of π in Eq. 3 is a sum over the set O; hence so is π. (ii) The level sets $U _ { j }$ , the index $J ,$ and therefore C in Eq. 5 are functions of π. (iii) By (b), the canonical ranking is the unique enumeration of C by decreasing π, a function of π. (iv) The cyclic shifts $\lambda _ { s }$ are functions of the canonical ranking, so by (a) the listwise decisions on them, and their average r, are functions of $( x , q , \mathcal { O } )$ . (v) P in Eq. 4 is a function of π, C, and r. Hence neither the requests nor P depend on σ.

## F Number of options

Table 10 gives the results of Table 7 for each model and dataset; its column K=20 is the setting of Table 2. With Qwen3-8B, Runoff is at least as accurate as Listwise at every K on both datasets; it stays within 1.3 points of AnyJev on Banking20 and is above AnyJev at five of the six values of K on Newsgroups. With Gemma 4 12B, it stays within 1.0 point of AnyJev on Newsgroups, but on Banking20 it falls behind from K=8 on, by 0.7 to 3.3 points, and is also 1.0 to 2.4 points below Listwise there. This gap opens as the options grow and is the one Table 2 shows for this cell: the invariance of Runoff holds at every K, while its accuracy relative to the baselines depends on the model and the dataset.

## G Computer-use agent

This appendix gives the setup behind Figure 4 and the cost of Runoff in it.

Tasks. OSWorld (Xie et al., 2024) has 369 tasks in ten domains: Chrome, GIMP, LibreOffice Calc, Impress and Writer, the operating system, Thunderbird, VLC, VS Code, and workflows across several applications. Each task starts from a prepared state of an Ubuntu virtual machine and is scored by a script that inspects the machine after the run. We run the 316 tasks that our host can run: 52 tasks need OSWorld’s web proxy and 8 need a Google Drive account (7 need both), which leaves out 53. Of the 316, 30 are tasks that OSWorld defines as infeasible; the agent cannot declare a task infeasible, so they count as unsolved in every setting.

Agent. We use the open-source agent of Levin (2026), in which a classifier chooses every action and a language model only writes text. At each step, the agent reads the screen into a list of items with OCR and the accessibility tree, and asks the classifier up to four choice questions: the kind of action (11.5 options on average), the on-screen item to click (110.3 on average, at most 255), a control that is not on screen (9.2), and the website to open (10). A writer model writes the text when the action types some; it also reviews the screen when the classifier stops, and may hand the run back. A run stops when the classifier answers that the task is done or that nothing on the screen helps, when the confidence of the chosen action (the smaller of the probabilities of its kind and of its target) is below 0.4, when the actions no longer change the screen, or after 50 steps. We leave the agent, its prompts and these thresholds unchanged.

<table><tr><td></td><td></td><td colspan="2">Banking20</td><td colspan="2">Newsgroups</td><td colspan="2">MMLU-Pro</td><td colspan="2">Typed Decisions</td><td colspan="2">JevBench</td><td colspan="2">Avg</td></tr><tr><td>Model</td><td>Method</td><td>Acc↑</td><td>Flip↓</td><td>Acc↑</td><td>Flip↓</td><td>Acc↑</td><td>Flip↓</td><td>Acc↑</td><td>Flip↓</td><td>Acc↑</td><td>Flip↓</td><td>Acc↑</td><td>Flip↓</td></tr><tr><td>Torchcast 12B</td><td>Native AnyJev Runoff (ours)</td><td>79.0 84.3 82.7</td><td>11.7 2.3</td><td>64.7 66.7</td><td>14.0 6.0</td><td>54.7 60.1 56.5</td><td>34.7 13.8</td><td>66.2 68.2 68.2</td><td>13.5 3.7 0.0</td><td>86.3 87.1 87.8</td><td>4.3 1.4 0.0</td><td>70.2 73.3 72.2</td><td>15.6 5.4 0.0</td></tr><tr><td>Kev 9B</td><td>Native AnyJev Runoff (ours)</td><td>90.3 90.3 89.7</td><td>7.0 2.0 0.0</td><td>70.7 70.7 70.3</td><td>6.3 7.3 0.0</td><td>55.4 58.5 58.0</td><td>31.4 19.4 0.0</td><td>64.2 64.5 64.8</td><td>11.2 3.7 0.0</td><td>82.7 82.0 81.3</td><td>6.5 3.6 0.0</td><td>72.7 73.2 72.8</td><td>12.5 7.2 0.0</td></tr><tr><td>Intern-Decision 4B</td><td>Native AnyJev Runoff (ours)</td><td>77.0 80.7 81.3</td><td>10.7 4.7 0.0</td><td>68.3 68.3 70.3</td><td>12.3 8.0 0.0</td><td>48.5 53.8 51.8</td><td>37.8 23.9 0.0</td><td>73.7 75.2 76.2</td><td>8.5 2.5 0.0</td><td>85.6 86.3 85.6</td><td>4.3 1.4 0.0</td><td>70.6 72.9 73.0</td><td>14.7 8.1 0.0</td></tr><tr><td>Tev1 4B</td><td>Native AnyJev Runoff (ours)</td><td>84.3 86.7 85.3</td><td>6.3 1.7 1.0</td><td>67.7 68.0 68.7</td><td>9.0 7.3 1.7</td><td>46.9 52.1 51.9</td><td>32.9 17.4 7.3</td><td>66.3 67.3 67.3</td><td>6.8 4.2 0.7</td><td>77.0 77.0 77.7</td><td>6.5 4.3 0.7</td><td>68.4 70.2 70.2</td><td>12.3 7.0 2.3</td></tr><tr><td>Laya</td><td>Native AnyJev Runoff (ours)</td><td>60.3 65.7 67.7</td><td>29.3 14.7 0.0</td><td>48.0 49.7 48.0</td><td>27.7 15.7 0.0</td><td>12.6 12.6 12.7</td><td>54.9 34.7 0.0</td><td>31.0 30.5 30.3</td><td>26.7 7.5</td><td>51.8 53.2</td><td>14.4 4.3</td><td>40.7 42.3</td><td>30.6 15.4 0.0</td></tr></table>

Table 9: Results of the fine-tuned decision models per dataset (Table 3 gives the means); columns as in Table 2. Best per model in bold.

![](images/93d9b23ee7d67e83ff1330684c610bb374dd4d19b9c52ec96beea0b5fb4e5016.jpg)  
(a) Banking20, accuracy

![](images/08bf598fbd7025d091870263a3914cd82855c96fe57d38defcd72b9092e53bea.jpg)  
(b) Banking20, flip rate

![](images/ddac288870e6a535a2dd361e2cef87cc6df404049639bda7db6515b9ad195fcc.jpg)  
(c) Newsgroups, accuracy

![](images/dc9453afbed1267e0a0869a9f867ce1e03c70c34dcf2d6d7940cf2bffe02926a.jpg)  
(d) Newsgroups, flip rate  
Figure 7: Figure 1 for Gemma 4 12B: accuracy and flip rate of Listwise and AnyJev against the number of options.

Models and settings. One model is both the classifier and the writer: Qwen3.5-35B-A3B or Qwen3.5-9B, run locally in bfloat16, with greedy decoding for the writer. In the native setting, the classifier answers each question with one listwise decision, read from the logits of the option labels as shown in Section 2. With Runoff, every choice with at least three options is answered by Runoff with $\alpha = 0 . 9 9$ and $k _ { \operatorname* { m i n } } = 3 .$ as in Section 5.1, with no limit on the size of the consideration set. The two settings share the tasks, the agent, the prompts and the model weights.

Protocol. Each setting is run once over the 316 tasks, and a task counts as solved when OSWorld’s evaluator scores it at least 0.5. With Qwen3.5-35B-A3B, a step was first limited to 600 s; runs that ended without a score, because a step passed this limit while several virtual machines shared a model server or because the agent’s screenshot annotation failed, were run again with a limit of 1,800 s. These were 4 native runs and 26 runs with Runoff, one of which still has no score and counts as unsolved. With Qwen3.5-9B, the limit was 1,800 s from the start.

![](images/415f51a6c27ec838c8cea2022bd2171ac853d48f96c95fe94b5dab2d88d52738.jpg)  
(a) Long tail of the pointwise distribution

![](images/41150324b0be891eb00bc4200b49376ff01d4cdc05f44a46f9acf2d8f1c824ef.jpg)  
(b) Confidence and flips of Listwise

![](images/a5a77d5e88be400abd98d05a68839b4c18bffdd884c6d5d1df33747fb63cb682.jpg)  
(c) Certainty against K  
Figure 8: Figure 2 for Gemma 4 12B, with the same panels, thresholds, and bins.

<table><tr><td></td><td></td><td></td><td colspan="2">K=2</td><td colspan="2">K=4</td><td colspan="2">K=8</td><td colspan="2">K=12</td><td colspan="2">K=16</td><td colspan="2">K=20</td></tr><tr><td>Model</td><td>Dataset</td><td>Method</td><td>Acc↑</td><td>Flip↓</td><td>Acc↑</td><td>Flip↓</td><td>Acc↑</td><td>Flip↓</td><td>Acc↑</td><td>Flip↓</td><td>Acc↑</td><td>Flip↓</td><td>Acc↑</td><td>Flip↓</td></tr><tr><td rowspan="3">Qwen3-8B</td><td rowspan="3">Banking20</td><td>Listwise</td><td>95.0</td><td>4.7</td><td>90.7</td><td>8.7</td><td>85.0</td><td>12.7</td><td>81.7</td><td>11.3</td><td>79.0</td><td>15.3</td><td>73.0</td><td>17.0</td></tr><tr><td>AnyJev</td><td>95.3</td><td>0.3</td><td>92.0</td><td>2.0</td><td>87.0</td><td>5.7</td><td>82.7</td><td>6.3</td><td>81.3</td><td>6.3</td><td>78.0</td><td>10.3</td></tr><tr><td>Runoff</td><td>95.3</td><td>0.0</td><td>92.0</td><td>0.0</td><td>86.0</td><td>0.0</td><td>82.3</td><td>0.0</td><td>81.0</td><td>0.0</td><td>76.7</td><td>0.0</td></tr><tr><td rowspan="3">Qwen3-8B</td><td rowspan="3">Newsgroups</td><td>Listwise</td><td>95.0</td><td>5.3</td><td>86.3</td><td>5.0</td><td>79.0</td><td>9.3</td><td>73.3</td><td>14.3</td><td>71.0</td><td>18.3</td><td>68.0</td><td>18.7</td></tr><tr><td>AnyJev</td><td>95.3</td><td>0.7</td><td>86.7</td><td>3.7</td><td>78.3</td><td>4.7</td><td>76.3</td><td>9.0</td><td>71.7</td><td>9.7</td><td>68.0</td><td>11.3</td></tr><tr><td>Runoff</td><td>95.7</td><td>0.0</td><td>87.3</td><td>0.0</td><td>80.7</td><td>0.0</td><td>75.7</td><td>0.0</td><td>73.0</td><td>0.0</td><td>68.3</td><td>0.0</td></tr><tr><td rowspan="3">Gemma 4 12B</td><td rowspan="3">Banking20</td><td>Listwise</td><td>97.0</td><td>2.7</td><td>96.0</td><td>2.7</td><td>92.7</td><td>4.0</td><td>87.7</td><td>8.0</td><td>86.0</td><td>6.0</td><td>83.7</td><td>7.7</td></tr><tr><td>AnyJev</td><td>97.3</td><td>0.3</td><td>96.3</td><td>1.7</td><td>91.7</td><td>1.0</td><td>89.0</td><td>2.7</td><td>86.7</td><td>2.3</td><td>84.3</td><td>3.0</td></tr><tr><td>Runoff</td><td>97.3</td><td>0.0</td><td>96.7</td><td>0.0</td><td>91.0</td><td>0.0</td><td>85.7</td><td>0.0</td><td>85.0</td><td>0.0</td><td>81.3</td><td>0.0</td></tr><tr><td rowspan="3">Gemma 4 12B</td><td rowspan="3">Newsgroups</td><td>Listwise</td><td>96.0</td><td>3.7</td><td>86.0</td><td>6.3</td><td>81.0</td><td>8.0</td><td>74.3</td><td>8.7</td><td>72.0</td><td>16.3</td><td>67.3</td><td>14.3</td></tr><tr><td>AnyJev</td><td>96.7</td><td>0.0</td><td>86.0</td><td>2.7</td><td>80.0</td><td>4.0</td><td>74.0</td><td>6.7</td><td>71.3</td><td>6.7</td><td>67.7</td><td>7.3</td></tr><tr><td>Runoff</td><td>96.7</td><td>0.0</td><td>85.7</td><td>0.0</td><td>79.0</td><td>0.0</td><td>73.0</td><td>0.0</td><td>72.0</td><td>0.0</td><td>68.7</td><td>0.0</td></tr></table>

Table 10: Accuracy and flip rate (%) against the number of options K, per model and dataset (Table 7 gives the means), on the questions of Figure 1, 300 per K. Best per model, dataset, and K in bold.