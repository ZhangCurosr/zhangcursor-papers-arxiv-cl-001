# A TRAINING CRITERION WITH TOKEN-LEVEL TOLERANCE TO TRANSCRIPTION AMBIGUITY FOR AUTOMATIC SPEECH RECOGNITION

Saurabh Kumar, Diptiman Mohanta, Prasanta Kumar Ghosh

Department of Electrical Engineering, Indian Institute of Science (IISc), Bangalore, India {saurabhk0317, diptimanmohanta7}@gmail.com, prasantg@iisc.ac.in

## ABSTRACT

Automatic speech recognition is typically trained assuming that the reference transcript is the only valid labeling of an utterance, yet even nominally verbatim transcripts contain localized differences in pronunciation, spelling, or lexical realization that the acoustics do not uniquely determine. Omni-temporal Classification (OTC) tolerates such noise by adding wildcard paths to the connectionist temporal classification (CTC) alignment graph, but its word-level arcs are too coarse, since bypassing one unsupported token discards supervision for the whole word. We move wildcard arcs to token granularity so unsupported tokens can be bypassed while the rest of the word stays supervised, and we combine token- and wordlevel arcs as complementary escape paths. Across 19 languages and three corpora, token-level OTC improves over CTC on all 25 tasks. We also replace epoch-indexed relaxation of the wildcard weights with a predictive-entropy-indexed schedule, which performs comparably while reducing dependence on training length. Combining this schedule with the hybrid graph gives the lowest mean word error rate (WER) on every corpus and a 9.45% average relative WER reduction over CTC. Independent validator transcriptions show that token-level models place significantly more wildcard-bypass probability than CTC on disputed characters, indicating that token-level tolerance targets localized transcript ambiguity.

Index Terms— Weakly supervised learning, connectionist temporal classification, weighted finite-state transducers, alignment tolerance, speech recognition.

## 1. INTRODUCTION

End-to-end automatic speech recognition (ASR) models trained with connectionist temporal classification (CTC) [1] sum over alignments that collapse to a single reference transcript, treating it as the only valid labeling. Yet even verbatim transcripts contain choices the acoustics do not settle, such as spelling variants, vowel-length or nasalization marks in Indic scripts, or inflections reduced in fast speech. Annotators often disagree on one or two characters of a word while agreeing on the rest, which ASR evaluation increasingly accounts for through multiple references or permissible spelling variants [2–4]. CTC training, however, still supervises every position as if the reference were certain, pushing the model toward labels the audio may not support.

Error-tolerant objectives for CTC and transducers [5–7] address several kinds of label noise. STC [8] and W-CTC [9] handle missing parts of the transcript, graph-based temporal classification [10] and alternative pseudo-labeling [11] handle uncertain pseudo-labels, and token-weighted RNN-T [12] down-weights unreliable tokens. Bypass temporal classification [13] and its extension, Omni-temporal

Classification (OTC) [14], add a wildcard ⋆ to the weighted finitestate transducer (WFST) alignment graph [15, 16], whose bypass arcs skip unsupported transcript units and self-loops absorb unexplained frames; similar arcs between adjacent tokens also make keyword spotting robust to noise [17]. These methods mostly target heavy label corruption or acoustic noise, whereas we focus on verbatim transcripts, where the reference is largely correct and disagreement is sparse and local.

Published OTC uses word-level arcs with weights tuned on transcripts containing 50% simulated errors and performs similarly to CTC on verbatim data [14]. To our knowledge, no OTC-related work reports a gain over CTC on verbatim LibriSpeech. Re-tuning the weights on clean speech helps only slightly since skipping one unsupported token still discards supervision for the whole word. We therefore place wildcard arcs at the token level so that a local deviation can be bypassed while the remaining tokens stay supervised, and we add word-level bypasses as a second escape path for wholly unsupported words. We also relax the wildcard weights by using held-out predictive entropy rather than epoch count to reduce dependence on training length.

WER gains alone do not show that this tolerance is used where transcripts are actually ambiguous. We therefore obtained independent transcriptions of the RESPIN [18] development and test sets, for which only a single reference is released, from three external validation vendors. Disputed characters and words receive substantially more bypass probability than agreed ones, and token-level models assign significantly more of it than CTC to disputed characters while leaving agreed characters nearly unchanged. This indicates that token-level tolerance is associated with localised transcript ambiguity rather than an indiscriminate relaxation of the objective.

Our contributions can be summarised as follows.

• Token-level wildcard arcs for OTC-based ASR training, which improve WER over CTC on all 25 tasks across 19 languages and three corpora, whereas existing word-level OTC performs similarly to CTC on LibriSpeech and RESPIN.

• A hybrid graph combining token- and word-level bypasses, trained with an entropy-indexed relaxation schedule, which gives the lowest mean WER on every corpus and a 9.45% average relative reduction over CTC.

• An analysis based on independent validator transcriptions of RESPIN, showing that token-level models become more tolerant specifically at characters where annotators disagree, which links the added tolerance to genuine transcript ambiguity in verbatim data.<sup>1</sup>

## 2. BACKGROUND

Let $\mathbf { y } = [ y _ { 1 } , \dots , y _ { U } ]$ be a transcript over vocabulary V including the CTC blank ∅, and let $P _ { t } ( v )$ be the model posterior of $\ u _ { 1 } \in \mathcal { V }$ at frame t of a T-frame utterance. CTC marginalises over the framelevel alignments that collapse to y, obtained by composing the CTC topology with a linear acceptor of y [1, 15, 16], so every valid path emits the transcript exactly (Fig. 1a).

## 2.1. Omni-temporal Classification

OTC [14] adds a wildcard ⋆ to this graph. For a segmentation ${ \bf y } =$ $u _ { 1 } \circ \cdots \circ u _ { K }$ with states $0 , \ldots , K$ , states $k - 1$ and k are joined by an arc emitting $u _ { k }$ and a parallel bypass arc labelled ⋆ of log-weight $w ^ { \mathrm { ( b y p ) } }$ , and every state carries a ⋆ self-loop of log-weight $w ^ { \mathrm { ( s e \bar { l } f ) } } ;$ wildcard emissions are scored by the mean non-blank posterior [8]. Let $\Pi ( \mathbf { y } )$ denote the alignments admitted by this graph composed with the CTC topology, and let $n _ { \mathrm { b y p } } ( \pi )$ and $n _ { \mathrm { s e l f } } ( \pi )$ count bypass and self-loop arcs on a path π, giving the wildcard-arc cost $\phi ( \pi ) =$ $n _ { \mathrm { b y p } } ( \pi ) w ^ { ( \mathrm { b y p } ) } + n _ { \mathrm { s e l f } } \bar { ( } \pi ) w ^ { ( \mathrm { s e l f } ) }$ <sup>)</sup>. The OTC objective is

$$
\mathcal { L } _ { \mathrm { O T C } } = - \log \sum _ { \pi \in \Pi ( \mathbf { y } ) } \left( \prod _ { t = 1 } ^ { T } P _ { t } ( \pi _ { t } ) \right) e ^ { \phi ( \pi ) } ,\tag{1}
$$

which reduces to CTC as both wildcard weights tend $\mathrm { t o \ - } \infty$ and wildcard paths are suppressed. The weights start strongly negative, so the transcript is trusted early in training, and are relaxed geometrically with epoch e:

$$
w ^ { ( j ) } ( e ) = w _ { 1 } ^ { ( j ) } \tau _ { j } ^ { e - 1 } , \quad j \in \{ \mathrm { b y p } , \mathrm { s e l f } \} , \quad \tau _ { j } \in ( 0 , 1 ) .\tag{2}
$$

In [14] each $u _ { k }$ is a word, so a bypass spans a whole word and selfloops occur only at word boundaries (Fig. 1b). More generally, the segmentation determines the wildcard granularity, which we move to token level in Sec. 3.1.

## 3. METHOD

## 3.1. Arc granularity

We target localized transcript variation, where a word differs from the acoustics in only one or a few tokens because of pronunciation, spelling, or inflectional variation, which motivates placing wildcard arcs at token rather than word granularity. Whereas the published OTC-W<sub>e</sub> formulation treats each word as one unit, OTC-T<sub>e</sub> sets $K = U$ , placing a bypass arc between every pair of consecutive tokens and a self-loop at every state (Fig. 1c); word boundaries are obtained from the SentencePiece word-start marker for BPE units and from the space symbol for character units.

Token-level arcs provide finer resolution and retain supervision: a single unsupported token can be bypassed without skipping the whole word, leaving the remaining tokens on the alignment path. They also make the escape cost less dependent on word length, since OTC-W replaces all n token emissions of a bypassed word with one ⋆ at a fixed penalty, whereas token-level arcs incur a cost that scales with the number of bypassed tokens. The two are therefore complementary: a wholly unsupported word can be handled by one wordlevel bypass, and a partially unsupported word by token-level bypasses. We combine both in OTC-TW, which adds one word bypass arc per word to the token-level graph (Fig. 1d). The self-loop absorbs extra audio frames and doesn’t need a separate word-level version because the token-level self-loop can already absorb any number of frames.

![](images/a3922ea500a6c156f3c0c75c8519da05d43be0dcc52f4ded944b1e15caa6c9f5.jpg)  
Fig. 1. Transcript graphs for words $w _ { 1 } ~ = ~ \left( y _ { 1 } , y _ { 2 } \right)$ and $\begin{array} { r l } { w _ { 2 } } & { { } = } \end{array}$ $( y _ { 3 } , y _ { 4 } )$ . (a) CTC. (b) OTC-W: wildcard arcs at word boundaries. (c) OTC-T: wildcard arcs at every token. (d) OTC-TW: (c) plus one word-level bypass per word.

## 3.2. Entropy-indexed decay

The geometric decay in Eq. (2) uses epoch as a proxy for training progress, so a schedule tuned for one training budget need not reach the same learning stage under another. We instead measure progress by predictive entropy on held-out data, computed without gradients so that it does not reflect memorization of the training transcripts. At each epoch, we remove the CTC blank and compute the normalized non-blank posterior and its entropy at each frame:

$$
q _ { t } ( v ) = \frac { P _ { t } ( v ) } { 1 - P _ { t } ( \emptyset ) } , \qquad H _ { t } = - \sum _ { v \neq \emptyset } q _ { t } ( v ) \log q _ { t } ( v ) .\tag{3}
$$

Averaging with the non-blank mass $\omega _ { t } = 1 - P _ { t } ( \emptyset )$ and normalizing by $\log ( V - 1 )$ , where $V = | \nu |$ , gives $\widehat { H } _ { e } \in [ 0 , 1 ]$

$$
\widehat { H } _ { e } = \frac { 1 } { \log ( V - 1 ) } \frac { \sum _ { t \in \mathcal { D } _ { \mathrm { v a l } } } \omega _ { t } H _ { t } } { \sum _ { t \in \mathcal { D } _ { \mathrm { v a l } } } \omega _ { t } } .\tag{4}
$$

Let $H _ { 0 }$ be $\widehat { H } _ { \epsilon }$ at the end of the first epoch. We smooth the entropy as $\bar { H } _ { e } = \alpha \widehat { H } _ { e } + ( 1 - \alpha ) \bar { H } _ { e - 1 }$ and convert it to progress using the reciprocal $r ( h ) = 1 / ( h + \epsilon )$

$$
p _ { e } = \operatorname* { m i n } \biggl \{ 1 , \operatorname* { m a x } \biggl \{ 0 , \frac { r ( \bar { H } _ { e } ) - r ( H _ { 0 } ) } { r ( H _ { \mathrm { l o w } } ) - r ( H _ { 0 } ) } \biggr \} \biggr \} .\tag{5}
$$

With $\beta _ { e } = p _ { e } ^ { \kappa }$ , the wildcard weights are interpolated geometrically between their initial and final magnitudes:

$$
w ^ { ( j ) } ( e ) = - \left| w _ { 1 } ^ { ( j ) } \right| ^ { 1 - \beta _ { e } } \left| w _ { \infty } ^ { ( j ) } \right| ^ { \beta _ { e } } , \quad j \in \{ \mathrm { b y p , s e l f } \} .\tag{6}
$$

Relaxation thus follows model confidence rather than elapsed epochs.

Table 1. Wildcard-arc settings. w<sub>1</sub>: initial weight; τ: per-epoch decay; E: entropy-indexed schedule (Sec. 3.2).
<table><tr><td rowspan="2">System</td><td colspan="2">Bypass</td><td colspan="2">Self-loop</td><td colspan="2">Word bypass</td></tr><tr><td> $w _ { 1 }$ </td><td>T</td><td> $_ { w _ { 1 } }$ </td><td>T</td><td> $w _ { 1 }$ </td><td>T</td></tr><tr><td>OTC-We</td><td>-19</td><td>0.975</td><td>3.75</td><td>0.999</td><td>一</td><td>一</td></tr><tr><td>OTC-W&#x27;</td><td>-19</td><td>0.925</td><td>-1</td><td>0.975</td><td>一</td><td>一</td></tr><tr><td> $\mathrm { O T C - T } _ { e } ^ { \circ }$ </td><td>-25</td><td>0.775</td><td>-5</td><td>0.775</td><td>1</td><td>一</td></tr><tr><td> $\mathrm { O T C - T } _ { H }$ </td><td>-25</td><td>E</td><td>-5</td><td>E</td><td></td><td></td></tr><tr><td> $\mathrm { O T C - T W } _ { e }$ </td><td>-25</td><td>0.775</td><td>-5</td><td>0.775</td><td>-19</td><td>0.925</td></tr><tr><td> $\mathrm { O T C - T W } _ { H }$ </td><td>-25</td><td>E</td><td>-5</td><td>E</td><td>-19</td><td>E</td></tr></table>

## 4. EXPERIMENTS AND RESULTS

We evaluate our methods on 19 languages from LibriSpeech [19], FLEURS [20], and RESPIN [18]. LibriSpeech uses train-clean-100 for training, dev-clean/other for validation, and test-clean/other for evaluation. FLEURS covers 14 languages across four families, 14 scripts, and diverse morphological types using the published splits. RESPIN uses the official dev/test sets and a 30-hour subset of train <lang> small, restricted to unique sentences. Language codes are listed in Table 2.

All systems use a 12-block E-Branchformer encoder [21] $( d = 2 5 6 ,$ , four heads) in ESPnet2 [22], with WFST losses computed in k2. RESPIN uses 80-dimensional filterbanks for 70 epochs, while FLEURS and LibriSpeech use frozen XLS-R 300M [23] and wav2vec 2.0 Base features [24], respectively, followed by linear projection and 30 epochs. FLEURS/RESPIN use character units, while LibriSpeech uses 200 BPE units. All systems apply SpecAugment [25], three-way speed perturbation, Adam with warmup, and validation-loss checkpoint averaging; only the loss function varies within each corpus. Decoding uses CTC prefix search without an external language model, and we report WER. To test whether the results depend on the encoder, we repeat the LibriSpeech experiments with a 12-block Conformer [26] of the same width and head count, changing only the encoder and adding early stopping (patience 5).

## 4.1. Training criteria

All systems share the same graph construction and WFST forward– backward implementation, differing only in wildcard arcs and schedules (Table 1); CTC disables both arc types. $\mathrm { O T C - W } _ { e }$ uses the published settings [14], raising $| w _ { 1 } ^ { ( \mathrm { b y p } ) } |$ only for divergent runs; $\mathrm { O T C - W } _ { e } ^ { \prime }$ re-tunes them on 10 h of LibriSpeech train-clean-100, and token-level settings come from the sweep in Sec. 4.3. Combined systems reuse the OTC-W<sup>′</sup> word bypass untuned. The entropy-indexed systems use $\alpha = 0 . 3 , H _ { \mathrm { l o w } } = 0 . 0 1 , \kappa = 1 . 3 ,$ and $\epsilon = 1 0 ^ { - 4 }$ throughout, relaxing toward $w _ { \infty } ^ { \mathrm { ( b y p ) } } = - 0 . 0 1$ and $w _ { \infty } ^ { \mathrm { ( s e l f ) } } = - 0 . 0 0 1$ . In $\mathrm { O T C - T W } _ { H }$ , the word bypass follows the same $p _ { e }$ toward $w _ { \infty } ^ { \mathrm { ( w b y p ) } } = - 1$ , so one entropy measurement controls all three arc types. The final weights remain negative, so escapes are never free, and rising held-out entropy makes the weights more restrictive.

## 4.2. Overall results

Table 2 compares the seven criteria on the 25 tasks. With the published settings, OTC- ${ \mathbf { } } . { \mathbf { } } { \mathbf { } } . { \mathbf { } } .$ has a slightly higher mean WER than CTC on LibriSpeech and RESPIN, and five runs required a larger bypass penalty to avoid divergence.<sup>2</sup> Re-tuning on clean speech (OTC-W<sup>′</sup> ) removes the divergence, but the average gain is small and does not carry over to RESPIN. Word-level arcs therefore remain limited on verbatim transcripts, since bypassing one token skips the whole word.

Table 2. WER (%) on 25 tasks; bold marks the row minimum. $W _ { e }$ $W _ { e } ^ { \prime } { : }$ word-level OTC (published, re-tuned weights); $T _ { e } , T _ { H } \mathrm { : }$ tokenlevel OTC (epoch-, entropy-indexed); $T W _ { e } , \bar { T } W _ { H } \colon$ combined. ∆: relative WER reduction (%) of OTC-TW vs. CTC. †: OTC-W diverged (see Sec. 4.2). Lavg, Favg, Ravg: mean WER over LibriSpeech, Fleurs, and RESPIN tasks; Avg: overall mean.
<table><tr><td>Lang.</td><td>CTC</td><td> ${ \bf W } _ { e }$ </td><td> ${ \mathbf { W } _ { e } ^ { \prime } }$ </td><td> $\mathbf { T } _ { e }$ </td><td> $\mathbf { T } _ { H }$ </td><td> $\mathbf { T W } _ { \epsilon }$ </td><td> $\mathbf { T } \mathbf { W } _ { H }$ </td><td> $\Delta$ </td></tr><tr><td>LibriSpeech</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>test-clean</td><td>5.96</td><td>6.13</td><td>5.68</td><td>5.49</td><td>5.51</td><td>5.24</td><td>5.38</td><td>9.73</td></tr><tr><td>test-other</td><td>13.19</td><td>13.91</td><td>13.09</td><td>13.02</td><td>12.99</td><td>12.71</td><td>12.46</td><td>5.53</td></tr><tr><td>FLEURS</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ar-eg</td><td>33.96</td><td>35.70</td><td>34.42</td><td>30.61</td><td>31.38</td><td>30.15</td><td>30.78</td><td>9.36</td></tr><tr><td> $\mathrm { { h y } \mathrm { { - a m } } }$ </td><td>26.56</td><td>23.84</td><td>25.36</td><td>23.25</td><td>22.50</td><td>22.97</td><td>22.89</td><td>13.82</td></tr><tr><td>bn_in</td><td>35.31</td><td>33.60</td><td>35.07</td><td>32.24</td><td>31.14</td><td>31.76</td><td>31.75</td><td>10.08</td></tr><tr><td> $\mathrm { e n \mathrm { _ { - } u s } }$ </td><td>33.83</td><td>32.32</td><td>31.35</td><td>31.35</td><td>31.83</td><td>30.74</td><td>30.65</td><td>9.40</td></tr><tr><td> $_ \mathrm { k a \_ g e }$ </td><td>48.78</td><td>45.75</td><td>47.51</td><td>43.18</td><td>43.53</td><td>43.80</td><td>43.07</td><td>11.71</td></tr><tr><td>el_gr</td><td>39.08</td><td>30.44</td><td>33.05</td><td>33.35</td><td>36.31</td><td>32.52</td><td>32.06</td><td>17.96</td></tr><tr><td>gu_in</td><td>36.49</td><td>34.24</td><td>34.42</td><td>31.92</td><td>31.68</td><td>32.04</td><td>31.85</td><td>12.72</td></tr><tr><td>hi_in</td><td>31.71</td><td>32.33</td><td>31.91</td><td>28.40</td><td>27.80</td><td>27.40</td><td>27.69</td><td>12.68</td></tr><tr><td>kn_in</td><td>35.87</td><td>33.77</td><td>35.62</td><td>31.96</td><td>32.58</td><td>31.97</td><td>31.93</td><td>10.98</td></tr><tr><td>mlLin</td><td>37.48</td><td>35.71†</td><td>37.19</td><td>34.81</td><td>34.74</td><td>34.38</td><td>34.57</td><td>7.76</td></tr><tr><td>pa_in</td><td>39.56</td><td>42.68</td><td>39.18</td><td>36.67</td><td>36.99</td><td>36.25</td><td>36.89</td><td>6.75</td></tr><tr><td>ru_ru</td><td>42.05</td><td>38.53</td><td>40.96</td><td>37.48</td><td>36.94</td><td>36.77</td><td>35.91</td><td>14.60</td></tr><tr><td>ta_in</td><td>54.76</td><td>52.29†</td><td>53.66</td><td>50.72</td><td>54.66</td><td>51.08</td><td>50.23</td><td>8.27</td></tr><tr><td>te_in</td><td>48.01</td><td>44.76</td><td>46.65</td><td>42.69</td><td>42.61</td><td>42.06</td><td>41.93</td><td>12.66</td></tr><tr><td>RESPIN</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>bh</td><td>20.74</td><td>20.31</td><td>20.46</td><td></td><td>19.87 19.98</td><td>20.05</td><td>19.59</td><td>5.54</td></tr><tr><td>bn</td><td>22.02</td><td>21.75</td><td>23.00</td><td>20.94</td><td>20.14</td><td>22.02</td><td>20.18</td><td>8.36</td></tr><tr><td>ch</td><td>14.43</td><td>14.93</td><td>14.42</td><td>13.70</td><td>13.96</td><td>13.98</td><td>13.76</td><td>4.64</td></tr><tr><td>hi</td><td>15.69</td><td>15.33</td><td>15.45</td><td>13.62</td><td>13.98</td><td>14.81</td><td>14.03</td><td>10.58</td></tr><tr><td>kn</td><td>30.96</td><td>31.14†</td><td>32.69</td><td>29.77</td><td>30.34</td><td>32.11</td><td>30.32</td><td>2.07</td></tr><tr><td>mg</td><td>26.70</td><td>26.92</td><td>26.57</td><td>25.94</td><td>26.39</td><td>26.18</td><td>25.65</td><td>3.93</td></tr><tr><td>mr</td><td>20.00</td><td>20.44†</td><td>22.84</td><td>18.78</td><td>18.61</td><td>20.76</td><td>19.03</td><td>4.85</td></tr><tr><td>mt</td><td>23.42</td><td>23.89</td><td>23.11</td><td>22.38</td><td>21.88</td><td>22.76</td><td>22.03</td><td>5.94</td></tr><tr><td>te</td><td>27.39</td><td>27.73†</td><td>30.51</td><td>27.20</td><td>26.87</td><td>29.16</td><td>27.15</td><td>0.88</td></tr><tr><td>Lavg</td><td>9.58</td><td>10.02</td><td>9.39</td><td>9.26</td><td>9.25</td><td>8.98</td><td>8.92</td><td>6.84</td></tr><tr><td>Favg</td><td>38.82</td><td>36.85</td><td>37.60</td><td>34.90</td><td>35.34</td><td>34.56</td><td>34.44</td><td>11.27</td></tr><tr><td>Ravg</td><td>22.37</td><td>22.49</td><td>23.23</td><td>21.36</td><td>21.35</td><td>22.43</td><td>21.30</td><td>4.77</td></tr><tr><td>Avg</td><td>30.56</td><td>29.54</td><td>30.17</td><td>27.97</td><td>28.21</td><td>28.15</td><td>27.67</td><td>9.45</td></tr></table>

Table 3. Conformer WERs (%) on LibriSpeech for all methods reported in Table 2. ∆: relative WER reduction over CTC (%).
<table><tr><td></td><td>CTC</td><td> ${ \bf W } _ { e }$ </td><td> ${ \mathbf { W } _ { e } ^ { \prime } }$ </td><td> $\mathbf { T } _ { e }$ </td><td> $\mathbf { T } _ { H }$ </td><td>TWe</td><td> $\mathbf { T } \mathbf { W } _ { H }$ </td></tr><tr><td>test-clean</td><td>6.43</td><td>6.17</td><td>5.62</td><td>5.54</td><td>5.52</td><td>5.31</td><td>5.49</td></tr><tr><td>test-other</td><td>14.12</td><td>14.07</td><td>13.23</td><td>13.04</td><td>12.94</td><td>12.79</td><td>12.60</td></tr><tr><td>Avg</td><td>10.28</td><td>10.12</td><td>9.43</td><td>9.29</td><td>9.23</td><td>9.05</td><td>9.05</td></tr><tr><td>∆</td><td>1</td><td>1.5</td><td>8.3</td><td>9.6</td><td>10.2</td><td>11.9</td><td>12.0</td></tr></table>

Token-level tolerance gives a substantially different result. Both $\mathrm { O T C - T } _ { e }$ and entropy-indexed $\mathrm { O T C - T } _ { H }$ improve over CTC on all 25 tasks without per-language tuning. Their similar performance across all three corpora also shows that the gain is not specific to the choice of epoch- or entropy-indexed decay. The consistent gains support our hypothesis that allowing individual unsupported tokens to be bypassed is better suited to localised transcript variation than bypassing whole words. These comparisons also bound how much of the gain can be attributed to generic relaxation of the objective. OTC-W<sub>e</sub> and OTC- ${ \boldsymbol { \cdot } } { \boldsymbol { \mathrm { W } } } _ { e } ^ { \prime }$ add wildcard escapes to the same objective, the latter with weights re-tuned on clean speech, yet every token-level criterion has a lower mean WER than both of them on all three corpora, and OTC-$\mathbf { W } _ { \epsilon }$ is worse than CTC on two. Relaxing the alignment constraint is therefore not sufficient on its own; what separates the criteria is where the escape is placed, which Sec. 4.4 examines directly.

![](images/177b583993586bcc5e98f8ae43e55ac5b6b5b81b4c1c950a672679d326540437.jpg)  
Fig. 2. Sensitivity of (a) OTC-T<sub>e</sub> to the decay parameter τ and (b) OTC-T<sub>H</sub> to the exponent κ on LibriSpeech dev-clean+dev-other. WER (%) is reported for models trained on a random 10 h subset of train-clean-100 with the SSL front-end frozen. Dashed line: CTC; circled point: best setting.

The combined graph further shows that the two granularities are complementary. $\mathrm { O T C - T W } _ { \epsilon }$ improves over the token-only and wordonly criteria on LibriSpeech and FLEURS, but its epoch-indexed relaxation degrades on RESPIN, falling 0.24% below CTC on average, the same failure mode seen in $\mathrm { O T C - W } _ { e }$ and $\mathrm { O T C - W } _ { e } ^ { \prime }$ . Replacing the epoch index with predictive entropy in $\mathrm { O T C - T W } _ { H }$ removes this degradation: it gives the lowest mean WER on all three corpora and, at 9.45% relative, the largest average reduction over CTC of any criterion in Table 2, ahead of OTC-T<sub>e</sub> (8.46%) and OTC-T<sub>H</sub> (7.67%). It also has the lowest WER on 9 of the 25 individual tasks, more than any other single criterion.

Table 3 repeats the LibriSpeech comparison with the Conformer encoder. The ranking of Table 2 is reproduced exactly except that OTC- $. \mathbf { W } _ { e }$ and CTC exchange places, indicating that the benefit of token-level tolerance is not specific to the E-Branchformer.

## 4.3. Sensitivity to the schedule

Figure 2 examines the schedule parameters on a randomly selected 10 h subset of LibriSpeech $\mathtt { t r a i n - c l e a n - 1 0 0 }$ . Epoch-indexed OTC-T<sub>e</sub> is sensitive to τ , with overly slow relaxation performing worse than CTC and an optimum near $\tau ~ = ~ 0 . 7 7 5$ . In contrast, $\mathrm { O T C - T } _ { H }$ is substantially less sensitive to κ and improves over CTC throughout the tested range, with its best value at $\kappa = 1 . 3$ We therefore use $\tau = 0 . 7 7 5$ and $\kappa = 1 . 3$ in the remaining experiments and select the $\mathrm { { O T C - W _ { e } ^ { \prime } } }$ word-level constants on the same subset. No per-task tuning is used in Table 2.

## 4.4. Bypass probability and annotator disagreement

RESPIN releases one reference transcript per utterance [18]; we additionally obtained three independent transcriptions of its dev and test sets from external validation vendors (in-house, not yet public). Aligning each validator transcript to the reference at the character level, we call a character or word disputed when at least one validator changes it and the validators do not all propose the same alternative, since a unanimous correction indicates an erroneous rather than an ambiguous reference.

For each model we compose its frame posteriors with the wildcard transcript graph of the reference (Sec. 2.1) and compute by forward–backward the posterior probability $m _ { k }$ that an alignment takes the bypass arc at unit k,

Table 4. Mean wildcard-bypass probability pooled over 9 RESPIN languages (test set). Columns 1–3 split the disputed characters by no. of dissenting validators (13.6k/3.5k/390 units). Char.: 1,014k agree / 17.5k disputed; Word: 174k agree / 18.9k disputed.
<table><tr><td></td><td colspan="5">Character</td><td colspan="2">Word</td></tr><tr><td></td><td>Agree</td><td>Disp.</td><td>1</td><td>2</td><td>3</td><td>Agree</td><td>Disp.</td></tr><tr><td>CTC</td><td>0.020</td><td>0.221</td><td>0.182</td><td>0.351</td><td>0.399</td><td>0.005</td><td>0.020</td></tr><tr><td> ${ \mathrm { O T C - W } } _ { e }$ </td><td>0.020</td><td>0.234</td><td>0.194</td><td>0.369</td><td>0.419</td><td>0.005</td><td>0.021</td></tr><tr><td> $\mathrm { O T C - W _ { \it e } ^ { \prime } }$ </td><td>0.024</td><td>0.230</td><td>0.193</td><td>0.359</td><td>0.397</td><td>0.013</td><td>0.039</td></tr><tr><td> $\mathrm { O T C - T } _ { e } ^ { \circ }$ </td><td>0.021</td><td>0.281</td><td>0.237</td><td>0.431</td><td>0.470</td><td>0.006</td><td>0.030</td></tr><tr><td> $\mathrm { O T C - T } _ { H }$ </td><td>0.022</td><td>0.302</td><td>0.257</td><td>0.454</td><td>0.488</td><td>0.006</td><td>0.034</td></tr><tr><td> $\mathrm { O T C - T W } _ { \epsilon }$ </td><td>0.023</td><td>0.279</td><td>0.235</td><td>0.428</td><td>0.462</td><td>0.011</td><td>0.041</td></tr><tr><td> $\mathrm { O T C - T W } _ { H } ^ { \mathrm { ~ - ~ } }$ </td><td>0.022</td><td>0.301</td><td>0.257</td><td>0.451</td><td>0.479</td><td>0.007</td><td>0.035</td></tr></table>

$$
m _ { k } = \frac { \sum _ { \pi \in \Pi _ { k } ( \mathbf { y } ) } \left( \prod _ { t } P _ { t } ( \pi _ { t } ) \right) e ^ { \phi ( \pi ) } } { \sum _ { \pi \in \Pi ( \mathbf { y } ) } \left( \prod _ { t } P _ { t } ( \pi _ { t } ) \right) e ^ { \phi ( \pi ) } } ,\tag{7}
$$

where $\Pi _ { k } ( \mathbf { y } ) \subset \Pi ( \mathbf { y } )$ are the alignments that bypass k and the denominator is the sum in Eq. (1). Thus $m _ { k }$ reflects the training objective rather than the decoded output. All systems are probed at the same fixed weight $( w ^ { ( \mathrm { b y p } ) } = 0 )$ , so differences between rows of Table 4 reflect the posteriors alone; CTC, never trained with wildcard arcs, is scored through the same graph as a control.

Every system, CTC included, places an order of magnitude more bypass probability on disputed than on agreed characters (11× for CTC), and the amount rises with the number of dissenting validators (Table 4). Token-level arcs raise the disputed-character mean by 27–36% over CTC, with the combined criteria behaving alike, while agreed characters are essentially unchanged; the gain holds in all nine languages on both splits and is significant under a paired bootstrap over utterances (10 000 resamples, $p < 0 . 0 0 1 )$ .

Word-level values are far smaller, and the criteria carrying word arcs raise agreed words as much as disputed ones, so their higher disputed means reflect a uniform increase rather than selectivity. We do not read much into these differences, since a whole word is rarely unsupported in verbatim transcripts, where half of all disputed words differ in a single character of an average 4.8 and a third only by an insertion. This plausibly explains why word-level arcs help little here: a word bypass must discard supervision for the remaining characters to absorb a disagreement that a token arc handles far more cheaply, the mechanism hypothesised in Sec. 3.1.

## 5. CONCLUSION

Moving OTC’s wildcard arcs from words to tokens lets an unsupported token be bypassed while the rest of the word stays supervised. With one setting for all corpora, token-level OTC improved over CTC on all 25 tasks, whereas published word-level OTC stayed close to CTC on LibriSpeech and RESPIN; the trends held with a Conformer encoder. Combining token- and word-level bypasses with entropy-indexed decay gave the lowest mean WER on every corpus, a 9.45% average relative reduction over CTC, and avoided the RESPIN degradation of its epoch-indexed counterpart. Validator transcriptions show that disagreement is local, usually a single character or an inserted space, and token-level models assign more bypass probability to disputed characters while leaving agreed ones essentially unchanged. In future work, we will aim to learn the wildcard weights instead of scheduling them and evaluate against multiple references.

## 6. ACKNOWLEDGMENTS

This work was partly supported by the RESPIN project, funded by the Gates Foundation. We thank the RESPIN contributors for data collection and validation. AI models (Claude Opus 5, Claude Sonnet 5, and GPT-5.6 Luna) assisted with writing, code, plots, tables, and references; all AI-assisted content was reviewed and verified by the authors.

## 7. REFERENCES

[1] Alex Graves, Santiago Fernandez, Faustino Gomez, and J´ urgen¨ Schmidhuber, “Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks,” in Proc. Int. Conf. Mach. Learn. (ICML), 2006, pp. 369–376.

[2] Ahmed Ali, Walid Magdy, Peter Bell, and Steve Renals, “Multi-Reference WER for Evaluating ASR for Languages with No Orthographic Rules,” in Proc. IEEE Autom. Speech Recognit. Understanding Workshop (ASRU), 2015, pp. 576– 580.

[3] Quinten McNamara et al., “Style-Agnostic Evaluation of ASR Using Multiple Reference Transcripts,” arXiv preprint arXiv:2412.07937, 2024.

[4] Kaushal Santosh Bhogale et al., “Towards Orthographically-Informed Evaluation of Speech Recognition Systems for Indian Languages,” arXiv preprint arXiv:2603.00941, 2026.

[5] Alex Graves, “Sequence Transduction with Recurrent Neural Networks,” in Proc. ICML Workshop Represent. Learn., 2012.

[6] Aleksandr Laptev, Vladimir Bataev, Igor Gitman, and Boris Ginsburg, “Powerful and Extensible WFST Framework for RNN-Transducer Losses,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process. (ICASSP), 2023, pp. 1–5.

[7] Dongji Gao et al., “WST: Weakly Supervised Transducer for Automatic Speech Recognition,” arXiv preprint arXiv:2511.04035, 2025.

[8] Vineel Pratap, Awni Hannun, Gabriel Synnaeve, and Ronan Collobert, “Star Temporal Classification: Sequence Modeling with Partially Labeled Data,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2022.

[9] Xingyu Cai, Jiahong Yuan, Yuchen Bian, Guangxu Xun, Jiaji Huang, and Kenneth Church, “W-CTC: A Connectionist Temporal Classification Loss with Wild Cards,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2022.

[10] Niko Moritz, Takaaki Hori, and Jonathan Le Roux, “Semi-Supervised Speech Recognition via Graph-Based Temporal Classification,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process. (ICASSP), 2021, pp. 6548–6552.

[11] Han Zhu, Dongji Gao, Gaofeng Cheng, Daniel Povey, Pengyuan Zhang, and Yonghong Yan, “Alternative Pseudo-Labeling for Semi-Supervised Automatic Speech Recognition,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 31, 2023.

[12] Gil Keren, Wei Zhou, and Ozlem Kalinli, “Token-Weighted RNN-T for Learning from Flawed Data,” in Proc. IEEE Spoken Lang. Technol. Workshop (SLT), 2024.

[13] Dongji Gao, Matthew Wiesner, Hainan Xu, Leibny Paola Garcia, Daniel Povey, and Sanjeev Khudanpur, “Bypass Temporal Classification: Weakly Supervised Automatic Speech Recognition with Imperfect Transcripts,” in Proc. INTERSPEECH, 2023, pp. 924–928.

[14] Dongji Gao, Hainan Xu, Desh Raj, Leibny Paola Garcia Perera, Daniel Povey, and Sanjeev Khudanpur, “Learning from Flawed Data: Weakly Supervised Automatic Speech Recognition,” in Proc. IEEE Autom. Speech Recognit. Understanding Workshop (ASRU), 2023, pp. 1–8.

[15] Mehryar Mohri, Fernando Pereira, and Michael Riley, “Speech Recognition with Weighted Finite-State Transducers,” in Springer Handbook ofSpeech Processing. Springer, 2008.

[16] Awni Hannun, Vineel Pratap, Jacob Kahn, and Wei-Ning Hsu, “Differentiable Weighted Finite-State Transducers,” arXiv preprint arXiv:2010.01003, 2020.

[17] Yu Xi et al., “NTC-KWS: Noise-Aware CTC for Robust Keyword Spotting,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process. (ICASSP), 2025, pp. 1–5.

[18] Saurabh Kumar et al., “RESPIN-S1.0: A Read Speech Corpus of 10000+ Hours in Dialects of Nine Indian Languages,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), Datasets and Benchmarks Track, 2025.

[19] Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur, “LibriSpeech: An ASR Corpus Based on Public Domain Audio Books,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process. (ICASSP), 2015, pp. 5206–5210.

[20] Alexis Conneau et al., “FLEURS: Few-Shot Learning Evaluation of Universal Representations of Speech,” in Proc. IEEE Spoken Lang. Technol. Workshop (SLT), 2023, pp. 798–805.

[21] Kwangyoun Kim et al., “E-Branchformer: Branchformer with Enhanced Merging for Speech Recognition,” in Proc. IEEE Spoken Lang. Technol. Workshop (SLT), 2023.

[22] Shinji Watanabe et al., “ESPnet: End-to-End Speech Processing Toolkit,” in Proc. INTERSPEECH, 2018.

[23] Arun Babu et al., “XLS-R: Self-Supervised Cross-Lingual Speech Representation Learning at Scale,” arXiv preprint arXiv:2111.09296, 2021.

[24] Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli, “wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2020.

[25] Daniel S. Park et al., “SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition,” in Proc. INTERSPEECH, 2019.

[26] Anmol Gulati et al., “Conformer: Convolution-Augmented Transformer for Speech Recognition,” in Proc. INTER-SPEECH, 2020, pp. 5036–5040.