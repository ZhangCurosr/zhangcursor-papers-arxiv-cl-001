# Using Letter Positional Probabilities to Assess French Word Complexity

Michael Dalvean

Albany, Western Australia

October 2026

admin@michaeldalvean.com

## Abstract

We use Letter Positional Probabilities (LPPs) to examine the orthographic diferences between simple and complex French words. We draw from a sample of 38,000 French words scored by reaction time on a lexical decision task. We take the 5000 highest reaction time words and the 5000 lowest as a proxy for high and low word complexity groups respectively. We create 18 variables based on the structure of the words, such as length and number of syllables. We also reduce each word to a string of binary values by one-hot encoding each letter according to the 42 characters of the standard French alphabet up to a length of 25 letters. Thus, each word is represented by 42 × 25 binary characters. We then create LPPs by looking at the average binary scores for simple as opposed to complex words for all letter/positions. We find that there is a high number of LPPs that are associated with word complexity. For example, we find that most accented letters are associated with simpler words. We examine the efects of structural and LPP variables in classifiers and find that word length alone enables a classification accuracy of 77.27%, while adding the binary variables from the first 6 letter positions provided an accuracy of 83.95%. We discuss possible explanations as to how complexity could be encoded in orthography.

Keywords: French; word complexity; letter positional probability; linguistics

## 1 Introduction

This paper examines French word complexity using letter positional probabilities (LPPs). In [1] it was demonstrated across three diferent datasets that LPPs can be used to assess English word complexity. In this study we apply the same concept to French word complexity. The underlying hypothesis here is that there is a latent quality of words which we call word complexity. This quality can be approximated with a number of proxies such as word length, number of syllables, age of acquisition, frequency, concreteness, orthographic characteristics and reaction times on various lexical decision criteria [2, 3, 4, 5, 6, 7, 8]. The idea of using LPPs is that these may also be proxies for word complexity.

We use the reaction time on a lexical decision task as a proxy for word complexity to select a sample of 5000 short reaction time (low complexity) words and 5000 long reaction time (high complexity) words from a database of over 38000 French words. Thus, the samples are drawn from the extreme ends of the frequency distribution. In this way, we approximate a sample that represents the extreme ends of the complexity range.

An important caveat that we must consider is the extent to which reaction times are a proxy for complexity. It is certainly established that reaction times are associated with the usual proxies noted above. Given that these are used as proxies for word complexity, it seems reasonable to claim that reaction times tap into significant elements of word complexity.

We generate LPPs by ‘one-hot encoding’, for each letter position, a binary for 1 - 26 Latin characters plus 16 accented letters and ligatures. We do this for word lengths of 3 - 25 letters. The purpose of this is to represent the fine grain ’atomic’ structure of the word. We find that there are many LPPs that are highly statistically significant in distinguishing between high and low complexity words.

## 2 Previous Work

This study extends the approach in [1] where LPPs were used to assess English word complexity over three datasets. In the first two datasets, 66 LPPS were found to be common as between the two datasets (p < .001), and these were then used to create a classifier that was able to correctly classify a third dataset with an accuracy of 70%. Thus, LPPs seem to carry information relating to word complexity. However, these experiments used English words. The current study attempts to find a similar phenomenon with French words.

The approach that is most similar to the current study is that exemplified by [9], in which phonotactic segments of words were used rather than individual letters. The idea here is that phonotactic segments that are more common in a language are more easily recognised. In that study, it was shown that phonotactic segments that are common in a language are more easily recognised in spoken words than those that are not, and this facilitates word recognition. However, there is an opposing influence in that, if there is a high density of words that have similar phonotactic segments, recognition time can slow due to the efort required to disambiguate the target word from other competing words.

Phonological segment recognition has been found to occur in infants as young as 9 months[10]. In that study, infants were found to be able to recognise phonotactic segments from their own language. Similarly, native speakers of Dutch use phonotactic segments to separate individual words in fluent speech [11]. There is evidence that phonotactic probability also afects word production. In [12], subjects named pictures with more probable phonotactic segments words more quickly than words with lower phonotactic probability. However, those target words with high numbers of orthographical neighbours where the initial phoneme is shared by the target word (high onset density), were responded to more slowly than target words with fewer orthographic neighbours sharing an initial phoneme (low onset density). Finally, in [13], those nonwords that contained high probability phonological segments were judged to be more ’wordlike’ than nonwords made of low-probability segments.

In [14] an algorithm was designed to score synonyms by simplicity on a continuous scale. The training took place with two sources with graded levels of vocabulary for younger/older students and French as a foreign language students. It was evaluated by comparing the rankings of synonyms by simplicity coded manually. They used 300- dimension word embeddings as features. Here the embeddings were vector representations of the word environment in which a target word appears in a large training corpus. They combined word embeddings combined with number of characters, number of phonemes and log frequency. In isolation, embeddings perform better than other features, but the best set of predictors was number of characters and + log frequency.

In [15], using the same graded corpora as used in [14], the authors calculated nine structural variables such as number of characters, number of syllables, phonemes etc. They also used orthographic neighbourhood, phoneme-grapheme correspondence - based on a three point scale. The second set of features was based on morphology, such as number of morphemes, sufixation, prefixation etc. A third set of features consisted of two measures of polysemy. Finally, they include two measures of word frequency. In both corpora, frequency was the feature with the highest correlation with complexity. The overall finding was that word frequency provided more signal than the other variables and the addition of other variables provided only marginal improvement. They found that intermediate classes are the most dificult to classify, with the extremes more easy. The authors acknowledged that the labels may themselves have been noisy, given that the same word may be inlcuded in or excluded from diferent corpora based on textbook selection and teaching approaches. They also point out that the morphology data may have been inaccurate.

In [16] the authors used word embeddings in order to capture word meanings. The idea, as in other work using embeddings, is that a word is given its function in relation to other words, the diference being that they include a context for each word in their word embedding procedure. Thus, the word ‘run’ has 10 entries, and this includes two noun forms and eight verbs. The lowest level of language learner (A1) learns the word in the context "He can run very fast" but at C2, "He would like to run for mayor". The procedure used involved presenting each target word with a contextual sentence. Contextual embeddings improved the Macro-F1 score across A1 - C2 language learning levels by 22 percentage points in comparison to using frequency and word length alone. An attempt to use multilingual BERT to score a corpus of French words using the model trained on English words was less successful, with only the beginner word class being correctly scored. The authors mentioned possible reasons for this as being diferences in morphology between English and French and the lack of availability of French lexical sources.

Alfter et al. [17] examined word-sense dificulty as perceived among French native and non-native speakers. Participants ranked word senses as presented in individual sentences from easiest to most dificult. There was a high degree of correlation between the native and non-native French speakers. Furthermore, the word sense dificulty rank ings were broadly correlated with vocabulary dificulty established in language-learning environments. A subsequent attempt to find an association between word-sense dificulty and the complexity of dictionary definitions was inconclusive.

In [18], subjects were given a prime and were subsequently given a lexical decision task. The format of the primes, for a given target ABCDE, was %BCDE, A%CDE, AB%DE, ABC%E and ABCD%E. They found that some partial word primes were more efective than others at eliciting the correct response. The important factor was the letter positional frequency. Where a given letter is unlikely to appear in a given position in words in the language in general, there is less room to move in terms of the decision as to the correct target, and so the prime is more eficient.

In [19], it was found that more frequent words are ‘better formed’. A word is well formed if its sequence of letters, based on a trigram search algorithm, is probable given other sequences in words in the language. They also found that more frequent words had more orthographic neighbours.

In [20], a compression algorithm (CAFTS) designed to speed up text search was used to compress texts from various languages to approximately 41% of their original size. The algorithm is based on a series of dictionaries of trigrams. Using the algorithm, French had a compression ratio of 3.18 bits/character, which was lower on average than several other languages, with an average across 8 languages of 3.31. We reproduce the table below as Table 1 as it includes the compression data for several languages and compression ratios.

Table 1: Compression ratios (in bits/character), compression and decompression times (in seconds).
<table><tr><td></td><td>Gzip</td><td>Compress</td><td>DLETDC</td><td>ETDC</td><td>SCDC</td><td>CAFTS</td></tr><tr><td>Dutch</td><td>2.86</td><td>3.04</td><td>2.91</td><td>2.84</td><td>2.80</td><td>3.23</td></tr><tr><td>English</td><td>3.06</td><td>3.22</td><td>2.82</td><td>2.78</td><td>2.70</td><td>3.23</td></tr><tr><td>Finnish</td><td>3.03</td><td>3.25</td><td>4.12</td><td>4.02</td><td>3.99</td><td>3.36</td></tr><tr><td>French</td><td>2.87</td><td>3.11</td><td>3.15</td><td>3.09</td><td>3.04</td><td>3.18</td></tr><tr><td>German</td><td>2.98</td><td>3.24</td><td>3.40</td><td>3.32</td><td>3.28</td><td>3.45</td></tr><tr><td>Italian</td><td>3.10</td><td>3.24</td><td>3.29</td><td>3.21</td><td>3.17</td><td>3.29</td></tr><tr><td>Spanish</td><td>2.93</td><td>3.15</td><td>3.33</td><td>3.26</td><td>3.21</td><td>3.32</td></tr><tr><td>Turkish</td><td>3.09</td><td>3.43</td><td>4.16</td><td>4.08</td><td>4.01</td><td>3.44</td></tr><tr><td>Average ratio</td><td>2.99</td><td>3.21</td><td>3.40</td><td>3.33</td><td>3.28</td><td>3.31</td></tr><tr><td>Comp&#x27; time</td><td>20.65</td><td>5.15</td><td>4.83</td><td>6.24</td><td>6.32</td><td>36.20</td></tr><tr><td>Decomp&#x27; time</td><td>1.66</td><td>1.55</td><td>1.95</td><td>1.94</td><td>1.91</td><td>1.26</td></tr></table>

Source: Carus and Mesut [20, p. 4363].

## 3 Data and Methods

## 3.1 Data

The data for this study comes from [21]. This source contains 38,840 French words and 38,840 non-words. The study was designed to determine the time taken to decide whether a given word was a real word or a non word. We used the z scores for the reaction times to rank the words.

## 3.2 Data Cleaning

With 38,840 words, we undertake the following data cleaning steps.

• convert characters to lowercase. The original words were in lowercase, but this step was undertaken to ensure there were no capitals

• remove all words containing any character other than those of the 42-character standard French alphabet

• remove lemmas with non-alphabetical characters

• remove words shorter than 3 characters or longer than 25 characters

After these operations we are left with 38,287 words. Of these, we extract the 5000 with the lowest reaction times and the 5000 with the highest reaction times, giving us a sample of n = 10000.

## 3.3 Variables

The fundamental analysis in this study is based on LPPs. The method of calculating these is as follows.

For any word of up to 25 letters, each letter in positions 1...25 is represented by 42 binary variables, each of which is a binary representing the presence/absence of one of the 42 letters of the alphabet. Thus, we have 25 positions × 42 alphabetic variables. For each position, if a given letter of the alphabet is in that position, the binary for that letter takes the value of 1 for that letter and 0 for all other 41 letters in that position. For example, for the word ’cat’, with $\mathrm { \ ' } _ { \mathrm { c } } \mathrm { \ ' }$ in the first position (the first 42 variables) the letter $\mathrm { \ ' } _ { \mathrm { c } } \mathrm { \ ' }$ takes a binary value of 1 and the other letters take a 0. For position two (variables 43 - 84), the letter ’a’ takes a binary value of 1, and the other letters take a value of 0. Finally, for position 3 (variables 85 - 126), the $\overrightarrow { \mathrm { \Delta t } } ^ { \prime }$ takes a binary value of one and all other letters take a value of 0. For the remaining letter positions from 4 - 25, all values = 0.

Thus, for any word w

$$
w = ( w _ { 1 } , w _ { 2 } , \dots , w _ { L } ) ,
$$

given $3 \leq L \leq 2 5$ , each character $w _ { i }$ belongs to the set of French alphabet characters

$$
\begin{array} { r }  \mathcal { A } = \{ a , b , c , d , e , f , g , h , i , j , k , l , m , n , o , p , q , r , s , t , u , v , w , x , y , z , \hat { \mathbf { a } } , \hat { \mathbf { a } } , \mathbf { a } , \mathbf { s } , \varsigma , \varsigma , \hat { \mathbf { e } } , \hat { \mathbf { e } } , \hat { \mathbf { e } } , \hat { \mathbf { , \} } , \hat { \mathbf { l } } , \hat { \mathbf { o } } , \infty , \hat { \mathbf { u } } , \hat { \mathbf { u } } , \ddot { \mathbf { y } } \} , } \end{array}
$$

and $| { \mathcal { A } } | = 4 2$

We define the binary positional variable as

$$
x _ { i j } ( w ) = \left\{ { \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ } } i \leq L { \mathrm { ~ a n d ~ } } w _ { i } = a _ { j } , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } , } \end{array} } \right.
$$

where

$$
i = 1 , \ldots , 2 5
$$

is the character position in the word, and

$$
j = 1 , \ldots , 4 2
$$

is the index of alphabet characters.

The unaccented characters in A above are represented by the first 26 indices:

$$
a _ { 1 } = a , \quad a _ { 2 } = b , \quad . . . , \quad a _ { 2 6 } = z .
$$

The accented characters and ligatures in A above are represented by the final 16

indices. characters and ligatures:

$$
a _ { 2 7 } = \mathrm { \hat { a } } , ~ a _ { 2 8 } = \mathrm { \hat { a } } , ~ a _ { 2 9 } = \mathrm { \Phi } \mathrm { \Phi } , ~ a _ { 3 0 } = \mathrm { \Phi } \mathrm { \Phi } ,
$$

$$
a _ { 3 1 } = \not \mathrm { e } , \quad a _ { 3 2 } = \not \mathrm { e } , \quad a _ { 3 3 } = \hat { \mathrm { e } } , \quad a _ { 3 4 } = \ddot { \mathrm { e } } ,
$$

$$
a _ { 3 5 } = \hat { 1 } , \quad a _ { 3 6 } = \ddot { 1 } , \quad a _ { 3 7 } = \hat { 0 } , \quad a _ { 3 8 } = \infty ,
$$

$$
a _ { 3 9 } = \mathrm { { \hat { u } } , ~ \ } a _ { 4 0 } = \mathrm { { \hat { u } } , ~ \ } a _ { 4 1 } = \mathrm { { \ddot { u } } , ~ \ } a _ { 4 2 } = \mathrm { { \ddot { y } } . }
$$

Given that there are 25 possible positions and 42 possible characters at each position, the complete set of binary variables for each word contains

$$
2 5 \times 4 2 = 1 0 5 0
$$

positional indicators.

For words of length < 25, the binary values beyond the position of its final letter are given a value of 0. It should be noted that, in the modeling stage, any binary variable that has zero values for all words is removed from the analysis. In this way we are able to reduce the number of variables from a potential 1050 to 459 for the corpus under study.

The above describes the creation of binaries for each letter position. These binary values are used in the creation of classifiers, where the target is a 1 for complex words and 0 for simple words. The LPPs are calculated by averaging the binaries for each of the simple and complex groups. This gives the probabilities that a given letter will occur at a given position for a simple as opposed to a complex word.

Given a group of words selected for their simplicity/complexity G, the LPP for the alphabet character $a _ { j }$ at the position i is the average of the binary indicators for the same alphabet character at the same position for all words in that group:

$$
P _ { i j } ^ { ( G ) } = \frac { 1 } { N _ { G } } \sum _ { w \in G } x _ { i j } ( w ) ,
$$

where $N _ { G }$ is the number of words in group G.

The group means calculated in this way are used for the two tailed t-tests in the descriptive statistical analysis below.

For each of the two groups, we calculate the diference in positional probability for a given character at a given position as

$$
\Delta P _ { i j } = P _ { i j } ^ { ( \mathrm { h i g h } ) } - P _ { i j } ^ { ( \mathrm { l o w } ) } .
$$

Where $\Delta P _ { i j }$ is positive, the $a _ { j }$ character occurs more often at position i in complex words, and a negative value indicates more common occurrence in simple words.

We also calculate 18 variables based on the structure of the words that are associated with word complexity.

• LENGTH = n; measures basic orthographic size.

• SYLLABLES = number of syllables in the word. Note that the number of syllables is not calculated from the Python script because the method of calculating syllables from French words was not accurate. The number of syllables was therefore taken from the original data source ([21]).

• MEAN\_SYLLABLE\_LENGTH; measures average syllable size.

• UNIQUE\_LETTERS; counts the number of diferent letters occurring in the word.

• LETTER\_DIVERSITY; measures normalized character diversity.

• REPEATED\_LETTERS; measures repetition/redundancy of characters.

• DOUBLE\_LETTERS; counts adjacent identical letter pairs.

• VOWELS; counts vowels in the word.

• CONSONANTS; number of alphabetic characters that are not vowels.

• VOWEL\_CONSONANT\_RATIO; measures Vowel/Consonant composition

• CONSONANT\_CLUSTERS; measures number of uninterrupted consonant sequences of length >= 2.

• MAX\_CONSONANT\_CLUSTER; maximum length of any uninterrupted consonant sequence.

• CV\_TRANSITIONS; counts changes between vowel and consonant categories.

• BIGRAM\_DIVERSITY; measures adjacent letter-pair diversity.

• TRIGRAM\_DIVERSITY; measures three-letter-sequence diversity.

• LETTER\_ENTROPY; measures uncertainty/diversity of letters within the word.

• REPETITION\_RATIO; measures internal redundancy.

• ORTH\_NEIGHBOURS; number of same-length words difering in exactly one letter.

With 10,000 words, we create a matrix of 1050 binary variables and 18 continuous variables. Of the 1050 binary variables, some have empty columns. Removing all empty columns gives us 441 LPP variables. Thus, the total number of variables is 459.

The target column consists of binary values, with 0 being assigned to the low complexity group and 1 to the high complexity group.

There is a large number of independent variables under analysis. In subsequent analysis, we therefore make a Bonferroni correction to all p-values by multiplying them by 459.

## 4 Results

After Bonferroni correction, there are 163 variables that are significant at the $\mathrm { p } { < } . 0 1$ level. These are reproduced at Appendix A. Inspection of the table shows that several of the standard word complexity measures are significantly discriminative of the high and low frequency groups, with the top 12 most significant variables being complexity measures. Of the 163 significant variables, 145 are LPPs, and all structural variables were found to be significant.

In order to contextualise the results for the LPPs, we should consider that the average word length of the low complexity group is 6.69 and that of the high complexity group is 9.77. What this means is that letter positions above 6 are more likely to be significant purely because their presence is indicative of greater length, and length is highly associated with complexity. This is why the first 27 significant LPPs are from positions ${ > } 6$ . We will examine the efects of this below when we create classifiers from the LPP data. At this stage, we will examine what is happening in the first 6 positions because these are more properly associated with orthography than length.

The significant LPPs in positions $< 7$ are presented in Table 2, in which there are 42 significant variables from positions $< 7 .$

Table 2: Low- and high-complexity French words: mean values for positional letter features at positions <7.
<table><tr><td>Variable</td><td>Mean – Low Complexity  $( n = 5 , 0 0 0 )$ </td><td>Mean – High Complexity (n = 5,000)</td><td></td><td>p value (Bonf. ×459)</td></tr><tr><td>06</td><td>0.011</td><td>0.051</td><td></td><td>&lt; .0001</td></tr><tr><td>e5</td><td>0.168</td><td>0.092</td><td></td><td>&lt; .0001</td></tr><tr><td>n2</td><td>0.035</td><td>0.086</td><td></td><td>&lt; .0001</td></tr><tr><td>a6</td><td>0.042</td><td>0.094</td><td></td><td>&lt; .0001</td></tr><tr><td>é6</td><td>0.056</td><td>0.017</td><td></td><td>&lt; .0001</td></tr><tr><td>i1</td><td>0.020</td><td>0.060</td><td></td><td>&lt; .0001</td></tr><tr><td>i6</td><td>0.053</td><td>0.105</td><td></td><td>&lt; .0001</td></tr><tr><td>e6</td><td>0.161</td><td>0.097</td><td></td><td>&lt; .0001</td></tr><tr><td>02</td><td>0.195</td><td>0.129</td><td></td><td>&lt; .0001</td></tr><tr><td>05</td><td>0.040</td><td>0.078</td><td></td><td>&lt; .0001</td></tr><tr><td>p6</td><td>0.004</td><td>0.020</td><td></td><td>&lt; .0001</td></tr><tr><td>j1</td><td>0.019</td><td>0.005</td><td></td><td>&lt; .0001</td></tr><tr><td>v1</td><td>0.044</td><td>0.021</td><td></td><td>&lt; .0001</td></tr><tr><td>16</td><td>0.028</td><td>0.052</td><td></td><td>&lt; .0001</td></tr><tr><td>i5</td><td>0.089</td><td>0.127</td><td></td><td>&lt; .0001</td></tr><tr><td>z6</td><td>0.011</td><td>0.002</td><td></td><td>&lt; .0001</td></tr><tr><td>12</td><td>0.043</td><td>0.022</td><td></td><td>&lt; .0001</td></tr><tr><td>a4</td><td>0.069</td><td>0.102</td><td></td><td>&lt; .0001</td></tr><tr><td>m6</td><td>0.014</td><td>0.031</td><td></td><td>&lt; .0001</td></tr><tr><td>ê2</td><td>0.007</td><td>0.000</td><td></td><td>&lt; .0001</td></tr><tr><td>m2</td><td>0.015</td><td>0.032</td><td></td><td>&lt; .0001</td></tr><tr><td>m5</td><td>0.012</td><td>0.028</td><td></td><td>&lt; .0001</td></tr><tr><td>r5</td><td>0.077</td><td>0.108</td><td></td><td>&lt; .0001</td></tr><tr><td>y3</td><td>0.014</td><td>0.004</td><td></td><td>&lt; .0001</td></tr><tr><td>c6</td><td>0.018</td><td>0.034</td><td></td><td>&lt; .0001</td></tr><tr><td>z5</td><td>0.007</td><td>0.001</td><td></td><td>&lt; .0001</td></tr><tr><td>f6</td><td>0.003</td><td>0.012</td><td></td><td>1.68 × 10−4</td></tr><tr><td>u3</td><td>0.087</td><td>0.060</td><td></td><td>1.88 × 10−4</td></tr><tr><td>r4</td><td>0.061</td><td>0.088</td><td></td><td>2.22 × 10−4</td></tr><tr><td>â2</td><td>0.008</td><td>0.001</td><td></td><td>2.44 × 10−4</td></tr><tr><td>11</td><td>0.032</td><td>0.017</td><td></td><td>2.59 × 10−4</td></tr><tr><td>é5</td><td>0.040</td><td>0.023</td><td></td><td>7.14 × 10−4</td></tr><tr><td>h3</td><td>0.005</td><td>0.014</td><td></td><td>1.08 × 10−3</td></tr><tr><td>a1</td><td>0.063</td><td>0.088</td><td></td><td>1.47 × 10−3</td></tr><tr><td>t3</td><td>0.061</td><td>0.085</td><td></td><td>1.48 × 10−3</td></tr><tr><td>04</td><td>0.060</td><td>0.084</td><td></td><td>1.48 × 10−3</td></tr><tr><td>û3</td><td>0.005</td><td>0.000</td><td></td><td>2.54 × 10−3</td></tr><tr><td>g6</td><td>0.014</td><td>0.027</td><td></td><td>2.72 × 10−3</td></tr><tr><td>f1</td><td>0.053</td><td>0.034</td><td></td><td>2.88 × 10−3</td></tr><tr><td>r3</td><td>0.082</td><td>0.108</td><td></td><td>3.01 × 10−3</td></tr><tr><td>t6</td><td>0.059</td><td>0.082</td><td></td><td>3.70 × 10−3</td></tr><tr><td>x3</td><td>0.008</td><td>0.002</td><td></td><td>4.11 × 10−3</td></tr><tr><td>â6</td><td>0.000</td><td>0.004</td><td></td><td> $5 . 8 5 \times 1 0 ^ { - 3 }$ </td></tr></table>

One interesting observation is that accented vowels, other than â6, are associated with low complexity.

Table 3: Low- and high-complexity French words: Significant accented-character features at positions $< 7$
<table><tr><td>Feature</td><td>Low Mean</td><td>High Mean</td><td>Direction</td></tr><tr><td>é6</td><td>0.056</td><td>0.017</td><td>Low ↑</td></tr><tr><td>ê2</td><td>0.007</td><td>0.000</td><td>Low ↑</td></tr><tr><td>â2</td><td>0.008</td><td>0.001</td><td>Low ↑</td></tr><tr><td>é5</td><td>0.040</td><td>0.023</td><td>Low ↑</td></tr><tr><td>û3</td><td>0.005</td><td>0.000</td><td>Low ↑</td></tr><tr><td>â6</td><td>0.000</td><td>0.004</td><td>High ↑</td></tr></table>

On the other hand, the six significant unaccented vowels are associated with high

complexity.

Table 4: Low- and high-complexity French words: Significant unaccented-character features at positions < 7
<table><tr><td>Feature</td><td>Low Mean</td><td>High Mean</td><td>Direction</td></tr><tr><td>a4</td><td>0.069</td><td>0.102</td><td>High ↑</td></tr><tr><td>a6</td><td>0.042</td><td>0.094</td><td>High ↑</td></tr><tr><td>i5</td><td>0.089</td><td>0.127</td><td>High ↑</td></tr><tr><td>i6</td><td>0.053</td><td>0.105</td><td>High ↑</td></tr><tr><td>05</td><td>0.040</td><td>0.078</td><td>High ↑</td></tr><tr><td>06</td><td>0.011</td><td>0.051</td><td>High ↑</td></tr></table>

In [22] it was found that accents do not significantly afect lexical processing, so it is an open question why the above associations might exist, and it is beyond the scope of the current study to examine possible reasons. However, these exploratory findings indicate that there is information encoded in orthography that carries a simplicity/complexity signal.

Position 6 is the most highly represented out of the first 6 positions: there are 6 vowels; o6, a6, é6, i6, e6, â6, and 8 consonants; p6, l6, z6, m6, c6, f6, g6, t6. There are four vowels, o, a, â6 and i, that are associated with high complexity words, while é6,and e6 are associated with low complexity words.

Table 5: Low- and high-complexity French words: Significant vowel and consonant features at position 6.
<table><tr><td>Feature</td><td>Low Mean</td><td>High Mean</td><td>Direction</td></tr><tr><td>Vowels</td><td></td><td></td><td></td></tr><tr><td>06</td><td>0.011</td><td>0.051</td><td>High↑</td></tr><tr><td>a6</td><td>0.042</td><td>0.094</td><td>High ↑</td></tr><tr><td>i6</td><td>0.053</td><td>0.105</td><td>High ↑</td></tr><tr><td>â6</td><td>0.000</td><td>0.004</td><td>High ↑</td></tr><tr><td>e6</td><td>0.161</td><td>0.097</td><td>Low ↑</td></tr><tr><td>é6</td><td>0.056</td><td>0.017</td><td>Low ↑</td></tr><tr><td>Consonants</td><td></td><td></td><td></td></tr><tr><td>p6</td><td>0.004</td><td>0.020</td><td>High ↑</td></tr><tr><td>16</td><td>0.028</td><td>0.052</td><td>High ↑</td></tr><tr><td>m6</td><td>0.014</td><td>0.031</td><td>High ↑</td></tr><tr><td>c6</td><td>0.018</td><td>0.034</td><td>High ↑</td></tr><tr><td>f6</td><td>0.003</td><td>0.012</td><td>High ↑</td></tr><tr><td>g6</td><td>0.014</td><td>0.027</td><td>High ↑</td></tr><tr><td>t6</td><td>0.059</td><td>0.082</td><td>High ↑</td></tr><tr><td>z6</td><td>0.011</td><td>0.002</td><td>Low ↑</td></tr></table>

Seven of the eight consonants at position 6 are associated with high complexity, with z

the exception.

At position 1, there is a distinct division of vowel and consonant associations, with the vowels a and i being associated with complex words and the consonants j,l, v and f associated with low complexity words.

Table 6: Low- and high-complexity French words: Significant character features at position 1.
<table><tr><td>Feature</td><td>Low Mean</td><td>High Mean</td><td>Direction</td></tr><tr><td>i1</td><td>0.020</td><td>0.060</td><td>High ↑</td></tr><tr><td>j1</td><td>0.019</td><td>0.005</td><td>Low ↑</td></tr><tr><td>v1</td><td>0.044</td><td>0.021</td><td>Low ↑</td></tr><tr><td>11</td><td>0.032</td><td>0.017</td><td>Low ↑</td></tr><tr><td>a1</td><td>0.063</td><td>0.088</td><td>High ↑</td></tr><tr><td>f1</td><td>0.053</td><td>0.034</td><td>Low ↑</td></tr></table>

The significant features at position 2 are as depicted in Table 7

Table 7: Low- and high-complexity French words: Significant character features at position 2.
<table><tr><td>Feature</td><td>Low Mean</td><td>High Mean</td><td>Direction</td></tr><tr><td>n2</td><td>0.035</td><td>0.086</td><td>High ↑</td></tr><tr><td>02</td><td>0.195</td><td>0.129</td><td>Low ↑</td></tr><tr><td>12</td><td>0.043</td><td>0.022</td><td>Low ↑</td></tr><tr><td>ê2</td><td>0.007</td><td>0.000</td><td>Low ↑</td></tr><tr><td>m2</td><td>0.015</td><td>0.032</td><td>High ↑</td></tr><tr><td>â2</td><td>0.008</td><td>0.001</td><td>Low ↑</td></tr></table>

We note that, as in the tables above, n and m are associated with high complexity. This occurs for the full cohort of results with m and n being associated with complexity for all significant occurrences of these letters in all positions.

At position 3, high complexity is associated with the consonants h, t and r, while low complexity is associated with the vowels u, û, y, and the only consonant associated with low complexity is x.

Table 8: Low- and high-complexity French words: Significant character features at position 3
<table><tr><td>Feature</td><td>Low Mean</td><td>High Mean</td><td>Direction</td></tr><tr><td>y3</td><td>0.014</td><td>0.004</td><td>Low ↑</td></tr><tr><td>u3</td><td>0.087</td><td>0.060</td><td>Low ↑</td></tr><tr><td>h3</td><td>0.005</td><td>0.014</td><td>High ↑</td></tr><tr><td>t3</td><td>0.061</td><td>0.085</td><td>High ↑</td></tr><tr><td>û3</td><td>0.005</td><td>0.000</td><td>Low ↑</td></tr><tr><td>r3</td><td>0.082</td><td>0.108</td><td>High ↑</td></tr><tr><td>x3</td><td>0.008</td><td>0.002</td><td>Low ↑</td></tr></table>

Position 4 has two vowels, a and o, and one consonant, r, all associated with high complexity.

Table 9: Low- and high-complexity French words: Significant character features at position 3
<table><tr><td>Feature</td><td>Low Mean</td><td>High Mean</td><td>Direction</td></tr><tr><td>a4</td><td>0.069</td><td>0.102</td><td>High ↑</td></tr><tr><td>r4</td><td>0.061</td><td>0.088</td><td>High ↑</td></tr><tr><td>04</td><td>0.060</td><td>0.084</td><td>High ↑</td></tr></table>

At position 5, there is no apparent vowel/consonant pattern, however it is worth noting that e and é both move in the same (low complexity) direction, as we noted above in position 6, indicating that both e and é in positions $< 7$ are associated with low complexity.

Table 10: Low- and high-complexity French words: Significant character features at position 5
<table><tr><td>Feature</td><td>Low Mean</td><td>High Mean</td><td>Direction</td></tr><tr><td>e5</td><td>0.168</td><td>0.092</td><td>Low ↑</td></tr><tr><td>05</td><td>0.040</td><td>0.078</td><td>High ↑</td></tr><tr><td>i5</td><td>0.089</td><td>0.127</td><td>High ↑</td></tr><tr><td>m5</td><td>0.012</td><td>0.028</td><td>High ↑</td></tr><tr><td>r5</td><td>0.077</td><td>0.108</td><td>High ↑</td></tr><tr><td>z5</td><td>0.007</td><td>0.001</td><td>Low ↑</td></tr><tr><td>é5</td><td>0.040</td><td>0.023</td><td>Low ↑</td></tr></table>

In [23] it was found that that consonants play a more important role in English word recognition during reading than vowels. In [24] the study design involved target words with either consonants in first, thrd and fifth place (e.g., acero), or vowels in the same positions (e.g., farol). Primes consisted of the actual letters in those positions $\left( \mathrm { a , c , o / f , r , l } \right)$ voewls/consonants that did not appear in the word $\left( \mathrm { l } , \mathrm { u } , \mathrm { i } / \mathrm { t } , \mathrm { s } , \mathrm { b } \right)$ , or by the actual word (acero/farol). Using measurements of Event Related Potentials, it was found that the consonant primes had a similar Event Related Potentials efects to the actual word, while vowel primes had efects similar to the unrelated primes. Thus, there is some evidence that lexical processing may be diferent for the vowels as opposed to consonants. Brain activation has been found to be diferent in relation to vowel and consonant in several contexts [25]. Again, the evidence from the research is that there are possible reasons why vowels and consonants are diferentially associated with complexity. However, there is no direct evidence of a link between vowels/consonants and word complexity.

## 5 Classifiers

The results above indicate that there are significant variables that are likely to be able to be used to create classifiers to distinguish between low and high complexity words. By introducing diferent sets of variables, we can determine which variables carry the greatest signal. On the basis of this, we can determine whether the signals in the data described above are merely statistical phenomena, or whether there is something about the orthography of French words that carries information about word complexity.

The features used for the classifiers are the variables as described above. Thus, we have 441 LPP features, 18 structural features, and the target is a binary with 1 representing the high complexity words and 0 representing the low complexity words. Details of the classifiers are presented in Table 11

Table 11: Details of SVM, RF and NN classifiers
<table><tr><td>Classifier</td><td></td><td>PreprocessingModel specification</td></tr><tr><td>Machine</td><td>tion; standardisa- 10,000 iterations.</td><td>Support Vector Median imputa- Linear SVM (LinearSVC); C = 1.0; maximum</td></tr><tr><td></td><td>tion</td><td>Random Forest Median imputa- 500 decision trees; square-root feature sampling at</td></tr><tr><td></td><td>tion</td><td>each split; all available processor cores used during training. Neural Network Median imputa- Feed-forward network with three hidden layers tion; standardisa- containing 128, 64 and 32 units, respectively</td></tr></table>

Note: All classifiers were evaluated using 10-fold cross-validation. Training data within each fold were used to estimate preprocessing parameters. The results represent the combined predictions from the 10 held-out folds.

The groups of features used in the following analysis are:

• Structural - These are the 18 variables calculated from the letters of words. They include LENGTH, SYLLABLES etc. Note that the number of syllables is taken from the original data source ([21]) because the tested methods of calculating syllables from French words were not eficient.

• LENGTH - Number of letters. This is the same variable extracted from the 18 structural variables. However, we examine word length as a single predictor of complexity and in combination with LPP variables because complexity is highly associated with length.

• First 6 LPP positions (n = 202) - These are the first 6 LPP positions calculated as above. Note that these include both the variables that were found to be significant after Bonferroni correction and those that were not. Potentially, this would give us a set of variables $= 4 2 \times 6 = 2 5 2$ . However, some of these did not have data and so were removed giving a final figure of 202.

• All LPP Positions - This is the full cohort of LPP variables. There are potentially 1050 ( 42 × 25) variables. However, the variables with no data were removed from the matrix of results giving a final figure of 441.

## 5.1 Classifier Results

Table 12 shows the results from the classifier experiments

Table 12: Classification accuracy details
<table><tr><td>Feature Set</td><td>NN</td><td>SVM</td><td>RF</td></tr><tr><td>Length</td><td>0.7727</td><td>0.7727</td><td>0.7727</td></tr><tr><td>Structural</td><td>0.7785</td><td>0.7775</td><td>0.7620</td></tr><tr><td>Structural + All Positions</td><td>0.8111</td><td>0.8031</td><td>0.8085</td></tr><tr><td>First 6 Positions</td><td>0.7657</td><td>0.6932</td><td>0.8012</td></tr><tr><td>All Positions</td><td>0.8091</td><td>0.7966</td><td>0.8364</td></tr><tr><td>Structural + First 6 Positions</td><td>0.8184</td><td>0.7951</td><td>0.8031</td></tr><tr><td>First 6 Positions + Length</td><td>0.8173</td><td>0.7859</td><td>0.8395</td></tr></table>

The results show that Length alone accounts for 77.27% of correct classifications. All three classifiers score the same with this one variable, suggesting a high degree of monotonicity. All classifiers have the same distribution of errors, with sensitivity = 68.30% and specificity = 86.24

The Structural variables, which include Length, attain a classification accuracy of 77.85% with the neural network. Given the small improvement from Length alone, the structural variables other than length add very little to the signal.

The Structural and All Positions features boost these results to 81.11%, with the Neural Network, indicating that the positional data carry significant information. The Random Forest results are lower than the RF result for All Positions features without the Structural features, indicating that there may be some degradation in classifier accuracy due to high levels of redundancy.

The highest accuracy is achieved by the First 6 Positions + Length in the Random Forest at 83.95%. This is marginally higher than All Positions on their own at 83.64%. What this indicates is that there is inherent length data in the All Positions data (due to the mere presence of positional data at positions > 6), but the purely orthographic data in the First 6 positions carries a significant signal. When we add the Length variable to the First 6 Positions features we include the important orthographical information as well as the length information.

## 6 Discussion

We have seen that there are LPPs that are highly statistically associated with complexity. We have also seen that there are patterns of vowels, consonants and articulation phenomena that are associated with complexity. The indication is that there is information encoded in orthography that is beyond mere words structural information. Importantly, this information can be used to determine the simplicity/complexity of words. A classifier with LPPs from the first 6 positions along with word length was found to be the most predictive of word complexity out of a number of feature sets, including one consisting of all positions.

We can draw from these observations that there is something in orthography that carries a signal associated with complexity. Certainly length also carries a signal, but, as we have seen, combining length with positional data provides the most robust signal.

In reviewing the work on phonotactic probability, we noted above that phonotactic probability is an important influence on the way words are processed, both in producing speech and understanding written and spoken speech. In the current study we posit that letter positional probabilities, rather than phonotactic probabilities, carry information about word complexity.

We also noted that position is important. In [18] above, we noted similarly that speakers of a language are sensitive to the frequencies at which letters occur at diferent positions. Thus, given a prime of %BCDE, and a prime of A%CDE and a target of ABCDE, the first prime will be more eficient than the second if ’A’ in first position is less frequent in the language and ’B’ in second position. Again, this overlaps with the current study.

The general approach of the majority of discussed studies is that native language users are sensitive to the atomic structure of words. The current study does not disagree with this, but the fundamental question is about word complexity. In the current study, the two extremes of the complexity word sample were chosen on the basis of a lexical decision experiment, and this means that the sample instantiates a phenomenon about word recognition that is similar to the other probability-based studies. The major diference is that the current study, after creating the sample based on lexical decisions, asks what it is about the structure of words at the letter level that distinguishes between simple and complex words. The latter use the structure of words to explain lexical decision criteria. In short, the process of the current study and the other studies is reversed. This is the fundamental diference between the two approaches. However, none of these, including the current study, directly address what is possibly the more interesting question: given that the structure of words carries significant information, what explains how words evolved to have such a structure? The question here is not whether complex words have a diferent structure compared to simple words, as we have already established this. The question is why this should be so. Although a full analysis of this is beyond the scope of this paper, this is an important issue to address.

One possibility is that words change over time and their orthography follows. A possible method for this to take place was given by Zipf [26]. He found that pronunciation changes over time according to how frequently the word is used. The idea here is that we look for ways to reduce our efort in communication, so words that are used more often are likely to sufer simplification because there they will be recognised without highly complex articulation forms because they are so common. The argument from Zipf is that 1) a given locution is used often, 2) the phonemes are repeated often due to frequent use 3), ’shortcuts’ in articulation take place, but because of the frequency of use, other speakers of the language still understand the locution 4), phonetic ’weakening’ takes place, 5) the weakening becomes conventional 6) the lexical form of the original form is altered. Thus, more frequent terms become more easy to pronounce over time.

There is one seemingly contradictory finding from [27] found that across 200 word meanings from 87 Indo-European languages, common words were less likely to change. However, this is consistent with Zipf’s argument in that we might consider that the simplified words, once having achieved their simplified forms, represent collections of phonemes that are the easiest to articulate in the language. There is no behavioural reason for changing them from this form.

The implication here is that, because the simplification process took place over time, the now simple words had a more complex phonological structure at some time in the past. The words that were complex in the past and remain complex now represent the default level of complexity, and those that have become simplified represent the changed words a simplified new phonology. This suggests that the significant LPP variables are picking up on a change that has occurred over time. It is not a case that there is was initially a structural diference between simple and complex words, but it is true that the

structural diference emerged over time.

Hypothetically, the reason that there are significant LPPs is that the simplification process is not arbitrary. The words that are more common change in a way that is systematic for a given language. As [19] found, words that have a highly probable (more likely) form within the language are more frequent, as are words with more orthographic neighbours.

Finally, we need to consider the possibility that the concept of complexity we have been using is actually frequency. As noted above, frequency is often used as a proxy for complexity. We have used a sample consisting of what we have taken to be high and low complexity words. However it is most likely that these also represent the extremes of the frequency range. It is theoretically possible to test for frequency efects by including in the modeling procedure a measure of frequency (although choosing what corpus to base this on might be problematic). Again, this is a possible avenue of enquiry in subsequent work.

## 7 Conclusion

This study has examined the role of LPPs in carrying a simplicity/complexity signal in French words. We examined the distribution of LPPs in simple as opposed to complex words. We found that there were 145 LPPs that were significant out of a potential cohort of 441. Because the average length of simple words was 6.69, we focused on the first 6 positions because we wished to reduce the confounding efect of length of LPPs in positions >6. We also found that several structural variables were associated with complexity, with length being particularly predictive of complexity in our sample.

We found that there were a number of phenomena associated with LPPs in relation to complexity. We found that 5 out of 6 of the significant LPPs containing an accented letter were associated with lower complexity, while 6 out of 6 of the significant LPPs containing an unaccented letter were associated with higher complexity. The overall finding was that there are significant diferences in the orthography of simple as opposed to complex words.

We went on to create classifiers using LPPs and structural data as features to predict binary simple and complex words. We found that length alone achieves an accuracy of 77.27%. Additional structural variables added little to classification accuracy. The best classifier achieved an accuracy of 83.95% and consisted of the first 6 LPP features plus Length. We concluded from this that both orthography and length are significant features associated with complexity.

The question we went on to examine is how orthography can encode complexity. Certainly it is conventionally accepted that word length is associated with complexity, and in the current study it was indeed the variable most closely associated with complexity. However, positing the idea that the placement of individual letters is predictive of complexity is significantly more controversial.

We considered an explanation from Zipf that more common words become simplified over time due to articulatory economy. The important point here for our purposes is that simplicity is associated with frequency in that words that are repeated often become simplified.

We observed that the question of the association between vowel/consonant patterns and complexity may be worth considering in subsequent work, as is the possibility that there may be a dificulty in disentangling the diference between word complexity and word frequency.

## Funding

This research received no external funding.

## Conflict of Interest

The author declares no conflict of interest.

## Data Availability

The Python scripts used to generate the data used in this study can be obtained from the author at admin@michaeldalvean.com.

## References

1. Dalvean M. Using Letter Positional Probabilities to Assess Word Complexity. arXiv preprint arXiv:2404.07768 2024

2. Ferrand L, Brysbaert M, Keuleers E, New B, Bonin P, Méot A, Augustinova M, and Pallier C. Comparing word processing times in naming, lexical decision, and progressive demasking: Evidence from Chronolex. Frontiers in Psychology 2011; 2:306. doi: 10.3389/fpsyg.2011.00306

3. Balota DA, Yap MJ, Cortese MJ, Hutchison KA, Kessler B, Loftis B, Neely JH, Nelson DL, Simpson GB, and Treiman R. The English Lexicon Project. Behavior Research Methods 2007; 39:445–59. doi: 10.3758/BF03193014

4. Yap MJ and Balota DA. Visual word recognition of multisyllabic words. Journal of Memory and Language 2009; 60:502–29. doi: 10.1016/j.jml.2009.02.001

5. Brysbaert M, Stevens M, Mandera P, and Keuleers E. The impact of word prevalence on lexical decision times: Evidence from the Dutch Lexicon Project 2. Journal of Experimental Psychology: Human Perception and Performance 2016; 42:441–58. doi: 10.1037/xhp0000159

6. Balota DA and Chumbley JI. Are lexical decisions a good measure of lexical access? The role of word frequency in the neglected decision stage. Journal of Experimental Psychology: Human Perception and Performance 1984; 10:340–57. doi: 10.1037/ 0096-1523.10.3.340

7. Rayner K and Dufy SA. Lexical complexity and fixation times in reading: Efects of word frequency, verb complexity, and lexical ambiguity. Memory & Cognition 1986; 14:191–201. doi: 10.3758/BF03197692

8. Keuleers E, Lacey P, Rastle K, and Brysbaert M. The British Lexicon Project: Lexical decision data for 28,730 monosyllabic and disyllabic English words. Behavior Research Methods 2012; 44:287–304. doi: 10.3758/s13428-011-0118-4

9. Vitevitch MS and Luce PA. Probabilistic phonotactics and neighborhood activation in spoken word recognition. Journal of memory and language 1999; 40:374–408

10. Jusczyk PW, Friederici AD, Wessels JM, Svenkerud VY, and Jusczyk AM. Infants<sup>′</sup> sensitivity to the sound patterns of native language words. Journal of memory and language 1993; 32:402–20

11. McQueen JM. Segmentation of continuous speech using phonotactics. Journal of memory and language 1998; 39:21–46

12. Vitevitch MS, Armbrüster J, and Chu S. Sublexical and lexical representations in speech production: efects of phonotactic probability and onset density. Journal of Experimental Psychology: Learning, Memory, and Cognition 2004; 30:514

13. Frisch SA, Large NR, and Pisoni DB. Perception of Wordlikeness: Efects of Segment Probability and Length on the Processing of Nonwords. Journal of Memory and Language 2000; 42:481–96. doi: 10.1006/jmla.1999.2692

14. Soler G, Garí A, Apidianaki M, and Allauzen A. A comparative study of word embeddings and other features for lexical complexity detection in French. Actes de la Conférence TALN. Volume 1-Articles longs, articles courts de TALN. 2018 :499–508

15. Gala N, François T, Bernhard D, and Fairon C. A model to predict lexical complexity and to grade words (Un modèle pour prédire la complexité lexicale et graduer les mots)[in French]. Proceedings of TALN 2014 (Volume 1: Long Papers). 2014 :91–102

16. Aleksandrova D and Pouliot V. Cefr-based contextual lexical complexity classifier in english and french. Proceedings of the 18th workshop on innovative use of nlp for building educational applications (bea 2023). 2023 :518–27

17. Alfter D, Cardon R, and François T. A dictionary-based study of word sense dificulty. Proceedings of the 2nd Workshop on Tools and Resources to Empower People with REAding DIficulties (READI) within the 13th Language Resources and Evaluation Conference. 2022 :17–24

18. Grainger J and Jacobs AM. Masked partial-word priming in visual word recognition: efects of positional letter frequency. Journal of experimental psychology: human perception and performance 1993; 19:951

19. Mahowald K, Dautriche I, Gibson E, and Piantadosi ST. Word forms are structured for eficient use. Cognitive science 2018; 42:3116–34

20. Carus A and Mesut A. A new compression algorithm for fast text search. Turkish Journal of Electrical Engineering and Computer Sciences 2016; 24:4355–67

21. Ferrand L, New B, Brysbaert M, Keuleers E, Bonin P, Méot A, Augustinova M, and Pallier C. The French Lexicon Project: Lexical decision data for 38,840 French words and 38,840 pseudowords. Behavior research methods 2010; 42:488–96

22. Mathey S and Zagar D. The orthographic neighbourhood frequency efect in French: A letter-case manipulation study. Canadian Journal of Experimental Psychology/Revue canadienne de psychologie expérimentale 2006; 60:159

23. New B, Araújo V, and Nazzi T. Diferential processing of consonants and vowels in lexical access through reading. Psychological Science 2008; 19:1223–7

24. Carreiras M, Duñabeitia JA, and Molinaro N. Consonants and vowels contribute diferently to visual word recognition: ERPs of relative position priming. Cerebral Cortex 2009; 19:2659–70

25. Carreiras M and Price CJ. Brain activation for consonants and vowels. Cerebral Cortex 2008; 18:1727–35

26. Zipf GK. Relative frequency as a determinant of phonetic change. Harvard studies in classical philology 1929; 40:1–95

27. Pagel M, Atkinson QD, and Meade A. Frequency of word-use predicts rates of lexical evolution throughout Indo-European history. Nature 2007; 449:717–20

## Appendix A Low- and High-Complexity Means for Significant Variables

Table 13: Mean values for low- and high-complexity French words and Bonferroni-adjusted p-values.
<table><tr><td>Variable</td><td>Mean – Low Complexity (n = 5,000)</td><td>Mean – High Complexity (n = 5,000)</td><td>p value (Bonf. ×459)</td></tr><tr><td>LENGTH</td><td>6.694</td><td>9.766</td><td>&lt; .0001</td></tr><tr><td>SYLLABLES</td><td>2.071</td><td>3.050</td><td>&lt; .0001</td></tr><tr><td>UNIQUE_LETTERS</td><td>5.922</td><td>7.562</td><td>&lt; .0001</td></tr><tr><td>LETTER_DIVERSITY</td><td>0.897</td><td>0.790</td><td>&lt; .0001</td></tr><tr><td>REPEATED_LETTERS</td><td>0.773</td><td>2.204</td><td>&lt; .0001</td></tr><tr><td>VOWELS</td><td>2.955</td><td>4.284</td><td>&lt; .0001</td></tr><tr><td>CONSONANTS</td><td>3.740</td><td>5.481</td><td>&lt; .0001</td></tr><tr><td>CV_TRANSITIONS</td><td>4.138</td><td>6.339</td><td>&lt; .0001</td></tr><tr><td>LETTER_ENTROPY</td><td>2.485</td><td>2.796</td><td>&lt; .0001</td></tr><tr><td>REPETITION_RATIO</td><td>0.103</td><td>0.210</td><td>&lt; .0001</td></tr><tr><td>ORTH_NEIGHBOURS</td><td>0.697</td><td>0.112</td><td>&lt; .0001</td></tr><tr><td>CONSONANT</td><td>0.888</td><td>1.470</td><td>&lt; .0001</td></tr><tr><td>CLUSTERS</td><td></td><td></td><td></td></tr><tr><td>e10</td><td>0.009</td><td>0.102</td><td>&lt; .0001</td></tr><tr><td>MAX_CONSONANT_</td><td>1.761</td><td>1.990</td><td>&lt; .0001</td></tr><tr><td>CLUSTER</td><td></td><td></td><td></td></tr><tr><td>i9</td><td>0.003</td><td>0.082</td><td>&lt; .0001</td></tr><tr><td>i8</td><td>0.012</td><td>0.098</td><td>&lt; .0001</td></tr><tr><td>e11</td><td>0.003</td><td>0.074</td><td>&lt; .0001</td></tr><tr><td>a8</td><td>0.007</td><td>0.081</td><td>&lt; .0001</td></tr><tr><td>i7</td><td>0.027</td><td>0.120</td><td>&lt; .0001</td></tr><tr><td>i10</td><td>0.001</td><td>0.058</td><td>&lt; .0001</td></tr><tr><td>a9</td><td>0.002</td><td>0.061</td><td>&lt; .0001</td></tr><tr><td>e9</td><td>0.035</td><td>0.124</td><td>&lt; .0001</td></tr><tr><td>DOUBLE_LETTERS</td><td>0.157</td><td>0.308</td><td>&lt; .0001</td></tr><tr><td>t10</td><td>0.007</td><td>0.071</td><td>&lt; .0001</td></tr><tr><td>BIGRAM_DIVERSITY</td><td>0.995</td><td>0.982</td><td>&lt; .0001</td></tr><tr><td>e12</td><td>0.000</td><td>0.051</td><td>&lt; .0001</td></tr><tr><td>n11</td><td>0.002</td><td>0.056</td><td>&lt; .0001</td></tr></table>

Continued on next page

Table 13 – continued from previous page
<table><tr><td>Variable</td><td>Mean – Low Complexity (n = 5,000)</td><td>Mean – High Complexity (n = 5,000)</td><td>p value (Bonf. ×459)</td></tr><tr><td>s10</td><td>0.009</td><td>0.069</td><td>&lt; .0001</td></tr><tr><td>t12</td><td>0.001</td><td>0.046</td><td>&lt; .0001</td></tr><tr><td>s11</td><td>0.003</td><td>0.050</td><td>&lt; .0001</td></tr><tr><td>a7</td><td>0.023</td><td>0.091</td><td>&lt; .0001</td></tr><tr><td>t11</td><td>0.002</td><td>0.047</td><td>&lt; .0001</td></tr><tr><td>n10</td><td>0.005</td><td>0.054</td><td>&lt; .0001</td></tr><tr><td>n9</td><td>0.013</td><td>0.069</td><td>&lt; .0001</td></tr><tr><td>i11</td><td>0.000</td><td>0.039</td><td>&lt; .0001</td></tr><tr><td>s12</td><td>0.000</td><td>0.037</td><td>&lt; .0001</td></tr><tr><td>a10</td><td>0.000</td><td>0.036</td><td>&lt; .0001</td></tr><tr><td>MEAN_SYLLABLE_</td><td>3.560</td><td>3.293</td><td>&lt; .0001</td></tr><tr><td>LENGTH</td><td></td><td></td><td></td></tr><tr><td>e13</td><td>0.000</td><td>0.033</td><td>&lt; .0001</td></tr><tr><td>t9</td><td>0.020</td><td>0.071</td><td>&lt; .0001</td></tr><tr><td>n12</td><td>0.001</td><td>0.031</td><td>&lt; .0001</td></tr><tr><td>t8</td><td>0.030</td><td>0.085</td><td>&lt; .0001</td></tr><tr><td>n8</td><td>0.024</td><td>0.073</td><td>&lt; .0001</td></tr><tr><td>06</td><td>0.011</td><td>0.051</td><td>&lt; .0001</td></tr><tr><td>s13</td><td>0.000</td><td>0.025</td><td>&lt; .0001</td></tr><tr><td>e5</td><td>0.168</td><td>0.092</td><td>&lt; .0001</td></tr><tr><td>07</td><td>0.006</td><td>0.039</td><td>&lt; .0001</td></tr><tr><td>18</td><td>0.008</td><td>0.042</td><td>&lt; .0001</td></tr><tr><td>n13</td><td>0.000</td><td>0.023</td><td>&lt; .0001</td></tr><tr><td>19</td><td>0.002</td><td>0.029</td><td>&lt; .0001</td></tr><tr><td>n2</td><td>0.035</td><td>0.086</td><td>&lt; .0001</td></tr><tr><td>s9</td><td>0.027</td><td>0.072</td><td>&lt; .0001</td></tr><tr><td>a6</td><td>0.042</td><td>0.094</td><td>&lt; .0001</td></tr><tr><td>t13</td><td>0.001</td><td>0.023</td><td>&lt; .0001</td></tr><tr><td>é6</td><td>0.056</td><td>0.017</td><td>&lt; .0001</td></tr><tr><td>i1</td><td>0.020</td><td>0.060</td><td>&lt; .0001</td></tr><tr><td>r10</td><td>0.005</td><td>0.031</td><td>&lt; .0001</td></tr><tr><td>a11</td><td>0.000</td><td>0.019</td><td>&lt; .0001</td></tr><tr><td>i6</td><td>0.053</td><td>0.105</td><td>&lt; .0001</td></tr><tr><td>e6</td><td>0.161</td><td>0.097</td><td>&lt; .0001</td></tr></table>

Continued on next page

Table 13 – continued from previous page
<table><tr><td>Variable</td><td>Mean – Low Complexity (n = 5,000)</td><td>Mean – High Complexity (n = 5,000)</td><td>p value (Bonf. ×459)</td></tr><tr><td>r9</td><td>0.013</td><td>0.045</td><td>&lt; .0001</td></tr><tr><td>i12</td><td>0.000</td><td>0.017</td><td>&lt; .0001</td></tr><tr><td>t14</td><td>0.000</td><td>0.017</td><td>&lt; .0001</td></tr><tr><td>e14</td><td>0.000</td><td>0.017</td><td>&lt; .0001</td></tr><tr><td>02</td><td>0.195</td><td>0.129</td><td>&lt; .0001</td></tr><tr><td>n7</td><td>0.037</td><td>0.078</td><td>&lt; .0001</td></tr><tr><td>o10</td><td>0.001</td><td>0.019</td><td>&lt; .0001</td></tr><tr><td>m8</td><td>0.001</td><td>0.018</td><td>&lt; .0001</td></tr><tr><td>r11</td><td>0.001</td><td>0.017</td><td>&lt; .0001</td></tr><tr><td>m9</td><td>0.001</td><td>0.017</td><td>&lt; .0001</td></tr><tr><td>17</td><td>0.018</td><td>0.047</td><td>&lt; .0001</td></tr><tr><td>05</td><td>0.040</td><td>0.078</td><td>&lt; .0001</td></tr><tr><td>e8</td><td>0.082</td><td>0.132</td><td>&lt; .0001</td></tr><tr><td>08</td><td>0.006</td><td>0.027</td><td>&lt; .0001</td></tr><tr><td>111</td><td>0.000</td><td>0.013</td><td>&lt; .0001</td></tr><tr><td>s14</td><td>0.000</td><td>0.012</td><td>&lt; .0001</td></tr><tr><td>110</td><td>0.001</td><td>0.014</td><td>&lt; .0001</td></tr><tr><td>d8</td><td>0.001</td><td>0.015</td><td>&lt; .0001</td></tr><tr><td>09</td><td>0.003</td><td>0.018</td><td>&lt; .0001</td></tr><tr><td>u10</td><td>0.001</td><td>0.014</td><td>&lt; .0001</td></tr><tr><td>e15</td><td>0.000</td><td>0.010</td><td>&lt; .0001</td></tr><tr><td>u9</td><td>0.004</td><td>0.019</td><td>&lt; .0001</td></tr><tr><td>r8</td><td>0.033</td><td>0.063</td><td>&lt; .0001</td></tr><tr><td>011</td><td>0.000</td><td>0.011</td><td>&lt; .0001</td></tr><tr><td>p6</td><td>0.004</td><td>0.020</td><td>&lt; .0001</td></tr><tr><td>m11</td><td>0.000</td><td>0.010</td><td>&lt; .0001</td></tr><tr><td>n14</td><td>0.000</td><td>0.010</td><td>&lt; .0001</td></tr><tr><td>m10</td><td>0.001</td><td>0.011</td><td>&lt; .0001</td></tr><tr><td>i13</td><td>0.000</td><td>0.009</td><td>&lt; .0001</td></tr><tr><td>r12</td><td>0.000</td><td>0.009</td><td>&lt; .0001</td></tr><tr><td>j1</td><td>0.019</td><td>0.005</td><td>&lt; .0001</td></tr><tr><td></td><td>0.000</td><td>0.008</td><td>&lt; .0001</td></tr><tr><td>a12</td><td></td><td>0.021</td><td>&lt; .0001</td></tr><tr><td>v1 012</td><td>0.044 0.000</td><td>0.008</td><td>&lt; .0001</td></tr></table>

Continued on next page

Table 13 – continued from previous page
<table><tr><td>Variable</td><td>Mean – Low Complexity (n = 5,000)</td><td>Mean – High Complexity (n = 5,000)</td><td>p value (Bonf. ×459)</td></tr><tr><td>TRIGRAM_DIVERSITY</td><td>1.000</td><td>0.998</td><td>&lt; .0001</td></tr><tr><td>u12</td><td>0.000</td><td>0.008</td><td>&lt; .0001</td></tr><tr><td>16</td><td>0.028</td><td>0.052</td><td>&lt; .0001</td></tr><tr><td>d7</td><td>0.005</td><td>0.018</td><td>&lt; .0001</td></tr><tr><td>é9</td><td>0.006</td><td>0.019</td><td>&lt; .0001</td></tr><tr><td>i5</td><td>0.089</td><td>0.127</td><td>&lt; .0001</td></tr><tr><td>é10</td><td>0.000</td><td>0.009</td><td>&lt; .0001</td></tr><tr><td>112</td><td>0.000</td><td>0.008</td><td>&lt; .0001</td></tr><tr><td>z6</td><td>0.011</td><td>0.002</td><td>&lt; .0001</td></tr><tr><td>12</td><td>0.043</td><td>0.022</td><td>&lt; .0001</td></tr><tr><td>u11</td><td>0.000</td><td>0.008</td><td>&lt; .0001</td></tr><tr><td>VOWEL_CONSONANT_</td><td>0.872</td><td>0.828</td><td>&lt; .0001</td></tr><tr><td>RATIO</td><td></td><td></td><td></td></tr><tr><td>c8</td><td>0.002</td><td>0.012</td><td>&lt; .0001</td></tr><tr><td>z7</td><td>0.013</td><td>0.002</td><td>&lt; .0001</td></tr><tr><td>a4</td><td>0.069</td><td>0.102</td><td>&lt; .0001</td></tr><tr><td>s15</td><td>0.000</td><td>0.007</td><td>&lt; .0001</td></tr><tr><td>t7</td><td>0.050</td><td>0.079</td><td>&lt; .0001</td></tr><tr><td>c7</td><td>0.007</td><td>0.021</td><td>&lt; .0001</td></tr><tr><td>t15</td><td>0.000</td><td>0.007</td><td>&lt; .0001</td></tr><tr><td>m6</td><td>0.014</td><td>0.031</td><td>&lt; .0001</td></tr><tr><td>m7</td><td>0.007</td><td>0.021</td><td>&lt; .0001</td></tr><tr><td>ê2</td><td>0.007</td><td>0.000</td><td>&lt; .0001</td></tr><tr><td>n15</td><td>0.000</td><td>0.006</td><td>&lt; .0001</td></tr><tr><td>m2</td><td>0.015</td><td>0.032</td><td>&lt; .0001</td></tr><tr><td>m5</td><td>0.012</td><td>0.028</td><td>&lt; .0001</td></tr><tr><td>c9</td><td>0.001</td><td>0.008</td><td>&lt; .0001</td></tr><tr><td>r5</td><td>0.077</td><td>0.108</td><td>&lt; .0001</td></tr><tr><td>b10</td><td>0.000</td><td>0.006</td><td>&lt; .0001</td></tr><tr><td>é11</td><td>0.000</td><td>0.006</td><td>&lt; .0001</td></tr><tr><td>y3</td><td>0.014</td><td>0.004</td><td>&lt; .0001</td></tr><tr><td>c6</td><td>0.018</td><td>0.034</td><td>&lt; .0001</td></tr><tr><td>h9</td><td>0.000</td><td>0.005</td><td>&lt; .0001</td></tr><tr><td>z5</td><td>0.007</td><td>0.001</td><td>&lt; .0001</td></tr></table>

Continued on next page

Table 13 – continued from previous page
<table><tr><td>Variable</td><td>Mean – Low Complexity  $( n = 5 , 0 0 0 )$ </td><td>Mean – High Complexity  $( n = 5 , 0 0 0 )$ </td><td>p value  $\mathbf { ( B o n f . \nabla \times 4 5 9 ) }$ </td></tr><tr><td></td><td>0.000</td><td>0.006</td><td> $1 . 1 5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>p9 r7</td><td>0.048</td><td>0.073</td><td> $1 . 4 3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>b8</td><td>0.001</td><td>0.007</td><td> $1 . 4 9 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>c11</td><td>0.000</td><td>0.005</td><td> $1 . 4 9 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>f6</td><td>0.003</td><td>0.012</td><td> $1 . 6 8 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>u3</td><td>0.087</td><td>0.060</td><td> $1 . 8 8 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>r4</td><td>0.061</td><td>0.088</td><td> $2 . 2 2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>â2</td><td>0.008</td><td>0.001</td><td> $2 . 4 4 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>11</td><td>0.032</td><td>0.017</td><td> $2 . 5 9 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>é5</td><td>0.040</td><td>0.023</td><td> $7 . 1 4 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>r13</td><td>0.000</td><td>0.005</td><td> $7 . 1 6 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>d9</td><td>0.001</td><td>0.007</td><td> $8 . 1 0 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>h10</td><td>0.000</td><td>0.006</td><td> $9 . 0 6 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>h3</td><td>0.005</td><td>0.014</td><td> $1 . 0 8 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>g7</td><td>0.003</td><td>0.011</td><td> $1 . 1 2 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>é12</td><td>0.000</td><td>0.004</td><td> $1 . 2 1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>113</td><td>0.000</td><td>0.004</td><td> $1 . 2 1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>s8</td><td>0.052</td><td>0.074</td><td> $1 . 2 3 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>a1</td><td>0.063</td><td>0.088</td><td> $1 . 4 7 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>t3</td><td>0.061</td><td>0.085</td><td> $1 . 4 8 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>04</td><td>0.060</td><td>0.084</td><td> $1 . 4 8 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>u8</td><td>0.011</td><td>0.023</td><td> $2 . 0 3 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>û3</td><td>0.005</td><td>0.000</td><td> $2 . 5 4 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>g6</td><td>0.014</td><td>0.027</td><td> $2 . 7 2 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>f1</td><td>0.053</td><td>0.034</td><td> $2 . 8 8 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>r3</td><td>0.082</td><td>0.108</td><td> $3 . 0 1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>t16</td><td>0.000</td><td>0.004</td><td> $3 . 4 5 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>t6</td><td>0.059</td><td>0.082</td><td> $3 . 7 0 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>x3</td><td>0.008</td><td>0.002</td><td> $4 . 1 1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>m12</td><td>0.000</td><td>0.004</td><td> $5 . 3 1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>â6</td><td>0.000</td><td>0.004</td><td> $5 . 8 5 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>b11</td><td>0.000</td><td>0.004</td><td> $5 . 8 5 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>u13</td><td>0.000</td><td>0.004</td><td> $5 . 8 5 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>g8</td><td>0.002</td><td>0.009</td><td> $8 . 2 1 \times 1 0 ^ { - 3 }$ </td></tr></table>

Continued on next page

Table 13 – continued from previous page
<table><tr><td>Variable</td><td>Mean - Low Complexity  $( n = 5 , 0 0 0 )$ </td><td>Mean – High Complexity  $( n = 5 , 0 0 0 )$ </td><td>p value  $\mathbf { ( B o n f . \nabla \times 4 5 9 ) }$ </td></tr><tr><td>è8</td><td>0.000</td><td>0.004</td><td> $8 . 9 7 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>q11</td><td>0.000</td><td>0.004</td><td> $9 . 9 1 \times 1 0 ^ { - 3 }$ </td></tr></table>