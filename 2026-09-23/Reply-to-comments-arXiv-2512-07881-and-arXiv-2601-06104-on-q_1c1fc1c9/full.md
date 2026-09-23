# Reply to comments arXiv:2512.07881 and arXiv:2601.06104 on quantum structure in human and AI-generated language

Massimiliano Sassoli de Bianchi<sup>∗</sup> and Roberto Leporini<sup>†</sup>

## Abstract

We reply to the comments by M. Sienicki and K. Sienicki (arXiv:2512.07881) and by K. Sienicki (arXiv:2601.06104) on our work on quantum-mechanical statistics in human language (arXiv:2407.14924) and on quantum structure in AI-generated language (arXiv:2511.21731). We thank the authors for their careful reading and address what we consider to be the main points of criticism: the exploratory nature of the protocol used in the experiments with large language models; the role of marginal-law violations, and of the Contextuality-by-Default criterion, in the identification of entanglement; the limited diagnostic value of a Bose-Einstein fit taken in isolation; the meaning of assigning the lowest energy levels to the most frequent words; and the relation between the vector spaces used by LLMs and quantum state spaces. We also correct a typographical error in Table 3 of arXiv:2511.21731, which does not afect the reported CHSH value.

The comment arXiv:2512.07881 (Sienicki & Sienicki, 2025) concerns our analysis of Bose-Einstein statistics in human language (Aerts et al., 2025), while arXiv:2601.06104 (Sienicki, 2026a) concerns our study of quantum structure in AI-generated language (Aerts et al., 2026), and refers to the former for part of its arguments. K. Sienicki also privately sent us a longer manuscript (Sienicki, 2026b), in which some of these objections are developed further and a few new ones are raised, notably those concerning marginal laws and vector-space representations. Since these are closely related to the points made in the published comments, we address them here as well.

Experimental design. A first criticism, raised in Sienicki (2026a), is that the experiments with the two LLMs were not conducted under a suficiently stringent protocol, particularly with regard to the independence of repeated responses and the overall data-collection procedure. We agree that future experiments should employ more rigorous controls, including independent sessions, randomized presentation orders, explicit reset conditions, and a more systematic assessment of run-to-run variability. The aim of the present study, however, was more limited and exploratory. Since the same conceptual-combination tests had already been performed with human participants, our primary objective was to determine whether comparable response patterns, and in particular CHSH violations, would also arise readily in LLMs. The reported results should therefore be understood as a first phenomenological comparison rather than as a definitive experimental characterization of LLM cognition.

Marginal laws. Let us now consider the criticism that, because our data are inconsistently connected and violate the marginal laws, their analysis within the Contextuality-by-Default framework does not reveal contextuality beyond direct contextual influences and therefore does not establish a genuine situation of entanglement comparable to that observed in physics. This objection to our understanding of entanglement in human and artificial cognition is not new. We refer, for instance, to our response in (Aerts et al., 2018a) to the criticisms raised in (Dzhafarov & Kujala, 2014; Kujala et al., 2014), and to the general analysis in (Aerts et al., 2019), where we examined entanglement in both physical and cognitive systems, including the role of marginal-law violations in its representation within the quantum formalism.

Without entering into the technical details here, our approach understands entanglement primarily as arising from a connection between the entities involved. In the case of conceptual entities, this is a connection of meaning. Correlations are actualized through coincidence measurements on the basis of this connection, and the marginal laws may or may not be satisfied depending on the symmetry and isotropy of the experimental arrangement. From this perspective, the Contextuality-by-Default criterion addresses a precise and important question (whether contextuality remains after direct contextual influences have been taken into account) but it does not exhaust all possible operational interpretations of entanglement. Consequently, subtracting the degree of inconsistent connectedness from the usual CHSH expression should not, in our view, be regarded as a universally decisive test for the presence or absence of entanglement in the broader sense adopted in our work.

It is also worth recalling that theoretical descriptions in physics often rely on idealizations that are only approximately realized in laboratory conditions. Violations of marginal laws have repeatedly been reported in analyses of Bell-type experiments and, in our view, cannot always be dismissed a priori as mere experimental errors; see, for example, (Adenier & Khrennikov, 2007; De Raedt et al., 2012, 2013; Adenier & Khrennikov, 2017; Bednorz, 2017; Kupczynski, 2017). The usual assumption that the measurements in a Bell-test scenario are suficiently separated to admit a single fixed tensor-product representation may therefore fail in actual experimental implementations. One should consequently be cautious about identifying the full phenomenon of entanglement exclusively with the most idealized no-signaling scenario.

A related example concerns photon indistinguishability. For photons to display indistinguishable boson behavior at a beam splitter, their frequencies and arrival times must be suficiently close; otherwise, they behave as distinguishable entities. See the discussion in (Aerts & Beltran, 2020) and the references therein. Thus, even for qualitatively identical photons, indistinguishability is context-dependent. This illustrates more generally why ideal mathematical conditions and their concrete experimental realization should not be conflated.

Word shufling. As noted in Sienicki & Sienicki (2025), the words of a text can be arbitrarily shufled while preserving exactly the same word frequencies. The shufled text will therefore display the same Bose-Einstein fit as the original one, showing that the fit alone cannot diagnose the presence of narrative meaning. We agree with this observation and have not claimed otherwise. A Bose-Einstein-type frequency distribution is not, by itself, a suficient condition for semantic organization or understanding. At most, it is one statistical signature that becomes relevant when considered together with independent evidence that a system generates and adapts meaningfully to diferent semantic contexts. This point was stated explicitly in (Sassoli de Bianchi & Sassoli de Bianchi, 2026):

It is important to note that the mere existence of a program capable of producing texts that comply with Bose-Einstein statistics is not, in itself, evidence of semantic understanding, since even a program that simply produces collages from pre-existing texts would be capable of doing so. The result becomes significant, however, if the program is also able to adapt dynamically to diferent semantic contexts. In other words, the claim defended here concerns the emergence of deeper organizational structures that cannot be reduced to frequency-based regularities alone. It is the presence of the latter in combination with these deeper organizational structures that confers upon LLMs a distinct cognitive status. This is why quantum notions such as superposition, contextuality, entanglement-like correlations, and bosonic amplification mechanisms become explanatorily relevant. At the same time, although both LLMs and quantum models rely on vector-space formalisms, the precise relation between the semantic spaces of LLMs and the more richly structured spaces of quantum theory remains an open question for further investigation.

Thus, the word-shufling and similar arguments establish a limitation of the rank-frequency analysis, but it does not render that analysis irrelevant. It shows that the Bose-Einstein fit must be interpreted as one component of a broader evidential framework, rather than as a stand-alone proof of the presence of meaning.

Energy ordering. Both comments question the identification of word energies with ranks (Sienicki & Sienicki, 2025; Sienicki, 2026a). More specifically, Sienicki asks whether the assignment of word energies is merely a rank convention or whether the energies are intended to have semantic or physical content. If the latter is intended, he argues that the energy should ideally be defined by a rule independent of the observed word frequencies, for example through a Hamiltonian.

We agree that the construction of an independently motivated Hamiltonian would be desirable. Zipf’s pioneering work already pointed in this direction through his principle of least efort, proposed as an explanation of the power law that bears his name. According to Zipf, speakers and hearers tend to minimize their efort while preserving eficient communication and mutual understanding. This requires a balance between the diversification and unification of words, and therefore an optimization of ambiguity, understood as the capacity of words to express diferent meanings in diferent contexts. Zipf also observed that shorter words generally require less physical efort and therefore tend to be used more frequently (Zipf, 1949).

Following this line of thought, a Hamiltonian associated with words in a text would plausibly contain at least two contributions. The first would be related to the physical or computational cost of producing a word, and could depend on quantities such as the number of letters, phonemes, syllables, articulatory gestures, or an empirically estimated production time. The second would be a cognitive-semantic contribution associated with the optimization of understanding and with the position of the word within the global semantic organization of the text. Because a word’s semantic versatility depends on its relations to the entire semantic environment, this second term would naturally have a mean-field-like or self-consistent character.

These considerations indicate why defining a meaningful Hamiltonian independently of observed word frequencies is a dificult problem. Developing such a construction is an objective for future work and requires a clearer account of how the notion of energy (which historically predates its modern physical formalization) should be extended to cognitive and cultural systems. Nevertheless, irrespective of the precise form that such models may ultimately take, it remains certainly meaningful to assign the lowest energy levels to the most frequently occurring words in a text, and these assignments should not be regarded as a mere convention based on rank.

Vector spaces. We agree with Sienicki that the existence of a vector-space representation does not, by itself, imply a quantum structure. Classical physics also makes extensive use of vectors, without thereby becoming quantum. A vector-space representation becomes specifically relevant to quantum modeling only when the vectors represent states, when measurements and probabilities are defined through an appropriate non-Kolmogorovian structure, and when genuinely nonclassical features such as incompatibility, interference, or entanglementlike correlations are present.

Our claim is therefore not that the real vector spaces used internally by LLMs are automatically Hilbert spaces of quantum theory. Rather, the claim is that the probabilistic and semantic behavior of language can be modeled by quantum-inspired structures defined on vector spaces. This is also the central methodological idea developed in quantum information retrieval (Van Rijsbergen, 2004; Melucci, 2015; Aerts et al., 2018b) and quantum cognition (Busemeyer & Bruza, 2012). Of course, the precise mathematical relation between the internal real vector representations learned by LLMs and the more highly structured complex state spaces of quantum theory remains to this day an open problem.

Erratum. Finally, we take this opportunity to correct a typographical error in Table 3 of Aerts et al. (2026), pointed out in Sienicki (2026b), which may give the impression of an arithmetic error in one of the reported calculations. The calculation itself is correct; the error occurs in the transcription of the data in Table 3. More precisely, it is Cat Growls that should have probability $P ( A _ { 2 } ^ { \prime } , B _ { 1 } ) = 0 . 2 2 2$ , whereas Cat Whinnies should have probability $P ( A _ { 2 } ^ { \prime } , B _ { 2 } ) = 0$ . With this correction, the expectation value $E ( A ^ { \prime } , B ) = 0 . 5 5 6$ and the corresponding CHSH value reported in the article remain unchanged.

## References

Adenier, G. & Khrennikov, A. (2007). Is the fair sampling assumption supported by EPR experiments? J. Phys. B: Atomic, Molecular and Optical Physics 40, 131–141.

Adenier, G. & Khrennikov, A. (2017). Test of the no-signaling principle in the Hensen loophole-free CHSH experiment. Fortschritte der Physik (Progress in Physics) 65, 1600096.

Aerts, D. and Beltran, L. (2020). Quantum structure in cognition: Human language as a Boson gas of entangled words. Foundations of Science 25, 755–802.

Aerts, D., Aerts Argu¨elles, J., Beltran, L., Geriente, S., Sassoli de Bianchi, M., Sozzo, S. & Veloz, T. (2018). Spin and Wind Directions II: A Bell state quantum model. Foundations of Science 23, 337–365.

Aerts, D., Aerts Argu¨elles, J., Beltran, L., Beltran, L., Distrito, I., Sassoli de Bianchi, M., Sozzo, S. & Veloz, T. (2018). Towards a Quantum World Wide Web. Theoretical Computer Science 752, 116–131.

Aerts, D., Aerts Argu¨elles, J., Beltran, L., Geriente, S., Sassoli de Bianchi, M., Sozzo, S. and Veloz, T. (2019). Quantum entanglement in physical and cognitive systems: a conceptual analysis and a general representation. The European Physical Journal Plus 134, 493.

Aerts, D., Aerts Argu¨elles, J., Beltran, L., Geriente, S., Leporini, R., Sassoli de Bianchi, M., & Sozzo, S. (2026). Identifying Quantum Structure in AI Language: Evidence for Evolutionary Convergence of Human and Artificial Cognition. Entropy, 28(6), 622. See also: arXiv:2511.21731 [cs.CL].

Aerts, D., Aerts Argu¨elles, J., Beltran, L., Sassoli de Bianchi, M. and Sozzo, S. (2025). The Origin of Quantum Mechanical Statistics: Some Insights from the Research on Human Language. Philos Trans A Math Phys Eng Sci 383 (2311): 20230285. See also: arXiv:2407.14924 [q-bio.NC].

Bednorz A. (2017). Analysis of assumptions of recent tests of local realism. Phys. Rev. A 95, 042118.

Busemeyer, J. and Bruza, P. (2012). Quantum Models of Cognition and Decision. Cambridge: Cambridge University Press.

De Raedt, H., Michielsen, K. & Jin, F. (2012). Einstein-Podolsky-Rosen-Bohm laboratory experiments: Data analysis and simulation. AIP Conf. Proc. 1424, 55–66.

De Raedt H., Jin, F. & Michielsen, K. (2013). Data analysis of Einstein-Podolsky-Rosen-Bohm laboratory experiments. Proc. of SPIE 8832, The Nature of Light: What are Photons? V, 88321N; doi: 10.1117/12.2021860.

Dzhafarov, E. N., & Kujala, J. V. (2014). Selective influences, marginal selectivity, and Bell/CHSH inequalities. Topics in Cognitive Science, 6, 121–128.

Dzhafarov, E. N., Kujala, J. V., Cervantes, V. H., Zhang, R., & Jones, M. (2016). On contextuality in behavioural data. Phil. Trans. R. Soc. A 374 (2068), 20150234.

Kupczynski, M. (2017). Is Einsteinian no-signalling violated in Bell tests? Open Phys. 15, 739–753.

Melucci, M. (2015). Introduction to Information Retrieval and Quantum Mechanics. The Information Retrieval Series 35. Springer-Verlag Berlin Heidelberg.

Sassoli de Bianchi, L. & Sassoli de Bianchi, M. (2026). Understanding without Consciousness: Quantum Structures and the Autonomy of Meaning. Foundations of Science. https://

doi.org/10.1007/s10699-026-10039-2.

Sienicki, M. & Sienicki, K. (2025). Revised comment on the paper titled “The Origin of Quantum Mechanical Statistics: Insights from Research on Human Language”. arXiv:2512.07881 [q-bio.NC].

Sienicki, K. (2026a). Comment on arXiv:2511.21731v1: Identifying Quantum Structure in AI Language: Evidence for Evolutionary Convergence of Human and Artificial Cognition. arXiv:2601.06104 [cs.AI].

Sienicki, K. (2026b). Context Dependence Is Not Yet Quantum Structure: A Critical Comment on “Identifying Quantum Structure in AI Language.” Unpublished manuscript, privately communicated by the authors on July 13, 2026.

Van Rijsbergen, C. J. (2004). The Geometry of Information Retrieval. Cambridge University Press.

Zipf, G. K. (1949). Human Behavior and the Principle of Least Efort. Cambridge: Addison-Wesley.