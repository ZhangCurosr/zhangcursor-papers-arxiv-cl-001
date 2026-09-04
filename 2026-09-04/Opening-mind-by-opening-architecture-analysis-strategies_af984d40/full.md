# Opening mind by opening architecture: analysis strategies

Francesco Vitucci<sup>1</sup>, Giuseppe Silvi<sup>1</sup>, Daniele Giuseppe Annese<sup>1</sup>, Francesco Scagliola<sup>1</sup> and Anthony Di Furia<sup>1</sup>

<sup>1</sup> Conservatorio “N. Piccinni” - Bari anthonydifuria.sound@gmail.com

Abstract. In numerical signal processing for electroacoustic composition, the progressive loss of specific development and research environments caused by the increasing use of digital market tools has favoured the dominance of the closed-architecture audio processor model. This model, while powerful, envisions the possibility of describing output data about its perceived characteristics, but at the cost of ignoring its internal process and interacting systems, which become complex, powerful environments but closed in an inscrutable black box, a loss we must consider. Any digital signal processing technique tells a story. Just as the words of a language incorporate social, historical and technical polysemic layers, a signal processor has its own story of implementation, a gradual technological achievement with its inevitable aesthetic consequences. Through the looking-glass of literature, one can access those environments with renewed awareness by reestablishing a scientific method and an attitude to research. In this specific case, starting from the case study of Manfred Schroeder’s historical reverbs, we illustrate the process of building analytical evaluation tools, as well as practical implementation, at the basis of a conscious study path.

Keywords: Reverberator, Faust, Porting, Impulse Response

## 1 Introduction

The process of IT democratization started with the introduction of the Personal Computer can be observed from multiple social perspectives. Unconditional accessibility to digital signal processing tools features a wide range of signal processors (DSP) to an ever wider range audience of users. This opening has generated a new category of producer users, who, fascinated by the ease of use and the speed of obtaining results, have become accustomed to the black box approach. This closed architecture model represents a paradigm in which users can use a process without necessarily understanding its internal workings: they evaluate the output based on perceived characteristics, ignoring implementation details, in one consequent social normalization in which the user, or the music producer, «è in realtà un music prosumer (che ‘produce’ solo consumando servizi e dispositivi tagliati su misura)»<sup>1</sup>. [4] Parallel to this trend is an ever-growing lack of specific research and development environments, in an extended sense of awareness and critical consciousness. The word environment, denotes «the conditions that you live or work in and the way that they influence how you feel or how efectively you can work»<sup>2</sup> and applies to the specific discussion, teaching or work environment, to design and implementation software, to large research centres, once the only environments where everything that is being discussed here was possible. Speaking of software, it is worth underlining that, the adoption of the model black box, with increasingly high-level function libraries, ready for use, cannot be inspected to understand how it works, promotes a superficiality in learning and using the tools sound processing.

In this scenario, training and education play a fundamental role in promoting an open architecture approach, encouraging students to explore and understand the theoretical and practical foundations of audio signal processing, in order to develop critical skills and creativity. This model, also called white box, [5] predicts that the focus is on understanding the underlying mechanisms of the process, promoting critical awareness and in-depth understanding. The purpose of this discussion is therefore to show a path of understanding, acquisition and potential creative development. At the same time, we want to chart a course for future artistic and musical research works scientists who can find fertile ground in the use of the proposed paradigm to take root.

## 2 Analysis methodology

The implementation path that will follow is based on a series of specific choices. First of all, we chose to observe the reverberations from the point of view of the contents because the literature in this regard is open and accessible, in line with the proposed paradigm, as well as among the richest: precisely this richness has its violent counterpart inaccessible in prosumer use. It is precisely this method of use that, for example, has led to the use of convolution reverbs, the closed application of a mathematical system that does not contain the reverberation process but only imitates one of its results. On the contrary, historical algorithmic implementations have from time to time recombined basic reverberant components to obtain diferent results and, precisely in this process, a possible creative outlet was identified, in continuity with the attitude of the past: these were historical moments in which implementation primarily meant dealing with the limitations imposed by the machines available, thus generating an impulse for continuous investigation.

The starting point of the research was the implementation of M. R. Schroeder reverberator. In 1962, he developed the first digital reverb algorithm in history [6]. It is made of two basic components: a delay in feedback loop, or comb filter and an all-pass filter. The former is built up by inserting a simple delay line into a feedback loop; by doing so, one can produce multiple echoes, with exponential decay, as shown in Fig. 1. By mixing the direct sound and the delayed sound the comb filter is converted to an all-pass filter, a basic reverberant unit with flat frequency response.

The creation of this path was done in a textual programming environment, a choice attributable to educational needs, as the understanding of all the pieces of classical programming is more direct: declarations of variables, the definition of functions and their concatenation. In this specific case, we initially chose to use the Faust language (Functional Audio Stream) which is a functional programming language for sound synthesis and audio processing.<sup>3</sup> In addition to being textual, this language has a series of specific characteristics of an open architecture model: the highest level functions available are implemented using the same language, allowing the operation to be observed in every greater depth, up to the basic “primitive” components; the tools provided have analytical capabilities that allow both to visualize the sample-by-sample behaviour of a processor and to generate its block diagram. This represents the background of the “Reverberators” project<sup>4</sup> [14], a path of reconstruction and conscious implementation of reverberators from historical literature.

The idea therefore of porting the project into a Csound<sup>5</sup> [1] environment, in an attempt to preserve the chosen methodology, was immediately measured with the ability to build precise analysis tools for the impulse responses of the components under construction (Sec. 3). However, it is not simply a question of replacing a language with an alternative, but rather the integration of the various experiences, merging the strengths of the various approaches: from this perspective it is possible to observe the reconstruction of some of Faust’s composition structures, such as par and seq<sup>6</sup>, indispensable for the implementation path (Sec. 4). Precisely in this step, we encountered diferent eficiency limitations between the two languages. Therefore, understanding that these are teaching study strategies, we want to underline how the construction of compiled csound opcodes in C is the next step which, in the growth of the project, will represent a future stage of development.

![](images/b29317188d6e6f6c3693543d9de6fb996945ea47a7f10d324e34ac1476406809.jpg)  
Fig. 1. Plot of the impulse response of the FDL component, obtained from within Csound with the proposed analysis routine.

## 3 Impulse Response

As already explored in other research works [8], once a component has been developed, it is necessary to test its impulse response (IR). To create a precise IR analysis routine, an appropriate orchestra of two instruments was developed, as can be seen in the code snippet below. instr 1 is responsible for generating a Dirac pulse, testing the component (Schroeder FDL<sup>7</sup>), writing the IR in a globally accessible table and scheduling instr 2. The latter plots the values in the console and a txt file; then it can be parsed to extract the desired amplitude values (for this purpose a script in Wolfram Language<sup>8</sup> was used, which also automates the graphic representation process (Fig. 1).

Analysis routine and plot of the impulse response

<CsoundSynthesizer>   
<CsOptions>   
-n -d -+rtmidi=NULL -M0 -m0d   
</CsOptions>   
<CsInstruments>   
ksmps = 32   
nchnls = 2   
0dbfs = 1   
#include "schroeder.udo"   
instr 1   
setksmps 1   
ires system\_i 1, "mkdir -p PLOT"   
iT = 1;time   
iG = 1/ sqrt(2);gain   
iSR = sr;sample rate   
aDirac mpulse 1,1   
aR = 0   
iLenTable = 32   
giFDL\_PLOT ftgen 2, 0, iLenTable, -2, 0   
aFDL\_PLOT = FDL\_SCH(aDirac, iT, iG, iSR);process to test   
aIndex phasor sr/iLenTable   
tablew aFDL\_PLOT, aIndex \* iLenTable, giFDL\_PLOT   
schedule 2, iLenTable/sr, 1   
endin   
instr 2   
ftprint giFDL\_PLOT   
ftsave "PLOT/1FDL\_PLOT.txt", 1, giFDL\_PLOT   
ires system\_i 1, "wolframscript -script plot.wls"   
exitnow   
endin   
</CsInstruments>   
<CsScore>   
i1 0 100   
</CsScore>   
</CsoundSynthesizer>

## 4 Compositional Structures: par and seq

Once the evaluation and analysis tools have been acquired and the actual efectiveness of the constructed basic components has been tested, the blocks can be composed. If we observe Schroeder’s reverb implementation model (Fig. 2), we identify two behaviours that deserve attention: a first construct with structurally identical blocks (but with diferent parameters) positioned in parallel precedes a second in which the blocks are chained in series.

![](images/7528f8f9d5a7936ce790099214f421da946495ce39e82546d58a484e1c28887a.jpg)  
Fig. 2. Faust Block diagram of Schroeder’s reverb implementation: a section of four comb filters in paralle precedes a chain of two all-pass filters in series.

These two compositional structures are already supported by Faust (see Sec. 2), but are not present in canonical Csound language, so two User Defined Opcodes were specifically written (as can be seen in the code snippet below). At the moment the proposed UDOs are not generic, as Faust par and seq; nevertheless, they are easily adaptable for any other opcode to compose in parallel and in series.

par and seq User Defined Opcodes

```lua
----PAR FDL--
opcode PAR_FDL_SCH, a, ai[]i[]iii
aPulse, iTfdl[], iGfdl[], iSR, iN, icnt xin
icnt = icnt + 1
aFDL = FDL_SCH(aPulse, iTfdl[icnt - 1], iGfdl[icnt - 1], iSR)
aMix init 0
if icnt < iN then
aMix PAR_FDL_SCH aPulse, iTfdl, iGfdl, iSR, iN, icnt
endif
xout aMix + aFDL
endop
;--------SEQ APF--
opcode SEQ_APF_SCH, a, ai[]i[]iii
aPulse, iTapf[], iGapf[], iSR, iN, icnt xin
icnt = icnt + 1
aAPF = APF_SCH(aPulse , iTapf[icnt - 1], iGapf[icnt - 1], iSR)
if icnt < iN then
aAPF SEQ_APF_SCH aAPF, iTapf, iGapf, iSR, iN, icnt
endif
xout aAPF
endop
```

Of course, a simple chain like the one shown in Fig. 2 can also be created without the illustrated UDOs. However, it becomes dificult to ignore them if the number of components in series or parallel becomes much higher, to research and study the creative expansion of historical processes, one of the topics underlying the “Reverberators” project. [8]

## 5 Conclusions

It is appropriate to underline the strong didactic aspects of the proposed paradigm. Opening architectures means opening paths of understanding and learning; it means colliding with and circumventing external problems and limitations; it therefore means not consuming but producing the environment with which to interact, in a virtuous mechanism of musical, creative and professional growth.

Schroeder’s work [6] clearly shows architectures impulse responses that led the first Faust implementation. [8] Its Csound porting [1] produced an analysis environment, necessary for the understanding and consequent conscious writing of open architectures and further creative applications.

## References

1. Accessed: 2024/07/29. https://github.com/s-e-a-m/csound-libraries.

2. Lazzarini, V. et al.: Csound: A Sound and Music Computing System. Springer (2016)

3. J. Heintz, Floss Manual Csound, 2023. Accessed: 2024/07/29. https://flossmanual.csound.com.

4. A. Di Scipio, Circuiti del Tempo, p. 560. Libreria Musicale Italiana srl, 2021.

5. M. E. Khan and F. Khan, “A comparative study of white box, black box and grey box testing techniques”, International Journal of Advanced Computer Science and Applications, vol. 3, no. 6, 2012.

6. M. R. Schroeder, “Natural sounding artificial reverberation”, J. Audio Eng. Soc, vol. 10, no. 3, pp. 219–223, 1962.

7. M. R. Schroeder and B. F. Logan, “colorless artificial reverberation”, IRE Transactions on Audio, no. 6, pp. 209–214, 1961.

8. D. G. Annese, F. Vitucci, A. Di Furia, F. Scagliola, and G. Silvi, “Archeotopologie: implementazione critica di memorie senza colore,” Atti del XXIV Colloquio di Informatica Musicale, 2024.

9. M. R. Schroeder, “New method of measuring reverberation time”, Acoustical Society of America, 1964.

10. M. R. Schroeder, “Digital simulation of sound transmission in reverberant spaces”, The Journal of the acoustical society of america, vol. 47, no. 2A, pp. 424–431, 1970.

11. C. Roads, The computer music tutorial. MIT press, 1996.

12. Accessed: 2024/07/29. https://ccrma.stanford.edu/\~dattorro/Griesinger.pdf.

13. Accessed: 2024/07/29. https://ccrma.stanford.edu/\~dattorro/Griesinger.jpg.

https://github.com/s-e-a-m/faust-libraries/blob/master/src/seam.schroeder.lib

15. J. Dattorro, “Efect design, part 1: Reverberator and other filters”, J. Audio Eng. Soc, vol. 45, no. 9, pp. 660–684, 1997.

16. W. G. Gardner, “A realtime multichannel room simulator”, J. Acoust. Soc. Am, vol. 92, no. 4, p. 2395, 1992.

17. W. C. Sabine, Collected papers on acoustics. Harvard university press, 1922.

18. R. Vermeulen, “Stereo-reverberation”, J. Audio Eng. Soc, vol. 6, no. 2, pp. 124–130, 1958.

19. M. R. Schroeder, “Listening with two ears”, Music Perception: An Interdisciplinary Journal, vol. 10, no. 3, pp. 255–280, 1993.

20. J. O. Smith, “Physical audio signal processing”, 2010, Accessed: 2024/05/07.

21. J. A. Moorer, “About this reverberation business”, Computer music journal, pp. 13–28, 1979.