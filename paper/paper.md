XXX-X-XXXX-XXXX-X/XX/$XX.00 ©20XX IEEE

# Arabic Noisy Text Correction Using T

# Transformers

Abstract **_—_** In this paper, we introduce a new method of
correcting noisy Arabic text based on the encoder-decoder
transformer model. We web-scraped 100,000 news articles from
**اليوم السابع** ("youm7.com") and added noise at character and
punctuation levels in a systematic fashion to generate noisy-clean
text pairs in parallel. We trained from-scratch a transformer
model that reached 89.36% accuracy in correcting artificially
inserted noise following 10 epochs of training. This model shows
effectiveness in correcting frequent Arabic spelling errors,
character replacements, and punctuation mistakes. Evaluation on
test examples reveals a character error rate (CER) drop of up to
87.5% relative to the noisy input. This work is a contribution to
Arabic natural language processing in that it offers a new parallel
noisy-clean Arabic news text corpus of modern Egyptian Arabic
and an effective correction model that is applicable to practical
use..

Keywords **_—_** Arabic spelling correction, transformers, T
model, natural language processing, soft errors

```
I. INTRODUCTION
```

## Arabic text processing faces unique

## challenges due to the language's complex

## morphology and orthography. While previous

## work has focused on standard Arabic corpora,

## modern web content—particularly news articles—

## presents distinct characteristics that require

## specialized handling. This paper addresses the

## problem of correcting noisy Arabic text in news

## articles, which may contain errors from various

## sources including OCR mistakes, typing errors,

## and automatic transcription artifacts. Our

## contributions include Creation of a parallel corpus

## of 10,0000 clean-noisy Arabic news articles from

## اليوم السابع, Development of a systematic noise

## injection methodology tailored for Arabic text,

## Implementation and evaluation of an encoder-

## decoder transformer model for Arabic text

## correction and Analysis of model performance on

## different error types and text lengths

## The Arabic language itself poses built-in

## challenges in spelling correction automation since

## it has a complex morphology, complex

## orthography, and high frequency of visually

## confusable characters whose substitution alters the

## meaning. Soft spelling errors such as Alef form

## confusion (ا vs. ى), Hamza placement mistakes (أ,

## ئ, ؤ), and Teh Marbuta (ة) vs. Heh (ه) interchange

## are some of the most common errors. These errors

## usually occur due to the contextual ambiguity of

## the Arabic writing in which one character

## difference could change the semantic sense of a

## word. For example, the word "ىضَ َق" ("he judged")

## spelled incorrectly as "اضَق" becomes nonsensical,

## and "جامعة" ("university") spelled incorrectly as

## "جامعه" loses grammaticality.

## During the implementation phase, we also

## encountered a few other kinds of errors besides the

## normal soft errors, which checked the model's

## ability to correct. The morpho-syntactic errors

## involved swapping feminine and masculine forms,

## incorrect tense in the verb, or word reordering

## within a sentence. For instance, in the sentence

## ""اشتريت خمس كتب, the word "خمس" must be in

## feminine form "خمسة" for the sentence to be

## grammatically correct. Lexical and semantic errors

## included typographical errors due to writing

## phonetically instead of correctly, such as in the

## sentence "البرمائيات من الظفدع يعتبر" where there is a

## lexical error in the word "ظفدع", which is not

## correct; the right word is "ضفدع", indicating

## confusion between the letters "ظ" and "ض". Soft

## errors also surfaced by confusing character shapes

## such as Hamza shapes (ء, ئ, ؤ), Alef shapes (أ, إ, آ),

## and Teh (ت) vs. Heh (ه), especially while typing

## morphologically rich verbs. For example, the noun

## "تفاؤل" bears a Hamza over Waw (ؤ), while the noun

## is changed to the verb "تفاءل", the Hamza stands

## alone. These difficulties highlighted the

## importance of a strong model that incorporates

## both syntactic and semantic contextual knowledge

## to facilitate correct correction.

#### II. RELATED WORK

```
Arabic spelling correction has long been a
challenging problem because of the richness of the
morphology and orthography of the language. Numerous
researchers have addressed the problem based on rule-based
and machine learning methods.
```

```
Al- Qaraghuli and Jaafar [1] proposed a
transformer model based on T5 to use in Arabic soft spelling
correction. Their work proved that sequence-to-sequence
models could effectively cope with a large variety of
orthographic mistakes by generating the corrected text in a
direct fashion. Their system was, however, constrained by
the use of only character-level context and lacked deeper
```

semantic modeling. Building on their work, we integrate
AraBERT in contextual reranking to achieve improved
processing of ambiguous correction where syntax and
meaning come into play.

Abandah et al.'s [2] suggestion of employing a
BiLSTM model for soft spelling correction is effective in
modeling sequential context, though the BiLSTM model is
not good with distant context and is not highly adaptable to
diversity in actual writing style. From their method, we
build upon by utilising the more expressive attention of
transformer models and training with a noisy Arabic
spelling problem-specific large-scale data set.

Raffel et al. [3] also proposed the T5 architecture in
a multilingual context where all NLP tasks were considered
as text-to-text transformations. This design influenced our
use of the base model, with the use of T5 being chosen
because of its flexibility and performance. But we design the
architecture specifically with Arabic in mind by
implementing a custom tokenizer and contextual reranking
and adapting it to the specific challenges of the language.

Guo et al. [4] provided the Wiki-40B corpus
containing a massive cleansed version of Arabic Wikipedia.
While the above dataset is helpful in general language
modeling, it does not contain the error diversity and richness
needed for spelling correction tasks. For comparison, our
100,000 Youm7 news article dataset has synthetically
injected realistic errors inspired from actual mistake patterns
and is thus more geared towards correction training.

Aloyaynaa and Kotb [5] explored grammatical
error detection with pre-trained Arabic transformers. While
their study is aimed at general grammatical correctness, in
our research we limit the scope to spelling-level problems
and to soft errors and set up a two-model scheme with a
combination of T5 and AraBERT to solve these problems
with high accuracy.
Finally, works by Al-Saqqaf and Al-Busaidi [6]
and Alsaawi [7] examine typical spelling errors of Arabic-
speaking learners. These papers confirm the high frequency
of errors of type misplacing Hamza and confusion of Alef
and justify our interest in such error types. Though their
research is diagnostic in scope, the current paper suggests a
machine-based correction scheme specifically trained to
cope with such patterns of error.

#### III. METHODOLOGY

First, Our approach builds on the T5 model, treating
spelling correction as a text-to-text task. We injected artificial
errors into the Arabic subset of the dataset and augmented it
with a new dataset we scraped from youm7.com, simulating
common soft spelling mistakes.

A. Data Collection pipeline

The starting point of our methodology involved
constructing a wide-ranging Arabic text corpus that reflects
the linguistic richness of Modern Standard Arabic with local
variations. This was attained through the creation of a refined
web scraping pipeline of Youm7.com, one of Egypt's most

```
reliable news sources with broad coverage in various fields. In
selection, we gave priority to websites posting professionally
edited material to guarantee grammatical correctness and
natural linguistic patterns.
```

```
We used Python's asynchronous aiohttp with
BeautifulSoup to implement the scraping system and
engineered the crawler to conduct polite scraping by
integrating a number of important features such as randomized
request delays of 1-3 seconds, automated retry on failed
requests, and strict robots.txt following. The system filtered
the article pages through various validation phases, initially
removing boilerplate information such as navigation menus
and advertisements via carefully constructed CSS selectors
and later checking text integrity with Unicode character range
checks.
```

```
Out of the original pool of 120,000 raw articles, we
employed strict quality filters to come up with our end corpus.
Articles of less than three-paragraphs length or those with less
than 50 words were automatically filtered out, together with
those with too much non-Arabic text (above 15% Latin
characters and symbols). Normalization of text format was
also performed during the cleaning to remove extra spaces and
normalize Arabic punctuation marks. This yielded a carefully
handpicked collection of 100,000 high-quality articles from
news, education, health, and technology categories with
publication dates largely in 2020-2025 to guarantee the use of
current language.
```

```
With regard to enabling scalable research and being
mindful of copyright constraints, we created a two-tiered data
release plan. Clean/noisy pairs of the entire corpus will be
accessible to research scholars via a secure access program,
and a representative sample of 1,000 articles with
comprehensive metadata will be openly released. Each article
in the set we created has full provenance information that
includes original URL, publishing date, and word count that
allow for focused investigation of style variations in various
contexts and timeframes.
Dataset #Articles Sources Features Availability
```

Youm 7 (^) 100,000 Youm
Egypt
Raw news
articles;
potential for
soft spelling
error analysis
Private
SANAD (^) 194,797 Alkhaleej Single-label
classification;
curated for
text
categorization
tasks
Public
Ultimate
Arabic
News
193,000 Multiple Preprocessed
and raw
versions;
suitable for
various NLP
tasks
Public
ANAD (^) 500,000 12 diverse
arabicnews
websites
Annotated
articles;
comprehensiv
e coverage of
modern
standard
Public

```
Arabic
```

Amina (^) 1,850,
00

#### 9

```
newspapers
across Arab
andnon-
Arab
countries
```

```
Includes
metadata
(e.g., author,
date, images);
supports
multimodal
research
```

```
Public
```

```
Wiki-
40B
```

#### ~1.

```
million
```

```
Arabic
wikipedia
```

```
Cleaned and
aligned for
language
modeling
tasks
```

```
Public
```

B. Noise Injection Process

Building realistic noisy-clean text pairs necessitated the
construction of a highly advanced error injection method that
is more than plain character substitutions. Developing our
noise model was guided by rigorous study of genuine Arabic
spell errors from three sources: learner error studies in
academia, study of social media writing, and professional
editors' correction logs. This work identified seven main error
categories representing more than 85% of frequent Arabic
spelling errors that formed the bulk of our generation of
synthetic errors.

At the character level, our model works with a multi-step
transformation process. It starts by detecting error positions
based on linguistic susceptibility - such as words terminating
in ة or having hamza variants. It follows that with context-
sensitive transformations in a weighted probability matrix that
considers (1) visual similarity between characters (such as أ
and إ), ( 2 ) proximity in sound, and (3) grammatical function.
The system has a 7% error rate at the character level that is
carefully distributed to prevent clustering and creating
unrealistic patterns of error.

A stand-alone noise component was used for punctuation
and spacing that introduces naturalistic spacing variations,
comma use, and quotation marks. This encompasses culturally
dependent patterns such as doubled punctuation marks to
express emphasis (used in Arabic informal text) and alternate
comma forms ( ،vs ,.). Punctuation noise is set to operate at a
15% modification level but does maintain semantic
boundaries to avoid sentence fragmentation.

The noise injection process produces parallel text pairs
that preserve the original meaning while introducing
pedagogically relevant errors. For example, the clean sentence
"مواعيد التسجيلأعلنت كلية الصيدلة عن " might transform to "أعلنت
كليه الصيدله عن مواعيد التسجيل", demonstrating both ه→ة
substitution and alef omission errors. We validated the noise
patterns through multiple rounds of expert review, with
Arabic linguists rating over 90% of the synthetic errors as
"plausible" or "highly plausible" for native speaker writing.

```
Example pairs:
```

```
Noisy: بـ جامعة حلوان على تعزز ا فذ إطار حرص كلضة الصيدلة ...
```

```
Clean: بـ جامعة حلوان على تعزيز ا في إطار حرص كلية الصيدلة ...
```

```
Noisy: انلاق مبادرة م ，，الودلية أعلنت مدارء النيل امصرية ...
```

```
Clean: انطلاق مبادرة م ，أعلنت مدارس النيل المصرية الدولية ...
```

```
Noisy: كلأاب نعشلاب نيمؤتهملا ةداسلاب ةرهاقلا ةعماج بيهُت...
```

```
Clean: كلأا نأشلاب نيمتهملا ةداسلاب ةرهاقلا ةعماج بيهُت...
```

### C. Model Architecture and Training

```
Our encoder-decoder transformer model includes:
```

```
 Embedding Layer: 256-dimensional character
embeddings
 Positional Encoding: Added to preserve sequence
order
 Encoder: 3 layers with 8 attention heads each
 Decoder: 3 layers with 8 attention heads each
 Output layer: Linear projection to vocabulary size
```

```
a) T5 for Sequence-to-Sequence Correction
```

```
The core of our system is a custom T5 model, which we
have trained on Arabic. We employed T5 because of its
flexibility in text-to-text tasks, enabling us to frame soft
spelling correction as a translation task: translating "noisy"
Arabic sentences to "clean" ones.
```

## This diagram illustrates the encoder-decoder

## flow with attention heads, showing the data

## transformation from noisy input to corrected

## output.

```
The architecture was designed in particular to cope with the
complexities of Arabic scriptThe architecture was designed
in particular to cope with the complexities of Arabic script:
```

- Decoder Layers: 3 layers each, trading off
  expressive power and training efficiency.
- Model Dimension: 512-dimensional hidden states,
  sufficiently large to incorporate rich morphological signals
  like affixation and internal root modification.
- Feedforward Layer: Dimension of 512 with ReLU
  activation for deeper feature extraction.
- Multi-Head Attention: 8 attention heads to allow the
  model to focus on parallel syntactic and orthographic
  structures across the sentence.
- Dropout: A dropout of 0.1 to prevent overfitting,
  especially necessary in consideration of the sparsity of
  certain Arabic error types.

A special tokenizer was also created to differentiate between
visually analogous but semantically different characters
(e.g., ء vs. أ, ؤ vs. ئ). This level of precision is crucial in
Arabic because diacritic position and letter shape can
profoundly change meaning.

```
b) Training Strategy
```

```
We implemented:
```

- Scheduled sampling
- Gradient scaling for mixed-precision training
- Check-pointing
- Character-level accuracy and loss monitoring
  The model achieved 89.36% accuracy after 10 epochs of
  training on an NVIDIA GPU.

```
c) Optimization for Deployment
```

To make the system viable for real-world applications (e.g.,
browser-based tools or educational platforms), we
performed several deployment-oriented optimizations:

1. Quantization: Reduced the model size to improve
   inference speed with minimal accuracy loss.
2. Graph Pruning: Removed redundant computation
   paths to reduce memory usage and boost latency
   performance.
3. Transformers Pipeline:

```
We adopted a 3-layer encoder-decoder T5 model with the
following configuration:
```

```
 Model Dimension: 512
```

```
 Feed-forward Dimension: 512
```

```
 Attention Heads: 8
```

```
 Dropout Rate: 0.
```

```
The model was trained using Adam optimizer, a learning
rate of 1e-4, and sparse categorical cross entropy loss.
```

```
IV. EVALUATION METRICS
```

```
A. Primary Metrics
CER is our primary evaluation metric, particularly well-
suited for character-level correction tasks like soft spelling
errors in Arabic. It measures the minimum number of
insertions (I), deletions (D), and substitutions (S) required to
transform the predicted text into the reference (ground truth)
normalized by the number of characters (N) in the reference
text. The formula is given as:
```

#### CER =

#### I+D+S

#### N

#### × 100%

```
This diagram shows how Character Error Rates (CER) are
distributed across the test set, indicating the model's
consistency.
```

```
This confusion matrix visualizes how often each
Arabic character was correctly predicted versus
misclassified, helping us understand model errors.
```

This pie chart breaks down the types of errors
(Insertions, Deletions, Substitutions) the model corrects,
highlighting common mistakes.

#### V. EXPERIMENTAL RESULTS AND DISCUSSION

Our comprehensive evaluation demonstrates that
the hybrid T5 architecture achieves state-of-the-art
performance in Arabic

A. Dataset Characteristics

Our training set was composed of 10,000 pairs of articles
with their clean and artificially corrupted (noisy) forms. On
average, each article was some 420 words long to allow
good context modeling of local and distant character
dependencies. In applying a balanced noise type distribution
between insertions, deletions, and substitutions to the
articles, we intended to replicate realistic error patterns of
user-generated Arabic text and OCR results.

The noise injection ensured linguistic realism through
prioritization of typical typing mistakes, which include
similar letter-forms of Arabic letters (i.e., "ب" and "ت")
being mistaken for each other, similar-sounding letters, and
most common punctuation mistakes. This confusion enabled
the model to generalize among various sources of textual
noise and mimic realistic use effectively.

B. Results

The final evaluation on the test set showed:

Character Accuracy: 89.36%
CER Reduction: Up to 87.5% improvement compared to the
noisy input
WER Improvement: Significant reduction compared to the
baseline noisy text WER

The model achieved consistent correction capability across
different article lengths, with shorter sentences achieving
slightly higher correction rates. Longer sentences presented
more challenges, especially when multiple noise types
coexisted within the same input.

```
C. Comparison with Baseline
```

```
We compared our model against two baselines:
```

1. Rule-Based Correction Systems\*: These systems
   achieved only 30-50% character accuracy. They were
   inconsistent and failed to generalize when facing unseen
   noise patterns.
2. BiLSTM Models: Bidirectional LSTM architectures
   performed better, reaching around 80-85% character
   accuracy. However, they struggled with long-range
   dependencies and complex error patterns.

```
Our Transformer-based approach outperformed both
baselines significantly. Thanks to the self-attention
mechanism, the model could effectively capture both local
and global dependencies, allowing it to correct not only
isolated mistakes but also systematic errors spread across a
sentence.
```

```
Moreover, the Transformer’s parallel processing capabilities
provided faster inference times during evaluation compared
to sequential models like BiLSTMs.
```

#### VI. CONCLUSION

```
This work has introduced an end-to-end Arabic soft spelling
correction solution, resolving some of the root issues of
Arabic natural language processing with a carefully
designed process for data acquisition, model development,
and diligent testing. This work makes three significant
contributions to Arabic NLP:
```

```
First, we established a new pipeline for generating high-
quality parallel corpora for Arabic spelling correction,
demonstrated through our dataset of 100,000 Youm7 articles
with exactly injected noise patterns. This specially designed
dataset to record real-world spelling errors in modern
Egyptian Arabic news text addresses an important gap in
Arabic NLP resources and provides a useful benchmark for
future research. Our noise injection approach, informed by
empirical analysis of common Arabic spelling mistakes,
offers a reproducible method of creating realistic training
data for resource-poor languages.
```

```
Second, we constructed and optimized a transformer
architecture specifically for Arabic spelling correction. Our
design choices—e.g., the 3-layer encoder-decoder model,
512 - dim embeddings, and Arabic-specific tokenization—
were chosen specifically to meet the complex orthography
of the language while being computationally lightweight.
The model's 89.36% character accuracy and 87.5% CER
reduction evidently outperform other existing methods,
particularly in terms of handling hard cases like hamza
placement and teh marbuta/heh distinction.
```

```
Third, our comprehensive evaluation approach, with
quantitative metrics (CER, WER) and qualitative human
assessment, offers a model for future work in Arabic text
```

correction. The findings not only support our technical
approach but also reveal valuable lessons about the salient
issues of Arabic NLP, specifically dealing with dialectical
variations and out-of-vocabulary proper nouns.

The practical application of this work is immediate and
significant. Educational institutions may integrate our model
in writing support software to allow students to become
accomplished in Arabic orthography. News organizations
and content websites may implement the system to
guarantee excellent standards of written Arabic. Our method
also provides a framework for developing similar systems
for other languages with complex morphology and
orthography.

Future work should be targeted in three areas: (1) extending
the model's capacity to deal with dialectal Arabic through
focused data gathering and dialect identification modules,
(2) including grammar-based error detection to construct
more solid writing support systems, and (3) designing more
sophisticated confidence estimation mechanisms to make it
more reliable for production environments.

This work advances Arabic NLP by demonstrating the
effectiveness of well-designed transformer architectures and
high-quality domain-specific data to address long-standing
Arabic spelling correction challenges. The resources and

```
methods published will help to facilitate further progress in
Arabic language technologies and resulting applications
across the Arab world.
```

#### VII. REFERENCES

```
[1] M. Al-Qaraghuli and O. A. Jaafar, "Arabic Soft Spelling Correction with
T5," Jordanian Journal of Computers and Information Technology, vol.
10, no. 1, 2024.
[2] G. Abandah, A. Suyyagh, and M. Z. Khedher, "Correcting Arabic Soft
Spelling Mistakes Using BiLSTM-based Machine Learning,"
International Journal of Advanced Computer Science and Applications,
vol. 13, no. 5, 2022.
[3] C. Raffel et al., "Exploring the Limits of Transfer Learning with a
Unified Text-to-Text Transformer," Journal of Machine Learning
Research, vol. 21, pp. 1-67, 2020.
[4] M. Guo et al., "Wiki-40B: Multilingual Language Model Dataset,"
Proceedings of the 12th Language Resources and Evaluation
Conference, 2020.
[5] S. Aloyaynaa and Y. Kotb, "Arabic Grammatical Error Detection Using
Transformers-based Pre-trained Language Models," ITM Web of
Conferences, vol. 56, 2023.
[6] Al-Saqqaf, A. H., & Al-Busaidi, S. (2015). English Spelling Errors
Made by Arabic-Speaking Students. English Language Teaching, 8(7),
181 – 199.
[7] Alsaawi, A. (2015). Spelling Errors Made by Arab Learners of
English. International Journal of Linguistics, 7(5), 55–65.
```

#### .
