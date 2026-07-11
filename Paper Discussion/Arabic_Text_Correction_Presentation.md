<!-- Slide number: 1 -->

![Picture 2.png](Picture2.jpg)
Intelligent Methods, Systems, and Applications
(IMSA)
An End-to-End Arabic Text Correction Framework:
Comparing Custom Character-Level Transformers
and Pre-trained AraBART
Mohamed Soltan
Abdelmonem Hatem
Mohamed Taha
Youssef Khalaf
Dept. of Computer Science
MSA University
Dept. of Computer Science
MSA University
Dept. of Computer Science
MSA University
Dept. of Computer Science
MSA University

![Picture 6.jpg](Picture12.jpg)
October University for Modern Sciences and Arts (MSA)
Cairo, Egypt
IEEE / IMSA 2026

### Notes:
Good morning. We present an end-to-end Arabic text correction framework comparing a compact custom Transformer with a pre-trained AraBART multi-modal service.

<!-- Slide number: 2 -->

![Picture 2.png](Picture2.jpg)
Outline
• Introduction & Motivation
• Problem Statement & Soft Errors
• Related Work & Research Gap
• Proposed End-to-End Framework
• Data, Normalization & Noise Injection
• Custom Transformer Architecture
• AraBART Multi-modal Deployment
• Experiments, Metrics & Baselines
• Results, Error Analysis & Discussion
• Conclusion & Future Work

### Notes:
Walk through the agenda briefly. Emphasize the two-path story: controlled benchmarking vs. real-world deployment.

<!-- Slide number: 3 -->

![Picture 2.png](Picture2.jpg)
Introduction
• Arabic text correction is critical for education, media QA, accessibility, and moderation.
• Challenges: orthographic variants, visually confusable letters, optional diacritics, and spacing that changes meaning.
• OCR and ASR add broken words, missing characters, merged tokens, and repeated substitutions.
• Core design question: compact task-specific model vs. larger pre-trained Arabic model?
• This work studies that trade-off with two complementary correction paths.

![soft_errors.png](Picture4.jpg)

<!-- Slide number: 4 -->

![Picture 2.png](Picture2.jpg)
Motivation

OCR Noise
ASR Noise
Broken tokens, substitutions, and spacing errors from scanned Arabic documents.
Speech transcripts introduce phonetic and boundary mistakes.

Typing Soft Errors
Deployment Gap
Alef/Hamza/Teh Marbuta confusions are frequent and meaning-changing.
Most Arabic studies report scores, few ship multi-modal user-facing systems.

<!-- Slide number: 5 -->

![Picture 2.png](Picture2.jpg)
Problem Statement
• Small character-level errors in Arabic can change meaning, break token boundaries, or reduce grammatical fluency.
• Correction is more than dictionary lookup — it needs sequence context at character and sentence levels.
• Existing resources often lack controlled noisy-clean pairs tailored to modern Arabic news orthography.
• Benchmarks alone are insufficient: real systems must also handle OCR images, audio, and live speech.
• Goal: a reproducible end-to-end workflow covering data curation, noise synthesis, training, evaluation, and deployment.

<!-- Slide number: 6 -->

![Picture 2.png](Picture2.jpg)
Arabic Soft Error Categories

![soft_errors.png](Picture3.jpg)
Guided by learner studies, social-media patterns, and editor correction logs (~85% of common Arabic spelling errors).

<!-- Slide number: 7 -->

![Picture 2.png](Picture2.jpg)
Contributions

![contributions.png](Picture3.jpg)

<!-- Slide number: 8 -->

![Picture 2.png](Picture2.jpg)
Related Work — Research Timeline

![related_timeline.png](Picture3.jpg)

### Notes:
Trace the field from QALB shared tasks to BiLSTM, AraBART+GED, and T5 soft spelling — then introduce our end-to-end deployment contribution.

<!-- Slide number: 9 -->

![Picture 2.png](Picture2.jpg)
Related Work — Comparative Landscape

![related_comparison_table.png](Picture3.jpg)

### Notes:
Highlight Abandah BiLSTM (CER 1.28%), Al-Qaraghuli T5 (CER 0.77% on Test200), and Alhafni AraBART+GED SOTA on QALB. Emphasize that none deliver our full multi-modal workflow.

<!-- Slide number: 10 -->

![Picture 2.png](Picture2.jpg)
Related Work — Capability Positioning

![related_positioning.png](Picture3.jpg)
Qualitative coverage view: prior systems are strong on soft spelling or GEC, but weak on multi-modal deployment + controlled end-to-end analysis together.

<!-- Slide number: 11 -->

![Picture 2.png](Picture2.jpg)
Research Gap

![research_gap.png](Picture3.jpg)

### Notes:
Prior art solves pieces of the problem. Our gap fill is the reproducible workflow that couples controlled modeling with OCR/ASR deployment.

<!-- Slide number: 12 -->

![Picture 2.png](Picture2.jpg)
Proposed Framework Overview

![two_tier.png](Picture3.jpg)

### Notes:
This is the key conceptual slide: Branch A optimizes controlled character fidelity; Branch B optimizes real-world multi-modal usability.

<!-- Slide number: 13 -->

![Picture 2.png](Picture2.jpg)
End-to-End Pipeline

![pipeline1.png](Picture3.jpg)
Shared data path splits into Branch A (custom Transformer) and Branch B (AraBART multi-modal service).

### Notes:
Walk left-to-right: Youm7 corpus, cleaning, normalization, synthetic noise, then the two branches.

<!-- Slide number: 14 -->

![Picture 2.png](Picture2.jpg)
Correction Examples

Noisy
Clean
اعلنت كليه الصيدله عن مواعبد التسجيل
أعلنت كلية الصيدلة عن مواعيد التسجيل
→
Teh Marbuta / Heh + Alef + letter swaps

Noisy
Clean
انلقت مبادرة مدارس النيل المصرية الودلية
انطلقت مبادرة مدارس النيل المصرية الدولية
→
Dropped / substituted characters

Noisy
Clean
هزا كتاب مفيد
هذا كتاب مفيد
→
Visual soft substitution

<!-- Slide number: 15 -->

![Picture 2.png](Picture2.jpg)
Data Acquisition & Curation
• Scraped Youm7 Arabic news with async aiohttp + BeautifulSoup.
• Polite crawling: 1–3s randomized delays, retries, robots.txt compliance.
• Filters: remove boilerplate, short articles, and samples with >15% non-Arabic.
• Final clean corpus: 100,000 high-quality articles (news, education, health, tech).
• Prepared 10,000 noisy-clean pairs for controlled modeling.
• Split: 8,000 train / 1,000 val / 1,000 held-out test.

![dataset_split.png](Picture4.jpg)

<!-- Slide number: 16 -->

![Picture 2.png](Picture2.jpg)
Arabic Corpus Context
| Dataset | Scale | Source | Use / Availability |
| --- | --- | --- | --- |
| Youm7 (this work) | 100,000 | Youm7 Egypt | Noisy-clean generation / Private |
| SANAD | 194,797 | AlKhaleej+ | Classification / Public |
| Ultimate Arabic News | 193,000 | Multiple outlets | General NLP / Public |
| ANAD | 500,000 | 12 news sites | Annotated news / Public |
| Amina | ~1.85M | Regional newspapers | Multimodal metadata / Public |
| Wiki-40B (AR) | ~1M+ | Arabic Wikipedia | Language modeling / Public |

<!-- Slide number: 17 -->

![Picture 2.png](Picture2.jpg)
Normalization & Noise Injection

![noise_pipeline.png](Picture3.jpg)
Deterministic normalization + 20% corruption budget enables reproducible character-level supervision.

<!-- Slide number: 18 -->

![Picture 2.png](Picture2.jpg)
Corruption Budget & Soft Categories

![corruption_budget.png](Picture3.jpg)
• 10% substitutions from confusable maps
• 5% deletions + 5% insertions
• Keyboard-adjacency & punctuation drift
• Hamza/Alef, Teh Marbuta–Heh, Yaa/ى
• Spacing boundary shifts included
• Designed for typing + OCR + ASR realism

<!-- Slide number: 19 -->

![Picture 2.png](Picture2.jpg)
Custom Seq2Seq Transformer (Branch A)

![seq2seq.png](Picture3.jpg)

<!-- Slide number: 20 -->

![Picture 2.png](Picture2.jpg)
Transformer Encoder–Decoder Intuition

![example.jpg](Picture3.jpg)
• Noisy Arabic → embeddings + positions
• Encoder builds contextual memory
• Decoder generates corrected chars
• Cross-attention links source ↔ target
• Softmax over character vocabulary

<!-- Slide number: 21 -->

![Picture 2.png](Picture2.jpg)
Model Configuration & Training

Architecture
Training Protocol
• Character-level encoder-decoder Transformer
• Embedding dim: 256
• Encoder / Decoder layers: 3 / 3
• Attention heads: 8
• FFN dimension: 512
• Dropout: 0.1
• Max sequence length: 128
• Optimizer: Adam (lr = 1e-4)
• Loss: sparse categorical cross-entropy
• Scheduled sampling: teacher forcing 1.0 → 0.3
• Mixed precision + checkpointing
• Batch size: 32 | Epochs: 10
• GPU training ≈ 100 minutes
• Checkpoint size ≈ 50 MB

<!-- Slide number: 22 -->

![Picture 2.png](Picture2.jpg)
AraBART Deployment Service (Branch B)

![multimodal_flow.png](Picture3.jpg)
Model: CAMeL-Lab/arabart-qalb15-gec-ged-13 via Hugging Face + Streamlit (OCR.space + Whisper).

<!-- Slide number: 23 -->

![Picture 2.png](Picture2.jpg)
Why AraBART for Deployment?
• Pretrained Arabic seq2seq model fine-tuned for GEC/GED.
• Strong fluency on orthography + grammar-oriented edits.
• Zero-shot adoption in this study (no project fine-tuning).
• Supports text, files, OCR images, audio, and live speech.
• Trade-off: heavier footprint (~1.5 GB) vs. broader usability.
• Complements the compact custom Transformer path.

![footprint.png](Picture4.jpg)

<!-- Slide number: 24 -->

![Picture 2.png](Picture2.jpg)
Evaluation Metrics & Protocol

![eval_protocol.png](Picture3.jpg)

<!-- Slide number: 25 -->

![Picture 2.png](Picture2.jpg)
Primary Metric: Character Error Rate

• CER suits character-level soft spelling correction.
• BLEU used as secondary lexical/sequence quality metric.
• Baselines: normalization-only + dictionary correction.
• Project reference ranges: rule-based & BiLSTM-style.
• Always report setting A and setting B separately.
CER Definition
CER = (S + D + I) / N

S = substitutions
D = deletions
I = insertions
N = reference length (characters)

<!-- Slide number: 26 -->

![Picture 2.png](Picture2.jpg)
Baseline Context

![baseline_bars.png](Picture3.jpg)
Custom Transformer reaches 89.36% character accuracy on synthetic project runs — best in-domain fidelity.

<!-- Slide number: 27 -->

![Picture 2.png](Picture2.jpg)
Headline Results

![metric_cards.png](Picture3.jpg)
Read as two settings: in-domain reconstruction fidelity vs. zero-shot multi-modal deployment.

<!-- Slide number: 28 -->

![Picture 2.png](Picture2.jpg)
Training Dynamics

![training_curve.png](Picture3.jpg)
• Epoch 1: 38.38%
• Epoch 6 peak: 91.69%
• Epoch 10: 89.36%
• Fast early learning of dominant patterns.
• Later slight drop suggests mild overfitting to synthetic noise.
• Earlier stopping / stronger regularization recommended.

<!-- Slide number: 29 -->

![Picture 2.png](Picture2.jpg)
Side-by-Side Results
| Attribute | Custom Transformer | AraBART Service |
| --- | --- | --- |
| Setting | In-domain synthetic | Zero-shot multi-modal |
| Model | Char seq2seq Transformer | arabart-qalb15-gec-ged-13 |
| CER | 0.0364 | 0.0950 |
| BLEU | 0.292 | Not reported |
| Peak token acc. | 91.69% | N/A (pre-trained) |
| Fine-tuning | Yes | No |
| Training time | ~100 min GPU | Inference only |
| Size | ~50 MB | ~1.5 GB |
| Inputs | Text pipeline | Text, file, OCR, audio, speech |

### Notes:
Stress that these are two operating settings, not a single fair leaderboard. Custom wins CER in-domain; AraBART wins deployment breadth.

<!-- Slide number: 30 -->

![Picture 2.png](Picture2.jpg)
CER Comparison Across Settings

![cer_comparison.png](Picture3.jpg)
• Lower CER = better reconstruction.
• Custom model wins in-domain fidelity.
• AraBART remains preferred for fluency and multi-modal robustness.
• Not a like-for-like contest — two operating settings.

<!-- Slide number: 31 -->

![Picture 2.png](Picture2.jpg)
CER Distribution

![cer.png](Picture3.jpg)
• Most samples concentrate in a low-CER band.
• Indicates stable correction, not luck on a few cases.
• Right tail = heavy corruption samples.
• Useful for reliability assessment in deployment.

<!-- Slide number: 32 -->

![Picture 2.png](Picture2.jpg)
Character Confusion Matrix

![cm.png](Picture3.jpg)
• Strong diagonal = solid identity mapping.
• Off-diagonal cells reveal persistent substitutions.
• Hard cases: Alef/Hamza variants, Yaa/Alif-Maqsura, Teh Marbuta/Heh.
• Whitespace boundary errors amplify local drift.
• Guides targeted augmentation and loss design.

<!-- Slide number: 33 -->

![Picture 2.png](Picture2.jpg)
Error Analysis

![error_pie.png](Picture3.jpg)
• Substitutions dominate (~60–70%).
• Insertions and deletions each ~15–20%.
• Matches Arabic soft-error behavior: near-equivalent character choice.
• Future work should prioritize whitespace + Alef/Hamza contrasts.
• Curriculum / class-weighted objectives can reduce hard confusions.

<!-- Slide number: 34 -->

![Picture 2.png](Picture2.jpg)
Discussion & Deployment Implications

![takeaway_quadrant.png](Picture3.jpg)
• Custom Transformer: best fidelity + tiny checkpoint.
• AraBART: best fluency across modalities.
• Recommended production pattern: two-tier routing.
• Next: shared QALB fair comparison.

<!-- Slide number: 35 -->

![Picture 2.png](Picture2.jpg)
Conclusion
• Delivered a reproducible Arabic correction workflow: collect → normalize → synthesize → train → evaluate → deploy.
• Custom character-level Transformer: strong in-domain reconstruction (CER 0.0364, peak accuracy 91.69%).
• AraBART Streamlit service: better practical fluency across text, OCR, audio, and live speech.
• Remaining errors concentrate in whitespace and Alef/Hamza-style confusions.
• Best practice: two-tier system — compact model for control, pre-trained service for real-world use.

### Notes:
Close with the two-tier takeaway and the three contribution pillars: workflow, dual paths, and strong in-domain CER.

<!-- Slide number: 36 -->

![Picture 2.png](Picture2.jpg)
Future Work

Targeted Augmentation
Regularization & Early Stop
Focus synthetic noise on Alef/Hamza, Teh Marbuta/Heh, and whitespace boundaries.
Preserve peak generalization near epoch 6 and reduce late overfitting.

Shared Benchmarks
Dialect Coverage
Fair comparison on external datasets such as QALB.
Extend to dialectal Arabic with dialect ID + focused data collection.

Grammar Layer
Confidence Estimation
Add grammar-aware detection for broader writing-support systems.
Production-ready uncertainty signals for safer corrections.

<!-- Slide number: 37 -->

![Picture 2.png](Picture2.jpg)
Selected References
• Mohit et al. / Rozovskaya et al., QALB-2014 & QALB-2015 Arabic GEC Shared Tasks.
• Abandah et al., Correcting Arabic Soft Spelling Mistakes Using BiLSTM, IJACSA, 2022.
• Al-Qaraghuli & Jaafar, Arabic Soft Spelling Correction with T5, JJCIT, 2024.
• Alhafni, Inoue, Khairallah & Habash, Arabic GED/GEC with AraBART, EMNLP 2023.
• Antoun et al., AraBERT; Inoue et al., CAMeL Tools / Arabic pretraining.
• Vaswani et al., Attention Is All You Need, NeurIPS, 2017.
• Lewis et al., BART; Raffel et al., T5 — denoising seq2seq pretraining.
• CAMeL-Lab AraBART (arabart-qalb15-gec-ged-13), Hugging Face.
• Radford et al., Whisper; OCR.space API for multi-modal ingestion.

<!-- Slide number: 38 -->

![Picture 2.png](Picture1.jpg)
Thank You
Questions & Discussion
Mohamed Soltan  •  Abdelmonem Hatem  •  Mohamed Taha  •  Youssef Khalaf
October University for Modern Sciences and Arts (MSA)

![Picture 6.jpg](Picture5.jpg)