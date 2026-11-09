---
language:
- uz
library_name: nemo
datasets:
- USC-2.0
- mozilla-foundation/common_voice_17_0
- UzbekVoice
thumbnail: null
tags:
- automatic-speech-recognition
- speech
- audio
- streaming
- Transducer
- RNNT
- FastConformer
- call-center
- telephony
- low-resource
- pytorch
- NeMo
- hf-asr-leaderboard
license: cc-by-4.0
model-index:
- name: stt_uz_fastconformer_transducer_medium_streaming
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: USC 2.0 Test-Clean
      type: USC-2.0
      config: uz
      split: test-clean
      args:
        language: uz
    metrics:
    - name: Test WER
      type: wer
      value: 12.9
    - name: Test CER
      type: cer
      value: 3.5
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: USC 2.0 Test-CallCenter (8 kHz telephony)
      type: USC-2.0
      config: uz
      split: test-callcenter
      args:
        language: uz
    metrics:
    - name: Test WER
      type: wer
      value: 19.4
    - name: Test CER
      type: cer
      value: 5.8
---

# Streaming FastConformer-Transducer for Uzbek (uz)

| [![Model architecture](https://img.shields.io/badge/Model_Arch-FastConformer--Transducer-lightgrey#model-badge)](#model-architecture)
| [![Mode](https://img.shields.io/badge/Mode-Streaming-lightgrey#model-badge)](#intended-use)
| [![Language](https://img.shields.io/badge/Language-uz-lightgrey#model-badge)](#training-data)

A **streaming** automatic speech recognition (ASR) model for Uzbek, built for
real-time telephony in call-center conditions. The model is the ASR stage of the
real-time voice-to-voice RAG question-answering pipeline described in the paper
*"Enhancing Retrieval Robustness through ASR-Aware Query Rewriting in Real-Time
Voice-to-Voice RAG for Uzbek"* (see [Citation](#citation)). In that pipeline, the ASR
transcript is passed to a fine-tuned flan-t5-large query-rewriting module before
hybrid retrieval, which reduces the propagation of ASR errors into the RAG stage.

It is a **FastConformer-Transducer (Medium)** model, fine-tuned from the pre-trained
English FastConformer-Transducer (Medium) checkpoint on **1,903.53 hours** of Uzbek
speech (the USC 2.0 corpus). Unlike attention encoder-decoder models such as Whisper,
which must buffer a full 30-second window before producing output, the transducer
decodes frame-synchronously and emits partial hypotheses as audio arrives — giving a
**First-Token Latency (FTL) of 110 ms**.

> **Output format.** This model outputs lowercase Uzbek Latin text **without
> punctuation and without capitalization**. Training transcripts were normalized by
> punctuation removal and case folding. If you need punctuation and casing, apply a
> separate post-processing / restoration step.

## Intended use

- Real-time (streaming) transcription of Uzbek telephone speech
- Front-end ASR for voice bots, intelligent IVR and call-center automation
- A starting checkpoint for further fine-tuning on another Uzbek domain

## Technologies

**This model (ASR stage)**

| Area | Technology |
|---|---|
| Framework | NVIDIA NeMo, PyTorch |
| Architecture | FastConformer-Transducer (Medium), RNN-T loss |
| Tokenizer | SentencePiece (subword) |
| Augmentation | Gaussian noise, speed perturbation, SpecAugment |
| Pseudo-labeling | ROVER over E2E-Conformer and two Whisper Medium models |
| Training hardware | 2 × NVIDIA Tesla V100S (32 GB), FP16 |

**Full voice-to-voice RAG pipeline (described in the paper)**

| Stage | Technology |
|---|---|
| Streaming ASR | FastConformer-Transducer (this model) + VAD |
| Query rewriting | flan-t5-large, fine-tuned with LoRA (PEFT) |
| Intent recognition | ModernUzBERT + linear classifier |
| Dense retrieval | ModernUzBERT embeddings + FAISS |
| Sparse retrieval | BM25 (k1 = 1.5, b = 0.75) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Generator | Qwen2.5-7B, CPT + SFT with QLoRA (4-bit, LoRA rank 32) |
| Speech synthesis | VITS with HiFi-GAN V1 decoder (Coqui TTS) |
| Data preparation | Speaker diarization, XLM-RoBERTa NER for PII masking, Gemini 2.5 Flash for transcript editing |
| Telephony | FreeSWITCH (live testing in progress) |

## Download the checkpoint

The `.nemo` checkpoint (≈438 MB) is **not stored in this Git repository**. Download it
from the [latest release](https://github.com/Mannon-Ochilov/STT/releases/latest):

```bash
curl -L -o stt_uz_fastconformer_transducer_medium_streaming.nemo \
  https://github.com/Mannon-Ochilov/STT/releases/download/v1.0/stt_uz_fastconformer_transducer_medium_streaming.nemo
```

PowerShell:

```powershell
Invoke-WebRequest -Uri "https://github.com/Mannon-Ochilov/STT/releases/download/v1.0/stt_uz_fastconformer_transducer_medium_streaming.nemo" -OutFile "stt_uz_fastconformer_transducer_medium_streaming.nemo"
```

## How to use

### Install NVIDIA NeMo

```bash
pip install nemo_toolkit['all']
```

Install the latest PyTorch first.

### Load the model

From the downloaded `.nemo` checkpoint:

```python
import nemo.collections.asr as nemo_asr

asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(
    "stt_uz_fastconformer_transducer_medium_streaming.nemo"
)
```

### Transcribe a file (offline / batch)

```python
output = asr_model.transcribe(["audio_file.wav"])
print(output[0].text)
```

### Transcribe many files

```bash
python [NEMO_GIT_FOLDER]/examples/asr/transcribe_speech.py \
  model_path="stt_uz_fastconformer_transducer_medium_streaming.nemo" \
  audio_dir="<DIRECTORY CONTAINING AUDIO FILES>"
```

### Streaming (cache-aware) inference

```bash
python [NEMO_GIT_FOLDER]/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
  --asr_model=stt_uz_fastconformer_transducer_medium_streaming.nemo \
  --audio_file=audio_file.wav \
  --compare_vs_offline
```

### Input

16 kHz mono-channel audio (WAV). Telephone audio recorded at 8 kHz must be
resampled to 16 kHz — the model was trained on 8 kHz call-center material that was
upsampled to 16 kHz, so this matches the training condition.

### Output

A transcription string in lowercase Uzbek Latin script, without punctuation.

## Model architecture

FastConformer is an optimized Conformer variant using 8× depthwise-separable
convolutional subsampling (256 channels) and a reduced convolution kernel size
(9 instead of 31), which makes the encoder roughly **2.4× faster** than a standard
Conformer encoder while keeping accuracy.

The model has three parts:

- **Encoder** — 8× subsampling module, a fully connected layer, Conformer blocks with
  relative positional encoding, and a final fully connected layer. Each Conformer block
  contains layer normalization, feed-forward modules, multi-head self-attention (global
  features) and a lightweight depthwise-separable convolution module (local features),
  with residual connections throughout.
- **Prediction network** — an embedding layer, an LSTM block and a fully connected
  layer; acts as an internal language model over the emitted token history.
- **Joint network** — combines the acoustic vector and the linguistic vector, applies a
  fully connected layer with `tanh`, then `softmax` over the vocabulary plus the
  `blank` symbol.

The model is trained with the **RNN-T loss**, which sums over all valid alignment paths
between the `T` acoustic frames and the `U` target tokens and is optimized as the
negative log-likelihood of that sum, computed with the forward-backward algorithm.

The tokenizer is a **SentencePiece** subword model trained on an Uzbek text corpus of
roughly **15 million sentences (~80 million words)**, which addresses the
out-of-vocabulary problem caused by the agglutinative morphology of Uzbek.

## Training data

The model was trained on **USC 2.0** — 1,903.53 hours of transcribed Uzbek speech,
1,456,276 utterances, 9,854,394 running words and 497,807 unique words.

| Source / method | Hours | Type |
|---|---:|---|
| Initial USC corpus (crowdsourcing + audiobooks) | 105 | Verified |
| Extended corpus (urban cadaster, geographical names) | 95 | Verified |
| Common Voice 8.0 + SpeechOcean | 227 | Verified |
| UzbekVoice platform | 600 | Pseudo-labeled |
| Common Voice 24.0 (pseudo-labeling) | 100 | Pseudo-labeled |
| Common Voice 24.0 (validated) | 101 | Verified |
| Call-center recordings (diarization + pseudo-labeling) | 500 | Pseudo-labeled |
| Crowdsourcing (Telegram bots) | 175.53 | Verified |
| **Total** | **1,903.53** | |

Clips were de-duplicated across Common Voice versions by clip identifier: clips already
present in the 227-hour Common Voice 8.0 + SpeechOcean set were removed, so the Common
Voice rows do not overlap.

Pseudo-labels were produced with the **ROVER** (Recognizer Output Voting Error
Reduction) algorithm, which aligns the outputs of three independent Uzbek ASR systems
into a word transition network and votes per column to build a single high-confidence
transcription. The three systems were an E2E-Conformer trained on 1,000 hours, a
Whisper Medium model fine-tuned on 1,600 hours, and a Whisper Medium model trained on
Common Voice 17.0 (~475 hours). Call-center audio (Public Services Centers of the
Samarkand region, obtained under an NDA) was processed through a semi-automated
pipeline: speaker diarization → ASR → LLM-based syntactic/contextual editing (personal
data masked with an NER model beforehand) → human verification against the original
audio → normalization.

### Data splits

| Split | Hours | Description |
|---|---:|---|
| Train | 1,856.83 | Parameter optimization |
| Validation | 16.4 | Hyperparameter tuning, overfitting control |
| Test-Clean | 15.2 | Standard, clean acoustic conditions |
| Test-CallCenter | 15.1 | 8 kHz telephone channel, room acoustics, background noise |

Segments from the same original recording or conversation never appear in more than one
split. Test-CallCenter was built from conversations entirely disjoint from the training
recordings, and speakers were kept disjoint across the training, validation and test
splits.

## Training

Fine-tuning of all parameters, starting from the pre-trained English
FastConformer-Transducer (Medium) checkpoint, on **2 × NVIDIA Tesla V100S (32 GB)**.

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (with weight decay) |
| LR schedule | Cosine annealing, peak 7e-4 |
| Warmup steps | 4,000 |
| Effective batch size | 96 (gradient accumulation) |
| Epochs | 4 |
| Dropout | 0.1 |
| Precision | FP16 |
| SpecAugment | Enabled |
| Speed perturbation | 0.9 – 1.1 |
| Additive Gaussian noise (σ) | 0.001 – 0.015 |

Three augmentation strategies were combined to cover telephone-channel distortion:
zero-mean Gaussian noise added to the waveform, speed perturbation (expanding the
training data threefold), and SpecAugment frequency/time masking at the mel-spectrogram
level.

## Performance

WER / CER in percent; FTL is First-Token Latency (time from speech onset to the first
emitted text chunk).

| Architecture | Augmentation | Train | Val | Test-Clean | Test-CallCenter | FTL |
|---|---|---|---|---|---|---:|
| Standard Conformer (baseline) | None | 12.4 / 3.1 | 15.8 / 4.2 | 16.5 / 4.6 | 32.4 / 11.2 | 240 ms |
| Standard Conformer | All augmentations (proposed) | 9.8 / 2.4 | 12.8 / 3.4 | 13.7 / 3.8 | 21.1 / 6.5 | 240 ms |
| FastConformer (Medium) | None | 13.2 / 3.4 | 16.2 / 4.5 | 17.1 / 4.9 | 34.2 / 11.9 | 110 ms |
| FastConformer (Medium) | Gaussian noise | 13.8 / 3.6 | 16.0 / 4.4 | 17.5 / 5.1 | 26.8 / 8.5 | 110 ms |
| FastConformer (Medium) | Speed perturbation | 10.5 / 2.5 | 14.1 / 3.8 | 14.8 / 4.1 | 30.5 / 10.1 | 110 ms |
| FastConformer (Medium) | SpecAugment | 11.1 / 2.8 | 13.9 / 3.6 | 14.2 / 3.9 | 28.3 / 9.2 | 110 ms |
| **FastConformer (Medium)** | **All augmentations (proposed)** | **9.2 / 2.1** | **12.4 / 3.2** | **12.9 / 3.5** | **19.4 / 5.8** | **110 ms** |

Most of the accuracy gain comes from the **augmentation strategies**: on call-center
audio they reduce WER from 34.2% to 19.4% for FastConformer (−14.8 points) and from
32.4% to 21.1% for the standard Conformer, while on clean speech the gain is 3–4 points.
The latency reduction comes from the **FastConformer architecture** (240 ms → 110 ms
FTL); without augmentation it is slightly less accurate than the Conformer baseline.
Under the same augmentation setting, FastConformer is 1.7 WER points better than the
Conformer on Test-CallCenter at less than half the first-token latency.

## Limitations

- Output is **lowercase and unpunctuated**; punctuation and capitalization restoration
  requires a separate model.
- Accuracy degrades on speech containing technical terminology, heavy code-switching or
  dialects not represented in the training data.
- Test-CallCenter WER (19.4%) is substantially higher than Test-Clean (12.9%) — degraded
  telephone channels remain the hardest condition.
- A share of the training data is pseudo-labeled (1,200 of 1,903.53 hours), so label
  noise is present.
- Telephone-channel conditions were approximated by resampling and additive noise;
  codec artifacts were not simulated.
- The call-center recordings (training and Test-CallCenter) are not publicly available
  due to an NDA, which limits independent reproduction of the Test-CallCenter results.
- Evaluated on Uzbek Latin script only.

---

# O'zbek tili uchun oqimli FastConformer-Transducer modeli

Bu — o'zbek tili uchun **oqimli (streaming)** avtomatik nutqni aniqlash modeli.
Real vaqt rejimida ishlaydigan telefoniya va call-center sharoitlari uchun
mo'ljallangan. Model yuqorida keltirilgan maqoladagi real vaqtli "ovozdan-ovozga" RAG
savol-javob tizimining ASR bosqichi hisoblanadi. Tizimda ASR transkripti qidiruvdan
oldin flan-t5-large asosidagi so'rovni qayta yozish moduliga uzatiladi, bu ASR
xatolarining RAG bosqichiga o'tishini kamaytiradi.

**Asosiy ko'rsatkichlar**

- Arxitektura: FastConformer-Transducer (Medium), ingliz tilidagi tayyor checkpoint
  asosida to'liq fine-tuning qilingan
- O'quv korpusi: **USC 2.0**, 1 903,53 soat, 1 456 276 ta ibora, 497 807 ta noyob so'z
- Aniqlik: **Test-Clean WER 12,9%** (CER 3,5%), **Test-CallCenter WER 19,4%** (CER 5,8%)
- Kechikish: **FTL 110 ms** (Conformer bazaviy modelida 240 ms)
- Kirish: 16 kHz mono WAV (8 kHz telefon audiosi 16 kHz ga qayta namunalanadi)
- Chiqish: kichik harflarda, **tinish belgilarisiz** o'zbek lotin matni
- Texnologiyalar: NVIDIA NeMo, PyTorch, SentencePiece; to'liq tizimda flan-t5-large,
  ModernUzBERT, FAISS, BM25, RRF, Qwen2.5-7B (CPT + SFT, QLoRA), VITS (Coqui TTS),
  FreeSWITCH

**Checkpoint'ni yuklab olish**

`.nemo` fayli (≈438 MB) git repozitoriysida saqlanmaydi — uni
[Releases](https://github.com/Mannon-Ochilov/STT/releases/latest) bo'limidan yuklab
oling.

**Muhim eslatmalar**

- Model tinish belgilari va katta harflarni chiqarmaydi — o'quv transkripsiyalari
  normallashtirilgan (tinish belgilari olib tashlangan, harflar kichiklashtirilgan).
  Kerak bo'lsa alohida tiklash moduli qo'shiladi.
- O'quv ma'lumotlarining 1 200 soati psevdo-belgilangan (ROVER algoritmi orqali),
  shu sababli ma'lum darajada yorliq shovqini mavjud.
- Telefon kanali sharoitida aniqlik sezilarli pasayadi (12,9% → 19,4%).
- Aniqlikdagi asosiy yutuq augmentatsiyadan keladi (call-center to'plamida WER
  34,2% → 19,4%), kechikishning kamayishi esa FastConformer arxitekturasidan
  (240 → 110 ms). Bir xil augmentatsiyada FastConformer Conformer'dan 1,7 punkt
  aniqroq.
- Telefon kanali qayta namunalash va qo'shimcha shovqin orqali taqlid qilingan,
  kodek artefaktlari modellashtirilmagan.

## Citation

```bibtex
@article{khujayarov2026queryrewriting,
  title   = {Enhancing Retrieval Robustness through ASR-Aware Query Rewriting in
             Real-Time Voice-to-Voice RAG for Uzbek},
  author  = {Khujayarov, Ilyos and Ochilov, Mannon and Kholmatov, Orzimurod and
             Jumanov, Vokhid and Juraev, Dilshod and Musaev, Muhammadjon},
  journal = {Technologies},
  year    = {2026}
}
```

## References

1. [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition](https://arxiv.org/abs/2305.05084)
2. [Google SentencePiece Tokenizer](https://github.com/google/sentencepiece)
3. [NVIDIA NeMo Toolkit](https://github.com/NVIDIA/NeMo)
4. [SpecAugment: A Simple Data Augmentation Method for ASR](https://arxiv.org/abs/1904.08779)
5. ROVER: J. G. Fiscus, *A post-processing system to yield reduced word error rates:
   Recognizer Output Voting Error Reduction*, IEEE ASRU, 1997.

## License

Released under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
