---
tags: [llm, nlp, machine-learning, huggingface, transformers]
created: 2026-09-24
status: in-progress
---

# Large Language Models (LLMs) — Learning Notes

> [!info] Purpose
> Notes built around the project's learning objectives: what LLMs are, how they learn, how they're used, and the key building blocks (tokens, embeddings, fine-tuning) plus the Hugging Face ecosystem (pipelines, RoBERTa, GPT-2, ViT, BLIP).

## Table of Contents
- [[#1. What is a Large Language Model (LLM)?]]
- [[#2. How do LLMs learn to understand and generate language?]]
- [[#3. Main real-world uses of LLMs]]
- [[#4. LLMs in chatbots and virtual assistants]]
- [[#5. Summarization and translation]]
- [[#6. Tokens — how LLMs process text]]
- [[#7. Embeddings — how LLMs understand meaning]]
- [[#8. Factors affecting LLM accuracy and performance]]
- [[#9. Ethical concerns and risks]]
- [[#10. Fine-tuning and domain adaptation]]
- [[#Hugging Face ecosystem]]
- [[#Key model families]]
- [[#Glossary]]
- [[#Resources checklist]]

---

## 1. What is a Large Language Model (LLM)?

A **Large Language Model** is a neural network (almost always a **Transformer**) trained on massive amounts of text to predict the next piece of text (a *token*) given everything that came before it. "Large" refers to two things at once:

- **Scale of parameters** — billions to trillions of learnable weights.
- **Scale of training data** — huge internet-scale text corpora (books, code, articles, web pages).

Because next-token prediction forces the model to internalize grammar, facts, reasoning patterns, and style, a single trained model can be reused for many downstream tasks (translation, summarization, Q&A, coding, etc.) without being built specifically for each one.

> [!tip] Mental model
> An LLM is a very sophisticated autocomplete engine — but because "predicting the next word well" requires implicitly learning syntax, semantics, and world knowledge, that simple objective produces surprisingly general capabilities.

---

## 2. How do LLMs learn to understand and generate language?

Training happens in stages:

1. **Pre-training (self-supervised learning)**
   - No manually labeled data needed — the text labels itself.
   - Two common objectives:
     - **Causal/Autoregressive Language Modeling** (used by GPT-family): predict the *next* token given all previous tokens. Good for generation.
     - **Masked Language Modeling (MLM)** (used by BERT/RoBERTa): randomly hide (mask) tokens in a sentence and train the model to predict them from the surrounding context (both left and right). Good for understanding/representation tasks.
   - This stage teaches grammar, facts, associations, and some reasoning, purely from raw text.

2. **The Transformer architecture makes this possible**
   - Built on **self-attention**: for every token, the model computes how much "attention" to pay to every other token in the sequence, regardless of distance. This solves the long-range dependency problem that older RNN/LSTM models struggled with.
   - Stacks of attention + feed-forward layers let the model build increasingly abstract representations of meaning.

3. **Fine-tuning / alignment (optional but common for modern assistants)**
   - **Supervised Fine-Tuning (SFT)**: train on curated input/output examples for a target behavior (e.g., "follow instructions").
   - **RLHF (Reinforcement Learning from Human Feedback)** or similar preference-based methods: humans rank model outputs, and the model is optimized to produce outputs people prefer — improving helpfulness and safety.

4. **Inference (using the trained model)**
   - The model generates text one token at a time: it predicts a probability distribution over the vocabulary, samples/selects the next token, appends it to the sequence, and repeats.

---

## 3. Main real-world uses of LLMs

- **Content generation** — drafting emails, articles, marketing copy, code.
- **Question answering** — open-domain and document-grounded (RAG) Q&A.
- **Summarization** — condensing long documents, meeting notes, articles.
- **Translation** — between languages, including low-resource pairs.
- **Classification** — sentiment analysis, topic labeling, spam detection.
- **Search and retrieval** — semantic search using embeddings.
- **Coding assistance** — autocomplete, bug fixing, code explanation.
- **Conversational agents** — customer support bots, virtual assistants.
- **Data extraction** — pulling structured fields out of unstructured text (invoices, resumes, contracts).

---

## 4. LLMs in chatbots and virtual assistants

- The LLM acts as the **reasoning/language core**: it interprets user intent from natural language and produces natural-language responses.
- **Prompt engineering** shapes behavior without retraining — system prompts define persona, tone, and constraints.
- **Context window** holds conversation history so the assistant maintains continuity across turns.
- **Tool use / function calling** lets the LLM call external APIs (calendars, databases, search) rather than relying only on memorized knowledge — this is how assistants answer real-time or account-specific questions.
- **Retrieval-Augmented Generation (RAG)** grounds responses in external documents to reduce hallucination and add up-to-date or proprietary knowledge.
- **Guardrails / moderation layers** filter unsafe or off-policy outputs before they reach the user.

---

## 5. How LLMs help with summarization or translation

- **Summarization**: framed as a *sequence-to-sequence* task — the model reads a long input and generates a shorter output that preserves key meaning. Approaches:
  - *Extractive*: pick out important existing sentences (less common with modern LLMs).
  - *Abstractive*: generate new sentences that paraphrase and compress the content (what most LLMs do).
- **Translation**: also sequence-to-sequence — encode the source-language sequence, decode into the target language. Specialized models like the **Helsinki-NLP (Opus-MT)** family are trained on many language pairs specifically for this task, while general-purpose LLMs can translate zero-shot via prompting.
- In Hugging Face terms, this maps to `summarization` and `translation` pipelines, both built on **Text2Text Generation** (encoder-decoder) or general text-generation models.

---

## 6. What role do tokens play in how LLMs process text?

- LLMs don't read raw characters or whole words — they operate on **tokens**, chunks of text produced by a **tokenizer** (e.g., Byte-Pair Encoding, WordPiece, SentencePiece).
- A token might be a whole word, part of a word, a punctuation mark, or a byte — this lets the model handle rare words, typos, and multiple languages with a manageable, fixed-size vocabulary.
- The text pipeline is: **raw text → tokenizer → token IDs (integers) → embedding lookup → model**.
- **Context window / max tokens** is measured in tokens, not words or characters — this limits how much text (input + generated output) a model can handle in a single pass.
- Special tokens matter operationally: e.g., `pad_token`, `eos_token` (end-of-sequence). Some models (like GPT-2) don't define a pad token by default, so practitioners commonly **set `pad_token_id` = `eos_token_id`** to enable batched generation.
- Cost and latency in LLM APIs are typically billed and measured per token (input tokens + output tokens).

---

## 7. What are embeddings, and how do they help LLMs understand meaning?

- An **embedding** is a dense vector (a list of numbers) that represents a token, word, sentence, or document in a continuous, high-dimensional space.
- Embeddings are learned such that **semantically similar items end up close together** in that vector space (e.g., "king" and "queen" are nearer to each other than "king" and "banana").
- Inside a Transformer:
  - Each token ID is converted to an embedding vector via a lookup (embedding matrix).
  - **Positional encodings** are added so the model knows token *order* (since attention itself is order-agnostic).
  - These embeddings are refined layer by layer through self-attention into increasingly context-aware representations (the embedding of "bank" differs depending on whether the sentence is about rivers or money).
- Embeddings also power practical applications outside generation:
  - **Semantic search** (finding documents by meaning, not keyword overlap).
  - **Clustering / recommendation** (grouping similar items).
  - **Retrieval-Augmented Generation** (finding relevant context to feed the LLM).

---

## 8. What factors affect the accuracy and performance of an LLM?

- **Training data quality and diversity** — noisy, biased, or narrow data limits capability and introduces errors/bias.
- **Model size (parameters)** — generally more capacity to learn patterns, but with diminishing returns and higher cost.
- **Training data volume/quality relative to model size** — a model can be under- or over-trained relative to its data.
- **Architecture and training objective choices** — e.g., MLM vs. causal LM suits different downstream tasks.
- **Fine-tuning / domain adaptation** — a base model fine-tuned on domain-specific data usually outperforms the generic model on that domain.
- **Prompt design** — the same model can perform very differently depending on how a task is phrased (few-shot examples, clear instructions, chain-of-thought prompting).
- **Decoding strategy at inference** — greedy decoding, beam search, temperature, top-k/top-p sampling all affect output quality, creativity, and determinism.
- **Context length / truncation** — if relevant information falls outside the context window, the model can't use it.
- **Evaluation/benchmark choice** — performance numbers are only as meaningful as the benchmark's relevance to the real task.
- **Inference optimizations** — quantization, caching (KV-cache), and batching affect speed/cost, and can slightly trade off accuracy for efficiency.

---

## 9. Ethical concerns and risks with LLMs

- **Hallucination** — models can generate fluent, confident, but factually incorrect content.
- **Bias and fairness** — models can reproduce or amplify societal biases present in training data (gender, race, culture, etc.).
- **Privacy** — models may memorize and regurgitate sensitive information seen during training; user conversations may also raise data-handling concerns.
- **Misinformation and misuse** — LLMs can be used to generate spam, propaganda, phishing content, or deepfake-adjacent text at scale.
- **Copyright and intellectual property** — training data provenance and generated-content ownership are unsettled legal/ethical areas.
- **Environmental cost** — training and running large models consumes significant energy and compute.
- **Job displacement** — automation of writing, coding, and support tasks raises workforce concerns.
- **Over-reliance / deskilling** — users trusting LLM output without verification, especially in high-stakes domains (medical, legal, financial).
- **Security** — prompt injection, jailbreaking, and adversarial inputs can bypass intended safeguards.
- **Mitigations** typically include RLHF/alignment training, content filters, transparency about limitations, human-in-the-loop review, and clear sourcing/citations (e.g., via RAG).

---

## 10. How can LLMs be adapted or fine-tuned for specific domains?

- **Full fine-tuning**: continue training all model weights on domain-specific labeled data (e.g., legal contracts, medical notes). Most effective but expensive and requires enough quality data.
- **Parameter-efficient fine-tuning (PEFT)**: methods like **LoRA** (Low-Rank Adaptation) train small additional weight matrices while freezing the base model — cheaper, faster, easier to store/swap per domain.
- **Prompt-based adaptation** (no weight updates):
  - **Zero-shot prompting** — ask directly, relying on the base model's general knowledge.
  - **Few-shot prompting** — include a handful of examples in the prompt to steer behavior.
  - **Prompt/instruction tuning** — optimize a fixed instruction template or soft prompt.
- **Retrieval-Augmented Generation (RAG)**: keep the base model frozen, but feed it relevant domain documents retrieved at query time — good when the domain knowledge changes often or is proprietary.
- **Domain-specific pretraining**: continue the self-supervised pretraining objective (MLM/causal LM) on an in-domain corpus before task fine-tuning — used for things like biomedical or legal LLMs.
- Choice depends on: amount of labeled data available, budget/compute, how often the domain knowledge changes, and how much behavior change (vs. just knowledge) is needed.

---

## Hugging Face ecosystem

**Hugging Face** is the de facto hub for open-source ML: a model repository, dataset repository, and libraries (`transformers`, `datasets`, `tokenizers`) that make using pretrained models straightforward.

### `pipeline()` — the fast path to inference
The `pipeline` abstraction wraps: tokenizer → model → post-processing into one call. Common pipeline tasks:

| Task | Pipeline name | Typical model family |
|---|---|---|
| Sentiment/text classification | `sentiment-analysis` / `text-classification` | RoBERTa, DistilBERT |
| Fill-in-the-blank | `fill-mask` | BERT, RoBERTa |
| Free-form generation | `text-generation` | GPT-2, GPT-family |
| Rewriting/transforming text (translation, summarization) | `text2text-generation`, `translation`, `summarization` | T5, BART, Helsinki-NLP (MarianMT) |
| Image classification | `image-classification` | ViT |
| Image captioning / VQA | — | BLIP |

**AutoClasses** (`AutoTokenizer`, `AutoModel`, `AutoModelForSequenceClassification`, etc.) automatically pick the right architecture based on a model's config, so the same code pattern works across many checkpoints.

**`torch.no_grad()`**: when only doing inference (not training), wrapping the forward pass in `no_grad()` disables gradient tracking — reduces memory use and speeds up inference since backpropagation isn't needed.

### Text Generation vs. Text2Text Generation
- **Text Generation (decoder-only, causal)**: e.g., GPT-2. Given a prompt, it continues the text. One input stream, autoregressive.
- **Text2Text Generation (encoder-decoder)**: e.g., T5, BART, MarianMT. Explicitly maps an input sequence to a *different* output sequence — natural fit for translation, summarization, paraphrasing.

---

## Key model families

### RoBERTa (Robustly Optimized BERT Pretraining Approach)
- A retrained/optimized version of BERT: same architecture, but trained longer, on more data, with larger batches, and **removes the Next Sentence Prediction (NSP)** objective, keeping only **Masked Language Modeling (MLM)**.
- Since it's an encoder-only model built on MLM, it's strong at *understanding* tasks: classification, sentiment analysis (e.g., tweet sentiment), named entity recognition, fill-mask.
- Not designed for open-ended text generation (no causal/decoder structure).

### GPT-2 / DistilGPT2
- Decoder-only, causal language model — trained purely to predict the next token.
- Good at open-ended **text generation**; the basis for understanding how larger GPT-family models work.
- **DistilGPT2** is a distilled (smaller, faster, knowledge-compressed) version of GPT-2 — trades a little quality for significantly lower size/latency.
- Practical note: GPT-2 has no default pad token, so `pad_token_id` is commonly set to `eos_token_id` to allow batched generation.

### Helsinki-NLP (Opus-MT) translation models
- A large collection of encoder-decoder translation models, each specialized for a specific language pair (or small group of pairs), trained on the OPUS parallel-corpus collection.
- Useful when you need dedicated, efficient translation rather than a general-purpose LLM doing translation via prompting.

### Vision Transformer (ViT)
- Applies the Transformer architecture (originally built for text) to images: an image is split into fixed-size **patches**, each patch is flattened and linearly embedded (like a "visual token"), positional encodings are added, and the sequence of patch embeddings is processed by a standard Transformer encoder.
- Used for image classification and as a visual backbone for multimodal models.

### Vision-Language Models (VLMs) and BLIP
- **VLMs** combine a vision encoder (often ViT-based) with a language model so the system can jointly reason over images and text — enabling tasks like image captioning, visual question answering (VQA), and image-text retrieval.
- **BLIP** (Bootstrapping Language-Image Pre-training) unifies understanding and generation: it can both *understand* image-text pairs (retrieval-style tasks) and *generate* text from images (captioning), using a bootstrapped/cleaned captioning process during pretraining to improve data quality.

---

## Glossary

- **Token** — smallest unit of text the model processes (word, subword, or character piece).
- **Embedding** — dense vector representation capturing meaning/context.
- **Self-attention** — mechanism letting each token weigh the relevance of every other token.
- **Context window** — max number of tokens a model can process at once (input + output).
- **Fine-tuning** — further training a pretrained model on task/domain-specific data.
- **Zero/few-shot learning** — performing a task with no/few examples, relying on the base model's general knowledge.
- **Hallucination** — confident but factually wrong model output.
- **RAG (Retrieval-Augmented Generation)** — grounding generation in retrieved external documents.
- **Inference** — running a trained model to produce predictions/output (as opposed to training).

---

## Resources checklist
Use this to track which linked resources (from the course) you've gone through — links omitted here since they're intranet-only, but titles kept for reference.

- [ ] What are large language models (LLMs)?
- [ ] Roadmap to Learning LLMs for Beginners
- [ ] Large Language Models Explained
- [ ] What is LLM (Large Language Model)?
- [ ] What is Hugging Face?
- [ ] Mastering Masked Language Models
- [ ] Masked language modeling
- [ ] Getting Started with Transformers and Pipelines
- [ ] A Hands-On Guide to Hugging Face Pipelines
- [ ] Introducing RoBERTa Base Model
- [ ] Overview of RoBERTa model
- [ ] Tweets sentiment analysis with RoBERTa
- [ ] Unlocking Language Barriers with Helsinki-NLP Translation Models
- [ ] Text Generation v/s Text2Text Generation
- [ ] Scientific text generation with HF-DistilGPT2 in TensorFlow
- [ ] Vision Transformers (ViT) in Image Recognition
- [ ] Vision Transformers (ViT) Tutorial
- [ ] IBM: What are vision language models (VLMs)?
- [ ] NVIDIA: What Are Vision Language Models
- [ ] BLIP: Bridging the Gap Between Vision-Language Tasks
- [ ] Understanding BLIP: A Huggingface Model

---

## Related notes
- [[Transformers Architecture]]
- [[Hugging Face Pipelines Cheatsheet]]
- [[Prompt Engineering]]
- [[Fine-tuning vs RAG]]