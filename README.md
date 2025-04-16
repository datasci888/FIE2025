# FIE_2025_AgenticAI_Quiz
FIE 2025 AgenticAI Quiz Generation Paper
Abstract
This research-to-practice full paper investigates how Large Language Models (LLMs) with long-context window and enhanced retrieval efficiency can generate context-specific quizzes to address high attrition in engineering education. We aim to enable LLMs to process large, multidisciplinary Artificial Intelligence (AI) datasets, covering topics like Machine Learning, Generative AI, and Neural Networks, from foundational to advanced concepts. A systematic literature review identified gaps in Retrieval-Augmented Generation (RAG) systems, which often retrieve irrelevant chunks due to context limitations, leading to inaccurate or hallucinated responses \cite{ke2025retrieval, agrawal2024mindfulrag}. Traditional quiz generation lacks modular design, limiting scalability and interpretability. To address this, we developed an agentic long-context RAG architecture using Gemini 1.5’s one-million-token window, integrating retrieval, reasoning, and evaluation in a unified pipeline.

Our methodology employed a modular Agentic AI system. A Parsing Agent extracts text from academic sources, followed by a Chunking & Storage Agent segmenting content with 500-character overlaps. An Embedding & Indexing Agent generates and indexes vector embeddings, verified by a Verification Agent for topical alignment. For quiz generation, a Retriever Agent uses cosine similarity and multilingual re-ranking, a Selector Agent filters meaningful chunks, a Response Agent leverages cached ground-truth MCQs with Gemini 1.5 Flash, and an Evaluator Agent assesses outputs using Correctness, Faithfulness, Non-Hallucination, ROUGE, and BERTScore metrics. Experiments on a 150-question benchmark showed accuracy improvements: 78.00\% (raw), 84.00\% (chunks), 89.33\% (chunks+cache), and 93.33\% (1M context+cache) for Gemini, with GPT-4o and Claude Sonnet 3.7 revealing complementary strengths in precision and confidence. This system fosters active recall, supporting conceptual mastery. Future work includes deploying an interactive quiz application and expanding domain-specific datasets across engineering fields.

# Agentic AI Quiz-Based Learning System: Enhancing MCQ Generation via Long-Context Cached Retrieval-Augmented Generation

## 📘 Project Overview
This repository accompanies the IEEE FIE 2025 paper **"Agentic AI Quiz-Based Learning System: Enhancing MCQ Generation via Long-Context Cached Retrieval-Augmented Generation"**, which presents a novel Agentic AI system for generating multiple-choice questions (MCQs) that reinforce core concepts in engineering education. The system addresses high attrition rates by integrating advanced AI components to deliver personalized, feedback-rich quizzes across domains like algorithms, machine learning, and recommender systems.

## 🧠 Key Contributions
- Developed a modular agentic architecture integrating retrieval, caching, and long-context LLMs (Gemini Flash 1.5) for scalable MCQ generation.
- Demonstrated application of long-context LLMs and agentic orchestration for adaptive learning, illustrated using AI interview preparation.
- Constructed a 4,404-question dataset combining curated and Grok-3-generated MCQs across multidisciplinary AI topics.
- Introduced a novel evaluation pipeline including Answer Correctness, Faithfulness, Hallucination, and Explanation Similarity.
- Integrated state-of-the-art models like Gemini 1.5 Flash, Claude 3.7, and GPT-4o with efficient prompt caching and retrieval.

## 🔍 System Architecture
The system is composed of two Agentic AI pipelines:
- **Agentic System I**: Parses educational content and generates vector embeddings using HNSW-based indexing (Vectara). Embeddings are used for similarity-based retrieval.
- **Agentic System II**: Retrieves relevant context, applies filtering and caching mechanisms, and feeds data to long-context LLMs to generate and evaluate MCQ responses.

Advanced agents handle:
- **Chunking and Storage**
- **Context Retrieval and Filtering**
- **Prompt Cache Management**
- **Answer Generation and Evaluation**

## 🧪 Evaluation Metrics
| Metric                | Description                                                      |
|-----------------------|------------------------------------------------------------------|
| Exact Match           | Measures if the model-selected answer  matches ground truth.     |
| Faithfulness          | Checks factual grounding of the explanation to retrieved context.|
| Explanation Similarity| Evaluates semantic alignment of generated explanation with GT.   |
| Hallucination         | Measures deviation from source context.                          |
| Uncertainty           | Evaluates confidence in generated answers.                       |

## 🧱 Technologies and Tools
| Category                       | Tools Used                                               |
|--------------------------------|----------------------------------------------------------|
| Cloud Platform                 | Google Cloud Platform (GCP)                              |
| Dataset Expansion \& Evaluation| Grok-3 (Think Mode), Grok-2                              |
| MCQ Generation                 | Claude Sonnet 3.7, Gemini 1.5 Flash, GPT-4o              |
| Retrieval and Indexing         | Vectara with HNSW, hybrid lexical-semantic search        |
| Embedding Generation           | text-embedding-005 (Google, VertexAI)                    |
| File/Directory Parsing         | CrewAI Tools (FileReadTool, DirectoryReadTool)           |
| Deduplication                  | DuplicateCheckerTool (BaseTool, Pydantic)                |
| Evaluation Agents              | Custom agents for BLEU, ROUGE, BERTScore, Face4RAG-style |

## 📜 Citation
If you use this code, please cite our IEEE FIE 2025 paper:
@inproceedings{fie2025agenticquiz, title={Agentic AI Quiz-Based Learning System: Enhancing MCQ Generation via Long-Context Cached Retrieval-Augmented Generation}, booktitle={Proceedings of the IEEE Frontiers in Education Conference (FIE)}, year={2025} }
