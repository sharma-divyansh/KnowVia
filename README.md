# KnowVia — AI-Powered Document Question Answering System

## 1. Project Overview

**KnowVia** is an AI-powered document question-answering application designed to help users retrieve accurate, relevant information from their own documents and other supported knowledge sources through natural-language questions.

The project implements Retrieval-Augmented Generation (RAG), combining document processing, semantic embeddings, similarity-based retrieval, and large language models (LLMs). Rather than relying exclusively on a model's pre-existing knowledge, KnowVia retrieves relevant information from available sources and uses that information to generate grounded responses.

The primary version of KnowVia is a Streamlit web application that uses Groq for language-model inference. The project also includes `chatbot.py`, a separate terminal-based implementation that demonstrates the core document retrieval and question-answering workflow using Ollama.

KnowVia was developed as a practical learning project to understand how modern AI applications process documents, retrieve relevant information, generate answers, and present source-supported results through a user-friendly interface.

## 2. Project Objectives

The primary objectives of KnowVia are:

- Build a functional document-based AI question-answering application.
- Understand and implement the Retrieval-Augmented Generation architecture.
- Support information retrieval from multiple document formats.
- Enable semantic search using vector embeddings.
- Generate concise, context-aware answers using an LLM.
- Provide source attribution to help users verify answers.
- Support OCR for scanned documents and images in the main application.
- Enable ingestion of publicly accessible URLs in the main application.
- Develop a professional, organized interface for document management and conversations.
- Evaluate answer accuracy, retrieval relevance, source consistency, and the handling of questions whose answers are unavailable.

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Main application framework | Streamlit |
| Main application LLM provider | Groq API |
| Main application retrieval architecture | Retrieval-Augmented Generation (RAG) |
| Embeddings and semantic search | Sentence Transformers and vector similarity |
| Terminal chatbot LLM | Ollama |
| Terminal chatbot model | `llama3.2:3b` |
| Terminal chatbot embedding model | `all-MiniLM-L6-v2` |
| PDF text extraction | PyMuPDF (`fitz`) |
| Word document extraction | `python-docx` |
| Text file processing | Python file handling |
| OCR | Image and scanned-document text extraction in the main application |
| Development environment | Visual Studio Code |
| Python environment | `.venv312` |

The exact supporting packages and configuration may vary between the two implementations.

## 4. Project Architecture

KnowVia follows a RAG-based architecture.

```text
                 KNOWVIA
                    |
          Document and URL Sources
                    |
          Document Ingestion Layer
                    |
       Text Extraction and Processing
                    |
       Chunking / Passage Preparation
                    |
          Embedding Generation
                    |
           Retrieval and Search
                    |
        Relevant Context Selection
                    |
           LLM Answer Generation
                    |
         Answer and Source Citations
                    |
              User Interface
```

### Main processing stages

**1. Input and ingestion**

Documents and supported public URLs are supplied to the main application. The terminal implementation reads supported files from the local project folders.

**2. Text extraction**

The application extracts readable text from supported formats. The main application additionally supports OCR for images and scanned documents.

**3. Text processing**

Extracted text is cleaned and divided into manageable passages or chunks. The purpose is to make relevant information easier to retrieve.

**4. Embedding generation**

Text passages are represented as numerical vectors that capture semantic information. These representations allow the system to identify passages related to a user's question.

**5. Retrieval**

The system searches for relevant passages and prepares the retrieved context for the language model.

**6. Answer generation**

The language model uses the retrieved context to construct a natural-language response.

**7. Source attribution**

The application provides source information where supported, helping users trace answers back to the material used to generate them.

## 5. Understanding the RAG Approach

Retrieval-Augmented Generation is the core concept behind KnowVia.

A conventional language model generates responses from information learned during training and the context provided in a prompt. A RAG application adds a retrieval stage that searches an external knowledge source before generating the answer.

For example, if a user asks, "What is the annual paid leave entitlement?", the system searches its available documents for the relevant policy passages. The retrieved passages are then supplied to the language model, which generates an answer based on that context.

### Benefits of RAG

- Uses information from user-provided knowledge sources.
- Makes document-based question answering practical without training a new model.
- Can provide source references for verification.
- Allows knowledge sources to be updated without fine-tuning the LLM.
- Helps reduce unsupported answers when retrieval and prompting are configured appropriately.

RAG does not guarantee correctness. Answer quality still depends on document quality, extraction, retrieval relevance, context selection, and model behavior.

## 6. Project Files and Their Responsibilities

### `app.py` — Main KnowVia application

`app.py` contains the primary Streamlit application and coordinates the user-facing document question-answering workflow.

Its responsibilities include:

- Rendering the Streamlit interface.
- Managing document-related interactions.
- Supporting document ingestion and processing.
- Supporting publicly accessible URL ingestion.
- Preparing document content for retrieval.
- Retrieving relevant passages for user questions.
- Sending prompts and context to the configured LLM provider.
- Displaying generated answers and available source information.
- Managing application settings and user interactions.
- Reporting processing and model-response errors.

The main application is the primary version of KnowVia.

### `chatbot.py` — Terminal-based document chatbot

`chatbot.py` is a separate, command-line implementation of a document question-answering chatbot. It demonstrates the fundamental RAG workflow using a local Ollama model.

Its responsibilities include:

- Locating the project directory and `documents/` folder.
- Loading a root-level `document.txt`, if present.
- Reading TXT, PDF, and DOCX files.
- Extracting text and splitting it into searchable passages.
- Retaining source filenames and location information.
- Creating semantic embeddings using Sentence Transformers.
- Retrieving the three most similar passages using cosine similarity.
- Constructing a prompt containing retrieved context and the user's question.
- Generating responses using Ollama's `llama3.2:3b` model.
- Displaying answers in the terminal.
- Handling file-reading and Ollama connection errors.

The script provides a simpler implementation for understanding document ingestion, embedding generation, similarity search, and context-based answer generation.

**Important distinction:** `chatbot.py` uses Ollama, whereas the primary `app.py` application uses Groq. They are separate implementations and do not necessarily share the same retrieval pipeline or configuration.

### `documents/` — Document repository

The `documents/` directory contains local files used as knowledge sources.

For `chatbot.py`, the supported file extensions are:

- `.txt`
- `.pdf`
- `.docx`

The main application supports additional input capabilities, including publicly accessible URLs and OCR-based text extraction.

### `document.txt` — Optional root-level document

The terminal implementation checks for an optional `document.txt` file in the project root and loads it when present.

### `README.md` — Project documentation

This file describes the purpose, architecture, technologies, setup instructions, functionality, testing, limitations, and future development of KnowVia.

### `.venv312/` — Python virtual environment

The virtual environment isolates the project's Python dependencies from other Python installations and projects on the machine.

## 7. Main Application Features

### 7.1 Document ingestion

The application processes supported knowledge sources so that their content can be used for question answering.

The main application supports document-based knowledge retrieval, OCR for images and scanned documents, and publicly accessible URL ingestion.

### 7.2 URL ingestion — Completed and verified

KnowVia supports ingesting content from publicly accessible URLs in the main application.

This capability allows the application to use supported web content as an additional knowledge source alongside local documents.

**Status: Implemented and tested by the developer.**

URL ingestion is a completed feature, not a pending task. The separate `chatbot.py` terminal implementation does not currently contain URL-ingestion logic.

### 7.3 OCR support

The main application supports extracting text from images and scanned documents using OCR.

OCR is useful when a PDF or image contains scanned pages rather than selectable digital text.

The extracted text can subsequently be used in the document question-answering workflow, subject to extraction quality.

### 7.4 Semantic retrieval

Semantic retrieval attempts to identify passages that are relevant in meaning, rather than relying only on exact keyword matches.

The `chatbot.py` implementation uses the `all-MiniLM-L6-v2` embedding model and cosine similarity to retrieve relevant passages.

The primary application's retrieval implementation is maintained separately in `app.py`.

### 7.5 Context-grounded answers

Retrieved passages are supplied to the language model as context for answer generation.

The terminal chatbot's prompt explicitly instructs the model not to invent facts, to acknowledge missing information, and to cite filenames and locations.

### 7.6 Source attribution

Source information helps users verify whether an answer is supported by the underlying knowledge material.

The terminal chatbot attaches filenames and location labels to retrieved passages. The main application provides source information through its own retrieval and presentation workflow.

### 7.7 Main application interface

The Streamlit application provides the user-facing experience for interacting with KnowVia.

The interface is organized around the following sections:

- **Chat:** Ask questions and review generated answers.
- **Document Library:** Manage and inspect knowledge sources.
- **Settings:** Access available application and processing settings.

The interface is intended to keep document management and question answering accessible without requiring users to interact with Python code directly.

## 8. How `chatbot.py` Works

The terminal chatbot follows this sequence.

### Step 1: Locate the project directory

Python's `pathlib` module identifies the directory containing the script and the associated `documents/` folder.

### Step 2: Load supported documents

The script searches for an optional root-level `document.txt` and recursively searches the `documents/` directory for TXT, PDF, and DOCX files.

### Step 3: Extract text

- TXT files are read using UTF-8 with a byte-order-mark-tolerant encoding.
- PDF files are processed page by page using PyMuPDF.
- DOCX files are processed paragraph by paragraph using `python-docx`.

### Step 4: Prepare passages

The script normalizes whitespace, ignores short passages, and stores passage text alongside its source filename and location.

### Step 5: Create embeddings

The `all-MiniLM-L6-v2` model converts document passages into vector representations.

### Step 6: Accept a question

The user enters a question in the terminal. The question is converted into an embedding using the same model.

### Step 7: Retrieve relevant passages

Cosine similarity scores are calculated between the question embedding and the document embeddings. Up to three passages with the highest similarity scores are selected.

### Step 8: Construct the prompt

The selected passages, source details, question, and response instructions are combined into a prompt.

### Step 9: Generate the answer

The prompt is sent to the local Ollama model, and the resulting answer is printed in the terminal.

### Step 10: Continue the conversation

The script accepts additional questions until the user enters `exit` or `quit`.

## 9. Environment Setup

The primary development environment is Windows with Visual Studio Code and Python.

### Prerequisites

Install or configure:

- Python compatible with the project's dependencies.
- Visual Studio Code.
- The project files.
- A Python virtual environment.
- A valid Groq API key for the main application.
- Ollama and the required local model if you intend to run `chatbot.py`.

### Activate the virtual environment

Open the VS Code terminal in the KnowVia project directory.

For PowerShell:

```powershell
.\.venv312\Scripts\Activate.ps1
```

For Command Prompt:

```cmd
.venv312\Scripts\activate.bat
```

After activation, the terminal generally displays the environment name in the prompt.

If PowerShell blocks activation because of its execution policy, use the Command Prompt activation command or follow your organization's approved Python environment setup procedure.

### Install dependencies

Install the packages required by the version you intend to run.

For the terminal chatbot, the main packages referenced by its code are:

```bash
pip install pymupdf python-docx sentence-transformers ollama
```

For the Streamlit application, install its complete set of dependencies. If the project contains a maintained `requirements.txt`, use:

```bash
pip install -r requirements.txt
```

The exact dependency list should match the current source code. Do not assume that the terminal chatbot and Streamlit application have identical requirements.

## 10. Configuring Groq for the Main Application

The primary application uses Groq for LLM inference.

### API key configuration

1. Obtain an API key through the official Groq developer platform.
2. Configure the key using the environment-variable or secrets mechanism supported by the current application.
3. Do not hard-code the key in source files.
4. Do not commit API keys or private configuration files to a public repository.
5. Restart the application after changing environment configuration if required.

If the application uses an environment variable named `GROQ_API_KEY`, configure it in the terminal session before launching the app, or use the configuration mechanism already implemented in the project.

Example for a temporary PowerShell session:

```powershell
$env:GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

Replace the placeholder with the actual key locally. Never include the real key in this README, screenshots, or source-control commits.

The exact model name and request settings should be taken from the current `app.py` configuration.

## 11. Running the Main Streamlit Application

From the project directory, activate the virtual environment and run:

```bash
streamlit run app.py
```

Streamlit should start a local web server and provide a local application URL in the terminal.

Open that URL in the browser to access KnowVia.

If Streamlit is not installed in the active environment, install it using:

```bash
pip install streamlit
```

If the application fails to start, verify the active Python environment and confirm that the required packages are installed in that environment.

## 12. Running the Terminal Chatbot

The `chatbot.py` implementation runs independently from the Streamlit application.

### Step 1: Install its dependencies

```bash
pip install pymupdf python-docx sentence-transformers ollama
```

### Step 2: Install and start Ollama

Install Ollama for the operating system and ensure its service is running.

### Step 3: Download the required model

```bash
ollama pull llama3.2:3b
```

### Step 4: Run the script

```bash
python chatbot.py
```

The script loads available documents, creates embeddings, and displays a prompt when the chatbot is ready.

Ask questions in the terminal and enter `exit` to stop the session.

**Note:** This setup is specifically for the Ollama-based terminal implementation. The main Streamlit application uses Groq and does not need to be switched to Ollama to use its existing functionality.

## 13. Testing and Evaluation

KnowVia has been tested against representative document-question-answering scenarios.

### Areas evaluated

**Basic factual accuracy:** Checks whether the answer correctly reflects information in the source material.

**Specific fact retrieval:** Checks whether the system can retrieve a particular value or detail from the documents.

**Missing-information handling:** Checks whether the system acknowledges when the available sources do not provide an answer.

**Source attribution:** Checks whether the cited source information is consistent with the retrieved passages.

**OCR extraction:** Checks whether text from images and scanned documents can be extracted and used.

**URL ingestion:** Verified by the developer as working in the main application.

### Test approach

1. Add known documents or other supported sources.
2. Ask questions whose answers are explicitly stated in those sources.
3. Compare the generated answers against the original material.
4. Inspect the associated source references.
5. Ask questions that cannot be answered from the available sources.
6. Test different question lengths and multiple-question inputs.
7. Record incorrect answers, missing responses, retrieval errors, and performance issues.

Previous checks of representative factual questions, specific-detail retrieval, missing-information handling, and source attribution were reported as passing. These results should not be interpreted as proof that every possible query is correct.

## 14. Current Known Issue

### Empty answers for larger multi-question requests in Groq

The main application has an intermittent issue in which a request containing multiple questions may return an empty answer from the Groq response.

Observed behavior has included:

- Single-question requests working.
- Some smaller multi-question requests working.
- Larger batches, including five-question batches, sometimes returning an empty answer.
- The interface displaying an empty-response message.

The underlying cause has not yet been confirmed.

Potential investigation areas include the model response's finish reason, returned content, token usage, request size, model-specific behavior, and context length. These are investigation areas, not confirmed causes.

The appropriate next step is to inspect the actual API response and diagnostic information before making changes to retrieval logic, request handling, or the user interface.

This issue is separate from URL ingestion, which has been verified as working.

## 15. Limitations and Considerations

- Answer quality depends on the accuracy and completeness of the available sources.
- Text extraction can lose structure or formatting from complex documents.
- OCR results depend on image quality, language, layout, and scan clarity.
- Semantic similarity does not guarantee that the retrieved passage contains the answer.
- The language model can still generate incorrect or unsupported statements.
- Source references help verification but do not independently guarantee correctness.
- The terminal chatbot retrieves up to three passages, which may not be sufficient for every question.
- The terminal implementation uses a local Ollama model, whereas the main application uses Groq.
- Large multi-question requests in the main application require further troubleshooting.
- URL access depends on the accessibility and compatibility of the target webpage.
- Sensitive or confidential documents should only be processed in accordance with organizational security and privacy requirements.

## 16. Security and Privacy

The following practices should be followed when developing and using KnowVia:

- Store API keys outside source code.
- Do not commit secrets or private environment files to source control.
- Use trusted documents and URLs.
- Avoid uploading confidential business information to services that are not approved for that data.
- Review the data handling, retention, and privacy policies of any external model provider.
- Validate document inputs and handle parsing errors safely.
- Keep dependencies updated and use a dedicated virtual environment.
- Test the system's ability to abstain when source material does not support an answer.
- Do not treat AI-generated answers as authoritative without checking the source where accuracy is important.

## 17. Development Progress

The project has progressed from a basic document-question-answering concept to a functional application with document retrieval, LLM integration, and source-oriented responses.

| Area | Status |
|---|---|
| Python-based project setup | Implemented |
| Streamlit main application | Implemented |
| Groq integration in the main application | Implemented; larger batch issue under investigation |
| Local Ollama terminal chatbot | Implemented in `chatbot.py` |
| TXT, PDF, and DOCX reading in terminal chatbot | Implemented |
| Semantic embeddings in terminal chatbot | Implemented |
| Cosine-similarity passage retrieval | Implemented |
| Source metadata in terminal chatbot | Implemented |
| Image and scanned-document OCR in main application | Implemented and tested |
| Public URL ingestion in main application | Verified working |
| Source attribution checks | Tested on representative examples |
| Basic factual and specific-detail checks | Tested on representative examples |
| Larger multi-question Groq response reliability | Requires further investigation |
| Additional evaluation and edge-case testing | Ongoing |

These statuses describe the development and testing reported so far, rather than a guarantee of production readiness.

## 18. Potential Future Enhancements

Possible improvements include:

1. Diagnose and resolve the intermittent empty-response issue for larger Groq batches.
2. Expand the evaluation dataset with more answerable and unanswerable questions.
3. Improve retrieval quality through systematic testing of chunking, ranking, and context selection.
4. Add structured evaluation metrics for answer correctness, retrieval relevance, and source faithfulness.
5. Improve error messages and diagnostic logging.
6. Add support for additional file formats where useful.
7. Improve handling of complex tables, layouts, and scanned PDFs.
8. Add regression tests to ensure that existing features continue working after code changes.
9. Document dependencies and setup steps in a maintained `requirements.txt`.
10. Review access controls, secret handling, and deployment requirements before using confidential data or deploying the application for wider organizational use.

These are potential enhancements, not claims that the features are already implemented.

## 19. Glossary

**LLM — Large Language Model:** An AI model that generates and interprets natural-language text.

**RAG — Retrieval-Augmented Generation:** A method that retrieves relevant external information and supplies it to a language model when generating an answer.

**Embedding:** A numerical vector representation of text used to compare semantic similarity.

**Chunking:** Dividing longer documents into smaller passages for retrieval.

**Semantic search:** Searching based on similarity in meaning rather than exact keyword matches alone.

**Cosine similarity:** A mathematical measure used to compare the direction of two vectors.

**OCR — Optical Character Recognition:** Technology that converts text appearing in images or scanned documents into machine-readable text.

**Context:** The information supplied to the model alongside a question to guide its response.

**Source attribution:** Identifying the document or passage supporting an answer.

**Hallucination:** An AI-generated statement that is incorrect or unsupported by the available evidence.

**API — Application Programming Interface:** A defined mechanism through which software systems communicate.

**Streamlit:** A Python framework for building interactive web applications.

## 20. Conclusion

KnowVia demonstrates how document processing, semantic retrieval, embeddings, and language-model inference can be combined to create a practical document question-answering application.

The main Streamlit application provides the primary user-facing experience, including Groq integration, document-based retrieval, OCR support, source-oriented answers, and verified URL ingestion. The additional `chatbot.py` script demonstrates the same general RAG concept through a simpler terminal-based implementation using Sentence Transformers and Ollama.

The project provides a foundation for further work in retrieval quality, model reliability, evaluation, observability, security, and user experience.

**Project name:** KnowVia  
**Primary interface:** Streamlit web application  
**Primary LLM provider:** Groq  
**Additional implementation:** Ollama-based terminal chatbot  
**Core approach:** Retrieval-Augmented Generation (RAG)  
**Development environment:** Python and Visual Studio Code
