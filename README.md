# RAG Application

This is a simple Retrieval Augmented Generation (RAG) app that lets you ask questions about your own documents. Instead of the AI just making stuff up, it actually reads your files first and uses that information to answer you.

## What is RAG?

Think of it like having a personal assistant who reads all your documents before answering your questions. You give it a question, it finds the relevant information from your files, and then it uses that knowledge to give you a more accurate answer. That's RAG.

## What You Need

First, install the required packages:

```
pip install llama-index-core llama-index-llms-openrouter llama-index-embeddings-huggingface python-dotenv
```

## Setting Up

1. Create a folder called `data` inside the RAG101 directory
2. Put your documents in there (PDFs, Word docs, text files, CSV files, whatever you have)
3. Create a `.env` file in the RAG101 directory with your OpenRouter API key like this:

```
OPENROUTER_API_KEY=your_api_key_here
```

## How to Run It

Open your terminal in the RAG101 folder and run:

```
python rag_app.py
```

The app will read all your documents, create an index of them, and then wait for you to ask a question. Type your question and hit enter. The AI will find the relevant info from your documents and answer based on that.

## Supported File Types

You can add any of these file types to the data folder and it will read them:

Text files, PDFs, Word documents, Excel spreadsheets, CSV files, Markdown files, HTML files, JSON files, and PowerPoint presentations.

## How It Works Behind the Scenes

When you run the app, three things happen:

First, it loads all your documents from the data folder.

Second, it breaks them down into smaller chunks and converts them into numbers that the AI can understand (called embeddings). This creates an index so the AI knows what's in each document.

Third, when you ask a question, it converts your question into the same number format, finds the most similar parts of your documents, and then feeds those relevant parts to the AI model along with your question.

The AI then uses all that context to give you a well informed answer instead of just guessing.

## Changing the Model

By default this uses Grok from x-ai. You can change it to any model available on OpenRouter. Just edit the rag_app.py file and change this line:

```
llm = OpenRouter(model="x-ai/grok-4.1-fast")
```

Check out openrouter.ai to see what models are available.

## Troubleshooting

If you get an error about the data folder not existing, just create one. If you get API key errors, make sure your .env file is in the RAG101 folder and has the correct API key. If something else breaks, the error message usually tells you what's wrong.
