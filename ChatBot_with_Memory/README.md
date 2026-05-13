# Chatbot with Conversational Memory

A custom chatbot built with LangChain and Groq, featuring persistent in-session memory and a Gradio chat interface.

## How it works

- A running list of messages (system + user + assistant) is maintained across turns
- The assistant is pre-loaded with user context at startup
- Each new message appends to history before invoking the LLM, giving full conversation context
- Token usage is returned alongside each response

## Stack

| Component | Tool |
|---|---|
| LLM | `openai/gpt-oss-20b` via Groq |
| Memory | In-session message list |
| Interface | Gradio `ChatInterface` |

## API Key Setup

**In Google Colab:** add `GROQ_API_KEY` to Secrets (🔑 left sidebar)

**Locally:** create a `.env` file:
```
GROQ_API_KEY=your_key_here
```

## Requirements

```bash
pip install langchain-groq gradio python-dotenv
```
