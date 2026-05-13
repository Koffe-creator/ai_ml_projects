# First LLM — LangChain + Groq

An introductory notebook exploring LLM inference using [LangChain](https://www.langchain.com/) and the [Groq](https://groq.com/) API, with two models: Llama 3.1 and GPT-o.

## Models Used

| Model | ID |
|---|---|
| Llama 3.1 8B | `llama-3.1-8b-instant` |
| GPT-o 120B | `openai/gpt-oss-120b` |

## What it covers

- Installing and configuring `langchain-groq`
- Invoking LLMs with a simple prompt
- Inspecting token usage metadata (`prompt_tokens`, `completion_tokens`, `total_time`)
- Testing multi-turn context awareness

## API Key Setup

### In Google Colab (recommended)
1. Open the **Secrets** panel (🔑 icon in the left sidebar)
2. Add a secret named `GROQ_API_KEY` with your key from [console.groq.com](https://console.groq.com)

### Running locally
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Fill in your key:
   ```
   GROQ_API_KEY=your_key_here
   ```
3. Install the extra dependency:
   ```bash
   pip install python-dotenv
   ```

The notebook auto-detects the environment and picks the right method.

## Requirements

```bash
pip install langchain-groq
```
