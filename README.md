# PDF to Spanish Translator

A tool for translating PDF documents from English to Spanish using LLMs

## How It Works

1. **Extract**: Converts PDF to clean markdown using PyMuPDF4LLM
2. **Chunk**: Splits markdown into semantic chunks (titles, headings, body text)
3. **Translate**: Uses AI services with type-aware prompts for better context
4. **Generate**: Creates Spanish PDFs from translated markdown

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key
- Anthropic API key (optional)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd pdf-to-es

# Install dependencies with uv
uv install

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys
```

### Usage

```bash
# Place your PDF in the pdfs/ folder (or update the path in main.py)
cp your-document.pdf pdfs/

# Run the translation
uv run python main.py
```

## Configuration

### Environment Variables

Create a `.env` file with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```
