# 🤖 AI Assistant - Multi-Provider CLI

A production-ready command-line AI assistant that supports both OpenAI and Anthropic LLM providers with comprehensive cost tracking, conversation management, and robust error handling.

## ✨ Features

- **Multi-Provider Support**: Seamlessly switch between OpenAI (GPT-4o-mini) and Anthropic (Claude 3.5 Haiku)
- **Conversation Management**: Maintains context across multiple turns
- **Cost Tracking**: Real-time cost estimation with configurable warnings and limits
- **Retry Logic**: Exponential backoff for handling rate limits and connection errors
- **Async Operations**: Non-blocking I/O for efficient API calls
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Type Safety**: Full type hints using Python 3.12+ syntax
- **Secure Configuration**: API keys managed via environment variables with Pydantic validation

## 📋 Requirements

- Python 3.12 or higher
- OpenAI API key
- Anthropic API key

## 🚀 Installation

### Step 1: Install UV Package Manager

**Linux/Mac:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 2: Clone and Setup Project

```bash
cd ai-assistant
uv sync
cp .env.example .env
```

### Step 3: Configure API Keys

Edit `.env` file:
```bash
OPENAI_API_KEY=sk-your-actual-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-actual-anthropic-key-here
```

## 🎮 Usage

```bash
uv run python src/main.py
```

### Available Commands

- `/help` - Show available commands
- `/clear` - Clear conversation history
- `/quit` or `/exit` - Exit application
- `/model <provider>` - Switch provider (openai or anthropic)
- `/cost` - Display cost summary
- `/history` - Show conversation history

## 🏗️ Architecture

```
ai-assistant/
├── src/
│   ├── main.py              # Main CLI application
│   ├── config.py            # Configuration management
│   ├── llm/
│   │   ├── base.py          # Abstract base class
│   │   ├── openai_provider.py
│   │   └── anthropic_provider.py
│   └── utils/
│       ├── cost_tracker.py  # Cost calculation
│       ├── conversation.py  # Chat history
│       └── logger.py        # Logging setup
├── tests/                   # Unit tests
├── pyproject.toml          # Dependencies
├── .env.example            # Environment template
└── README.md
```

## 💰 Cost Estimates

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| GPT-4o-mini | $0.150 | $0.600 |
| Claude 3.5 Haiku | $0.80 | $4.00 |

**Typical conversation cost:** $0.0001 - $0.001 per interaction

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src tests/

# Test individual components
uv run python src/config.py
uv run python src/utils/logger.py
uv run python src/utils/conversation.py
uv run python src/utils/cost_tracker.py
```

## 🔧 Configuration

Edit `.env` file to customize:

```bash
DEFAULT_PROVIDER=openai        # Default: openai
TEMPERATURE=0.7                # Range: 0.0-2.0
MAX_TOKENS=1000               # Maximum response length
COST_WARNING_THRESHOLD=0.10   # Warning at $0.10
COST_LIMIT_THRESHOLD=1.00     # Hard limit at $1.00
LOG_LEVEL=INFO                # DEBUG, INFO, WARNING, ERROR
```

## 🐛 Troubleshooting

### Issue: "Module not found" errors
**Solution:** Run `uv sync` to install dependencies

### Issue: "Invalid API key" errors
**Solution:** Verify `.env` file format and key validity

### Issue: "Rate limit exceeded"
**Solution:** Wait briefly; retry logic handles this automatically

### Issue: Import errors
**Solution:** Ensure running from project root directory

## 📚 Documentation

- [OpenAI Python Library](https://github.com/openai/openai-python)
- [Anthropic SDK](https://docs.anthropic.com/en/api/client-sdks)
- [UV Documentation](https://docs.astral.sh/uv/)
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)

## 🤝 Contributing

This is an educational project for the NexusBerry Full-Stack AI Engineer Course.

## 📄 License

Educational use only - NexusBerry Certified Full-Stack AI Engineer Course

## 🎓 Learning Objectives

This project demonstrates:
- Production-grade Python project structure
- Secure API key management
- Multi-provider LLM integration
- Async/await patterns
- Cost tracking and monitoring
- Comprehensive error handling
- Type hints and documentation
- Testing best practices

---

**Built with ❤️ for NexusBerry AI Engineering Course**