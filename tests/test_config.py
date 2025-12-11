"""
Simple test to check if config loading works.
Ye script bina API keys ke test karega.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set dummy environment variables for testing
os.environ['OPENAI_API_KEY'] = 'sk-test-key-1234567890'
os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-test-key-1234567890'
os.environ['DEFAULT_PROVIDER'] = 'openai'
os.environ['TEMPERATURE'] = '0.7'
os.environ['MAX_TOKENS'] = '1000'
os.environ['LOG_LEVEL'] = 'INFO'

try:
    from src.config import get_settings
    
    print("🔄 Loading configuration...")
    settings = get_settings()
    
    print("\n✅ Configuration loaded successfully!\n")
    print(f"📊 Settings:")
    print(f"  - Default Provider: {settings.default_provider}")
    print(f"  - Temperature: {settings.temperature}")
    print(f"  - Max Tokens: {settings.max_tokens}")
    print(f"  - Log Level: {settings.log_level}")
    print(f"  - Cost Warning: ${settings.cost_warning_threshold}")
    print(f"  - Cost Limit: ${settings.cost_limit_threshold}")
    
    # Check API keys (without showing them)
    print(f"\n🔑 API Keys:")
    print(f"  - OpenAI: {'✓ Set' if settings.openai_api_key else '✗ Missing'}")
    print(f"  - Anthropic: {'✓ Set' if settings.anthropic_api_key else '✗ Missing'}")
    
    print("\n✨ All checks passed! Configuration is working correctly.")
    
except Exception as e:
    print(f"\n❌ Error loading configuration:")
    print(f"   {type(e).__name__}: {e}")
    print("\n💡 Make sure:")
    print("   1. You have a .env file (copy from .env.example)")
    print("   2. API keys are set in .env file")
    print("   3. All dependencies are installed (run: uv sync)")
    sys.exit(1)