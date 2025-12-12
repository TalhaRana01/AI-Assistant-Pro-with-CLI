"""
Pytest configuration and shared fixtures.

Ye file pytest ke liye global configuration aur fixtures provide karti hai.
"""

import os
import sys

import pytest

# Add src directory to Python path for all tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]


@pytest.fixture
def mock_api_key():
    """Mock API key for testing."""
    return "sk-test-key-1234567890"


@pytest.fixture
def mock_anthropic_api_key():
    """Mock Anthropic API key for testing."""
    return "sk-ant-test-key-1234567890"