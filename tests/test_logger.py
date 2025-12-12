"""
Tests for logging functionality.
"""

from __future__ import annotations

import logging
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest

from src.utils.logger import setup_logger, get_logger, log_api_call, log_error


class TestLogger:
    """Test logger functionality."""

    def test_setup_logger_default(self):
        """Test setting up logger with defaults."""
        logger = setup_logger("test_logger_1")

        assert logger.name == "test_logger_1"
        assert logger.level == logging.INFO

    def test_setup_logger_custom_level(self):
        """Test setting up logger with custom level."""
        logger = setup_logger("test_logger_2", "DEBUG")

        assert logger.level == logging.DEBUG

    def test_setup_logger_creates_handlers(self):
        """Test that logger has handlers."""
        logger = setup_logger("test_logger_3")

        assert len(logger.handlers) > 0

    def test_setup_logger_idempotent(self):
        """Test that calling setup_logger multiple times doesn't duplicate handlers."""
        logger1 = setup_logger("test_logger_4")
        handler_count = len(logger1.handlers)

        logger2 = setup_logger("test_logger_4")

        assert len(logger2.handlers) == handler_count

    def test_get_logger(self):
        """Test getting global logger."""
        logger = get_logger()

        assert logger is not None
        assert isinstance(logger, logging.Logger)

    def test_log_api_call(self, caplog):
        """Test logging API call."""
        logger = setup_logger("test_api_call", "INFO")

        with caplog.at_level(logging.INFO):
            log_api_call(logger, "openai", "gpt-4o-mini", 100, 50, 0.00015)

        assert "API Call" in caplog.text
        assert "openai" in caplog.text
        assert "gpt-4o-mini" in caplog.text

    def test_log_error(self, caplog):
        """Test logging errors."""
        logger = setup_logger("test_error", "ERROR")

        try:
            1 / 0
        except Exception as e:
            with caplog.at_level(logging.ERROR):
                log_error(logger, e, "Test context")

        assert "Test context" in caplog.text
        assert "ZeroDivisionError" in caplog.text

    def test_log_error_without_context(self, caplog):
        """Test logging errors without context."""
        logger = setup_logger("test_error_2", "ERROR")

        try:
            raise ValueError("Test error")
        except Exception as e:
            with caplog.at_level(logging.ERROR):
                log_error(logger, e)

        assert "Test error" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])