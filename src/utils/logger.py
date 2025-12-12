"""
Logging configuration for the AI Assistant.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime


def setup_logger(name: str = "ai_assistant", level: str = "INFO") -> logging.Logger:
    """
    Create and configure a logger.
    """
    logger = logging.getLogger(name)

    # Do NOT duplicate handlers (tests check idempotency)
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    return logger


_global_logger: logging.Logger | None = None


def get_logger() -> logging.Logger:
    """
    Return global logger instance.
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_logger()
    return _global_logger


# -------------------------------------------------------
#  Required by tests: log_api_call(logger, provider, model, pt, ct, cost)
# -------------------------------------------------------
def log_api_call(
    logger: logging.Logger,
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost: float,
):
    """
    Log details of an API call.
    EXACT FORMAT expected by tests.
    """
    message = (
        f"API Call | provider={provider} | model={model} | "
        f"prompt_tokens={prompt_tokens} | completion_tokens={completion_tokens} | cost={cost}"
    )

    logger.info(message)
    return message  # not required but safe


# -------------------------------------------------------
# Required format for tests
# -------------------------------------------------------
def log_error(logger: logging.Logger, error: Exception, context: str = None):
    """
    Log error in the format required by tests.
    Must contain:
        - exception type
        - exception message
        - context (if provided)
    """
    exc_type = type(error).__name__
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if context:
        message = f"{timestamp} | ERROR | {exc_type}: {error} | Context: {context}"
    else:
        message = f"{timestamp} | ERROR | {exc_type}: {error}"

    logger.error(message)
    return message




# """
# Logging configuration for the AI Assistant.

# Ye file logging setup karti hai - debugging aur monitoring ke liye.
# """

# from __future__ import annotations

# import logging
# import sys
# from typing import Any


# def setup_logger(name: str = "ai_assistant", level: str = "INFO") -> logging.Logger:
#     """
#     Configure and return a logger instance.
    
#     Args:
#         name: Logger name (default: "ai_assistant")
#         level: Log level - DEBUG, INFO, WARNING, ERROR (default: "INFO")
    
#     Returns:
#         Configured logger instance
        
#     Example:
#         >>> logger = setup_logger("ai_assistant", "DEBUG")
#         >>> logger.info("Application started")
#     """
#     # Get or create logger
#     logger = logging.getLogger(name)
    
#     # Agar already configured hai toh return karo
#     if logger.handlers:
#         return logger
    
#     # Set log level
#     log_level = getattr(logging, level.upper(), logging.INFO)
#     logger.setLevel(log_level)
    
#     # Console handler banao (terminal mein print karne ke liye)
#     console_handler = logging.StreamHandler(sys.stdout)
#     console_handler.setLevel(log_level)
    
#     # Formatter banao (log message ka format)
#     formatter = logging.Formatter(
#         fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S",
#     )
#     console_handler.setFormatter(formatter)
    
#     # Handler ko logger se attach karo
#     logger.addHandler(console_handler)
    
#     # Prevent propagation to root logger (duplicate logs se bachne ke liye)
#     logger.propagate = False
    
#     return logger


# def log_api_call(
#     logger: logging.Logger,
#     provider: str,
#     model: str,
#     input_tokens: int,
#     output_tokens: int,
#     cost: float,
# ) -> None:
#     """
#     Log API call details.
    
#     Args:
#         logger: Logger instance
#         provider: LLM provider name
#         model: Model name
#         input_tokens: Number of input tokens
#         output_tokens: Number of output tokens
#         cost: API call cost in USD
#     """
#     logger.info(
#         f"API Call | Provider: {provider} | Model: {model} | "
#         f"Tokens: {input_tokens}→{output_tokens} | Cost: ${cost:.6f}"
#     )


# def log_error(logger: logging.Logger, error: Exception, context: str = "") -> None:
#     """
#     Log error with context.
    
#     Args:
#         logger: Logger instance
#         error: Exception that occurred
#         context: Additional context about where error occurred
#     """
#     error_msg = f"{context}: {type(error).__name__} - {str(error)}" if context else str(error)
#     logger.error(error_msg, exc_info=True)


# # Global logger instance
# _global_logger: logging.Logger | None = None


# def get_logger() -> logging.Logger:
#     """
#     Get the global logger instance.
    
#     Returns:
#         Global logger instance
#     """
#     global _global_logger
#     if _global_logger is None:
#         _global_logger = setup_logger()
#     return _global_logger


# if __name__ == "__main__":
#     # Test logging
#     logger = setup_logger("test_logger", "DEBUG")
    
#     logger.debug("This is a debug message")
#     logger.info("This is an info message")
#     logger.warning("This is a warning message")
#     logger.error("This is an error message")
    
#     # Test API call logging
#     log_api_call(logger, "openai", "gpt-4o-mini", 100, 50, 0.00015)
    
#     # Test error logging
#     try:
#         1 / 0
#     except Exception as e:
#         log_error(logger, e, "Division test")