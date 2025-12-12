"""
Tests for cost tracking functionality.

Test karte hain ke cost calculations sahi hain aur tracking kaam kar rahi hai.
"""

from __future__ import annotations

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from src.utils.cost_tracker import CostEntry, CostTracker, count_tokens


class TestCostEntry:
    """Test CostEntry dataclass."""

    def test_cost_entry_creation(self):
        """Test creating a cost entry."""
        entry = CostEntry(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=1000,
            output_tokens=500,
            cost=0.00045,
        )

        assert entry.provider == "openai"
        assert entry.model == "gpt-4o-mini"
        assert entry.input_tokens == 1000
        assert entry.output_tokens == 500
        assert entry.cost == 0.00045

    def test_cost_entry_str_representation(self):
        """Test string representation of cost entry."""
        entry = CostEntry(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=1000,
            output_tokens=500,
            cost=0.00045,
        )

        str_repr = str(entry)
        assert "openai" in str_repr
        assert "gpt-4o-mini" in str_repr
        assert "1000" in str_repr
        assert "500" in str_repr


class TestCostTracker:
    """Test CostTracker class."""

    def test_cost_tracker_creation(self):
        """Test creating a cost tracker."""
        tracker = CostTracker()

        assert len(tracker.entries) == 0
        assert tracker.warning_threshold == 0.10
        assert tracker.limit_threshold == 1.00

    def test_cost_tracker_custom_thresholds(self):
        """Test creating tracker with custom thresholds."""
        tracker = CostTracker(warning_threshold=0.50, limit_threshold=2.00)

        assert tracker.warning_threshold == 0.50
        assert tracker.limit_threshold == 2.00

    def test_calculate_cost_gpt4o_mini(self):
        """Test cost calculation for GPT-4o-mini."""
        cost = CostTracker.calculate_cost("gpt-4o-mini", 1000, 500)

        # Expected: (1000/1M * 0.150) + (500/1M * 0.600)
        # = 0.00015 + 0.0003 = 0.00045
        assert abs(cost - 0.00045) < 1e-10

    def test_calculate_cost_claude_haiku(self):
        """Test cost calculation for Claude Haiku."""
        cost = CostTracker.calculate_cost("claude-3-5-haiku-20241022", 1000, 500)

        # Expected: (1000/1M * 0.80) + (500/1M * 4.00)
        # = 0.0008 + 0.002 = 0.0028
        assert abs(cost - 0.0028) < 1e-10

    def test_calculate_cost_unknown_model(self):
        """Test cost calculation for unknown model defaults to GPT-4o-mini."""
        cost = CostTracker.calculate_cost("unknown-model", 1000, 500)

        # Should default to GPT-4o-mini pricing
        expected = CostTracker.calculate_cost("gpt-4o-mini", 1000, 500)
        assert cost == expected

    def test_add_entry(self):
        """Test adding a cost entry."""
        tracker = CostTracker()

        cost = tracker.add_entry("openai", "gpt-4o-mini", 1000, 500)

        assert len(tracker.entries) == 1
        assert cost == 0.00045
        assert tracker.entries[0].provider == "openai"
        assert tracker.entries[0].input_tokens == 1000
        assert tracker.entries[0].output_tokens == 500

    def test_get_total_cost(self):
        """Test getting total cost across entries."""
        tracker = CostTracker()

        tracker.add_entry("openai", "gpt-4o-mini", 1000, 500)
        tracker.add_entry("openai", "gpt-4o-mini", 2000, 1000)

        total = tracker.get_total_cost()

        # First: 0.00045, Second: 0.0009
        assert abs(total - 0.00135) < 1e-10

    def test_get_total_tokens(self):
        """Test getting total tokens."""
        tracker = CostTracker()

        tracker.add_entry("openai", "gpt-4o-mini", 1000, 500)
        tracker.add_entry("openai", "gpt-4o-mini", 2000, 1000)

        total_input, total_output = tracker.get_total_tokens()

        assert total_input == 3000
        assert total_output == 1500

    def test_should_warn_false(self):
        """Test warning threshold not exceeded."""
        tracker = CostTracker(warning_threshold=1.00)
        tracker.add_entry("openai", "gpt-4o-mini", 1000, 500)

        assert tracker.should_warn() is False

    def test_should_warn_true(self):
        """Test warning threshold exceeded."""
        tracker = CostTracker(warning_threshold=0.0001)
        tracker.add_entry("openai", "gpt-4o-mini", 1000, 500)

        assert tracker.should_warn() is True

    def test_should_stop_false(self):
        """Test limit threshold not exceeded."""
        tracker = CostTracker(limit_threshold=1.00)
        tracker.add_entry("openai", "gpt-4o-mini", 1000, 500)

        assert tracker.should_stop() is False

    def test_should_stop_true(self):
        """Test limit threshold exceeded."""
        tracker = CostTracker(limit_threshold=0.0001)
        tracker.add_entry("openai", "gpt-4o-mini", 1000, 500)

        assert tracker.should_stop() is True

    def test_get_summary(self):
        """Test getting cost summary."""
        tracker = CostTracker()
        tracker.add_entry("openai", "gpt-4o-mini", 1000, 500)

        summary = tracker.get_summary()

        assert "SESSION COST SUMMARY" in summary
        assert "Total API Calls: 1" in summary
        assert "$" in summary

    def test_reset(self):
        """Test resetting tracker."""
        tracker = CostTracker()
        tracker.add_entry("openai", "gpt-4o-mini", 1000, 500)

        assert len(tracker.entries) == 1

        tracker.reset()

        assert len(tracker.entries) == 0
        assert tracker.get_total_cost() == 0.0

    def test_str_representation(self):
        """Test string representation of tracker."""
        tracker = CostTracker()
        tracker.add_entry("openai", "gpt-4o-mini", 1000, 500)

        str_repr = str(tracker)

        assert "CostTracker" in str_repr
        assert "1 calls" in str_repr


class TestCountTokens:
    """Test count_tokens function."""

    def test_count_tokens_gpt_model(self):
        """Test token counting for GPT models."""
        text = "Hello, world!"
        tokens = count_tokens(text, "gpt-4o-mini")

        assert tokens > 0
        assert isinstance(tokens, int)

    def test_count_tokens_non_gpt_model(self):
        """Test token counting for non-GPT models (approximate)."""
        text = "Hello, world!" * 10  # 130 characters
        tokens = count_tokens(text, "claude-3-5-haiku")

        # Approximate: 130 / 4 = 32.5 -> 32
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_count_tokens_empty_string(self):
        """Test token counting for empty string."""
        tokens = count_tokens("", "gpt-4o-mini")
        assert tokens == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])