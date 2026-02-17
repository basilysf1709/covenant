"""Tests for working memory."""

import json

from covenant.core.working_memory import AgentState, WorkingMemory


def test_default_wm():
    wm = WorkingMemory()
    assert wm.state == AgentState.IDLE
    assert wm.budget == 10
    assert wm.goal == ""


def test_add_observation():
    wm = WorkingMemory()
    wm.add_observation("saw a cat")
    assert "saw a cat" in wm.recent_observations


def test_compaction():
    wm = WorkingMemory()
    # Fill with large observations to trigger compaction
    for i in range(100):
        wm.add_observation(f"observation {i}: {'x' * 50}")
    assert wm._byte_size() <= wm.MAX_SIZE_BYTES


def test_snapshot():
    wm = WorkingMemory(goal="test goal")
    snap = wm.snapshot()
    data = json.loads(snap)
    assert data["goal"] == "test goal"
    assert "MAX_SIZE_BYTES" not in data


def test_model_validator_compacts():
    """Large initial data gets compacted during construction."""
    big_observations = [f"obs {i}: {'y' * 100}" for i in range(50)]
    wm = WorkingMemory(recent_observations=big_observations)
    assert wm._byte_size() <= wm.MAX_SIZE_BYTES
