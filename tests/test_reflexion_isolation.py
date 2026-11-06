"""Regression tests for Reflexion cross-trial state isolation.

These tests exercise only stdlib.  They deliberately avoid importing the
real ``agents`` package (whose ``__init__`` transitively pulls in
``google-genai``, ``openai`` and ``anthropic``); see
``tests/_bootstrap.py``.

Deterministic stubs replace the LLM client so no API call is made.
"""

from __future__ import annotations

import unittest
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402  installs a namespace-only `agents` package

PromptBuffer = _bootstrap.prompt_buffer.PromptBuffer
ReflexionAgent = _bootstrap.reflexion.ReflexionAgent


LLMResponse = namedtuple(
    "LLMResponse",
    ["model_id", "completion", "stop_reason", "input_tokens", "output_tokens", "auxiliary"],
)


class StubClient:
    """Client stand-in that mirrors agents.client.LLMClientWrapper's surface.

    Records every ``generate`` call and returns a deterministic
    ``LLMResponse`` whose completion text contains a unique call index so
    assertions can pin down exactly what was injected where.
    """

    def __init__(self):
        self.calls = []
        self._counter = 0

    def generate(self, messages):
        self._counter += 1
        self.calls.append([{"role": m.role, "content": m.content, "attachment": m.attachment} for m in messages])
        return LLMResponse(
            model_id="stub",
            completion=f"RESP-{self._counter}",
            stop_reason="stop",
            input_tokens=10,
            output_tokens=5,
            auxiliary=None,
        )


def _make_config(**agent_kwargs):
    return SimpleNamespace(
        num_episodes=3,
        agent_config=SimpleNamespace(agent_name="reflexion", agent_kwargs=agent_kwargs),
    )


def _make_agent(**agent_kwargs):
    client = StubClient()
    agent = ReflexionAgent(lambda: client, _make_config(**agent_kwargs), "SYSTEM")
    return agent, client


class PromptBufferClearTests(unittest.TestCase):
    def test_clear_mutates_underlying_deque(self):
        pb = PromptBuffer(limits={"observation": 5, "action": 5, "previous_summaries": 5, "summary": 1})
        pb.add("observation", "obs-a")
        pb.add("action", "act-a")
        pb.clear("observation")
        self.assertEqual(pb.get("observation"), [])
        self.assertEqual(pb.get("action"), ["act-a"])

    def test_clear_preserves_maxlen(self):
        pb = PromptBuffer(limits={"observation": 3})
        for v in ["a", "b", "c"]:
            pb.add("observation", v)
        pb.clear("observation")
        pb.add("observation", "x")
        pb.add("observation", "y")
        self.assertEqual(pb.get("observation"), ["x", "y"])

    def test_clear_unknown_key_raises_keyerror(self):
        pb = PromptBuffer(limits={"observation": 5})
        with self.assertRaises(KeyError):
            pb.clear("nope")


class ReflexionResetIsolationTests(unittest.TestCase):
    """Drive one trial through a stubbed ReflexionAgent and inspect buffers
    after ``reset``.  ``reset`` for the very first trial short-circuits the
    archive branch (trial_index==0); subsequent resets archive the current
    summary into previous_summaries and then clear the per-trial buffers.
    """

    def _run_trial(self, agent, tag, num_steps):
        for i in range(num_steps):
            agent.act(f"OBS-{tag}-{i}", {})

    def test_reset_clears_raw_buffers(self):
        agent, _ = _make_agent(max_text_history=3, max_reflection_history=3)
        agent.reset(None, None)
        self._run_trial(agent, "T1", 4)
        agent.reset("TERMINAL-T1", None)
        self.assertEqual(agent.prompt_buffer.get("observation"), [])
        self.assertEqual(agent.prompt_buffer.get("action"), [])
        self.assertEqual(agent.prompt_buffer.get("summary"), [])

    def test_reset_preserves_previous_summaries(self):
        agent, _ = _make_agent(max_text_history=3, max_reflection_history=3)
        agent.reset(None, None)
        self._run_trial(agent, "T1", 4)
        agent.reset("TERMINAL-T1", None)
        previous = agent.prompt_buffer.get("previous_summaries")
        self.assertEqual(len(previous), 1, "exactly one trial should have been archived")
        self.assertTrue(all(isinstance(s, str) and s for s in previous))

    def test_previous_summaries_survive_multiple_resets(self):
        agent, _ = _make_agent(max_text_history=3, max_reflection_history=3)
        agent.reset(None, None)
        self._run_trial(agent, "T1", 4)
        agent.reset("TERMINAL-T1", None)
        self._run_trial(agent, "T2", 3)
        agent.reset("TERMINAL-T2", None)
        previous = agent.prompt_buffer.get("previous_summaries")
        self.assertEqual(len(previous), 2)
        self._run_trial(agent, "T3", 2)
        agent.reset("TERMINAL-T3", None)
        previous = agent.prompt_buffer.get("previous_summaries")
        self.assertEqual(len(previous), 3)


class ReflexionPromptInjectionTests(unittest.TestCase):
    """Verify the archived summary reaches the next trial's prompt exactly
    once and that raw per-trial records do not leak across the boundary.

    Assertions must inspect **every** message of every recorded LLM call,
    not a positional slice: the pre-fix leak injected trial-1 records into
    message slots that were not the second message, so a ``call[1]`` probe
    would have missed them.
    """

    def _run_trial(self, agent, tag, num_steps):
        for i in range(num_steps):
            agent.act(f"OBS-{tag}-{i}", {})

    @staticmethod
    def _concat_all_messages(messages):
        """Join the ``content`` of every message in a single recorded call."""
        return "\n".join(message["content"] for message in messages)

    @classmethod
    def _concat_every_call(cls, client):
        """Join every message in every LLM call the stub has recorded."""
        return "\n".join(cls._concat_all_messages(messages) for messages in client.calls)

    def test_trial_two_prompt_does_not_replay_trial_one_raw_history(self):
        agent, client = _make_agent(max_text_history=3, max_reflection_history=3)
        agent.reset(None, None)
        self._run_trial(agent, "T1", 4)
        agent.reset("TERMINAL-T1", None)
        client.calls.clear()

        agent.act("OBS-T2-0", {})
        # Inspect every message in every LLM call the agent has recorded since
        # the boundary; a positional probe on the second message would miss the
        # slots the pre-fix bug actually used.
        prompt_text = self._concat_every_call(client)
        for token in (
            "OBS-T1-0",
            "OBS-T1-1",
            "OBS-T1-2",
            "OBS-T1-3",
            "TERMINAL-T1",
            "Trial number: 1",
            "RESP-1",
            "RESP-2",
            "RESP-3",
            "RESP-4",
            "RESP-5",
            "RESP-6",
        ):
            self.assertNotIn(token, prompt_text, f"trial-1 artefact {token!r} leaked into trial-2 prompt")

    def test_archived_summary_is_injected_exactly_once(self):
        agent, client = _make_agent(max_text_history=3, max_reflection_history=3)
        agent.reset(None, None)
        self._run_trial(agent, "T1", 4)
        agent.reset("TERMINAL-T1", None)
        archived = agent.prompt_buffer.get("previous_summaries")[-1]
        self.assertTrue(archived, "archived summary should be a non-empty string")
        client.calls.clear()

        agent.act("OBS-T2-0", {})
        # The "first LLM call" of trial 2 is the only call recorded after the
        # clear; every message in it (not just the second) must be scanned.
        self.assertEqual(len(client.calls), 1, "trial 2's first act should record exactly one LLM call")
        first_call = client.calls[0]
        matches = [
            (idx, message["role"])
            for idx, message in enumerate(first_call)
            if archived in message["content"]
        ]
        self.assertEqual(
            len(matches),
            1,
            f"archived summary {archived!r} must appear in exactly one message of the first trial-2 LLM call; found in {matches}",
        )
        # The single match must be a previous_summaries block, i.e. labelled
        # "Summary of Trial 1:", never the "so far:" current-trial slot.
        matching_content = first_call[matches[0][0]]["content"]
        self.assertIn("Summary of Trial 1:", matching_content)
        self.assertNotIn("so far:", matching_content)


if __name__ == "__main__":
    unittest.main()
