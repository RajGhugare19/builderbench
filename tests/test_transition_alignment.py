"""F-06/F-07 transition-alignment regression tests.

The agents package __init__ chain-imports model-client dependencies that
are not installed here, so the agent modules are loaded through a
namespace-package bootstrap (no production file is modified).  All LLM
interaction goes through deterministic stub clients; no network calls.

Causal ground truth used throughout: act() receives OBS-k and produces
ACT-k, which causes OBS-(k+1).  OBS-1 is the initial observation and has
no causal action.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import types
import unittest
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = REPO_ROOT / "agents"


def _bootstrap_agents():
    if "agents" in sys.modules and hasattr(sys.modules["agents"], "__path__"):
        return
    pkg = types.ModuleType("agents")
    pkg.__path__ = [str(AGENTS_DIR)]
    sys.modules["agents"] = pkg
    for name in ("prompt_buffer", "base", "reflexion", "naive", "cot"):
        spec = importlib.util.spec_from_file_location(
            f"agents.{name}", AGENTS_DIR / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"agents.{name}"] = module
        spec.loader.exec_module(module)


_bootstrap_agents()

from agents.reflexion import ReflexionAgent  # noqa: E402
from agents.naive import NaiveAgent  # noqa: E402
from agents.cot import CoTAgent  # noqa: E402

LLMResponse = namedtuple(
    "LLMResponse", ["completion", "auxiliary", "input_tokens", "output_tokens"]
)

SUMMARY_MARKER = "Recent Events (since last summary)"


class StubClient:
    """Labels action responses ACT-1..ACT-n in call order; summary calls
    (identified by the summary prompt text) return SUMMARY-n instead."""

    def __init__(self, cot=False):
        self.calls = []
        self.cot = cot
        self.action_counter = 0
        self.summary_counter = 0

    def generate(self, messages):
        self.calls.append([(m.role, m.content) for m in messages])
        is_summary = any(SUMMARY_MARKER in content for _, content in self.calls[-1])
        if is_summary:
            self.summary_counter += 1
            return LLMResponse(f"SUMMARY-{self.summary_counter}", "", 1, 1)
        self.action_counter += 1
        label = f"ACT-{self.action_counter}"
        completion = (
            f"<thinking>plan {self.action_counter}</thinking>\n"
            f"<action>{label}</action>"
            if self.cot
            else label
        )
        return LLMResponse(completion, "", 1, 1)


def make_config(agent_name, kwargs, num_episodes=3):
    return SimpleNamespace(
        agent_config=SimpleNamespace(agent_name=agent_name, agent_kwargs=kwargs),
        num_episodes=num_episodes,
    )


def summary_event_tokens(message_content):
    """Extract the ordered event tokens from an update_summary prompt."""
    events = message_content.split(SUMMARY_MARKER + ":\n", 1)[1]
    events = events.split("\n\nAt this step", 1)[0]
    return re.findall(
        r"(?:Action taken:\n\n|Observation:\n\nTrial number: \d+\n\n)([^\n]+)",
        events,
    )


def summary_prompts(client):
    return [
        content
        for call in client.calls
        for role, content in call
        if role == "user" and SUMMARY_MARKER in content
    ]


def enter_trial_one(agent, client):
    """Trial 0 archives nothing; one warm-up act + reset moves to trial 1
    so reset() exercises the final-summary path.  The warm-up is discarded
    from the stub's counters so trial-1 actions are labelled ACT-1..n."""
    agent.act("WARMUP", {})
    agent.reset(terminal_obs="ignored", terminal_info={})
    client.action_counter = 0
    client.calls.clear()


class ReflexionPeriodicSummaryTests(unittest.TestCase):
    """F-06: periodic summary windows keep every causal action."""

    def _run_eight_acts(self):
        client = StubClient()
        agent = ReflexionAgent(
            lambda: client,
            make_config("reflexion", {"max_text_history": 5}),
            "SYS",
        )
        self.assertEqual(agent.summary_update_interval, 4)
        for k in range(1, 9):
            agent.act(f"OBS-{k}", {})
        return client, summary_prompts(client)

    def test_F06_step8_summary_contains_full_causal_sequence(self):
        _, prompts = self._run_eight_acts()
        self.assertEqual(len(prompts), 2)
        self.assertEqual(
            summary_event_tokens(prompts[1]),
            ["ACT-4", "OBS-5", "ACT-5", "OBS-6", "ACT-6", "OBS-7", "ACT-7", "OBS-8"],
        )

    def test_F06_step4_summary_remains_correct(self):
        _, prompts = self._run_eight_acts()
        self.assertEqual(
            summary_event_tokens(prompts[0]),
            ["OBS-1", "ACT-1", "OBS-2", "ACT-2", "OBS-3", "ACT-3", "OBS-4"],
        )

    def test_initial_observation_never_gets_fabricated_action(self):
        _, prompts = self._run_eight_acts()
        self.assertTrue(prompts[0].split(SUMMARY_MARKER + ":\n", 1)[1].startswith(
            "Observation:\n\nTrial number: 0\n\nOBS-1"
        ))


class ReflexionTerminationTests(unittest.TestCase):
    """F-07: the final trial summary keeps the terminal-producing action."""

    def _terminate_after(self, k):
        client = StubClient()
        agent = ReflexionAgent(
            lambda: client,
            make_config("reflexion", {"max_text_history": 5}),
            "SYS",
        )
        enter_trial_one(agent, client)
        for j in range(1, k + 1):
            agent.act(f"OBS-{j}", {})
        before = len(summary_prompts(client))
        agent.reset(terminal_obs="TERMINAL-T", terminal_info={})
        prompts = summary_prompts(client)
        self.assertEqual(len(prompts), before + 1)
        return summary_event_tokens(prompts[-1])

    def test_F07_aligned_termination_includes_terminal_action(self):
        # 4 acts: periodic summary already ran at step 4; remainder window
        # is the terminal observation plus the action that produced it.
        self.assertEqual(self._terminate_after(4), ["ACT-4", "TERMINAL-T"])

    def test_F07_unaligned_termination_full_remaining_sequence(self):
        self.assertEqual(
            self._terminate_after(5),
            ["ACT-4", "OBS-5", "ACT-5", "TERMINAL-T"],
        )

    def test_F07_unaligned_six_acts_full_remaining_sequence(self):
        self.assertEqual(
            self._terminate_after(6),
            ["ACT-4", "OBS-5", "ACT-5", "OBS-6", "ACT-6", "TERMINAL-T"],
        )

    def test_termination_boundary_scan_k1_to_k8_no_exceptions(self):
        for k in range(1, 9):
            tokens = self._terminate_after(k)
            self.assertEqual(tokens[-1], "TERMINAL-T", f"k={k}")
            self.assertIn(f"ACT-{k}", tokens, f"k={k}: terminal action missing")


class RoleSafetyTests(unittest.TestCase):
    """Offset=0 windows must never start with an assistant turn after a
    system-only prefix, yet the first causal action must stay visible."""

    def _full_buffer_midstep_naive(self, system_prompt):
        client = StubClient()
        agent = NaiveAgent(
            lambda: client,
            make_config("naive", {"max_text_history": 16}),
            system_prompt,
        )
        for k in range(1, 17):
            agent.act(f"OBS-{k}", {})
        # mid-step of act 17: observation added (evicts OBS-1), action not yet
        agent.prompt_buffer.add("observation", "OBS-17")
        return agent.get_prompt_from_buffer()

    def test_system_only_prefix_no_leading_assistant_action_embedded(self):
        messages = self._full_buffer_midstep_naive("SYS")
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[1].role, "user")
        self.assertNotIn("assistant", [m.role for m in messages[:2]])
        first_user = messages[1].content
        self.assertIn("Prior agent action (context):", first_user)
        self.assertIn("ACT-1", first_user)      # causal action of OBS-2
        self.assertIn("OBS-2", first_user)
        self.assertTrue(first_user.startswith("Prior agent action (context):"))
        # remaining window keeps the normal assistant/user alternation
        self.assertEqual(messages[2].role, "assistant")
        self.assertEqual(messages[2].content, "ACT-2")
        self.assertEqual(messages[3].role, "user")

    def test_empty_system_prompt_no_leading_assistant(self):
        messages = self._full_buffer_midstep_naive("")
        self.assertEqual(messages[0].role, "user")
        self.assertIn("ACT-1", messages[0].content)
        self.assertIn("Prior agent action (context):", messages[0].content)

    def test_reflexion_user_prefix_keeps_normal_assistant_sequence(self):
        client = StubClient()
        agent = ReflexionAgent(
            lambda: client,
            make_config("reflexion", {"max_text_history": 5}),
            "SYS",
        )
        for k in range(1, 6):
            agent.act(f"OBS-{k}", {})
        agent.prompt_buffer.add("observation", "OBS-6")
        messages = agent.get_prompt_from_buffer()
        roles = [m.role for m in messages]
        self.assertEqual(roles[0], "system")
        self.assertEqual(roles[1], "user")          # current-summary block
        self.assertEqual(roles[2], "assistant")     # restored first action
        self.assertEqual(messages[2].content, "ACT-1")
        self.assertEqual(roles[3], "user")
        self.assertIn("OBS-2", messages[3].content)
        self.assertNotIn("Prior agent action (context):", messages[3].content)
        self.assertEqual(roles[-1], "user")
        self.assertEqual(
            roles, ["system", "user"] + ["assistant", "user"] * 5
        )

    def test_naive_early_steps_identical_to_original_pairing(self):
        # offset=-1 regime: behaviour outside the target boundary unchanged
        client = StubClient()
        agent = NaiveAgent(
            lambda: client, make_config("naive", {"max_text_history": 16}), "SYS"
        )
        for k in range(1, 4):
            agent.act(f"OBS-{k}", {})
        agent.prompt_buffer.add("observation", "OBS-4")
        messages = agent.get_prompt_from_buffer()
        self.assertEqual(
            [(m.role, m.content.split("\n\n")[-1]) for m in messages],
            [
                ("system", "SYS"),
                ("user", "OBS-1"),
                ("assistant", "ACT-1"),
                ("user", "OBS-2"),
                ("assistant", "ACT-2"),
                ("user", "OBS-3"),
                ("assistant", "ACT-3"),
                ("user", "OBS-4"),
            ],
        )
        self.assertNotIn(
            "Prior agent action (context):",
            "\n".join(m.content for m in messages),
        )


class CoTAlignmentTests(unittest.TestCase):
    """Both CoT branches (reasoning-first, action-fallback) keep the
    first window's causal context without a leading assistant turn."""

    def _full_buffer_midstep_cot(self, max_cot_history):
        client = StubClient(cot=True)
        agent = CoTAgent(
            lambda: client,
            make_config(
                "cot",
                {"max_text_history": 16, "max_cot_history": max_cot_history},
            ),
            "SYS",
        )
        for k in range(1, 17):
            agent.act(f"OBS-{k}", {})
        agent.prompt_buffer.add("observation", "OBS-17")
        return agent.get_prompt_from_buffer()

    def test_reasoning_first_embeds_first_plan_as_context(self):
        messages = self._full_buffer_midstep_cot(16)
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[1].role, "user")
        first_user = messages[1].content
        self.assertTrue(first_user.startswith("Prior agent reasoning (context):"))
        self.assertIn("Previous plan:", first_user)
        self.assertIn("plan 1", first_user)         # reasoning that caused OBS-2
        self.assertIn("OBS-2", first_user)
        self.assertEqual(messages[2].role, "assistant")
        self.assertIn("Previous plan:", messages[2].content)
        self.assertIn("plan 2", messages[2].content)
        self.assertEqual(messages[3].role, "user")

    def test_action_fallback_embeds_first_action_as_context(self):
        messages = self._full_buffer_midstep_cot(1)
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[1].role, "user")
        first_user = messages[1].content
        self.assertTrue(first_user.startswith("Prior agent action (context):"))
        self.assertIn("ACT-1", first_user)
        self.assertIn("OBS-2", first_user)
        # reasoning buffer holds only the latest plan: the final assistant
        # turn before the current observation must still be reasoning-first
        assistants = [m for m in messages if m.role == "assistant"]
        self.assertIn("Previous plan:", assistants[-1].content)
        self.assertIn("plan 16", assistants[-1].content)

    def test_cot_early_steps_reasoning_sequence_unchanged(self):
        client = StubClient(cot=True)
        agent = CoTAgent(
            lambda: client,
            make_config("cot", {"max_text_history": 16, "max_cot_history": 16}),
            "SYS",
        )
        for k in range(1, 4):
            agent.act(f"OBS-{k}", {})
        agent.prompt_buffer.add("observation", "OBS-4")
        messages = agent.get_prompt_from_buffer()
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[1].role, "user")   # OBS-1: no fabricated cause
        self.assertNotIn("Prior agent", messages[1].content)
        self.assertEqual(messages[2].role, "assistant")
        self.assertIn("plan 1", messages[2].content)
        self.assertEqual(messages[3].role, "user")
        self.assertIn("OBS-2", messages[3].content)


class NegativeControlTests(unittest.TestCase):
    """Reconstruct the pre-fix logic in-line and prove the new assertions
    are load-bearing: the original code fails them, and the naive fix
    (drop i > 0 unconditionally) breaks API role safety."""

    def test_original_update_summary_omits_boundary_action(self):
        client = StubClient()
        agent = ReflexionAgent(
            lambda: client,
            make_config("reflexion", {"max_text_history": 5}),
            "SYS",
        )
        for k in range(1, 9):
            agent.act(f"OBS-{k}", {})
        # Buffers now hold the post-step-8 state; rebuild the step-8 window
        # exactly as the original code did (i > 0 guard).
        observations = agent.prompt_buffer.get("observation")[-4:]
        actions = agent.prompt_buffer.get("action")[-4:]
        offset = len(actions) - len(observations)
        event_lines = []
        for i in range(len(observations)):
            action_idx = i + offset
            if action_idx >= 0 and i > 0:           # ORIGINAL guard
                event_lines.append(f"Action taken:\n\n{actions[action_idx]}")
            event_lines.append(f"Observation:\n\n{observations[i]}")
        original_tokens = re.findall(
            r"(?:Action taken:\n\n|Observation:\n\nTrial number: \d+\n\n)([^\n]+)",
            "\n\n".join(event_lines),
        )
        self.assertNotIn("ACT-4", original_tokens)  # original fails F-06
        fixed_tokens = summary_event_tokens(summary_prompts(client)[1])
        self.assertIn("ACT-4", fixed_tokens)

    def test_original_reset_omits_terminal_action(self):
        client = StubClient()
        agent = ReflexionAgent(
            lambda: client,
            make_config("reflexion", {"max_text_history": 5}),
            "SYS",
        )
        enter_trial_one(agent, client)
        for k in range(1, 5):
            agent.act(f"OBS-{k}", {})
        # Original final-summary window: num_events=1, i > 0 guard.
        observations = (agent.prompt_buffer.get("observation") + ["T"])[-1:]
        actions = agent.prompt_buffer.get("action")[-1:]
        offset = len(actions) - len(observations)
        kept = [
            actions[i + offset]
            for i in range(len(observations))
            if i + offset >= 0 and i > 0            # ORIGINAL guard
        ]
        self.assertEqual(kept, [])                  # original fails F-07
        self.assertEqual(actions, ["ACT-4"])        # the lost causal action

    def test_original_naive_prompt_drops_first_window_action(self):
        client = StubClient()
        agent = NaiveAgent(
            lambda: client, make_config("naive", {"max_text_history": 16}), "SYS"
        )
        for k in range(1, 17):
            agent.act(f"OBS-{k}", {})
        agent.prompt_buffer.add("observation", "OBS-17")
        observations = agent.prompt_buffer.get("observation")
        actions = agent.prompt_buffer.get("action")
        offset = len(actions) - len(observations)
        original_assistants = [
            actions[i + offset]
            for i in range(len(observations))
            if i + offset >= 0 and i > 0            # ORIGINAL guard
        ]
        self.assertNotIn("ACT-1", original_assistants)
        fixed = agent.get_prompt_from_buffer()
        self.assertIn("ACT-1", fixed[1].content)    # fixed keeps it visible

    def test_unconditional_guard_removal_would_lead_with_assistant(self):
        # The tempting one-line fix (action_idx >= 0, no role check) is
        # invalid after a system-only prefix: prove it, so the embedded
        # fallback is shown to be necessary rather than decorative.
        client = StubClient()
        agent = NaiveAgent(
            lambda: client, make_config("naive", {"max_text_history": 16}), "SYS"
        )
        for k in range(1, 17):
            agent.act(f"OBS-{k}", {})
        agent.prompt_buffer.add("observation", "OBS-17")
        observations = agent.prompt_buffer.get("observation")
        actions = agent.prompt_buffer.get("action")
        offset = len(actions) - len(observations)
        roles = ["system"]
        for i in range(len(observations)):
            if i + offset >= 0:                     # UNSAFE variant
                roles.append("assistant")
            roles.append("user")
        self.assertEqual(roles[1], "assistant")     # invalid leading turn
        fixed = agent.get_prompt_from_buffer()
        self.assertEqual(fixed[1].role, "user")     # our fix stays valid


if __name__ == "__main__":
    unittest.main()