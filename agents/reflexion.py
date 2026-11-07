import re
import json
import copy
from dataclasses import dataclass

from agents.base import BaseAgent, resolve_agent_config
from agents.prompt_buffer import PromptBuffer, Message

@dataclass
class AgentConfig:
    max_text_history: int = 5
    max_reflection_history: int = 5

class ReflexionAgent(BaseAgent):
    """Agent implements the Reflexion strategy: the agent periodically summarises its
    observation-action history into a concise reflection, and carries those
    reflections across trials so that mistakes from earlier attempts inform
    future behaviour.  After every summary_update_interval the agent asks the LLM to
    produce a condensed summary of recent events and strategy.  At the end
    of each trial the running summary is archived into previous_summaries so
    that subsequent trials can learn from prior attempts.
    """

    def __init__(self, client_factory, config, env_system_prompt):
        """Initialize the agent with client factory, config, prompt buffers, and summary state."""
        super().__init__(client_factory)

        self.agent_config = resolve_agent_config(config)

        self.system_prompt = self.update_system_prompt(env_system_prompt, num_trials=config.num_episodes)

        self.prompt_buffer = PromptBuffer(
            limits={
                "observation": self.agent_config.max_text_history,
                "action": self.agent_config.max_text_history,
                "previous_summaries": self.agent_config.max_reflection_history,
                "summary": 1,
            })

        self.summary_update_interval = max(1, self.agent_config.max_text_history - 1)
        self.summary_instructions = (
            "At this step, combine both the previous summary and the recent observation-action history into a single, concise, and focused new summary. "
            "Do not summarize the environment, but rather summarize the relevant aspects of strategy and the steps you took to attempt to complete the task. "
            "Develop a brief and actionable new plan that addresses any mistakes and references specific strategies you should have employed. "
            "This summarization is an intermediate process, do not output an action at this step. "
            "Avoid unnecessary details to prevent the summary from growing excessively in size. "
            "The purpose of this summary is to facilitate reflection and improve your performance in this and future trials."
        )
        self.trial_index = 0
        self.step_index = 0
    
    def act(self, obs, info):
        """Buffer the observation, optionally update the summary, call the LLM, and return the action."""
        obs = f"Trial number: {self.trial_index}\n\n" + obs
        self.prompt_buffer.add("observation", obs)
        self.step_index += 1

        if self.step_index % self.summary_update_interval == 0:
            summary_response = self.update_summary()

        messages = self.get_prompt_from_buffer()

        naive_instruction = (
            "Always output actions in the JSON format provided in the Action schema."
        )

        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + naive_instruction

        response = self.client.generate(messages)

        final_answer = self._extract_final_answer(response)
        self.prompt_buffer.add("action", final_answer.completion)

        if self.step_index % self.summary_update_interval == 0:
            final_answer = final_answer._replace(
                auxiliary=self.prompt_buffer.get("summary")[-1],
                input_tokens=final_answer.input_tokens + summary_response.input_tokens,
                output_tokens=final_answer.output_tokens + summary_response.output_tokens,
            )
        
        return final_answer

    def get_prompt_from_buffer(self):
        """Build the message list from previous summaries, current summary, and recent observations/actions."""
        messages = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))

        previous_summaries =  self.prompt_buffer.get("previous_summaries")
        for i in range(len(previous_summaries)):
            message_parts = []
            message_parts.append(f"Summary of Trial {self.trial_index - len(previous_summaries) + i}:")
            message_parts.append(previous_summaries[i])
            content = "\n\n".join(message_parts)
            message = Message(role="user", content=content)
            messages.append(message)

        current_summary = "N/A"
        summaries = self.prompt_buffer.get("summary")
        if summaries:
            current_summary = summaries[-1]
        message_parts = []
        message_parts.append(f"Summary of Trial {self.trial_index} so far:")
        message_parts.append(current_summary)
        content = "\n\n".join(message_parts)
        message = Message(role="user", content=content)
        messages.append(message)

        observations = self.prompt_buffer.get("observation")
        actions = self.prompt_buffer.get("action")

        action_offset = len(actions) - len(observations)        # action offset will either be 0 or -1

        for i in range(len(observations)):
            action_idx = i + action_offset
            pending_action = actions[action_idx] if action_idx >= 0 else None
            if pending_action is not None and (messages and messages[-1].role == "user"):
                message = Message(role="assistant", content=pending_action)
                messages.append(message)
                pending_action = None

            message_parts = []
            if pending_action is not None:
                # No preceding user message: an assistant turn here would be
                # an invalid leading assistant, so keep the causal action as
                # labelled context inside the observation message instead.
                message_parts.append("Prior agent action (context):")
                message_parts.append(pending_action)
            if i == len(observations) - 1:
                message_parts.append("Current Observation:")
            else:
                message_parts.append("Observation:")
            message_parts.append(observations[i])
            content = "\n\n".join(message_parts)
            message = Message(role="user", content=content)
            messages.append(message)

        return messages
    
    def reset(self, terminal_obs=None, terminal_info=None):
        """Archive the current trial's summary into previous_summaries, clear per-trial buffers, and increment the trial index."""
        if self.trial_index > 0:
            terminal_obs = f"Trial number: {self.trial_index}\n\n" + terminal_obs
            self.prompt_buffer.add("observation", terminal_obs)
            remainder_events = 1 + self.step_index % self.summary_update_interval
            self.update_summary(num_events=remainder_events)

            previous_summary = self.prompt_buffer.get("summary")[-1]
            self.prompt_buffer.add("previous_summaries", previous_summary)

        for key in ["observation", "action", "summary"]:
            self.prompt_buffer.get(key).clear()
        
        self.trial_index += 1
        self.step_index = 0
        
    def update_system_prompt(self, system_prompt, num_trials):
        """Append the maximum trial count to the system prompt."""
        added_instructions = (
            f"Your will get a maximum of {num_trials} trials to solve the task."
        )
        system_prompt  = system_prompt + "\n\n" + added_instructions

        return system_prompt

    def update_summary(self, num_events=None):
        """Ask the LLM to condense recent events into a new running summary and store it in the buffer."""
        if num_events is None:
            num_events = self.summary_update_interval

        messages = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))

        previous_summary = "N/A"
        summaries = self.prompt_buffer.get("summary")
        if summaries:
            previous_summary = summaries[-1]

        observations = self.prompt_buffer.get("observation")[-num_events:]
        actions = self.prompt_buffer.get("action")[-num_events:]
        event_lines = []
        action_offset = len(actions) - len(observations)
        for i in range(len(observations)):
            action_idx = i + action_offset
            # Plain-text events have no API role constraint: always keep the
            # causal action, including the window's first one (offset == 0).
            if action_idx >= 0:
                event_lines.append(f"Action taken:\n\n{actions[action_idx]}")
            event_lines.append(f"Observation:\n\n{observations[i]}")
        events_text = "\n\n".join(event_lines).strip()

        user_content = (
            f"Summary of Trial {self.trial_index} so far:\n\n"
            f"{previous_summary}\n\n"
            f"Recent Events (since last summary):\n"
            f"{events_text}\n\n"
            f"{self.summary_instructions}"
        )

        messages.append(Message(role="user", content=user_content))
        summary_response = self.client.generate(messages)
        self.prompt_buffer.add("summary", summary_response.completion)
        return summary_response
    
    def _extract_final_answer(self, answer):
        """Return the LLM response as-is."""
        return answer