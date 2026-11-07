import re
import json
import copy
from dataclasses import dataclass

from agents.base import BaseAgent, resolve_agent_config
from agents.prompt_buffer import PromptBuffer, Message

@dataclass
class AgentConfig:
    max_text_history: int = 16
    max_cot_history: int = 1

class CoTAgent(BaseAgent):
    """Agent that elicits chain-of-thought reasoning before acting.
    At each step the agent builds a message sequence from the system prompt
    and the observation/action/reasoning history (kept in a prompt buffer).
    The agent appends CoT instructions to the latest
    observation in the message sequence which asks the LLM to reason and propose an action, then
    extracts the action from <action> tags.
    """

    def __init__(self, client_factory, config, env_system_prompt):
        """Initialize the agent with client factory, config, system prompt, and CoT instruction text."""
        super().__init__(client_factory)

        self.agent_config = resolve_agent_config(config)
        assert self.agent_config.max_cot_history <= self.agent_config.max_text_history, "max_cot_history cannot exceed max_text_history"

        self.system_prompt = self.update_system_prompt(env_system_prompt)

        self.prompt_buffer = PromptBuffer(
            limits={
                "observation": self.agent_config.max_text_history,
                "action": self.agent_config.max_text_history,
                "reasoning": self.agent_config.max_cot_history,
            })

        self.cot_instructions = (
                "You MUST structure your response into two distinct parts:\n"
                "1. First, a <thinking> block where you write out your step-by-step analysis of the best plan of action.\n"
                "2. Second, an <action> block containing the final valid action."
            )

    def act(self, obs, info):
        """Buffer the observation, generate chain-of-thought reasoning, extract and return the action."""
        self.prompt_buffer.add("observation", obs)

        messages = self.get_prompt_from_buffer()

        messages[-1].content += "\n\n" + self.cot_instructions

        cot_reasoning = self.client.generate(messages)
        self.prompt_buffer.add("reasoning", cot_reasoning.completion)

        final_answer = self._extract_final_answer(cot_reasoning)
        self.prompt_buffer.add("action", final_answer.completion)

        return final_answer
    
    def reset(self, terminal_obs=None, terminal_info=None):
        """Clear the prompt buffer to start a new episode."""
        self.prompt_buffer.reset()

    def update_system_prompt(self, system_prompt):
        """Return the system prompt unchanged."""
        return system_prompt

    def get_prompt_from_buffer(self):
        """Build an interleaved message list of observation + action | (action + reasoning)."""
        messages = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))
            
        observations = self.prompt_buffer.get("observation")
        actions = self.prompt_buffer.get("action")
        reasonings = self.prompt_buffer.get("reasoning")

        action_offset = len(actions) - len(observations)        # action offset will either be 0 or -1
        reasoning_offset = len(reasonings) - len(observations)      # reasoning offset could be between max_cot_history-max_text_history to 0 

        for i in range(len(observations)):
            reasoning_idx = i + reasoning_offset
            action_idx = i + action_offset

            # because reasoning contains the action, only fall back to the
            # action when no reasoning is available for this observation
            pending_label = None
            pending_content = None
            if reasoning_idx >= 0:
                pending_label = "Prior agent reasoning (context):"
                pending_content = "Previous plan:\n" + reasonings[reasoning_idx]
            elif action_idx >= 0:
                pending_label = "Prior agent action (context):"
                pending_content = actions[action_idx]

            if pending_content is not None and (messages and messages[-1].role == "user"):
                message = Message(role="assistant", content=pending_content)
                messages.append(message)
                pending_content = None

            message_parts = []
            if pending_content is not None:
                # No preceding user message: an assistant turn here would be
                # an invalid leading assistant, so keep the causal context
                # inside the observation message instead.
                message_parts.append(pending_label)
                message_parts.append(pending_content)
            if i == len(observations) - 1:
                message_parts.append("Current Observation:")
            else:
                message_parts.append("Observation:")
            message_parts.append(observations[i])
            content = "\n\n".join(message_parts)
            message = Message(role="user", content=content)
            messages.append(message)

        return messages
        
    def _extract_final_answer(self, reasoning):
        """Extract the action from <action> tags in the reasoning, falling back to the first JSON object."""
        final_answer = copy.deepcopy(reasoning)

        final_answer = final_answer._replace(auxiliary=reasoning.completion)

        completion_text = reasoning.completion
        match = re.search(r"<action>(.*?)</action>", completion_text, re.DOTALL)
        if match:
            extracted_action = match.group(1).strip()
        else:
            json_match = re.search(r"(\{.*\})", completion_text, re.DOTALL)
            if json_match:
                extracted_action = json_match.group(1).strip()
            else:
                extracted_action = "Failed to obtain a valid action from the reasoning."

        final_answer = final_answer._replace(completion=extracted_action)

        return final_answer