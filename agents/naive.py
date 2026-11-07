from dataclasses import dataclass

from agents.base import BaseAgent, resolve_agent_config
from agents.prompt_buffer import PromptBuffer, Message

@dataclass
class AgentConfig:
    max_text_history: int = 16

class NaiveAgent(BaseAgent):
    """Baseline agent that directly prompts the LLM for an action.

    At each step the agent builds a message sequence from the system prompt
    and the observation/action history (kept in a prompt buffer),
    appends a short instruction reminding the model to output valid JSON,
    and returns the raw LLM response.
    """
    
    def __init__(self, client_factory, config, env_system_prompt):
        """Initialize the agent with a client factory, config, and environment system prompt."""
        super().__init__(client_factory)

        self.agent_config = resolve_agent_config(config)

        self.system_prompt = self.update_system_prompt(env_system_prompt)

        self.prompt_buffer = PromptBuffer(
            limits={
                "observation": self.agent_config.max_text_history,
                "action": self.agent_config.max_text_history
            })

    def act(self, obs, info):
        """Buffer the observation, build the prompt, call the LLM, and return the action."""
        self.prompt_buffer.add("observation", obs)

        messages = self.get_prompt_from_buffer()

        naive_instruction = (
            "Always output actions in the JSON format provided in the Action schema."
        )

        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + naive_instruction

        response = self.client.generate(messages)

        final_answer = self._extract_final_answer(response)
        self.prompt_buffer.add("action", final_answer.completion)            

        return final_answer
        
    def reset(self, terminal_obs=None, terminal_info=None):
        """Clear the prompt buffer to start a new episode."""
        self.prompt_buffer.reset()

    def update_system_prompt(self, system_prompt):
        """Return the system prompt unchanged; subclasses may override to modify it."""
        return system_prompt

    def get_prompt_from_buffer(self):
        """Build an interleaved system/user/assistant message list from buffered observations and actions."""
        messages = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))
            
        observations = self.prompt_buffer.get("observation")
        actions = self.prompt_buffer.get("action")

        # action offset will either be 0 or -1
        action_offset = len(actions) - len(observations)

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
    
    def _extract_final_answer(self, answer):
        """Return the LLM response as-is; subclasses may parse or transform it."""
        return answer