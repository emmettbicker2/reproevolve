from typing import Optional
import litellm
from litellm.llms.anthropic.common_utils import AnthropicError
import random  # For choosing random model


class Model:
    primary_model: str
    secondary_model: str
    secondary_model_usage_rate: float
    system_prompt: str

    def __init__(
        self,
        primary_model: str,
        secondary_model: str,
        secondary_model_usage_rate: float,
        system_prompt: str,
    ):
        self.primary_model = primary_model
        self.secondary_model = secondary_model
        self.secondary_model_usage_rate = secondary_model_usage_rate
        self.system_prompt = system_prompt

    def generate_edit(self, model: str, user_prompt: str) -> Optional[str]:
        # Write the most recent prompt as a txt
        with open("most_recent_prompt.txt", "w") as f:
            f.write(f"{self.system_prompt}\n\n{user_prompt}")

        try:
            completion = litellm.completion(  # type: ignore
                model=model,
                messages=[
                    litellm.Message(content=self.system_prompt, role="system"),
                    litellm.Message(content=user_prompt, role="user"),
                ],
                temperature=0.7,
                top_p=0.95,
            )
        except (
            litellm.exceptions.APIConnectionError,
            litellm.exceptions.InternalServerError,
            AnthropicError,
        ) as e:
            print(
                f"Encountered {e} while generating model response. Skipping this iteration"
            )
            return None

        return completion.choices[0].message.content  # type: ignore

    def choose_model(self) -> str:
        if random.random() < self.secondary_model_usage_rate:
            return self.secondary_model
        return self.primary_model
