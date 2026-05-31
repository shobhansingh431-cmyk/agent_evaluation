import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from deepeval.models.base_model import DeepEvalBaseLLM


load_dotenv()

## Custom Class created for Groq model
class GroqModel(DeepEvalBaseLLM):

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("GROQ_EVAL_MODEL", "llama-3.1-8b-instant")
        self.usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }
        self.model = ChatGroq(
            model=self.model_name,
            temperature=0
        )

    def reset_usage(self):
        self.usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }

    def usage_snapshot(self) -> dict:
        return dict(self.usage)

    def _record_usage(self, message):
        usage = getattr(message, "usage_metadata", None) or {}
        response_metadata = getattr(message, "response_metadata", None) or {}
        token_usage = response_metadata.get("token_usage", {}) if isinstance(response_metadata, dict) else {}

        input_tokens = usage.get("input_tokens", token_usage.get("prompt_tokens", 0)) or 0
        output_tokens = usage.get("output_tokens", token_usage.get("completion_tokens", 0)) or 0
        total_tokens = usage.get("total_tokens", token_usage.get("total_tokens", input_tokens + output_tokens)) or 0

        self.usage["input_tokens"] += int(input_tokens)
        self.usage["output_tokens"] += int(output_tokens)
        self.usage["total_tokens"] += int(total_tokens)
        self.usage["calls"] += 1

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        message = self.model.invoke(prompt)
        self._record_usage(message)
        return message.content

    async def a_generate(self, prompt: str) -> str:
        message = await self.model.ainvoke(prompt)
        self._record_usage(message)
        return message.content

    def get_model_name(self):
        return self.model_name
    

groq_judge = GroqModel()
# print( groq_judge.generate("Write me a joke") )
