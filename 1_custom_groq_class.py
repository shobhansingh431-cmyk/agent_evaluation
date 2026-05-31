import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from deepeval.models.base_model import DeepEvalBaseLLM


load_dotenv()

## Custom Class created for Groq model
class GroqModel(DeepEvalBaseLLM):

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("GROQ_EVAL_MODEL", "llama-3.1-8b-instant")
        self.model = ChatGroq(
            model=self.model_name,
            temperature=0
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        res = await self.model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return self.model_name
    

groq_judge = GroqModel()
# print( groq_judge.generate("Write me a joke") )
