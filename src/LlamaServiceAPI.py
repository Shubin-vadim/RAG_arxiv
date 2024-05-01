from typing import Dict, Any
from llama_index.llms.llama_api import LlamaAPI

class LlamaServiceAPI:
    def __init__(self,
                 model: str ='llama-7b-chat',
                 api_key: str ='',
                 temperature: float = 0.01,
                 max_tokens: int = 2048,
                 kwargs: Dict[str, Any] = None
                 ) -> None:

        self.llm = LlamaAPI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=kwargs
        )

    def get_llm(self):
        return self.llm

    def send_query(self, query: str) -> str:
        return self.llm.complete(query)
