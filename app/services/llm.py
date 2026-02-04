from typing import List, Dict
import openai
from anthropic import Anthropic
from typer import prompt


class LLMService:
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-3.5-turbo",
        api_key: str = None,
    ):

        self.provider = provider.lower()
        self.model = model

        if provider == "openai":
            openai.api_key = api_key
        elif self.provider == "anthropic":
            self.client = Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unsupported LLM provider {provider}")

    def generate_answer(
        self, question: str, context_chunks: List[str], max_tokens: int = 500
    ) -> str:
        context = "\n\n".join(
            [
                f"Context {i+1}:{chunk}"
                for i, chunk in enumerate(context_chunks)
            ]
        )

        prompt = f"""Answer the question based only on the following context. 
        If the answer is not in the context, say 
        "I don't have enough information to answer this question."

        Context:{context}

        Question: {question}
        Answer:
        """

        if self.provider == "openai":
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based only on the provided context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()

        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        else:
            raise ValueError("Unsupported LLM provider")
