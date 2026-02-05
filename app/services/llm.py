from typing import List, Dict, Any, Union
from openai import OpenAI
from anthropic import Anthropic


def _extract_openai_content(message: Any, content: Any) -> str:
    """Extract text from OpenAI message content (string, list of parts, or object)."""
    if content is None:
        # Some models put text in message.model_dump() or other fields
        try:
            raw = message.model_dump() if hasattr(message, "model_dump") else {}
        except Exception:
            raw = {}
        content = raw.get("content")
        if content is None:
            return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", part.get("content", "")))
            elif hasattr(part, "text"):
                parts.append(getattr(part, "text", "") or "")
            elif hasattr(part, "content"):
                parts.append(getattr(part, "content", "") or "")
            else:
                parts.append(str(part))
        return " ".join(p for p in parts if p).strip()

    if hasattr(content, "text"):
        return (getattr(content, "text") or "").strip()
    if hasattr(content, "content"):
        return (getattr(content, "content") or "").strip()
    return (str(content) or "").strip()


class LLMService:
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: str = None,
    ):

        self.provider = provider.lower()
        self.model = model

        if self.provider == "openai":
            self.client = OpenAI(api_key=api_key)
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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based only on the provided context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=max_tokens,
                temperature=1,
            )
            if not response.choices:
                return ""
            message = response.choices[0].message
            content = message.content
            answer = _extract_openai_content(message, content)
            # Fallback: use refusal if content is empty (some models put reason there)
            if not answer and getattr(message, "refusal", None):
                answer = (message.refusal or "").strip()
            # Fallback: Responses API style
            if not answer and hasattr(response, "output") and response.output:
                try:
                    first_output = response.output[0]
                    if hasattr(first_output, "content") and first_output.content:
                        first_content = first_output.content[0]
                        if hasattr(first_content, "text"):
                            answer = (first_content.text or "").strip()
                except (IndexError, AttributeError, TypeError):
                    pass
            if not answer:
                answer = (
                    "The model did not return a response. Try shortening the context, "
                    "using a different model (e.g. gpt-4o-mini), or check the model name and API."
                )
            return answer

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
