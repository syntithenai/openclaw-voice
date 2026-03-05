"""Quick answer LLM client for fast factual responses."""
import logging
import httpx
from datetime import datetime
from typing import Optional


logger = logging.getLogger("orchestrator.gateway.quick_answer")


QUICK_ANSWER_SYSTEM_PROMPT = """You are a strict validation gatekeeper. Your sole objective is to provide immediate answers only when they are factual, indisputable, and concise.

Strict Response Protocol:
- Verification Requirement: Before answering, mentally verify the fact against your training or tools. If the information is subject to change, opinion-based, or requires nuance, you must fail the check.
- The "Uncertainty" Trigger: If there is even a 1% margin of doubt, or if the query involves complex reasoning, reply exactly with: USE_UPSTREAM_AGENT.
- Constraint: Answers must be exactly one to two sentences. No conversational filler, no "I believe," and no "As of my last update."
- Binary Outcome: Your output is either a short, definitive fact or the escalation code. Any middle ground is a failure of your instructions.

Current date and time: {current_datetime}"""


class QuickAnswerClient:
    """Client for getting quick factual answers from an LLM before escalating to the gateway."""
    
    def __init__(
        self,
        llm_url: str,
        api_key: Optional[str] = None,
        timeout_ms: int = 5000,
    ):
        """
        Initialize the quick answer client.
        
        Args:
            llm_url: OpenAI-compatible chat completions endpoint
            api_key: Optional API key for authentication
            timeout_ms: Request timeout in milliseconds
        """
        self.llm_url = llm_url
        self.api_key = api_key
        self.timeout_s = timeout_ms / 1000.0
        
    async def get_quick_answer(self, user_query: str) -> tuple[bool, str]:
        """
        Try to get a quick answer from the LLM.
        
        Args:
            user_query: The user's transcript/question
            
        Returns:
            Tuple of (should_use_upstream, response_text)
            - If should_use_upstream is True, response_text will be empty and gateway should be used
            - If should_use_upstream is False, response_text contains the quick answer
        """
        try:
            current_datetime = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
            system_prompt = QUICK_ANSWER_SYSTEM_PROMPT.format(current_datetime=current_datetime)
            
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            payload = {
                "model": "gpt-3.5-turbo",  # Model name (may be ignored by some endpoints)
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                "temperature": 0.0,  # Deterministic for factual answers
                "max_tokens": 100,  # Keep responses brief
            }
            
            logger.info("→ QUICK ANSWER: Querying LLM for: '%s'", user_query)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.llm_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_s,
                )
                
            if response.status_code != 200:
                logger.warning(
                    "Quick answer LLM returned status %d: %s",
                    response.status_code,
                    response.text[:200]
                )
                return True, ""  # Fall back to upstream
                
            response_data = response.json()
            
            # Extract the assistant's message
            if "choices" not in response_data or len(response_data["choices"]) == 0:
                logger.warning("Quick answer LLM response missing 'choices' field")
                return True, ""
                
            message = response_data["choices"][0].get("message", {})
            content = message.get("content", "").strip()
            
            if not content:
                logger.warning("Quick answer LLM returned empty content")
                return True, ""
            
            # Check if LLM wants to escalate to upstream
            if content == "USE_UPSTREAM_AGENT" or content.startswith("USE_UPSTREAM_AGENT"):
                logger.info("← QUICK ANSWER: LLM escalated to upstream agent")
                return True, ""
            
            logger.info("← QUICK ANSWER: Got response (%d chars): %s", len(content), content[:100])
            return False, content
            
        except httpx.TimeoutException:
            logger.warning("Quick answer LLM request timed out after %.1fs", self.timeout_s)
            return True, ""  # Fall back to upstream
        except Exception as exc:
            logger.error("Quick answer LLM failed: %s", exc)
            return True, ""  # Fall back to upstream
