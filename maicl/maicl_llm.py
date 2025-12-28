"""
LLM-related classes for MA-ICL system.
Contains GoogleAPIKeyManager and BatchedLLM classes.
"""

import os
import time
import threading
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from langchain.schema import HumanMessage
except Exception:
    from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    # Optional dependency (only needed for --llm_provider openai)
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

from .maicl_config import MAX_BATCH_SIZE, BATCH_TIMEOUT, MAX_PROMPT_PREVIEW_LENGTH

logger = logging.getLogger(__name__)


# =========================
# API KEY MANAGEMENT
# =========================

class GoogleAPIKeyManager:
    """Thread-safe Google API key manager with rotation and quota handling"""
    
    def __init__(self, api_keys: List[str], model_name: str = "gemini-2.0-flash", temperature: float = 0.8):
        if not model_name.lower().startswith('gemini'):
            raise ValueError(f"GoogleAPIKeyManager is only for Gemini models, not {model_name}")
        
        self.api_keys = api_keys
        self.model_name = model_name
        self.temperature = temperature
        self.current_key_index = 0
        self.exhausted_keys = set()
        self.llm_cache = {}
        self.key_lock = threading.Lock()
        self.cache_lock = threading.Lock()
        
        if not self.api_keys or self.api_keys == [None]:
            raise ValueError("No valid Google API keys provided.")
        
        logger.info(f"✓ Initialized Gemini API Key Manager with {len(self.api_keys)} key(s)")
    
    def get_current_llm(self, temperature: float = None) -> ChatGoogleGenerativeAI:
        """Get current LLM instance (thread-safe)
        
        Args:
            temperature: Optional temperature override. If provided and different from
                        the default, creates a new LLM instance (not cached).
        """
        with self.key_lock:
            current_key = self.api_keys[self.current_key_index]
        
        # If temperature is specified and different, create a new instance (not cached)
        if temperature is not None and temperature != self.temperature:
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=temperature,
                google_api_key=current_key
            )
        
        with self.cache_lock:
            if current_key not in self.llm_cache:
                self.llm_cache[current_key] = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    temperature=self.temperature,
                    google_api_key=current_key
                )
            return self.llm_cache[current_key]
    
    def switch_to_next_key(self) -> bool:
        """Thread-safe key switching"""
        with self.key_lock:
            if len(self.exhausted_keys) >= len(self.api_keys):
                logger.warning("⚠️ All API keys exhausted!")
                return False
            
            self.exhausted_keys.add(self.current_key_index)
            logger.warning(f"⚠️ API key {self.current_key_index + 1} exhausted")
            
            for i in range(len(self.api_keys)):
                next_index = (self.current_key_index + i + 1) % len(self.api_keys)
                if next_index not in self.exhausted_keys:
                    self.current_key_index = next_index
                    logger.info(f"✓ Switched to API key {next_index + 1}/{len(self.api_keys)}")
                    return True
        
        return False
    
    def is_quota_error(self, error: Exception) -> bool:
        """Check if error is a quota/resource exhaustion error"""
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        quota_indicators = [
            "resourceexhausted", "429", "quota exceeded", "quota limit",
            "exceeded your current quota", "rate limit"
        ]
        
        if error_type == "ResourceExhausted":
            return True
        
        return any(indicator in error_str for indicator in quota_indicators)


# =========================
# OPENAI API KEY MANAGEMENT
# =========================

class OpenAIAPIKeyManager:
    """Thread-safe OpenAI API key manager with rotation and basic rate-limit handling."""

    # NOTE: keep this default conservative and widely available.
    # Users can override via CLI/config (e.g., --model_name or MAICL_MODEL_NAME).
    def __init__(self, api_keys: List[str], model_name: str = "gpt-4o-mini", temperature: float = 0.8):
        # We allow any OpenAI model name, but explicitly disallow Gemini names here to avoid confusion.
        if model_name.lower().startswith("gemini"):
            raise ValueError(f"OpenAIAPIKeyManager is for OpenAI models, not {model_name}")

        if ChatOpenAI is None:
            raise RuntimeError(
                "OpenAI provider requested but langchain_openai is not installed.\n"
                "Install with: pip install langchain-openai openai"
            )

        self.api_keys = api_keys
        self.model_name = model_name
        self.temperature = temperature
        self.current_key_index = 0
        self.exhausted_keys = set()
        self.llm_cache = {}
        self.key_lock = threading.Lock()
        self.cache_lock = threading.Lock()

        if not self.api_keys or self.api_keys == [None]:
            raise ValueError("No valid OpenAI API keys provided.")

        logger.info(f"✓ Initialized OpenAI API Key Manager with {len(self.api_keys)} key(s)")

    def get_current_llm(self, temperature: float = None):
        """Get current OpenAI LLM instance (thread-safe).
        
        Args:
            temperature: Optional temperature override. If provided and different from
                        the default, creates a new LLM instance (not cached).
        """
        with self.key_lock:
            current_key = self.api_keys[self.current_key_index]

        # If temperature is specified and different, create a new instance (not cached)
        if temperature is not None and temperature != self.temperature:
            return ChatOpenAI(
                model=self.model_name,
                temperature=temperature,
                api_key=current_key,
            )

        with self.cache_lock:
            if current_key not in self.llm_cache:
                # Keep temperature aligned with Gemini usage for now.
                self.llm_cache[current_key] = ChatOpenAI(
                    model=self.model_name,
                    temperature=self.temperature,
                    api_key=current_key,
                )
            return self.llm_cache[current_key]

    def switch_to_next_key(self) -> bool:
        """Thread-safe key switching."""
        with self.key_lock:
            if len(self.exhausted_keys) >= len(self.api_keys):
                logger.warning("⚠️ All OpenAI API keys exhausted!")
                return False

            self.exhausted_keys.add(self.current_key_index)
            logger.warning(f"⚠️ OpenAI API key {self.current_key_index + 1} exhausted")

            for i in range(len(self.api_keys)):
                next_index = (self.current_key_index + i + 1) % len(self.api_keys)
                if next_index not in self.exhausted_keys:
                    self.current_key_index = next_index
                    logger.info(f"✓ Switched to OpenAI API key {next_index + 1}/{len(self.api_keys)}")
                    return True

        return False

    def is_quota_error(self, error: Exception) -> bool:
        """Check if error is a rate-limit/quota error."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        quota_indicators = [
            "rate limit",
            "ratelimit",
            "429",
            "insufficient_quota",
            "quota",
            "exceeded your current quota",
            "too many requests",
        ]

        # langchain/openai exceptions vary; string matching is the most robust here.
        if "ratelimit" in error_type or "rate_limit" in error_type:
            return True

        return any(indicator in error_str for indicator in quota_indicators)


# =========================
# BATCHED LLM
# =========================

class BatchedLLM:
    """Batched LLM wrapper with quota management and retry logic"""
    def __init__(self, llm_or_key_manager, print_samples: bool = None, max_samples_to_print: int = 2):
        if isinstance(llm_or_key_manager, (GoogleAPIKeyManager, OpenAIAPIKeyManager)):
            self.key_manager = llm_or_key_manager
            self.llm = self.key_manager.get_current_llm()
            self.use_key_manager = True
        else:
            self.llm = llm_or_key_manager
            self.key_manager = None
            self.use_key_manager = False
        self.call_count = 0
        self.batch_count = 0
        if print_samples is None:
            self.print_samples = os.getenv("MAICL_PRINT_LLM_SAMPLES", "0").lower() in ("1", "true", "yes")
        else:
            self.print_samples = print_samples
        self.max_samples_to_print = max_samples_to_print
        self._samples_printed_count = 0
    
    def invoke_single(self, message: HumanMessage, max_retries: int = 3, skip_print: bool = False) -> str:
        """Invoke single LLM call with retry logic"""
        self.call_count += 1
        
        # Validate message content is not empty or None
        if not message.content or not message.content.strip():
            error_msg = f"Empty or None content provided to LLM. Message content: {repr(message.content)}"
            logger.error(error_msg)
            return ""
        
        if not skip_print and self.print_samples and self._samples_printed_count < self.max_samples_to_print:
            self._samples_printed_count += 1
            prompt_preview = message.content[:MAX_PROMPT_PREVIEW_LENGTH] if len(message.content) > MAX_PROMPT_PREVIEW_LENGTH else message.content
            print(f"\n[LLM Prompt #{self._samples_printed_count}]:\n{prompt_preview}")
            if len(message.content) > 1000:
                print(f"[... truncated, {len(message.content)} chars total]\n")
        
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke([message])
                return response.content.strip()
            except Exception as e:
                if self.use_key_manager and self.key_manager.is_quota_error(e):
                    if self.key_manager.switch_to_next_key():
                        self.llm = self.key_manager.get_current_llm()
                        continue
                if attempt == max_retries - 1:
                    logger.error(f"LLM call failed after {max_retries} attempts: {e}")
                    return ""
                time.sleep(2 ** attempt)
        return ""
    
    def invoke_batch(self, messages: List[HumanMessage], max_workers: int = MAX_BATCH_SIZE) -> List[str]:
        """Batch invoke LLM calls with parallel execution"""
        if len(messages) == 0:
            return []
        if len(messages) == 1:
            return [self.invoke_single(messages[0])]
        
        if self.print_samples and self._samples_printed_count < self.max_samples_to_print:
            samples_to_print = min(self.max_samples_to_print - self._samples_printed_count, len(messages))
            for i in range(samples_to_print):
                self._samples_printed_count += 1
                prompt_preview = messages[i].content[:MAX_PROMPT_PREVIEW_LENGTH] if len(messages[i].content) > MAX_PROMPT_PREVIEW_LENGTH else messages[i].content
                print(f"\n[LLM Prompt #{self._samples_printed_count} (batch {i+1}/{len(messages)})]:\n{prompt_preview}")
                if len(messages[i].content) > 1000:
                    print(f"[... truncated, {len(messages[i].content)} chars total]\n")
        
        self.batch_count += 1
        results = [None] * len(messages)
        
        def call_api(idx_msg_tuple):
            idx, msg = idx_msg_tuple
            try:
                result = self.invoke_single(msg, skip_print=True)
                return idx, result
            except Exception as e:
                logger.error(f"Batch call {idx} failed: {e}")
                return idx, ""
        
        with ThreadPoolExecutor(max_workers=min(max_workers, len(messages))) as executor:
            future_to_idx = {executor.submit(call_api, (i, msg)): i for i, msg in enumerate(messages)}
            for future in as_completed(future_to_idx, timeout=BATCH_TIMEOUT):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    idx = future_to_idx[future]
                    results[idx] = ""
                    logger.error(f"Batch future {idx} failed: {e}")
        
        return results
    
    def get_stats(self) -> dict:
        """Get API call statistics"""
        return {
            "total_calls": self.call_count,
            "batch_operations": self.batch_count,
            "avg_batch_size": self.call_count / max(1, self.batch_count)
        }

