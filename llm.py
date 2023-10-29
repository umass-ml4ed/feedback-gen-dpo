import json
import os
from typing import Dict, List

from openai_api import get_batched_responses

def get_saved_cache(cache_filename: str):
    if os.path.exists(cache_filename):
        with open(cache_filename, encoding="utf-8") as cache_file:
            return json.load(cache_file)
    return {}

class LLM:
    _cache: Dict[str, str] = {}
    _cache_filename = ""
    _model_name = ""
    _mgt = 0
    _batch_size = 0

    @classmethod
    def load(cls, args):
        cls._model_name = args.model
        cls._batch_size = args.batch_size
        cls._mgt = args.max_gen_tokens
        cls._cache_filename = f"llm_cache_{cls._model_name}_mgt{cls._mgt}.json"
        cls._cache = get_saved_cache(cls._cache_filename)

    @classmethod
    def save_cached(cls):
        # Get updates from other processes and then save whole thing
        temp_cache = get_saved_cache(cls._cache_filename)
        temp_cache.update(cls._cache)
        print(f"Saving cache ({len(temp_cache)} entries)...")
        with open(cls._cache_filename, "w", encoding="utf-8") as cache_file:
            json.dump(temp_cache, cache_file, indent=2, ensure_ascii=False)

    @classmethod
    def generate(cls, prompts: List[str], temperature: int = 0, system_message: str = None, histories: List[dict] = None, show_progress: bool = False):
        use_chat_api = "gpt-3.5-turbo" in cls._model_name or "gpt-4" in cls._model_name
        use_cache = temperature == 0 and not histories
        uncached_prompts = [prompt for prompt in prompts if prompt not in cls._cache] if use_cache else prompts
        if uncached_prompts:
            results = get_batched_responses(
                uncached_prompts, cls._model_name, cls._mgt, cls._batch_size, temperature=temperature,
                system_message=system_message, histories=histories, use_parallel=use_chat_api, show_progress=show_progress)
            assert len(uncached_prompts) == len(results)
            result_texts: List[str] = []
            for prompt, result in zip(uncached_prompts, results):
                result_text = result["message"]["content"] if use_chat_api else result["text"]
                if use_cache:
                    cls._cache[prompt] = result_text
                else:
                    result_texts.append(result_text)
        if use_cache:
            return [cls._cache[prompt] for prompt in prompts]
        return result_texts

class LLMCM:
    def __init__(self, args):
        self.args = args

    def __enter__(self):
        LLM.load(self.args)

    def __exit__(self, exc_type, exc_value, traceback):
        LLM.save_cached()
