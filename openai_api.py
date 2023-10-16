from typing import List
import time
import os
from tqdm import tqdm
import concurrent.futures
import openai
from openai.error import RateLimitError, Timeout, APIError, ServiceUnavailableError, APIConnectionError

openai.api_key = os.getenv("OPENAI_API_KEY")

delay_time = 0.5
decay_rate = 0.8

def get_batched_responses(prompts: List[str], model: str, max_tokens: int, batch_size: int, temperature: int = 0,
                          system_message: str = None, histories: List[str] = None, use_parallel: bool = False, show_progress: bool = False):
    responses = []
    it = range(0, len(prompts), batch_size)
    if show_progress:
        it = tqdm(it)
    for batch_start_idx in it:
        batch = prompts[batch_start_idx : batch_start_idx + batch_size]
        histories_batch = histories[batch_start_idx : batch_start_idx + batch_size] if histories else None
        if use_parallel:
            responses += get_parallel_responses(batch, model, max_tokens, temperature=temperature,
                                                system_message=system_message, histories=histories_batch)
        else:
            responses += get_responses(batch, model, max_tokens, temperature=temperature)
    return responses

def get_parallel_responses(prompts: List[str], model: str, max_tokens: int, temperature: int = 0,
                           system_message: str = None, histories: List[dict] = None):
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        # Submit requests to threads
        futures = [
            executor.submit(get_responses, [prompt], model, max_tokens, temperature=temperature,
                            system_message=system_message, histories=[histories[prompt_idx]] if histories else None)
            for prompt_idx, prompt in enumerate(prompts)
        ]

        # Wait for all to complete
        concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

        # Accumulate results
        results = [future.result()[0] for future in futures]
        return results

def get_responses(prompts: List[str], model="code-davinci-002", max_tokens=400, temperature=0,
                  system_message=None, histories=None, logprobs=None, echo=False):
    global delay_time, cur_key_idx
    # import pdb; pdb.set_trace()

    # Wait for rate limit
    time.sleep(delay_time)

    # Send request
    try:
        if "gpt-3.5-turbo" in model or "gpt-4" in model:
            results = []
            for prompt_idx, prompt in enumerate(prompts):
                history = histories[prompt_idx] if histories else []
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_message or "You are a helpful assistant."
                        },
                        *history,
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    request_timeout=45
                )
                results.append(response["choices"][0])
        else:
            response = openai.Completion.create(
                model=model,
                prompt=prompts,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=["\n\n"],
                logprobs=logprobs,
                echo=echo
            )
            results = response["choices"]
        delay_time = max(delay_time * decay_rate, 0.1)
    except (RateLimitError, Timeout, APIError, ServiceUnavailableError, APIConnectionError) as exc:
        print(openai.api_key, exc)
        delay_time = min(delay_time * 2, 30)
        return get_responses(prompts, model, max_tokens, temperature=temperature, system_message=system_message,
                             histories=histories, logprobs=logprobs, echo=echo)
    except Exception as exc:
        print(exc)
        raise exc

    return results
