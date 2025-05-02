
import os
from typing import List, Dict, Callable, Any, Union, Optional
import aiohttp
import numpy as np

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    Timeout,
    AsyncAzureOpenAI,
)
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from transformers import AutoTokenizer, AutoModelForCausalLM

from .utils import (
    wrap_embedding_func_with_attrs,
    locate_json_string_body_from_string,
    safe_unicode_decode,
    logger,
)

import sys

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def azure_openai_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    base_url=None,
    api_key=None,
    api_version=None,
    **kwargs,
):
    if api_key:
        os.environ["AZURE_OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["AZURE_OPENAI_ENDPOINT"] = base_url
    if api_version:
        os.environ["AZURE_OPENAI_API_VERSION"] = api_version

    openai_async_client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    if prompt is not None:
        messages.append({"role": "user", "content": prompt})

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )
    content = response.choices[0].message.content

    return content

class GPTKeywordExtractionFormat(BaseModel):
    high_level_keywords: List[str]
    low_level_keywords: List[str]

async def azure_openai_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    result = await azure_openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )
    if keyword_extraction:  # TODO: use JSON API
        return locate_json_string_body_from_string(result)
    return result

async def fetch_data(url, headers, data):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            response_json = await response.json()
            data_list = response_json.get("data", [])
            return data_list

@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8191)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def azure_openai_embedding(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str = None,
    api_key: str = None,
    api_version: str = None,
) -> np.ndarray:
    if api_key:
        os.environ["AZURE_OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["AZURE_OPENAI_ENDPOINT"] = base_url
    if api_version:
        os.environ["AZURE_OPENAI_API_VERSION"] = api_version

    openai_async_client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )

    response = await openai_async_client.embeddings.create(
        model=model, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def azure_openai_complete_if_cache_stream(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    base_url=None,
    api_key=None,
    api_version=None,
    **kwargs,
):
    # Set environment variables if provided
    if api_key:
        os.environ["AZURE_OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["AZURE_OPENAI_ENDPOINT"] = base_url
    if api_version:
        os.environ["AZURE_OPENAI_API_VERSION"] = api_version

    openai_async_client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    if prompt is not None:
        messages.append({"role": "user", "content": prompt})

    # Enable streaming mode
    kwargs["stream"] = True

    # Iterate over streamed responses
    async for response in openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    ):
        # Each streamed response chunk may include a delta with content
        delta = response.choices[0].delta
        if "content" in delta:
            yield delta["content"]


async def azure_openai_complete_stream(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
):
    """
    A streamable version of your azure_openai_complete function.
    Note: Keyword extraction while streaming may require accumulating the complete result.
    """
    # This function currently streams tokens. If keyword_extraction is enabled,
    # you might need to collect the full result and then process it.
    async for token in azure_openai_complete_if_cache_stream(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    ):
        yield token


class Model(BaseModel):
    """
    This is a Pydantic model class named 'Model' that is used to define a custom language model.

    Attributes:
        gen_func (Callable[[Any], str]): A callable function that generates the response from the language model.
            The function should take any argument and return a string.
        kwargs (Dict[str, Any]): A dictionary that contains the arguments to pass to the callable function.
            This could include parameters such as the model name, API key, etc.

    Example usage:
        Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_1"]})

    In this example, 'openai_complete_if_cache' is the callable function that generates the response from the OpenAI model.
    The 'kwargs' dictionary contains the model name and API key to be passed to the function.
    """

    gen_func: Callable[[Any], str] = Field(
        ...,
        description="A function that generates the response from the llm. The response must be a string",
    )
    kwargs: Dict[str, Any] = Field(
        ...,
        description="The arguments to pass to the callable function. Eg. the api key, model name, etc",
    )

    class Config:
        arbitrary_types_allowed = True


class MultiModel:
    """
    Distributes the load across multiple language models. Useful for circumventing low rate limits with certain api providers especially if you are on the free tier.
    Could also be used for spliting across diffrent models or providers.

    Attributes:
        models (List[Model]): A list of language models to be used.

    Usage example:
        ```python
        models = [
            Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_1"]}),
            Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_2"]}),
            Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_3"]}),
            Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_4"]}),
            Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_5"]}),
        ]
        multi_model = MultiModel(models)
        rag = LightRAG(
            llm_model_func=multi_model.llm_model_func
            / ..other args
            )
        ```
    """

    def __init__(self, models: List[Model]):
        self._models = models
        self._current_model = 0

    def _next_model(self):
        self._current_model = (self._current_model + 1) % len(self._models)
        return self._models[self._current_model]

    async def llm_model_func(
        self, prompt, system_prompt=None, history_messages=[], **kwargs
    ) -> str:
        kwargs.pop("model", None)   
        kwargs.pop("keyword_extraction", None)
        kwargs.pop("mode", None)
        next_model = self._next_model()
        args = dict(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
            **next_model.kwargs,
        )

        return await next_model.gen_func(**args)


