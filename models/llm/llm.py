import logging
from collections.abc import Generator
from typing import Optional, Union
from decimal import Decimal

from dify_plugin import LargeLanguageModel
from dify_plugin.entities import I18nObject
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeError
)
from dify_plugin.entities.model import (
    AIModelEntity,
    FetchFrom,
    ModelType,
)
from dify_plugin.entities.model.llm import (
    LLMResult,
    LLMUsage,
    LLMResultChunk,
    LLMResultChunkDelta,
)
from dify_plugin.entities.model.message import (
    PromptMessage,
    PromptMessageTool,
    AssistantPromptMessage,
)

from dify_plugin.errors.model import (
    InvokeServerUnavailableError,
    InvokeConnectionError,
    InvokeRateLimitError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
)

import requests
import json
from typing import Dict, Any
        

logger = logging.getLogger(__name__)


class InfiniaiLargeLanguageModel(LargeLanguageModel):
    """
    Model class for infiniai large language model.
    """

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator[LLMResultChunk, None, None]]:
        """
        Invoke large language model

        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param model_parameters: model parameters
        :param tools: tools for tool calling
        :param stop: stop words
        :param stream: is stream response
        :param user: unique user id
        :return: full response or stream response chunk generator result
        """
        base_url = "https://cloud.infini-ai.com/maas/v1"
        headers = {
            "Authorization": f"Bearer {credentials['infiniai_api_key']}",
            "Content-Type": "application/json",
        }
        
        # 构建请求数据
        messages = []
        for message in prompt_messages:
            messages.append({
                "role": message.role.value,
                "content": message.content
            })
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **model_parameters
        }
        
        if stop:
            payload["stop"] = stop
        
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                def response_generator() -> Generator[LLMResultChunk, None, None]:
                    chunk_index = 0
                    for line in response.iter_lines():
                        if line:
                            if line.strip() == b"data: [DONE]":
                                break
                            if line.startswith(b"data: "):
                                data = json.loads(line.decode("utf-8").replace("data: ", ""))
                                if data["choices"][0]["finish_reason"] is not None:
                                    break
                                
                                # 创建 AssistantPromptMessage
                                message = AssistantPromptMessage(
                                    content=data["choices"][0]["delta"].get("content", "")
                                )
                                
                                # 创建 LLMUsage
                                usage = LLMUsage.empty_usage()
                                
                                # 创建 LLMResultChunkDelta
                                delta = LLMResultChunkDelta(
                                    index=chunk_index,
                                    message=message,
                                    usage=usage,
                                    finish_reason=data["choices"][0].get("finish_reason")
                                )
                                
                                # 创建并返回 LLMResultChunk
                                yield LLMResultChunk(
                                    model=model,
                                    prompt_messages=prompt_messages,
                                    system_fingerprint=None,
                                    delta=delta
                                )
                                
                                chunk_index += 1
                                
                return response_generator()
            else:
                response_json = response.json()
                
                # 创建 AssistantPromptMessage
                message = AssistantPromptMessage(
                    content=response_json["choices"][0]["message"]["content"]
                )
                
                # 从响应中获取使用情况
                usage_data = response_json.get("usage", {})
                usage = LLMUsage(
                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                    prompt_unit_price=Decimal("0.0"),
                    prompt_price_unit=Decimal("0.0"),
                    prompt_price=Decimal("0.0"),
                    completion_tokens=usage_data.get("completion_tokens", 0),
                    completion_unit_price=Decimal("0.0"),
                    completion_price_unit=Decimal("0.0"),
                    completion_price=Decimal("0.0"),
                    total_tokens=usage_data.get("total_tokens", 0),
                    total_price=Decimal("0.0"),
                    currency="RMB",
                    latency=0.0
                )
                
                return LLMResult(
                    model=model,
                    prompt_messages=prompt_messages,
                    message=message,
                    usage=usage,
                    system_fingerprint=response_json.get("system_fingerprint")
                )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise InvokeError(f"API request failed: {str(e)}")

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        """
        Get number of tokens for given prompt messages

        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param tools: tools for tool calling
        :return:
        """
        return 0

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials

        :param model: model name
        :param credentials: model credentials
        :return:
        """
        try:
            pass
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    def get_customizable_model_schema(
        self, model: str, credentials: dict
    ) -> AIModelEntity:
        """
        If your model supports fine-tuning, this method returns the schema of the base model
        but renamed to the fine-tuned model name.

        :param model: model name
        :param credentials: credentials

        :return: model schema
        """
        entity = AIModelEntity(
            model=model,
            label=I18nObject(zh_Hans=model, en_US=model),
            model_type=ModelType.LLM,
            features=[],
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={},
            parameter_rules=[],
        )

        return entity

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        """
        Map model invoke error to unified error
        The key is the error type thrown to the caller
        The value is the error type thrown by the model,
        which needs to be converted into a unified error type for the caller.

        :return: Invoke error mapping
        """

        return {
            InvokeConnectionError: [
            InvokeConnectionError
            ],
            InvokeServerUnavailableError: [
            InvokeServerUnavailableError
            ],
            InvokeRateLimitError: [
            InvokeRateLimitError
            ],
            InvokeAuthorizationError: [
            InvokeAuthorizationError
            ],
            InvokeBadRequestError: [
            InvokeBadRequestError
            ],
        }
