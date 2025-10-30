from __future__ import annotations
import os, json, pathlib, time
from typing import Any, Dict, Optional
from openai import OpenAI
from request_logger import log_request_response  # 新增导入

class LLMBase:
    """与 worldview_generator.py 风格对齐的基础类：
    - 使用 .env 中的 OPENAI_* / STRONG_TEXT_MODEL / WEAK_TEXT_MODEL
    - 提供统一的 JSON Schema 约束调用
    - 所有 Agent 继承它，以保持一致性
    """
    def __init__(self, env_path: str):
        from dotenv import load_dotenv
        load_dotenv(env_path)
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
        self.OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL")
        if not self.OPENAI_API_KEY:
            raise RuntimeError("Missing OPENAI_API_KEY (or API_KEY) in .env")
        self.STRONG_TEXT_MODEL = os.getenv("STRONG_TEXT_MODEL", "gpt-5")
        self.WEAK_TEXT_MODEL = os.getenv("WEAK_TEXT_MODEL", "gpt-5-mini")
        self.client = OpenAI(api_key=self.OPENAI_API_KEY,
                             base_url=self.OPENAI_BASE_URL if self.OPENAI_BASE_URL else None)

    def call_structured_json(self, *, model: str, system_prompt: str, user_prompt: str,
                              json_schema: Optional[Dict[str, Any]] = None, temperature: float = 0.7) -> Any:
        request_payload = {
            "component": "LLMBase.call_structured_json",
            "model": model,
            "temperature": temperature,
            "json_schema_name": json_schema.get("name") if json_schema and isinstance(json_schema, dict) else None,
            "system_prompt": system_prompt if system_prompt else None,
            "user_prompt": user_prompt if user_prompt else None,
        }

        try:
            if json_schema:
                # 如果提供了 JSON Schema，则使用结构化输出
                resp = self.client.responses.create(
                    model=model,
                    temperature=temperature,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": json_schema.get("name", "structured_output"),
                            "strict": False,
                            "schema": json_schema["schema"],
                        }
                    },
                )
                output = json.loads(resp.output_text)
            else:
                # 如果未提供 JSON Schema，则以常规文本形式输出
                resp = self.client.responses.create(
                    model=model,
                    temperature=temperature,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                output = resp.output_text
        except Exception as e:
            # 记录异常响应（简洁记录），不改变原始异常抛出行为
            try:
                log_request_response(request_payload, {"error": str(e)})
            except Exception:
                pass
            raise

        # 记录成功的请求/返回（简洁记录关键字段）
        response_payload = {
            "output_text": getattr(resp, "output_text", None),
            "raw": str(resp),
        }
        try:
            log_request_response(request_payload, response_payload)
        except Exception:
            pass

        return output
