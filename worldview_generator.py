# -*- coding: utf-8 -*-
"""
Meta（用户元设定）单独存储；LLM 基于 Meta 自由生成 Expansion（开放 Facets，无预置领域键）；
再由校验模型给出一次修订；最终组装 Final（Meta + Expansion）。
- 使用 .env 中 STRONG_TEXT_MODEL 与 WEAK_TEXT_MODEL
- 严禁人物/剧情抓手；仅世界背景
- JSON Schema 仅约束“结构与约束强度”，不预置具体领域字段
"""

import os
import json
import time
import random
import hashlib
import pathlib
from typing import Any, Dict
from openai import OpenAI

class WorldviewGenerator:
    def __init__(self, example_meta: Dict[str, Any], env_path: str, seed: int = None):
        from dotenv import load_dotenv
        load_dotenv(env_path)

        self.example_meta = example_meta
        self.seed = seed
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
        self.OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL")
        self.STRONG_TEXT_MODEL = os.getenv("STRONG_TEXT_MODEL", "gpt-5")
        self.WEAK_TEXT_MODEL = os.getenv("WEAK_TEXT_MODEL", "gpt-5-mini")
        self.META_STORE = pathlib.Path(os.getenv("META_STORE", "./meta_store"))
        self.META_STORE.mkdir(parents=True, exist_ok=True)

        if not self.OPENAI_API_KEY:
            raise RuntimeError("Missing OPENAI_API_KEY (or API_KEY) in .env")

        self.client = OpenAI(
            api_key=self.OPENAI_API_KEY,
            base_url=self.OPENAI_BASE_URL if self.OPENAI_BASE_URL else None,
        )

    # =======================
    # JSON Schemas
    # =======================
    META_SCHEMA: Dict[str, Any] = {
        "name": "WorldviewMeta",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "genre_tone": {"type": "string", "minLength": 1},
                "audience_rating": {"type": "string", "minLength": 1},
                "inspirations": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "null"},
                    ]
                },
                "themes": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ]
                },
                "medium": {"type": "string", "minLength": 1},
                "era_power_level": {"type": "string", "minLength": 1},
                "language_culture_flavor": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "null"},
                    ]
                },
                "constraints": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "hard": {"type": "array", "items": {"type": "string"}},
                        "soft": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["hard", "soft"],
                },
            },
            "required": [
                "genre_tone", "audience_rating", "themes", "medium",
                "era_power_level", "constraints"
            ],
        }
    }

    # 2) Expansion：开放式 Facets，不预置固定领域键
    #    - 必须有 facets（≥6），每个 facet 自命名，描述世界的某一维
    #    - mechanics 为自由对象，允许任意键（additionalProperties=True）
    #    - 保留 consistency_rules / glossary / warnings 等全局信息
    EXPANSION_SCHEMA: Dict[str, Any] = {
        "name": "WorldviewExpansion",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "facets": {
                    "type": "array",
                    "minItems": 6,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string", "minLength": 2},       # 例如："Temporal Taxation", "Gravito-Trade Protocol"
                            "kind": {"type": "string"},                        # 标签/类别名，自定：cosmology/tech/ecology/law/linguistics/…
                            "overview": {"type": "string"},                    # 本 facet 的总述
                            "axioms": {                                        # 该 facet 的基 axioms 或运行假设
                                "type": "array",
                                "minItems": 2,
                                "items": {"type": "string"}
                            },
                            "mechanics": {                                     # 关键机制：允许任意键，承载具体规则/流程/公式
                                "type": "object",
                                "additionalProperties": True
                            },
                            "limits": { "type": "array", "items": {"type": "string"}},      # 硬限制/边界
                            "risks": { "type": "array", "items": {"type": "string"}},       # 代价/风险
                            "metrics": { "type": "array", "items": {"type": "string"}},     # 可量化指标（若无可留空数组）
                            "implications": { "type": "array", "items": {"type": "string"}},# 对社会/生态/知识系统的推论
                            "open_questions": { "type": "array", "items": {"type": "string"}}
                        },
                        "required": ["name", "overview", "axioms", "mechanics"]
                    }
                },
                "consistency_rules": {
                    "type": "array",
                    "minItems": 5,
                    "items": {"type": "string"}
                },
                "glossary": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "term": {"type": "string"},
                            "definition": {"type": "string"},
                        },
                        "required": ["term", "definition"]
                    }
                },
                "randomization_notes": {"type": "string"},
                "warnings": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["facets", "consistency_rules"]
        }
    }

    # 3) Final：Meta + Expansion（为下游使用方便，这里把 Expansion 收进 "expansion" 键）
    FINAL_SCHEMA: Dict[str, Any] = {
        "name": "FinalWorldview",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                **META_SCHEMA["schema"]["properties"],
                "expansion": EXPANSION_SCHEMA["schema"]
            },
            "required": META_SCHEMA["schema"]["required"] + ["expansion"]
        }
    }

    # =======================
    # Prompts
    # =======================
    GENERATOR_SYSTEM = """\
    你是一名世界观工程师。你的输入是“元设定（Meta）”，你的输出是“设定拓展（Expansion）”。
    必须遵循：
    1) 仅限世界背景：不得包含主角、配角、具体剧情、任务、关卡、分镜、单一场景钩子。
    2) Expansion 采用开放式“facets”列表：每个 facet 必须自命名（name），并给出 overview、axioms（>=2）、mechanics（对象，可含任意键）、以及适当的 limits/risks/metrics/implications/open_questions。
    3) 不要重复 Meta 字段描述；而是以 facets 的形式补充新的、可扩展的世界维度。至少生成 6 个 facet。
    4) 给出全局的 consistency_rules（>=5），明确可检验的边界、守恒、条件与例外，避免隐形万能设定。
    5) 禁止使用“主角/配角/英雄/任务/剧情/章节/关卡/故事线/弧线”等词或相关内容。
    """

    GENERATOR_USER_TEMPLATE = """\
    以下是世界观的元设定（Meta）。请基于它生成开放式的“Expansion”（严格遵循 Expansion JSON Schema），
    重点通过多个自命名的 facets（≥6）充实世界背景，并保证自洽与可拓展性。

    {meta_json}
    """

    VALIDATOR_SYSTEM = """\
    你是一名“世界观一致性审阅者”。你的工作：
    1) 输入：Meta 与 Expansion（facets 列表）。
    2) 输出 JSON：
       - "issues": 按重要性排序的问题列表（含 severity/summary/affected_fields/rationale）
       - "improvements": 面向具体 facet/字段的条例化修订建议
       - "revised_expansion": 仅返回修订后的 Expansion（同 Expansion Schema）。不要改动或复述 Meta 字段。
    3) 严禁引入人物与剧情抓手；只作用于“世界背景与规则”。
    4) 修订目标：更强自洽性、清晰边界与代价、术语一致性、指标可量化。
    """

    VALIDATOR_USER_TEMPLATE = """\
    请审阅下列“Meta + Expansion（合并展示）”，并产出 issues / improvements / revised_expansion（严格符合各自 Schema）：

    ## Meta
    {meta_json}

    ## Expansion
    {expansion_json}
    """

    # =======================
    # Core Calls
    # =======================
    def call_structured_json(self, model: str, system_prompt: str, user_prompt: str, json_schema: Dict[str, Any], temperature: float) -> Dict[str, Any]:
        """Responses API with JSON schema-constrained output."""
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
        return json.loads(resp.output_text)

    # =======================
    # Pipeline Helpers
    # =======================
    def meta_hash(self, meta: Dict[str, Any]) -> str:
        dump = json.dumps(meta, ensure_ascii=False, sort_keys=True)
        import hashlib
        return hashlib.sha256(dump.encode("utf-8")).hexdigest()[:16]

    def save_meta(self, meta: Dict[str, Any]) -> pathlib.Path:
        # 轻量 schema 校验：强模型以 0 温度回传原 JSON（触发 schema 验证）
        _ = self.call_structured_json(
            model=self.STRONG_TEXT_MODEL,
            system_prompt="请把以下 JSON 原样返回（用于 Meta Schema 校验）。",
            user_prompt=json.dumps(meta, ensure_ascii=False),
            json_schema=self.META_SCHEMA,
            temperature=0.0
        )
        p = self.META_STORE / f"meta_{self.meta_hash(meta)}.json"
        p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    def load_meta(self, path: pathlib.Path) -> Dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def generate_expansion(self, meta: Dict[str, Any], seed: int = None) -> Dict[str, Any]:
        if seed is None:
            seed = int(time.time() * 1000) ^ random.getrandbits(32)
        random.seed(seed)
        up = self.GENERATOR_USER_TEMPLATE.format(meta_json=json.dumps(meta, ensure_ascii=False, indent=2))
        expansion = self.call_structured_json(
            model=self.STRONG_TEXT_MODEL,                 # 用强模型进行自由发挥与结构输出
            system_prompt=self.GENERATOR_SYSTEM,
            user_prompt=up,
            json_schema=self.EXPANSION_SCHEMA,
            temperature=0.95
        )
        return expansion

    def review_and_revise(self, meta: Dict[str, Any], expansion: Dict[str, Any]) -> Dict[str, Any]:
        review_schema = {
            "name": "WorldviewReview",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "issues": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "severity": {"type": "string", "enum": ["critical", "major", "minor"]},
                                "summary": {"type": "string"},
                                "affected_fields": {"type": "array", "items": {"type": "string"}},
                                "rationale": {"type": "string"},
                            },
                            "required": ["severity", "summary"]
                        }
                    },
                    "improvements": {"type": "array", "items": {"type": "string"}},
                    "revised_expansion": self.EXPANSION_SCHEMA["schema"]
                },
                "required": ["issues", "improvements", "revised_expansion"]
            }
        }
        up = self.VALIDATOR_USER_TEMPLATE.format(
            meta_json=json.dumps(meta, ensure_ascii=False, indent=2),
            expansion_json=json.dumps(expansion, ensure_ascii=False, indent=2)
        )
        # 评审用 Weak（更省），若对质量有高要求可切回 Strong
        review = self.call_structured_json(
            model=self.STRONG_TEXT_MODEL,
            system_prompt=self.VALIDATOR_SYSTEM,
            user_prompt=up,
            json_schema=review_schema,
            temperature=0.4
        )
        return review

    def assemble_final(self, meta: Dict[str, Any], expansion: Dict[str, Any]) -> Dict[str, Any]:
        final_obj = {**meta, "expansion": expansion}
        _ = self.call_structured_json(
            model=self.WEAK_TEXT_MODEL,  # 最终结构校验用强模型更稳
            system_prompt="请把以下 JSON 原样返回（用于 Final Schema 校验）。",
            user_prompt=json.dumps(final_obj, ensure_ascii=False),
            json_schema=self.FINAL_SCHEMA,
            temperature=0.0
        )
        return final_obj

    # =======================
    # Public API
    # =======================
    def run(self) -> Dict[str, Any]:
        meta_path = self.save_meta(self.example_meta)
        meta = self.load_meta(meta_path)
        draft_expansion = self.generate_expansion(meta, seed=self.seed)
        review = self.review_and_revise(meta, draft_expansion)
        final_expansion = review.get("revised_expansion", draft_expansion)
        final_worldview = self.assemble_final(meta, final_expansion)
        return {
            "meta_path": str(meta_path),
            "draft_expansion": draft_expansion,
            "review_report": {
                "issues": review.get("issues", []),
                "improvements": review.get("improvements", []),
            },
            "final_worldview": final_worldview
        }

# =======================
# Example
# =======================
if __name__ == "__main__":
    example_meta = {
        "genre_tone": "硬科幻 · 冷峻",
        "audience_rating": "青年～成人",
        "inspirations": ["三体"],
        "era_power_level": "星际早期扩张，亚光速为主",
        "medium": "小说",
        "language":"中文",
    }
    generator = WorldviewGenerator(example_meta, ".env", seed=20251029)
    result = generator.run()
    print("\n=== Meta stored at ===")
    print(result["meta_path"])
    print("\n=== Draft Expansion ===")
    print(json.dumps(result["draft_expansion"], ensure_ascii=False, indent=2))
    print("\n=== Final Worldview (Meta + Revised Expansion) ===")
    print(json.dumps(result["final_worldview"], ensure_ascii=False, indent=2))
    print("\n=== Review Issues (Top) ===")
    for i, issue in enumerate(result["review_report"]["issues"][:10], 1):
        sev = issue.get("severity", "-")
        print(f"{i}. [{sev}] {issue.get('summary','')}")
