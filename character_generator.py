# -*- coding: utf-8 -*-
"""
CharacterGenerator
- 输入: env路径、元数据(meta)、世界观(final_worldview)
- 输出: 主要/次要角色集合（含记忆、经历、时间线、关系、目标等），并经一次评审修订
- 模型选择：强模型 STRONG_TEXT_MODEL（创作/最终校验），弱模型 WEAK_TEXT_MODEL（评审/修订）
- JSON Schema：仅做必要约束，尽量留白，保障可扩展性
- 写作语言：中文
"""

import os
import json
import time
import random
from typing import Any, Dict, List, Optional
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

class CharacterGenerator:
    def __init__(
        self,
        env_path: str,
        meta: Dict[str, Any],
        worldview: Dict[str, Any],
        num_primary: int = 2,
        num_secondary: int = 8,
        seed: Optional[int] = None,
    ):
        load_dotenv(env_path)

        self.meta = meta
        self.worldview = worldview
        self.num_primary = num_primary
        self.num_secondary = num_secondary
        self.seed = seed if seed is not None else int(time.time() * 1000) ^ random.getrandbits(32)

        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
        self.OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL")
        self.STRONG_TEXT_MODEL = os.getenv("STRONG_TEXT_MODEL")
        self.WEAK_TEXT_MODEL   = os.getenv("WEAK_TEXT_MODEL")

        if not self.OPENAI_API_KEY:
            raise RuntimeError("Missing OPENAI_API_KEY (or API_KEY) in .env")

        self.client = OpenAI(
            api_key=self.OPENAI_API_KEY,
            base_url=self.OPENAI_BASE_URL if self.OPENAI_BASE_URL else None,
        )

        # ------------ Prompts ------------
        self.SYSTEM_PROMPT = """你是一名“人物设定工程师”。你将根据提供的世界观与写作目标，自由而一致地生成主要/次要角色。
请遵循以下创作导向（但不限制你的发挥）：
- 用“根/伤/链/面/变”作为灵感提示：
  · 根：角色与世界的连接（出身、地域、时代、技术/信仰环境）
  · 伤：心理创伤或情感空缺，优先用具体记忆与可观察行为而非抽象标签
  · 链：重要关系与记忆钩子（朋友、导师、敌人、前任、家人；允许留白）
  · 面：公众身份 vs 自我叙述的差距与张力（可以有自我欺骗）
  · 变：角色弧（信念从何而来、因何改变、可能走向）
- 时间线自洽、语境一致；尊重世界观中的硬约束与物理常识；允许带“可信的误记/偏见”。
- 输出必须是 JSON，字段允许扩展；除非必要，不要过度解释设置背后的理论。
- 用中文写作。"""

        self.USER_PROMPT_TEMPLATE = """
{worldview_and_meta}

【人物生成目标】
- 需要角色数：主要 {num_primary}，次要 {num_secondary}

【写作自由度】
- 在不违背硬约束与常识的前提下，可大胆进行社会/技术/心理延展；
- 人物须能直接服务剧情：给出“场景级可用”的细节与钩子，但不要被格式束缚；
- 为每个角色编写一段或多段背景故事（鼓励与其他角色产生关联）
- 为每个角色生成潜在目标（近期/中期/长期）
- 允许保留悬而未决的问题/伏笔，便于后续追踪与回收。

仅输出 JSON，不要附加任何解释。
"""

        self.REVIEW_SYSTEM = """你是一名“角色一致性审阅者”。你的任务：
1) 检查角色集合与世界观硬约束的符合性、内部自洽性、时间线与经历的合理性、术语一致性。
2) 输出 JSON：包含 issues（按重要性排序）、improvements（可执行建议）、revised_characters（修订后的角色集合）。
3) 不改变世界观与元设定；只在必要处修订角色字段与内容。
4) 鼓励保留适度的“可信误记/偏见”，但要明确其不确定性标注。
"""

        self.REVIEW_USER_TEMPLATE = """
【世界观摘要】
{worldview_json}

【元设定】
{meta_json}

【待审角色（草稿）】
{characters_json}

请产出 issues / improvements / revised_characters（严格遵循给定的 JSON Schema）。
"""

        # ------------ JSON Schemas ------------
        # 角色集合的 Schema：尽量留白，只约束必要字段与类型，允许扩展。
        # - characters: 数组，包含 primary/secondary
        # - 每个角色要求: id, role, display_name, background.story, memories[], timeline[]
        # - 大部分子字段 additionalProperties=True，给模型自由发挥空间
        self.CHARACTER_SET_SCHEMA: Dict[str, Any] = {
            "name": "CharacterSet",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "counts": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "primary": {"type": "integer", "minimum": 0},
                            "secondary": {"type": "integer", "minimum": 0}
                        },
                        "required": ["primary", "secondary"]
                    },
                    "characters": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": True,  # 留白，允许扩展任意自定义键
                            "properties": {
                                "id": {"type": "string"},
                                "role": {"type": "string", "enum": ["primary", "secondary"]},
                                "display_name": {"type": "string"},
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "root_injury_chain_face_change": {
                                    "type": "object",
                                    "additionalProperties": True,
                                    "properties": {
                                        "root": {"type": "string"},
                                        "injury": {"type": "string"},
                                        "chain": {"type": "string"},
                                        "face": {"type": "string"},
                                        "change": {"type": "string"},
                                    }
                                },
                                "background": {
                                    "type": "object",
                                    "additionalProperties": True,
                                    "properties": {
                                        "story": {"type": "string"},
                                        "culture_language_notes": {"type": "string"},
                                        "worldview_alignment": {
                                            "type": "array",
                                            "items": {"type": "string"}  # 可引用世界观中的 facet 名称或术语
                                        }
                                    },
                                    "required": ["story"]
                                },
                                "memories": {
                                    "type": "array",
                                    "minItems": 3,
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": True,
                                        "properties": {
                                            "type": {"type": "string", "enum": ["episodic", "semantic", "procedural", "flashbulb", "dreamlike"]},
                                            "content": {"type": "string"},
                                            "trigger": {"type": "string"},
                                            "salience": {"type": "number", "minimum": 0, "maximum": 1},
                                            "reliability": {"type": "string", "enum": ["certain", "uncertain", "contested"]},
                                            "time_hint": {"type": "string"}
                                        },
                                        "required": ["type", "content"]
                                    }
                                },
                                "timeline": {
                                    "type": "array",
                                    "minItems": 3,
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": True,
                                        "properties": {
                                            "when": {"type": "string"},           # 与世界观历法/时代描述相容的时间线文本
                                            "event": {"type": "string"},
                                            "facet_refs": {
                                                "type": "array",
                                                "items": {"type": "string"}
                                            },
                                            "certainty": {"type": "string", "enum": ["high", "medium", "low"]}
                                        },
                                        "required": ["when", "event"]
                                    }
                                },
                                "relationships": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": True,
                                        "properties": {
                                            "target_id": {"type": "string"},   # 指向其他角色 id（允许先行引用）
                                            "relation": {"type": "string"},
                                            "evidence_or_memory": {"type": "string"}
                                        }
                                    }
                                },
                                "goals": {
                                    "type": "object",
                                    "additionalProperties": True,
                                    "properties": {
                                        "short_term": {"type": "array", "items": {"type": "string"}},
                                        "mid_term": {"type": "array", "items": {"type": "string"}},
                                        "long_term": {"type": "array", "items": {"type": "string"}}
                                    }
                                },
                                "traits_skills_assets": {
                                    "type": "object",
                                    "additionalProperties": True,
                                    "properties": {
                                        "traits": {"type": "array", "items": {"type": "string"}},
                                        "skills": {"type": "array", "items": {"type": "string"}},
                                        "assets": {"type": "array", "items": {"type": "string"}},
                                        "constraints": {"type": "array", "items": {"type": "string"}}
                                    }
                                },
                                "secrets_and_hooks": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "unresolved_questions": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "pov_voice_hint": {"type": "string"},
                                "reliability_notes": {"type": "string"}
                            },
                            "required": ["id", "role", "display_name", "background", "memories", "timeline"]
                        }
                    }
                },
                "required": ["counts", "characters"]
            }
        }

        # 评审输出 Schema：只修订角色集合，不触碰 meta/worldview
        self.REVIEW_SCHEMA: Dict[str, Any] = {
            "name": "CharacterReview",
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
                                "rationale": {"type": "string"}
                            },
                            "required": ["severity", "summary"]
                        }
                    },
                    "improvements": {"type": "array", "items": {"type": "string"}},
                    "revised_characters": self.CHARACTER_SET_SCHEMA["schema"]
                },
                "required": ["issues", "improvements", "revised_characters"]
            }
        }

    # ---------------- Core LLM Call ----------------
    def _call_structured_json(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        json_schema: Dict[str, Any],
        temperature: float
    ) -> Dict[str, Any]:
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

    # ---------------- Builders ----------------
    def _build_generation_user_prompt(self) -> str:
        # 将世界观和元数据按“原样”并列给出，便于模型引用；不做剪裁以减少信息丢失
        worldview_and_meta = json.dumps(
            {
                "worldview": self.worldview,
                "meta": self.meta
            },
            ensure_ascii=False,
            indent=2
        )
        return self.USER_PROMPT_TEMPLATE.format(
            worldview_and_meta=worldview_and_meta,
            num_primary=self.num_primary,
            num_secondary=self.num_secondary
        )

    def _build_review_user_prompt(self, characters_json: Dict[str, Any]) -> str:
        return self.REVIEW_USER_TEMPLATE.format(
            worldview_json=json.dumps(self.worldview, ensure_ascii=False, indent=2),
            meta_json=json.dumps(self.meta, ensure_ascii=False, indent=2),
            characters_json=json.dumps(characters_json, ensure_ascii=False, indent=2)
        )

    # ---------------- Pipeline Steps ----------------
    def generate_characters(self) -> Dict[str, Any]:
        random.seed(self.seed)
        uprompt = self._build_generation_user_prompt()
        draft = self._call_structured_json(
            model=self.STRONG_TEXT_MODEL,        # 创作用强模型
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=uprompt,
            json_schema=self.CHARACTER_SET_SCHEMA,
            temperature=0.95
        )
        # 若模型未填 counts，则补上（容错）
        if "counts" not in draft:
            draft["counts"] = {"primary": self.num_primary, "secondary": self.num_secondary}
        return draft

    def review_and_revise(self, draft_characters: Dict[str, Any]) -> Dict[str, Any]:
        uprompt = self._build_review_user_prompt(draft_characters)
        review = self._call_structured_json(
            model=self.STRONG_TEXT_MODEL,
            system_prompt=self.REVIEW_SYSTEM,
            user_prompt=uprompt,
            json_schema=self.REVIEW_SCHEMA,
            temperature=0.4
        )
        return review

    def final_schema_check(self, characters: Dict[str, Any]) -> Dict[str, Any]:
        # 做一次结构校验（原样返回）
        checked = self._call_structured_json(
            model=self.WEAK_TEXT_MODEL,
            system_prompt="请把以下 JSON 原样返回（用于角色集合 Schema 校验）。",
            user_prompt=json.dumps(characters, ensure_ascii=False),
            json_schema=self.CHARACTER_SET_SCHEMA,
            temperature=0.0
        )
        return checked

    # ---------------- Public API ----------------
    def run(self) -> Dict[str, Any]:
        """
        流程：
        1) 生成角色草稿（主要/次要、记忆、时间线、关系、目标等）
        2) 进行一次一致性审阅并得到修订稿
        3) 最终结果通过 Schema 校验后返回
        """
        draft_characters = self.generate_characters()
        review = self.review_and_revise(draft_characters)
        final_characters = review.get("revised_characters", draft_characters)
        final_characters = self.final_schema_check(final_characters)

        return {
            "seed": self.seed,
            "counts_requested": {"primary": self.num_primary, "secondary": self.num_secondary},
            "draft_characters": draft_characters,
            "review_report": {
                "issues": review.get("issues", []),
                "improvements": review.get("improvements", [])
            },
            "final_characters": final_characters
        }

