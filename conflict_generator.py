# -*- coding: utf-8 -*-
"""
ConflictGenerator
- 输入: env 路径、世界观(final_worldview)、角色集合(final_characters)
- 输出: 目标-矛盾网络（goals + links + tensions），每个角色的短/中/长期目标相互勾连，形成张力
- 模型:
    STRONG_TEXT_MODEL: 创作 & 最终结构校验
    WEAK_TEXT_MODEL:   评审 & 修订（可按需切换为强模型）
- 约束:
    1) 仅世界背景与人物目标/动机；不强行写剧情分镜
    2) 每个目标必须至少有 1 条 link（出边或入边）
    3) 主要角色之间至少存在一种冲突/竞争/阻断关系
"""

import os
import json
import time
import random
from typing import Any, Dict, Optional, List
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


class ConflictGenerator:
    def __init__(
        self,
        env_path: str,
        worldview: Dict[str, Any],
        characters: Dict[str, Any],
        seed: Optional[int] = None,
    ):
        load_dotenv(env_path)

        self.worldview = worldview
        self.characters = characters
        self.seed = seed if seed is not None else int(time.time() * 1000) ^ random.getrandbits(32)

        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
        self.OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL")
        self.STRONG_TEXT_MODEL = os.getenv("STRONG_TEXT_MODEL", "gpt-5")
        self.WEAK_TEXT_MODEL   = os.getenv("WEAK_TEXT_MODEL", "gpt-5-mini")

        if not self.OPENAI_API_KEY:
            raise RuntimeError("Missing OPENAI_API_KEY (or API_KEY) in .env")

        self.client = OpenAI(
            api_key=self.OPENAI_API_KEY,
            base_url=self.OPENAI_BASE_URL if self.OPENAI_BASE_URL else None,
        )

        # ================= Prompts =================
        self.SYSTEM_PROMPT = """你是一名“矛盾与目标网络设计师”。请依据提供的世界观与角色卡：
- 为每位角色生成“当前动机与目标”，分为短/中/长期（short/mid/long）。
- 目标必须相互交织：通过关系边（links）形成网络；严禁相互独立。
- 允许多种关系：supports（支持/实现）、blocks（阻断/对立）、competes（竞争/排他）、depends（依赖/必要条件）、enables（使能）、mutual_exclusion（互斥）。
- 主要角色之间至少存在一种 blocks / competes / mutual_exclusion 关系。
- 目标应可操作、可验证（给出 success_conditions / failure_risks / metrics），且与世界观硬约束一致（引用 world_refs）。
- 输出中文，仅输出 JSON。

注意：不要写具体剧情分镜；聚焦“目标与张力的结构化设计”。"""

        self.USER_PROMPT_TEMPLATE = """
【世界观（摘录/原样）】
{worldview_json}

【角色卡（原样）】
{characters_json}

【建模目标】
- 依据世界观约束与角色背景，生成角色的当前动机与目标（short/mid/long）。
- 目标之间必须相互勾连：请构造包含多种关系的 links（supports/blocks/competes/depends/enables/mutual_exclusion）。
- 主要角色之间至少存在 1 条 blocks/competes/mutual_exclusion 边。
- 每个目标至少 1 条 link（出或入），禁止孤立目标。
- 对关键矛盾簇，给出紧张度说明与演化钩子（tensions / progression）。

仅输出 JSON，不要附加解释。
"""

        self.REVIEW_SYSTEM = """你是一名“目标网络一致性审阅者”。你的任务：
1) 检查目标层级（short/mid/long）的合理性、世界观一致性、连通性（无孤立目标）、主要角色之间是否存在冲突边。
2) 必要时修订：优化目标描述、修正层级、补充/重连 links、补充 success_conditions / failure_risks / metrics。
3) 输出 JSON：issues / improvements / revised_conflicts（仅修订冲突网络，不改世界观与角色卡）。"""

        self.REVIEW_USER_TEMPLATE = """
【世界观（提供以便一致性校验）】
{worldview_json}

【角色卡（提供以便一致性校验）】
{characters_json}

【待审冲突网络（草稿）】
{conflicts_json}

请输出 issues / improvements / revised_conflicts（严格遵循给定 JSON Schema）。
"""

        # ================= JSON Schemas =================
        # 仅约束骨架，尽量留白，让模型自由发挥。
        self.CONFLICT_SCHEMA: Dict[str, Any] = {
            "name": "ConflictNetwork",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "actors": {  # 参与者索引，便于对齐角色 id 与显示名
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                            "properties": {
                                "id": {"type": "string"},
                                "display_name": {"type": "string"},
                                "role": {"type": "string"},  # primary/secondary
                            },
                            "required": ["id", "display_name", "role"]
                        }
                    },
                    "goals": {
                        "type": "array",
                        "minItems": 4,
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                            "properties": {
                                "goal_id": {"type": "string"},
                                "owner_id": {"type": "string"},  # 指向 actors.id
                                "tier": {"type": "string", "enum": ["short", "mid", "long"]},
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "rationale": {"type": "string"},
                                "world_refs": {  # 与世界观/术语/规则的对齐（字符串数组即可）
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "constraints": {"type": "array", "items": {"type": "string"}},
                                "success_conditions": {"type": "array", "items": {"type": "string"}},
                                "failure_risks": {"type": "array", "items": {"type": "string"}},
                                "metrics": {"type": "array", "items": {"type": "string"}},
                                "time_horizon_hint": {"type": "string"},
                                "notes": {"type": "string"}
                            },
                            "required": ["goal_id", "owner_id", "tier", "title", "description"]
                        }
                    },
                    "links": {
                        "type": "array",
                        "minItems": 3,
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                            "properties": {
                                "source_goal_id": {"type": "string"},
                                "target_goal_id": {"type": "string"},
                                "relation": {
                                    "type": "string",
                                    "enum": ["supports","blocks","competes","depends","enables","mutual_exclusion"]
                                },
                                "weight": {"type": "number"},  # 可选：强度/置信
                                "notes": {"type": "string"}
                            },
                            "required": ["source_goal_id", "target_goal_id", "relation"]
                        }
                    },
                    "tensions": {  # 关键矛盾簇的摘要
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                            "properties": {
                                "label": {"type": "string"},
                                "involved_goal_ids": {"type": "array", "items": {"type": "string"}},
                                "why_it_matters": {"type": "string"},
                                "escalation_paths": {"type": "array", "items": {"type": "string"}},  # 可能升级方向
                                "deescalation_options": {"type": "array", "items": {"type": "string"}} # 降级/妥协方式
                            },
                            "required": ["label", "involved_goal_ids", "why_it_matters"]
                        }
                    },
                    "progression": {  # 张力演化阶段建议（非剧情分镜）
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                            "properties": {
                                "phase": {"type": "string"},
                                "goal_shifts": {"type": "array", "items": {"type": "string"}}, # 目标改写/迁移/升级
                                "link_shifts": {"type": "array", "items": {"type": "string"}}, # 关系改变（依赖->竞争等）
                                "risk_triggers": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["phase"]
                        }
                    },
                    "consistency_rules": {  # 网络层面的可检验规则
                        "type": "array",
                        "minItems": 5,
                        "items": {"type": "string"}
                    }
                },
                "required": ["actors", "goals", "links", "tensions", "consistency_rules"]
            }
        }

        self.REVIEW_SCHEMA: Dict[str, Any] = {
            "name": "ConflictReview",
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
                                "affected": {"type": "array", "items": {"type": "string"}},
                                "rationale": {"type": "string"}
                            },
                            "required": ["severity", "summary"]
                        }
                    },
                    "improvements": {"type": "array", "items": {"type": "string"}},
                    "revised_conflicts": self.CONFLICT_SCHEMA["schema"]
                },
                "required": ["issues", "improvements", "revised_conflicts"]
            }
        }

    # ================ Core LLM ================
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

    # ================ Internal Builders ================
    def _extract_actor_index(self) -> List[Dict[str, Any]]:
        """
        从角色集合提取 actors 索引（id, display_name, role）。
        兼容 self.characters 结构：期望为 CharacterGenerator 的 final_characters。
        """
        actors = []
        # 允许两种包装：{"final_characters": {...}} 或直接 {...}
        payload = self.characters.get("final_characters", self.characters)
        char_list = payload.get("characters", [])
        for c in char_list:
            actors.append({
                "id": c.get("id", ""),
                "display_name": c.get("display_name", ""),
                "role": c.get("role", "")
            })
        return actors

    def _build_generation_user_prompt(self) -> str:
        actors = self._extract_actor_index()
        scaffold = {
            "worldview": self.worldview,
            "actors_index": actors,
            "characters_full": self.characters.get("final_characters", self.characters)  # 完整角色卡以便引用记忆/经历
        }
        return self.USER_PROMPT_TEMPLATE.format(
            worldview_json=json.dumps(self.worldview, ensure_ascii=False, indent=2),
            characters_json=json.dumps(scaffold["characters_full"], ensure_ascii=False, indent=2),
        )

    def _build_review_user_prompt(self, conflicts_json: Dict[str, Any]) -> str:
        return self.REVIEW_USER_TEMPLATE.format(
            worldview_json=json.dumps(self.worldview, ensure_ascii=False, indent=2),
            characters_json=json.dumps(self.characters.get("final_characters", self.characters), ensure_ascii=False, indent=2),
            conflicts_json=json.dumps(conflicts_json, ensure_ascii=False, indent=2)
        )

    # ================ Pipeline Steps ================
    def generate_draft(self) -> Dict[str, Any]:
        random.seed(self.seed)
        uprompt = self._build_generation_user_prompt()
        draft = self._call_structured_json(
            model=self.STRONG_TEXT_MODEL,
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=uprompt,
            json_schema=self.CONFLICT_SCHEMA,
            temperature=0.95
        )
        return draft

    def review_and_revise(self, draft_conflicts: Dict[str, Any]) -> Dict[str, Any]:
        uprompt = self._build_review_user_prompt(draft_conflicts)
        review = self._call_structured_json(
            model=self.STRONG_TEXT_MODEL,
            system_prompt=self.REVIEW_SYSTEM,
            user_prompt=uprompt,
            json_schema=self.REVIEW_SCHEMA,
            temperature=0.4
        )
        return review

    def final_schema_check(self, conflicts: Dict[str, Any]) -> Dict[str, Any]:
        # 用强模型做一次结构校验（原样返回）
        checked = self._call_structured_json(
            model=self.WEAK_TEXT_MODEL,
            system_prompt="请把以下 JSON 原样返回（用于冲突网络 Schema 校验）。",
            user_prompt=json.dumps(conflicts, ensure_ascii=False),
            json_schema=self.CONFLICT_SCHEMA,
            temperature=0.0
        )
        return checked

    # ================ Public API ================
    def run(self) -> Dict[str, Any]:
        """
        1) 生成矛盾网络草稿（actors/goals/links/tensions/progression）
        2) 进行一次一致性审阅与修订
        3) 通过最终 Schema 校验并返回
        额外保障（由提示词与评审共同达成）：
            - 每个目标至少1条 link
            - 主要角色之间至少存在 blocks / competes / mutual_exclusion
        """
        draft = self.generate_draft()
        review = self.review_and_revise(draft)
        final_conflicts = review.get("revised_conflicts", draft)
        final_conflicts = self.final_schema_check(final_conflicts)

        return {
            "seed": self.seed,
            "draft_conflicts": draft,
            "review_report": {
                "issues": review.get("issues", []),
                "improvements": review.get("improvements", [])
            },
            "final_conflicts": final_conflicts
        }


# ============== Example ==============
if __name__ == "__main__":
    # 你可以将下述示例替换为 pipeline 中的真实文件加载
    env_path = ".env"

    # 示例世界观（通常来自你的 WorldviewGenerator 最终稿）
    example_worldview = {
        "genre_tone": "硬科幻 · 冷峻",
        "medium": "小说",
        "expansion": {"facets": [], "consistency_rules": ["能量守恒记录", "信息代价明确"]}
    }

    # 示例角色集合（通常来自 CharacterGenerator 最终稿）
    example_characters = {
        "characters": [
            {"id": "P1", "display_name": "林洵", "role": "primary"},
            {"id": "P2", "display_name": "安可", "role": "primary"},
            {"id": "S1", "display_name": "莱尔", "role": "secondary"},
            {"id": "S2", "display_name": "贺晚舟", "role": "secondary"}
        ]
    }

    gen = ConflictGenerator(env_path, example_worldview, example_characters, seed=20251029)
    result = gen.run()
    print(json.dumps(result["final_conflicts"], ensure_ascii=False, indent=2))
