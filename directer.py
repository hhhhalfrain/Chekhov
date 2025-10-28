# -*- coding: utf-8 -*-
"""
世界观生成 + 逻辑校验（单次迭代）脚本
- 使用 OpenAI Responses API
- 以 JSON Schema 约束产出
- 随机扩展额外维度，且禁止人物/剧情抓手
"""

import os
import json
import time
import random
import dotenv
from typing import Any, Dict

from openai import OpenAI

# ========== 配置区域 ==========
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# 可按账户开通情况替换为实际可用的最新最强模型
STRONG_TEXT_MODEL = os.getenv("STRONG_TEXT_MODEL", "gpt-5")
WEAK_TEXT_MODEL = os.getenv("WEAK_TEXT_MODEL", "gpt-5")
if not OPENAI_API_KEY or not OPENAI_BASE_URL:
    raise RuntimeError("Missing OPENAI_API_KEY (or API_KEY) in environment (.env)")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None,
)

# ========== JSON Schema（世界观）==========
WORLDVIEW_JSON_SCHEMA: Dict[str, Any] = {
    "name": "WorldviewSchema",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            # ---- 用户必填要素（保留原样）----
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

            # ---- 模型需“随机但有意义”地扩展的维度（每次可变子集，至少 6 个）----
            "cosmology_or_fundamental_rules": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "space_time": {"type": "string"},
                    "energy_sources": {"type": "string"},
                    "conservation_exceptions": {"type": "string"},
                },
                "required": ["space_time", "energy_sources"],
            },
            "tech_or_magic_system": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "classification": {"type": "string"},  # 例如：模组化科技 / 仪式魔法 / 奥义工程学
                    "capability_bounds": {"type": "string"},
                    "costs_and_risks": {"type": "string"},
                    "accessibility": {"type": "string"},
                },
                "required": ["classification", "capability_bounds", "costs_and_risks"],
            },
            "political_economic_landscape": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "dominant_structures": {"type": "string"},
                    "resource_constraints": {"type": "string"},
                    "trade_conflict_patterns": {"type": "string"},
                },
                "required": ["dominant_structures"],
            },
            "societies_and_cultures": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "social_strata": {"type": "string"},
                    "value_systems": {"type": "string"},
                    "rituals_and_customs": {"type": "string"},
                    "taboos": {"type": "string"},
                },
                "required": ["value_systems"],
            },
            "ecology_and_biomes": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "dominant_biomes": {"type": "string"},
                    "keystone_species": {"type": "string"},
                    "environmental_pressures": {"type": "string"},
                },
                "required": ["dominant_biomes"],
            },
            "sentient_kinds_catalog": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "label": {"type": "string"},
                        "physiology_outline": {"type": "string"},
                        "social_traits": {"type": "string"},
                        "interaction_protocols": {"type": "string"},
                    },
                    "required": ["label", "physiology_outline"],
                },
            },
            "language_and_writing": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "linguistic_features": {"type": "string"},
                    "writing_systems": {"type": "string"},
                    "translation_constraints": {"type": "string"},
                },
            },
            "knowledge_systems_and_education": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "institutions": {"type": "string"},
                    "canon_vs_heresy": {"type": "string"},
                    "innovation_channels": {"type": "string"},
                },
            },
            "law_morality_and_order": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "legal_principles": {"type": "string"},
                    "enforcement_mechanisms": {"type": "string"},
                    "notable_illegalities": {"type": "string"},
                },
            },
            "timekeeping_calendar": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "calendar_system": {"type": "string"},
                    "cycles_and_eras": {"type": "string"},
                    "festival_mapping": {"type": "string"},
                },
            },
            "geography_and_infrastructure": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "macro_topology": {"type": "string"},
                    "transport_and_networks": {"type": "string"},
                    "settlement_archetypes": {"type": "string"},
                },
            },
            "everyday_material_culture": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "household_technics": {"type": "string"},
                    "food_and_clothing": {"type": "string"},
                    "art_and_recreation": {"type": "string"},
                },
            },
            "consistency_rules": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3
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
                    "required": ["term", "definition"],
                },
            },

            # 元信息
            "randomization_notes": {"type": "string"},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "genre_tone", "audience_rating", "themes", "medium",
            "era_power_level", "constraints",
            "cosmology_or_fundamental_rules", "tech_or_magic_system",
            "political_economic_landscape", "societies_and_cultures",
            "ecology_and_biomes", "consistency_rules"
        ],
    }
}

# ========== 提示词（System / User）==========

GENERATOR_SYSTEM_PROMPT = """\
你是一名世界观工程师。你的目标是：在用户提供的“世界观要素”基础上，生成只包含“世界背景”的设定文本。
必须遵循：
1) 仅限世界背景，不得包含主角、配角、具体剧情、任务、关卡、分镜、单一场景钩子。
2) 若用户输入含人物或剧情抓手，忽略之；只吸纳对世界背景有效的部分。
3) 输出通过 JSON Schema 严格约束，字段必须完整且自洽；描述详实并可拓展；不得出现逻辑矛盾。
4) 每次生成需在用户要素外，随机抽样若干“额外维度”进行充实，至少 6 个，且内在合理、有趣，但不引入剧情线。
5) 采用具象可执行的规则与边界（如代价、风险、限制、传播路径），并显式给出“Consistency Rules”。
6) 禁止使用“主角/配角/英雄/任务/剧情/章节/关卡/故事线/弧线/冲突的推进”等词汇与对应内容。
"""

GENERATOR_USER_PROMPT_TEMPLATE = """\
以下为用户给定的世界观基础要素。请在此基础上完成世界背景生成（包含随机扩展的若干维度），并严格符合 JSON Schema：

- 题材/基调：{genre_tone}
- 受众与年龄分级：{audience_rating}
- 灵感参考（可为空）：{inspirations}
- 开放式主题（核心母题/议题）：{themes}
- 媒介形态：{medium}
- 时代与科技魔法水平：{era_power_level}
- 语言与文化风味（可为空）：{language_culture_flavor}
- 硬性与软性约束：
  - Hard：{hard_constraints}
  - Soft：{soft_constraints}

注意：若上面任何条目中出现了“主角/配角/剧情场景抓手”等，请忽略这些信息，不得写入输出。
"""

VALIDATOR_SYSTEM_PROMPT = """\
你是一名“世界观一致性审阅者”。你的工作：
1) 对输入的世界观 JSON 进行一致性与可拓展性检查：找出矛盾、自我冲突、隐含违例、边界未定义等。
2) 严格避免引入人物与剧情抓手；全部建议仅作用于“世界背景与规则”。
3) 输出一个 JSON，其中包含：
   - "issues": 按重要性排序的问题列表（每项含说明与受影响字段）
   - "improvements": 可执行的修订建议（条例化，具体到字段/子字段）
   - "revised_worldview": 按同一 JSON Schema 给出的修订版本（若无需修订，可原样拷贝）
4) 确保修订版满足：更强的自洽性、边界更清晰、术语一致、无人物与剧情钩子。
"""

VALIDATOR_USER_PROMPT_TEMPLATE = """\
请审阅以下世界观（JSON）并给出问题列表、改进建议，以及一次性修订版本：
{worldview_json}
"""

# ========== 生成与校验逻辑 ==========


def responses_json(model: str, system_prompt: str, user_prompt: str, json_schema: Dict[str, Any], temperature: float = 0.9) -> Dict[str, Any]:
    """调用 Responses API，要求按 JSON Schema 输出。"""

    resp = client.responses.create(
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

    text = resp.output_text
    return json.loads(text)

def generate_worldview(user_fields: Dict[str, Any], seed: int = None) -> Dict[str, Any]:
    """第一阶段：根据用户要素 + 随机扩展维度，产出世界观 JSON。"""
    if seed is None:
        seed = int(time.time() * 1000) ^ random.getrandbits(32)
    random.seed(seed)

    # 拼装 user prompt
    uprompt = GENERATOR_USER_PROMPT_TEMPLATE.format(
        genre_tone=user_fields.get("genre_tone", ""),
        audience_rating=user_fields.get("audience_rating", ""),
        inspirations=user_fields.get("inspirations", ""),
        themes=user_fields.get("themes", ""),
        medium=user_fields.get("medium", ""),
        era_power_level=user_fields.get("era_power_level", ""),
        language_culture_flavor=user_fields.get("language_culture_flavor", ""),
        hard_constraints=user_fields.get("constraints", {}).get("hard", []),
        soft_constraints=user_fields.get("constraints", {}).get("soft", []),
    )

    result = responses_json(
        model=STRONG_TEXT_MODEL,
        system_prompt=GENERATOR_SYSTEM_PROMPT,
        user_prompt=uprompt,
        json_schema=WORLDVIEW_JSON_SCHEMA,
        temperature=0.95  # 保持一定创意，同时由 Schema 兜底
    )

    # 轻量净化（避免人物/剧情术语混入）
    return result

def validate_and_revise_worldview(worldview: Dict[str, Any]) -> Dict[str, Any]:
    """第二阶段：用校验模型审阅并给出一次修订，返回 {issues, improvements, revised_worldview}。"""
    payload = json.dumps(worldview, ensure_ascii=False, indent=2)
    uprompt = VALIDATOR_USER_PROMPT_TEMPLATE.format(worldview_json=payload)

    # 校验输出的 Schema（简单定义）
    validator_schema = {
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
                "improvements": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "revised_worldview": WORLDVIEW_JSON_SCHEMA["schema"]
            },
            "required": ["issues", "improvements", "revised_worldview"]
        }
    }

    review = responses_json(
        model=STRONG_TEXT_MODEL,
        system_prompt=VALIDATOR_SYSTEM_PROMPT,
        user_prompt=uprompt,
        json_schema=validator_schema,
        temperature=0.4  # 偏向稳健评审
    )

    return review

def generate_with_single_iteration(user_fields: Dict[str, Any], seed: int = None) -> Dict[str, Any]:
    """完整流程：生成一次→校验并修订一次→返回最终稿与审阅报告。"""
    draft = generate_worldview(user_fields, seed=seed)
    review = validate_and_revise_worldview(draft)

    final_worldview = review.get("revised_worldview", draft)
    return {
        "seed": seed,
        "draft_worldview": draft,
        "review_report": {
            "issues": review.get("issues", []),
            "improvements": review.get("improvements", []),
        },
        "final_worldview": final_worldview
    }

# ========== 示例用法 ==========
if __name__ == "__main__":
    user_input_example = {
        "genre_tone": "硬科幻 · 冷峻",
        "audience_rating": "青年～成人（PEGI/ESRB 相当级别）",
        "inspirations": ["三体的黑暗森林假说", "阿西莫夫的基地式文明更迭", "生态科幻影像"],
        "themes": ["生存与信息非对称", "文明脆弱性", "道德在极限压力下的边界"],
        "medium": "CRPG",
        "era_power_level": "星际早期扩张，亚光速主导，高能物理可控但昂贵",
        "language_culture_flavor": ["中英混合命名体系", "受东亚与环太平洋文化启发"],
        "constraints": {
            "hard": [
                "不出现具体人物设定",
                "不编写任务/关卡/剧情推进",
                "宇宙学与能量守恒需在内在逻辑自洽"
            ],
            "soft": [
                "尽量引入可量化成本/风险",
                "术语与名词统一可扩展"
            ]
        }
    }

    result = generate_with_single_iteration(user_input_example, seed=20251028)
    # 仅演示：打印最终世界观与问题概览
    print("\n=== Final Worldview (JSON) ===")
    print(json.dumps(result["final_worldview"], ensure_ascii=False, indent=2))

    print("\n=== Review Issues (Summary) ===")
    for i, issue in enumerate(result["review_report"]["issues"][:10], 1):
        print(f"{i}. [{issue.get('severity','-')}] {issue.get('summary','')}")
