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
from llm_base import LLMBase

class WorldviewGenerator(LLMBase):  # 继承 LLMBase
    def __init__(self, example_meta: Dict[str, Any], env_path: str, seed: int = None):
        super().__init__(env_path, seed)  # 调用父类初始化
        self.example_meta = example_meta

    # =======================
    # JSON Schemas  （仅约束为 JSON 对象；不预定义字段）
    # =======================
    META_SCHEMA: Dict[str, Any] = {
        "name": "WorldviewMeta",
        "schema": {
            "type": "object",
            "properties": {},  # 必须显式声明，即使为空
            "additionalProperties": True
        }
    }

    EXPANSION_SCHEMA: Dict[str, Any] = {
        "name": "WorldviewExpansion",
        "schema": {
            "type": "object",
            "properties": {},  # 同样要求
            "additionalProperties": True
        }
    }

    FINAL_SCHEMA: Dict[str, Any] = {
        "name": "FinalWorldview",
        "schema": {
            "type": "object",
            "properties": {},  # 必须显式声明
            "additionalProperties": True
        }
    }

    # =======================
    # Prompts
    # =======================
    GENERATOR_SYSTEM = """\
    你是一名“世界观工程师（Worldbuilding Engineer）”。你的输入是元设定（Meta），你的输出是世界观（Worldview）。
    目标：输出一个结构清晰、信息密度高、具有强可写性的“世界背景规则集”。务必新鲜、有张力，能长期支撑连载创作。

    硬性约束（必须遵守）：
    1) 只产出“世界背景与运行规则”。严禁包含：主角、配角、具体剧情桥段、任务、章节、关卡、台词、独立场景钩子等。
    2) 结果必须是一个 JSON 对象（不要 Markdown、不要解释文字）。

    你需要思考读者在阅读过程中可能产生的疑问，并在世界观中提前准备。重点关注：
    - 现实层（Observable Layer）：世界的可见规则、社会结构、技术
    - 隐藏层（Hidden Layer）：潜在力量、秘密组织、未解之谜

    输出风格：
    - 直接输出 JSON 对象；键名自命名（中文或英文均可）；内容紧凑但具体，避免空话。
    - 严禁出现：主角、配角、任务、剧情、反派、弧线、关卡、台词等叙事词，减少出现复杂的数学公式、指标。
    """

    GENERATOR_USER_TEMPLATE = """\
    基于下列元设定（Meta），请生成一个“世界观”的 JSON 对象。
    要求：有强可写性与生长性；包含现实层、隐藏层；附带少量术语表。严禁引入人物与具体剧情。只输出 JSON。
    基于元设定，若有必要，生成一段符合世界观的历史背景介绍作为引导，包含历史大事件与现状概述。
    # Meta
    {meta_json}
    """

    VALIDATOR_SYSTEM = """\
    你是一名“世界观有趣度审阅者”。输入是 Meta（元设定）与 Worldview（世界观），二者均为 JSON。
    你的唯一输出是一个 JSON 数组（array），其中每个元素是一个对象，包含：
    - target_path：修改位置（使用清晰的字段定位描述）
    - suggestion：修改建议（简明、可操作；指出应增加/删减/重述/补充哪些要素）

    硬性约束：
    - 只输出 JSON 数组（不要额外解释、不要 Markdown）。
    - 严禁引入人物、情节、任务、章节或对话等叙事要素；仅针对“世界背景与运行规则”提出修改建议。
    
    想办法提升世界观的有趣度，使用包含但不限于以下策略：
    1) 增加冲突与张力：引入对立势力
    2) 丰富细节与设定：补充独特的文化、技术、生态等
    3) 引入悬念与谜团：设置未解之谜、隐藏力量等
    4) 明确边界与代价：定义规则的限制与违反的风险
    5) 历史与背景：提供世界的起源故事与重大事件
    """

    VALIDATOR_USER_TEMPLATE = """\
    请基于以下 Meta 与 Worldview（均为 JSON）进行审阅，返回一个 JSON 数组。数组内每个元素为一个对象，包含：
    - target_path（修改位置）
    - suggestion（修改建议）

    只输出 JSON 数组，禁止输出除数组以外的任何内容。

    ## Meta
    {meta_json}

    ## Worldview
    {expansion_json}
    """




    def generate_expansion(self, meta: Dict[str, Any], seed: int = None) -> Dict[str, Any]:
        if seed is None:
            seed = self.seed
        random.seed(seed)
        up = self.GENERATOR_USER_TEMPLATE.format(meta_json=json.dumps(meta, ensure_ascii=False, indent=2))
        expansion = self.call_structured_json(  # 使用父类的 call_structured_json 方法
            model=self.STRONG_TEXT_MODEL,
            system_prompt=self.GENERATOR_SYSTEM,
            user_prompt=up,
            json_schema=self.EXPANSION_SCHEMA,
            temperature=0.95
        )
        return expansion

    def review_and_revise(self, meta: Dict[str, Any], expansion: Dict[str, Any]) -> Dict[str, Any]:
        review_schema: Dict[str, Any] = {
            "name": "WorldviewReview",
            "schema": {
                "type": "object",
                "properties": {
                    "reviews": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "target_path": {"type": "string"},
                                "suggestion": {"type": "string"}
                            },
                            "required": ["issue_type", "target_path", "suggestion"],
                            "additionalProperties": False
                        },
                        "minItems": 1  # 可选：至少给一条
                    }
                },
                "required": ["reviews"],
                "additionalProperties": False
            }
        }

        up = self.VALIDATOR_USER_TEMPLATE.format(
            meta_json=json.dumps(meta, ensure_ascii=False, indent=2),
            expansion_json=json.dumps(expansion, ensure_ascii=False, indent=2)
        )

        review = self.call_structured_json(
            model=self.STRONG_TEXT_MODEL,
            system_prompt=self.VALIDATOR_SYSTEM,
            user_prompt=up,
            json_schema=review_schema,
            temperature=0.4
        )
        return review

    def assemble_final(self, draft: Dict[str, Any], suggestions: Any) -> Dict[str, Any]:
        """
        输入：
            draft: 世界观草稿（JSON 对象）
            suggestions: 审阅建议数组（每项包含 target_path / suggestion）
        输出：
            revised_final: 根据建议修订后的世界观（严格为 JSON，对 FINAL_SCHEMA 做结构约束）
        说明：
            - 不引入人物/剧情；仅修改“世界背景与运行规则”层。
            - 尽量“就地修订”：在 target_path 指定的节点进行补充、精炼或改写；
              若路径不存在，允许在最邻近可定位的上层键下新增合理子键。
            - 优先落实：边界、代价、失败模式、可观测信号、开放问题等“可写性杠杆”。
            - 保持原有命名与结构风格，避免无必要的字段大改名。
        """
        # 构造用于应用建议的 System 与 User 提示词
        patcher_system = """\
    你是一名“世界观补丁工程师”。你的任务是将审阅建议数组有序地应用到给定的世界观草稿上，产出修订版世界观（JSON）。
    必须遵守：
    1) 仅修改“世界背景与运行规则”；严禁引入人物、情节、任务、对话等叙事要素。
    2) 优先在 target_path 指向的位置进行就地修订
    3) 保留原有术语与结构风格，不做无谓重命名；必要时可在原字段下增加小型子结构以承载新信息。
    4) 只输出 JSON 对象；不要 Markdown、不要解释文字。
    """

        # 为了让模型可定位与改写，这里把 draft 和 suggestions 直接放入 user 提示
        patcher_user = """\
    # DRAFT (JSON)
    {draft_json}

    # SUGGESTIONS (JSON Array)
    {suggestions_json}

    # 产出要求
    - 输出修订后的“世界观 JSON 对象”（完整体，不只是差异）。
    - 尽量减少冗余与空话，保持信息密度与可写性。
    """.format(
            draft_json=json.dumps(draft, ensure_ascii=False, indent=2),
            suggestions_json=json.dumps(suggestions, ensure_ascii=False, indent=2),
        )

        # 让强模型做结构化修订，并用 FINAL_SCHEMA 做格式约束
        revised_final = self.call_structured_json(
            model=self.STRONG_TEXT_MODEL,
            system_prompt=patcher_system,
            user_prompt=patcher_user,
            json_schema=self.FINAL_SCHEMA,
            temperature=0.3,
        )
        return revised_final

    # =======================
    # Public API
    # =======================
    def run(self) -> Dict[str, Any]:
        """
        流程：
          1) 保存并读取 Meta
          2) 生成世界观草稿 draft_expansion
          3) 调用审阅器得到建议数组 suggestions（仅包含 target_path / suggestion）
          4) 调用 assemble_final(draft, suggestions) 应用补丁，产出最终世界观（用 FINAL_SCHEMA 约束）
        返回：
          - meta_path：保存的 Meta 路径
          - draft_expansion：初次生成的世界观草稿
          - review_suggestions：审阅建议数组（统一三字段）
          - final_worldview：按建议修订后的最终世界观 JSON
        """
        # 1) Meta
        meta = self.example_meta

        # 2) 生成草稿
        draft_expansion = self.generate_expansion(meta, seed=self.seed)

        # 3) 审阅（现在应返回“建议数组”）
        suggestions = self.review_and_revise(meta, draft_expansion)

        review_suggestions = suggestions
        final_worldview = self.assemble_final(draft_expansion, review_suggestions)

        return {
            "draft_expansion": draft_expansion,
            "review_suggestions": review_suggestions,
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
    print(json.dumps(result, ensure_ascii=False, indent=2))
