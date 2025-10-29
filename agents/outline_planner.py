from __future__ import annotations
import json
from typing import Any, Dict, List
from llm_base import LLMBase

OUTLINE_SCHEMA: Dict[str, Any] = {
    "name": "ChapterOutline",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "chapter_goal": {"type": "string"},
            "sections": {
                "type": "array",
                "minItems": 4,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "target_words": {"type": "integer"},
                        "section_goal": {"type": "string"},
                        "conflict_hook": {"type": "string"},
                        "pov": {"type": "string"},
                        "foreshadow_slots": {"type": "array", "items": {"type": "string"}},
                        "noise_budget": {"type": "string"},
                    },
                    "required": ["id", "target_words", "section_goal", "conflict_hook", "pov"]
                }
            },
            "notes": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["chapter_goal", "sections"]
    }
}

class OutlinePlanner(LLMBase):
    SYSTEM = """
你是纲要师。任务：基于导演决策与记忆卡，产出本章与各节纲要（4–8节，目标字数≈2000/节）。
- 不生成正文；只输出结构。
- 每节明确：section_goal / conflict_hook / pov / foreshadow_slots / noise_budget。
"""

    USER_TEMPLATE = """
## Meta（关键，必须遵循）
{meta_json}

## 导演决策
{director_json}

## 记忆卡（当章必需设定）
{cards_json}

请输出 ChapterOutline（严格遵循 Schema）。
"""

    def run(self, *, meta: Dict[str, Any], director_decision: Dict[str, Any], memory_cards: Dict[str, Any]) -> Dict[str, Any]:
        up = self.USER_TEMPLATE.format(
            meta_json=json.dumps(meta, ensure_ascii=False, indent=2),
            director_json=json.dumps(director_decision, ensure_ascii=False, indent=2),
            cards_json=json.dumps(memory_cards, ensure_ascii=False, indent=2)
        )
        outline = self.call_structured_json(
            model=self.STRONG_TEXT_MODEL,
            system_prompt=self.SYSTEM,
            user_prompt=up,
            json_schema=OUTLINE_SCHEMA,
            temperature=0.55
        )
        return outline

