# =============================
# agents/lore_assistant.py
# =============================
from __future__ import annotations
import json, os
from typing import Any, Dict, Optional
from llm_base import LLMBase

MEMORY_CARDS_SCHEMA: Dict[str, Any] = {
    "name": "MemoryCards",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "must_have_facts": {"type": "array", "items": {"type": "string"}},
            "volatile_risks": {"type": "array", "items": {"type": "string"}},
            "diction_guides": {"type": "array", "items": {"type": "string"}},
            "prior_updates": {"type": "array", "items": {"type": "string"}},  # 来自上一章 update.json
        },
        "required": ["must_have_facts", "volatile_risks"]
    }
}

class LoreAssistant(LLMBase):
    SYSTEM = """
你是设定助理。任务：在极大设定中筛出“当章必需记忆卡”，避免信息过载。
约束：
- 不生成正文；仅输出记忆卡。
- 优先包含：硬设定约束、术语使用规范（必要时）、可能冲突的设定警告。
- 若检测到上一章 update.json，则将可继承的变更摘要并入 prior_updates。
"""

    USER_TEMPLATE = """
## Meta（关键，必须遵循）
{meta_json}

## 导演决策
{director_json}

## 设定来源
- 世界观：\n{worldview_json}
- 角色与矛盾：\n{chars_conflicts}
- 上一章更新（可空）：\n{prev_update}

请返回 MemoryCards（严格遵循 Schema）。
"""

    def run(self, *, meta: Dict[str, Any], director_decision: Dict[str, Any],
            worldview: Dict[str, Any], characters: Optional[Dict[str, Any]], conflicts: Optional[Dict[str, Any]],
            update_json_path: Optional[str]) -> Dict[str, Any]:
        meta_json = json.dumps(meta, ensure_ascii=False, indent=2)
        director_json = json.dumps(director_decision, ensure_ascii=False, indent=2)
        world_json = json.dumps(worldview, ensure_ascii=False, indent=2)
        chars_conf = json.dumps({"characters": characters or {}, "conflicts": conflicts or {}}, ensure_ascii=False, indent=2)

        prev_update = {}
        if update_json_path and os.path.exists(update_json_path):
            try:
                with open(update_json_path, "r", encoding="utf-8") as f:
                    prev_update = json.load(f)
            except Exception:
                prev_update = {"_warn": "failed_to_load_update_json"}
        prompt = self.USER_TEMPLATE.format(
            meta_json=meta_json,
            director_json=director_json,
            worldview_json=world_json,
            chars_conflicts=chars_conf,
            prev_update=json.dumps(prev_update, ensure_ascii=False, indent=2)
        )
        cards = self.call_structured_json(
            model=self.STRONG_TEXT_MODEL,
            system_prompt=self.SYSTEM,
            user_prompt=prompt,
            json_schema=MEMORY_CARDS_SCHEMA,
            temperature=0.4
        )
        return cards

