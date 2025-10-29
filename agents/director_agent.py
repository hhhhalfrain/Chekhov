from __future__ import annotations
import json
from typing import Any, Dict, Optional
from llm_base import LLMBase

DIRECTOR_DECISION_SCHEMA: Dict[str, Any] = {
    "name": "DirectorDecision",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "writing_style": {"type": "string"},              # 写作手法：如 in media res / 多视角拼贴 / 慢燃悬疑
            "focalization": {"type": "string"},               # 视角策略
            "tone_curve": {"type": "string"},                 # 情绪曲线文字描述或参数化说明
            "info_budget": {"type": "integer", "minimum": 1},# 每节硬设定灌入上限
            "conflict_focus": {"type": "string"},            # 本章冲突入口
            "notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["writing_style", "focalization", "tone_curve", "info_budget", "conflict_focus"]
    }
}

class DirectorAgent(LLMBase):
    SYSTEM = """
你是章节导演。目标：决定本章的写作手法、视角策略、情绪曲线与信息滴灌预算，并明确冲突入口。
必须遵守：
1) 不生成剧情正文；只输出决策参数。
2) 以“张力最大化+因果可追溯”为准绳；避免万能视角导致的信息过载。
3) 输出字段必须符合 DirectorDecision JSON Schema。
"""

    USER_TEMPLATE_GENERIC = """
## Meta（关键，必须遵循）
{meta_json}

## 输入素材
- 上一章摘要（可空）：\n{prev_summary}\n
- 世界观 Final（长文，按需参考）：\n{worldview_json}\n
- 角色与矛盾（长文，按需参考）：\n{chars_conflicts}\n
请基于以上，产出本章的导演决策。
"""

    # 第一章的特殊提示词：必须向读者建立世界背景的最低可理解面
    USER_TEMPLATE_CH1 = """
## Meta（关键，必须遵循）
{meta_json}

你正在为“第一章”做导演决策。要求：
1) 全量读取世界观与角色信息，判断读者理解所需的最低背景，并将此转化为 info_budget 建议（不过载）。
2) 写作手法需兼顾引子：允许 in media res，但必须设计最低背景可推断路径。
3) 明确冲突入口（conflict_focus），保证人物能动性驱动场景。
4) 给出 3-6 条 notes，说明第一章在读者侧的“背景建立策略”。

## 世界观 Final：\n{worldview_json}

## 角色与矛盾总览：\n{chars_conflicts}
"""

    def run(self, *, meta: Dict[str, Any], worldview: Dict[str, Any],
            characters: Optional[Dict[str, Any]], conflicts: Optional[Dict[str, Any]],
            prev_chapter_summary: str = "", chapter_index: int = 1) -> Dict[str, Any]:
        meta_json = json.dumps(meta, ensure_ascii=False, indent=2)
        world_json = json.dumps(worldview, ensure_ascii=False, indent=2)
        chars_conf = json.dumps({
            "characters": characters or {},
            "conflicts": conflicts or {}
        }, ensure_ascii=False, indent=2)

        if chapter_index == 1:
            up = self.USER_TEMPLATE_CH1.format(
                meta_json=meta_json,
                worldview_json=world_json,
                chars_conflicts=chars_conf
            )
        else:
            up = self.USER_TEMPLATE_GENERIC.format(
                meta_json=meta_json,
                prev_summary=prev_chapter_summary or "",
                worldview_json=world_json,
                chars_conflicts=chars_conf
            )
        decision = self.call_structured_json(
            model=self.STRONG_TEXT_MODEL,
            system_prompt=self.SYSTEM,
            user_prompt=up,
            json_schema=DIRECTOR_DECISION_SCHEMA,
            temperature=0.6
        )
        return decision

