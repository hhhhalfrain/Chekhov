from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any, Dict, Optional
from agents.director_agent import DirectorAgent
from agents.lore_assistant import LoreAssistant
from agents.outline_planner import OutlinePlanner

class ChapterBootstrapPipeline:
    """阶段A：导演决策 → 设定记忆卡 → 章/节纲要
    负责：
    - 目录准备 output/{task_name}/runtime/chapter_{n}
    - 三个核心产物写盘：director_decision.json / memory_cards.json / chapter_outline.json
    - 每个 Prompt 中嵌入 Meta
    - 第一章使用 Director 特殊提示词
    - 设定助理会尝试读取 output/{task_name}/update.json 作为上一章增量
    """
    def __init__(self, *, env_path: str, task_name: str, chapter_index: int,
                 meta: Dict[str, Any], worldview: Dict[str, Any],
                 characters: Optional[Dict[str, Any]] = None,
                 conflicts: Optional[Dict[str, Any]] = None,
                 seed: Optional[int] = None):
        self.env_path = env_path
        self.task_name = task_name
        self.chapter_index = chapter_index
        self.meta = meta
        self.worldview = worldview
        self.characters = characters or {}
        self.conflicts = conflicts or {}
        self.seed = seed

        self.base_dir = Path(f"output/{task_name}")
        self.runtime_dir = self.base_dir / f"runtime/chapter_{chapter_index}"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

        self.prev_update_path = str(self.base_dir / "update.json")  # 若存在则纳入考虑

    # 可选：上一章摘要（如果你在别处生成过）
    def _load_prev_summary(self) -> str:
        p = self.base_dir / f"runtime/chapter_{self.chapter_index-1}/summary.txt"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def run(self) -> Dict[str, Any]:
        prev_summary = self._load_prev_summary() if self.chapter_index > 1 else ""

        # 1) 导演决策
        director = DirectorAgent(self.env_path, seed=self.seed)
        director_decision = director.run(
            meta=self.meta,
            worldview=self.worldview,
            characters=self.characters,
            conflicts=self.conflicts,
            prev_chapter_summary=prev_summary,
            chapter_index=self.chapter_index,
        )
        (self.runtime_dir / "director_decision.json").write_text(
            json.dumps(director_decision, ensure_ascii=False, indent=2), encoding="utf-8")

        # 2) 设定记忆卡
        lore = LoreAssistant(self.env_path, seed=self.seed)
        memory_cards = lore.run(
            meta=self.meta,
            director_decision=director_decision,
            worldview=self.worldview,
            characters=self.characters,
            conflicts=self.conflicts,
            update_json_path=self.prev_update_path
        )
        (self.runtime_dir / "memory_cards.json").write_text(
            json.dumps(memory_cards, ensure_ascii=False, indent=2), encoding="utf-8")

        # 3) 章/节纲要
        planner = OutlinePlanner(self.env_path, seed=self.seed)
        chapter_outline = planner.run(
            meta=self.meta,
            director_decision=director_decision,
            memory_cards=memory_cards
        )
        (self.runtime_dir / "chapter_outline.json").write_text(
            json.dumps(chapter_outline, ensure_ascii=False, indent=2), encoding="utf-8")

        # 汇总 runtime 索引
        index = {
            "task_name": self.task_name,
            "chapter_index": self.chapter_index,
            "artifacts": {
                "director_decision": str(self.runtime_dir / "director_decision.json"),
                "memory_cards": str(self.runtime_dir / "memory_cards.json"),
                "chapter_outline": str(self.runtime_dir / "chapter_outline.json"),
            }
        }
        (self.runtime_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        return index

