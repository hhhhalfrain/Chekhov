# python
import os
import json
from pathlib import Path
from worldview_generator import WorldviewGenerator
from character_generator import CharacterGenerator
from conflict_generator import ConflictGenerator
from pipelines.chapter_bootstrap import ChapterBootstrapPipeline

def check_and_continue(file_path: Path) -> bool:
    """检查文件是否存在，若存在则跳过生成流程"""
    if file_path.exists():
        print(f"File already exists at: {file_path}")
        return False
    return True

if __name__ == "__main__":
    # 初始输入
    example_meta = {
        "genre_tone": "悬疑",
        "audience_rating": "青年～成人",
        "inspirations": ["盗墓笔记"],
        "era_power_level": "现代社会",
        "medium": "小说",
        "language": "中文",
    }

    # 配置环境变量路径和随机种子
    env_path = ".env"
    seed = 20251029

    # 定义输出路径
    task_name = "example_task_2"  # 可根据需要修改任务名
    output_dir = Path(f"output/{task_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存元输入到 output/task_name/meta.json
    meta_path = output_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(example_meta, f, ensure_ascii=False, indent=2)
    print(f"Meta saved to: {meta_path}")

    worldview_path = output_dir / "worldview.json"

    # 检查文件是否已存在,若存在则跳过生成流程
    if check_and_continue(worldview_path):
        # 创建 WorldviewGenerator 实例
        worldview_generator = WorldviewGenerator(example_meta, env_path, seed)

        # 执行生成流程
        result = worldview_generator.run()

        # 保存结果到文件
        with worldview_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"Worldview saved to: {worldview_path}")

    # 读取
    with open(worldview_path, "r", encoding="utf-8") as f:
        worldview_payload = json.load(f)
    final_worldview = worldview_payload.get("final_worldview", worldview_payload)

    # 生成角色设定
    characters_path = output_dir / "characters.json"
    # 检查角色设定文件是否已存在,若存在则跳过生成流程
    if check_and_continue(characters_path):
        # 读取 Final Worldview
        with worldview_path.open("r", encoding="utf-8") as f:
            worldview_payload = json.load(f)
        final_worldview = worldview_payload.get("final_worldview", worldview_payload)  # 兼容两种包装

        # 生成角色
        char_gen = CharacterGenerator(env_path=env_path
                                      ,seed=seed
                                      ,worldview=final_worldview
                                      ,meta=example_meta)
        char_result = char_gen.run()

        with characters_path.open("w", encoding="utf-8") as f:
            json.dump(char_result, f, ensure_ascii=False, indent=2)

        print(f"Characters saved to: {characters_path}")
    with open(characters_path, "r", encoding="utf-8") as f:
        characters_payload = json.load(f)
    final_characters = characters_payload.get("final_characters", characters_payload)

    # 生成矛盾网络
    conflicts_path = output_dir / "conflicts.json"
    if check_and_continue(conflicts_path):

        # 生成矛盾网络
        conf_gen = ConflictGenerator(env_path=env_path, worldview=final_worldview, characters=final_characters,
                                     seed=seed)
        conf_result = conf_gen.run()

        # 保存
        with conflicts_path.open("w", encoding="utf-8") as f:
            json.dump(conf_result, f, ensure_ascii=False, indent=2)
        print(f"Conflicts saved to: {conflicts_path}")
    with open(conflicts_path, "r", encoding="utf-8") as f:
        conf_result = json.load(f)
    final_conflicts = conf_result.get("final_conflicts", conf_result)

    chapter_n = 1

    pipeline = ChapterBootstrapPipeline(
        env_path=env_path,
        task_name=task_name,
        chapter_index=chapter_n,
        meta=example_meta,
        worldview=final_worldview,
        characters=final_characters,
        conflicts=final_conflicts,
        seed=seed,
    )
    index = pipeline.run()

    # ---- 存储总览 ----
    runtime_dir = Path(index["artifacts"]["chapter_outline"]).parent
    summary = {
        "task_name": task_name,
        "chapter_index": chapter_n,
        "runtime_dir": str(runtime_dir),
        "artifacts": index["artifacts"],
    }
    print("\n[Stage-A Complete] Artifacts saved:")
    for k, v in summary["artifacts"].items():
        print(f"- {k}: {v}")
