# python
import os
import json
from pathlib import Path
from worldview_generator import WorldviewGenerator
from character_generator import CharacterGenerator

def check_and_continue(file_path: Path) -> bool:
    """检查文件是否存在，若存在则跳过生成流程"""
    if file_path.exists():
        print(f"File already exists at: {file_path}")
        return False
    return True

if __name__ == "__main__":
    # 初始输入
    example_meta = {
        "genre_tone": "硬科幻 · 冷峻",
        "audience_rating": "青年～成人",
        "inspirations": ["三体"],
        "era_power_level": "星际早期扩张，亚光速为主",
        "medium": "小说",
        "language": "中文",
    }

    # 配置环境变量路径和随机种子
    env_path = ".env"
    seed = 20251029

    # 定义输出路径
    task_name = "example_task"  # 可根据需要修改任务名
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
