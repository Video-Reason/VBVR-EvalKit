#!/usr/bin/env python3
"""
生成 Object Subtraction 任务的脚本
使用改进后的代码生成所有四个级别（L1, L2, L3, L4）的任务
"""

import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from vmevalkit.tasks.object_subtraction_task import create_dataset

# 配置：每个级别生成多少个任务
SAMPLES_PER_LEVEL = {
    "L1": 25,  # Level 1: 25个任务
    "L2": 25,  # Level 2: 25个任务
    "L3": 25,  # Level 3: 25个任务
    "L4": 25,  # Level 4: 25个任务
}

# 或者生成更多任务
# SAMPLES_PER_LEVEL = {
#     "L1": 50,
#     "L2": 50,
#     "L3": 50,
#     "L4": 50,
# }

output_dir = Path("data/questions/object_subtraction_task")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("🎯 生成 Object Subtraction 任务")
print("=" * 70)
print(f"📁 输出目录: {output_dir}")
print(f"📊 每个级别的任务数:")
for level, num in SAMPLES_PER_LEVEL.items():
    print(f"   {level}: {num} 个任务")
print()

total_generated = 0

# 为每个级别生成任务
for level, num_samples in SAMPLES_PER_LEVEL.items():
    print(f"\n{'='*70}")
    print(f"📊 生成 Level {level} 任务: {num_samples} 个")
    print(f"{'='*70}")
    
    # 生成数据集
    dataset = create_dataset(num_samples=num_samples, levels=[level])
    
    # 保存到文件夹
    base_dir = project_root
    for pair in dataset['pairs']:
        task_id = pair.get("id", f"object_subtraction_{level.lower()}_{total_generated:04d}")
        pair['id'] = task_id
        pair['domain'] = "object_subtraction"
        
        # 创建任务目录
        task_dir = output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制图片文件
        first_image_path = pair.get("first_image_path")
        final_image_path = pair.get("final_image_path")
        
        if first_image_path:
            src_first = base_dir / first_image_path
            dst_first = task_dir / "first_frame.png"
            if src_first.exists():
                shutil.copyfile(src_first, dst_first)
                pair['first_image_path'] = f"object_subtraction_task/{task_id}/first_frame.png"
            else:
                print(f"   ⚠️  警告: 找不到图片 {src_first}")
        
        if final_image_path:
            src_final = base_dir / final_image_path
            dst_final = task_dir / "final_frame.png"
            if src_final.exists():
                shutil.copyfile(src_final, dst_final)
                pair['final_image_path'] = f"object_subtraction_task/{task_id}/final_frame.png"
            else:
                print(f"   ⚠️  警告: 找不到图片 {src_final}")
        
        # 保存 prompt
        prompt_text = pair.get("prompt", "")
        (task_dir / "prompt.txt").write_text(prompt_text)
        
        # 保存 metadata
        pair['created_at'] = datetime.now().isoformat() + 'Z'
        metadata_path = task_dir / "question_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(pair, f, indent=2, default=str, ensure_ascii=False)
        
        total_generated += 1
        
        if total_generated % 10 == 0:
            print(f"   ✅ 已生成 {total_generated} 个任务...")
    
    print(f"   ✅ Level {level} 完成: {num_samples} 个任务")

print(f"\n{'='*70}")
print(f"🎉 任务生成完成!")
print(f"   📁 总任务数: {total_generated}")
print(f"   📂 保存位置: {output_dir}")
print(f"{'='*70}")

