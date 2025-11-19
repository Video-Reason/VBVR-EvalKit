# 📋 本地文件分类总结

## ❌ 不上传GitHub的文件（已添加到.gitignore）

### 开发和分析文档（临时文档）
这些是开发过程中的分析文档，只保留在本地：

1. **`ADDING_MORPHIC_MODEL_GUIDE.md`**
   - 添加Morphic模型的指南（开发文档）

2. **`IMPLEMENTATION_CHECK.md`**
   - 实现检查文档（开发过程中的检查清单）

3. **`SLIDING_PUZZLE_IMPLEMENTATION_PLAN.md`**
   - Sliding Puzzle实现计划（开发计划文档）

4. **`vmevalkit/tasks/ALL_TASKS_ANALYSIS.md`**
   - 所有任务的分析文档（分析笔记）

5. **`vmevalkit/tasks/object_subtraction_task/L4_CONCEPTUAL_ABSTRACTION.md`**
   - L4概念抽象分析（分析笔记）

6. **`vmevalkit/tasks/object_subtraction_task/LEVEL_ANALYSIS.md`**
   - 级别分析文档（分析笔记）

7. **`vmevalkit/tasks/object_subtraction_task/SPATIAL_REASONING_ANALYSIS.md`**
   - 空间推理分析（分析笔记）

### 临时脚本
8. **`generate_object_subtraction_tasks.py`**
   - 生成任务的临时脚本

9. **`test_morphic_integration.py`**
   - 测试脚本

---

## ✅ 应该上传GitHub的文件（正式文档）

### 项目主文档
- `README.md` - 项目主文档
- `CONTRIBUTING.md` - 贡献指南

### 任务文档（正式文档）
- `vmevalkit/tasks/chess_task/CHESS.md`
- `vmevalkit/tasks/maze_task/MAZE.md`
- `vmevalkit/tasks/raven_task/RAVEN.md`
- `vmevalkit/tasks/rotation_task/ROTATION.md`
- `vmevalkit/tasks/sudoku_task/SUDOKU.md`
- `vmevalkit/tasks/object_subtraction_task/OBJECT_SUBTRACTION.md` ✅
- `vmevalkit/tasks/sliding_puzzle_task/SLIDING_PUZZLE.md` ✅
- `vmevalkit/tasks/causality/CASUALITY.md`

### 文档目录（docs/）
- `docs/ADDING_TASKS.md`
- `docs/ADDING_MODELS.md`
- `docs/DATA_MANAGEMENT.md`
- `docs/INFERENCE.md`
- `docs/SCORING.md`
- `docs/THEORY.md`
- `docs/WEB_DASHBOARD.md`

### 其他文档
- `data/data_logging/README.md`
- `web/README.md`
- `paper/README.md`
- `examples/opensource/open_source.md`
- `submodules/*/README.md` (子模块文档)

---

## 📝 总结

**不上传的文件特点：**
- 开发过程中的分析文档
- 临时实现计划
- 开发检查清单
- 临时测试脚本

**上传的文件特点：**
- 正式的任务文档（如 `OBJECT_SUBTRACTION.md`, `SLIDING_PUZZLE.md`）
- 项目主文档和贡献指南
- 用户文档（docs/目录）
- 子模块文档

**已更新 `.gitignore`**，上述不上传的文件已被忽略。
