

## 📊 Supported Datasets

VMEvalKit provides access to **9 local task generation engines(quickly increasing)** and other external benchmark datasets (HuggingFace) [here](docs/tasks/README.md).

### Local Task Generation Engines

| Task | Description | Generation Method |
|------|-------------|-------------------|
| **Chess** | Strategic thinking & tactical patterns | Chess engine with mate-in-1 puzzles |
| **Maze** | Path-finding & navigation | Procedural maze generation (Kruskal's algorithm) |
| **Raven** | Abstract reasoning matrices | RAVEN dataset patterns |
| **Rotation** | 3D mental rotation | Procedural 3D object generation |
| **Sudoku** | Logical constraint satisfaction | Sudoku puzzle generator |
| **Object Subtraction** | Selective object removal | Multi-level cognitive reasoning |
| **Clock** | Time-based reasoning | Clock time increment |

### External Benchmarks (HuggingFace)

| Dataset | Tasks | Domains | Key Features |
|---------|-------|---------|--------------|
| **VideoThinkBench** | ~4,000 | 4 subsets | Vision-centric (ARC-AGI, Eyeballing, Visual Puzzles) + Text-centric reasoning |
| **MME-CoF** | 59 | 16 domains | Video Chain-of-Frame reasoning across cognitive domains |



**VideoThinkBench Subsets:**
- `arc_agi_2` - Abstract reasoning (1,000 tasks)
- `eyeballing_puzzles` - Visual estimation (1,050 tasks)  
- `visual_puzzles` - Pattern recognition (496 tasks)
- `text_centric_tasks` - Math & multimodal reasoning (1,453 tasks)



### Sync with Cloud
```bash
# AWS S3 (enterprise backup)
python data/s3_sync.py --log
```

**Tips:**
- Use `--task-id chess_0001` to run specific questions  