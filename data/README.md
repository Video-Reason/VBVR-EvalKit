# Data Format

VMEvalKit expects reasoning task data in this structure:

```
data/questions/{domain}_task/{task_id}/
├── first_frame.png          # Initial state image
├── final_frame.png          # Target state image  
├── prompt.txt              # Text instructions
└── ground_truth.mp4        # Optional ground truth video
```

**Example:**
```
data/questions/chess_task/chess_0001/
├── first_frame.png          # Chess board position
├── final_frame.png          # Board after checkmate move
├── prompt.txt              # "White to move and checkmate"
└── ground_truth.mp4        # Optional solution video
```

That's it! VMEvalKit will auto-discover domains and tasks from these folders.