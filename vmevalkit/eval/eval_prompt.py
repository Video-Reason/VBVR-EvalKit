"""
Prompt guidance for evaluation.
"""

TASK_PROMPTS = {
    "chess_task": "Check if the final board position matches the expected position after the correct move.",
    "maze_task": "Verify that the final frame shows a complete path from start to end that matches the expected solution.",
    "rotation_task": "Check if the final rotation angle and position match the expected result.",
    "raven_task": "Verify that the pattern completion in the final frame matches the expected pattern.",
    "sudoku_task": "Check if the numbers placed in the final frame match the expected solution.",
    "counting_objects_task": "Check if the count shown in the final frame matches the ground_truth_count. Award 1 point if counts match, 0 otherwise.",
    "letter_counting_task": "Check if the count shown in the final frame matches the ground_truth_count for the target letter. Award 1 point if counts match, 0 otherwise.",
    "subway_pathfinding_task": "Check if the agent icon in the final frame is at the correct destination_station. Award 1 point if destination matches, 0 otherwise."
    "object_subtraction_task": "Verify that the specified object(s) have been correctly removed from the scene, while other objects remain unchanged and the scene remains complete.",
    "object_permanence_task": "Verify that the object(s) remain unchanged in position, color, and shape, and the occluder is moved out of the frame.",
    "light_sequence_task": "Verify that the correct lights are on and all other lights are off in the final frame.",
    "tower_of_hanoi_task": "Check if exactly one disk moved between frames. Verify the move is legal (top disk moved to empty peg or larger disk). Compare final disk positions to expected."
}
