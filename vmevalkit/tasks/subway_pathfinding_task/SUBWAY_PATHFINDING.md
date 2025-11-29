# Subway Pathfinding Task

**Source**: Adapted from Tin's [simple_task_video_reasoning](https://github.com/tin-xai/simple_task_video_reasoning)

## Description

Spatial navigation task where models must track an agent's journey through a subway network from source station to destination station.

## Task Details

- **First Frame**: Complete subway map with multiple colored paths connecting stations A, B, C, D
- **Final Frame**: Same map with agent icon (red circle with yellow star) at destination
- **Goal**: Determine which station the agent reaches and which path was taken

## Parameters (from Tin's original)

- **Stations**: A (top), B (right), C (bottom), D (left)
- **Image sizes**: 512×512, 1024×1024
- **Line thickness**: 10, 20
- **Routes**: Automatically generated with 1-3 paths per station
- **Colors**: tab10 colormap for path differentiation

## Ground Truth

Each sample includes:
- `source_station`: Starting station (A/B/C/D)
- `destination_station`: Ending station (A/B/C/D)
- `path_color`: Color of the path connecting them
- `metadata`: All connections, route details, and coordinates

## Evaluation

Award 1 point if the model correctly identifies the destination_station, 0 otherwise.
This tests spatial navigation and path tracking ability.

