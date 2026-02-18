"""
VBVR-Bench Evaluators Module

This module provides task-specific evaluators for all 100 VBVR-Bench tasks.
Each evaluator implements rule-based evaluation following documented criteria.
"""

from typing import Dict, Any, Optional
from .base_evaluator import BaseEvaluator

# Import specialized Hidden_40 evaluators (Part 1)
from .hidden40_evaluators import (
    MultipleKeysForOneDoorEvaluator,
    SelectNextFigureAlternatingEvaluator,
    LocatePointInOverlappingAreaEvaluator,
    LocateTopmostFigureEvaluator as Hidden40LocateTopmostFigureEvaluator,
    IdentifyUniqueFigureEvaluator as Hidden40IdentifyUniqueFigureEvaluator,
    CircleLargestNumericalValueEvaluator as Hidden40CircleLargestEvaluator,
    MarkSecondLargestShapeEvaluator as Hidden40MarkSecondLargestEvaluator,
    SelectLongestPolygonSideEvaluator as Hidden40SelectLongestSideEvaluator,
    ArrangeCirclesByCircumferenceEvaluator as Hidden40ArrangeCirclesEvaluator,
    DrawNextSizedShapeEvaluator as Hidden40DrawNextShapeEvaluator,
)

# Import specialized Hidden_40 evaluators (Part 2)
from .hidden40_evaluators_part2 import (
    MarkWavePeaksEvaluator,
    FindIncorrectArrowDirectionEvaluator,
    CircleCentralDotEvaluator,
    IdentifyLargestAngleEvaluator,
    SelectLeftmostShapeEvaluator,
    OutlineInnermostSquareEvaluator,
    AddBordersToUnborderedEvaluator,
    IdentifyChineseCharacterEvaluator,
    MarkAsymmetricalShapeEvaluator,
    HighDensityLiquidEvaluator,
)

# Import specialized Hidden_40 evaluators (Part 3)
from .hidden40_evaluators_part3 import (
    PigmentColorMixingEvaluator as Hidden40PigmentMixingEvaluator,
    SymbolDeletionEvaluator as Hidden40SymbolDeletionEvaluator,
    GeometricTransformationEvaluator as Hidden40GeometricTransformEvaluator,
    ShapeScalingAnalogyEvaluator as Hidden40ShapeScalingEvaluator,
    ShapeColorThenMoveEvaluator as Hidden40ShapeColorMoveEvaluator,
    ConstructionStackEvaluator as Hidden40ConstructionStackEvaluator,
    MazePathfindingEvaluator as Hidden40MazeEvaluator,
    ObjectSubtractionEvaluator as Hidden40ObjectSubtractionEvaluator,
    ShapeSorterEvaluator as Hidden40ShapeSorterEvaluator,
    SymmetryCompletionEvaluator as Hidden40SymmetryEvaluator,
)

# Import specialized Hidden_40 evaluators (Part 4)
from .hidden40_evaluators_part4 import (
    ControlPanelEvaluator as Hidden40ControlPanelEvaluator,
    RavenMatrixEvaluator as Hidden40RavenMatrixEvaluator,
    SymbolDeleteEvaluator as Hidden40SymbolDeleteEvaluator,
    SymbolInsertEvaluator as Hidden40SymbolInsertEvaluator,
    SymbolSubstituteEvaluator as Hidden40SymbolSubstituteEvaluator,
    SymbolEditConstraintEvaluator as Hidden40SymbolEditEvaluator,
    GravityPhysicsEvaluator as Hidden40GravityPhysicsEvaluator,
    AnimalMatchingEvaluator as Hidden40AnimalMatchingEvaluator,
    AnimalSizeSortingEvaluator as Hidden40AnimalSizeSortingEvaluator,
    ObjectRotation2DEvaluator as Hidden40ObjectRotation2DEvaluator,
)

# Import specialized Open_60 evaluators (Part 1)
from .open60_evaluators import (
    StableSortEvaluator as Open60StableSortEvaluator,
    MultiObjectPlacementEvaluator as Open60MultiObjectPlacementEvaluator,
    TrackObjectMovementEvaluator as Open60TrackObjectMovementEvaluator,
    IdentifyObjectsInRegionEvaluator as Open60IdentifyObjectsInRegionEvaluator,
    GridNumberSequenceEvaluator as Open60GridNumberSequenceEvaluator,
    GridAvoidObstaclesEvaluator as Open60GridAvoidObstaclesEvaluator,
    GridGoThroughBlockEvaluator as Open60GridGoThroughBlockEvaluator,
    SeparateObjectsNoSpinEvaluator as Open60SeparateObjectsNoSpinEvaluator,
    GridShortestPathEvaluator as Open60GridShortestPathEvaluator,
    MultipleOcclusionsVerticalEvaluator as Open60MultipleOcclusionsEvaluator,
)

# Import specialized Open_60 evaluators (Part 2)
from .open60_evaluators_part2 import (
    SeparateObjectsSpinningEvaluator as Open60SeparateObjectsSpinningEvaluator,
    ChartExtremeEvaluator as Open60ChartExtremeEvaluator,
    DirectedGraphNavigationEvaluator as Open60DirectedGraphNavigationEvaluator,
    AttentionShiftEvaluator as Open60AttentionShiftEvaluator,
    GridHighestCostEvaluator as Open60GridHighestCostEvaluator,
    UnderstandSceneStructureEvaluator as Open60UnderstandSceneStructureEvaluator,
    KeyDoorMatchingEvaluator as Open60KeyDoorMatchingEvaluator,
    PredictNextColorEvaluator as Open60PredictNextColorEvaluator,
    ConnectingColorEvaluator as Open60ConnectingColorEvaluator,
    SelectNextFigureIncreasingEvaluator as Open60SelectNextFigureIncreasingEvaluator,
)

# Import specialized Open_60 evaluators (Part 3)
from .open60_evaluators_part3 import (
    SelectNextFigureLargeSmallEvaluator as Open60SelectNextFigureLargeSmallEvaluator,
    SpotUniqueColorEvaluator as Open60SpotUniqueColorEvaluator,
    IdentifyAllHollowPointsEvaluator as Open60IdentifyAllHollowPointsEvaluator,
    IdentifyNearestSquareRectangleEvaluator as Open60IdentifyNearestSquareRectangleEvaluator,
    LocateSegmentIntersectionEvaluator as Open60LocateSegmentIntersectionEvaluator,
    DrawMidpointPerpendicularEvaluator as Open60DrawMidpointPerpendicularEvaluator,
    ConstructConcentricRingEvaluator as Open60ConstructConcentricRingEvaluator,
    IdentifyPentagonsEvaluator as Open60IdentifyPentagonsEvaluator,
    MarkTangentPointEvaluator as Open60MarkTangentPointEvaluator,
    HighlightHorizontalLinesEvaluator as Open60HighlightHorizontalLinesEvaluator,
)

# Import specialized Open_60 evaluators (Part 4)
from .open60_evaluators_part4 import (
    ColorTripleIntersectionEvaluator as Open60ColorTripleIntersectionEvaluator,
    ShapeOutlineFillEvaluator as Open60ShapeOutlineFillEvaluator,
    ShapeColorThenScaleEvaluator as Open60ShapeColorThenScaleEvaluator,
    ShapeOutlineThenMoveEvaluator as Open60ShapeOutlineThenMoveEvaluator,
    ShapeScaleThenOutlineEvaluator as Open60ShapeScaleThenOutlineEvaluator,
    BallBounceEvaluator as Open60BallBounceEvaluator,
    ColorAdditionEvaluator as Open60ColorAdditionEvaluator,
    ConstructionBlueprintEvaluator as Open60ConstructionBlueprintEvaluator,
    GlassRefractionEvaluator as Open60GlassRefractionEvaluator,
    MirrorReflectionEvaluator as Open60MirrorReflectionEvaluator,
)

# Import specialized Open_60 evaluators (Part 5)
from .open60_evaluators_part5 import (
    DominoChainBranchEvaluator as Open60DominoChainBranchEvaluator,
    DominoChainGapEvaluator as Open60DominoChainGapEvaluator,
    LEGOConstructionEvaluator as Open60LEGOConstructionEvaluator,
    MoveObjectsToTargetEvaluator as Open60MoveObjectsToTargetEvaluator,
    BallColorEvaluator as Open60BallColorEvaluator,
    BookshelfEvaluator as Open60BookshelfEvaluator,
    BallEatingEvaluator as Open60BallEatingEvaluator,
    RollingBallEvaluator as Open60RollingBallEvaluator,
    CountingObjectEvaluator as Open60CountingObjectEvaluator,
    DotToDotEvaluator as Open60DotToDotEvaluator,
)

# Import specialized Open_60 evaluators (Part 6)
from .open60_evaluators_part6 import (
    GridShiftEvaluator as Open60GridShiftEvaluator,
    LightSequenceEvaluator as Open60LightSequenceEvaluator,
    MajorityColorEvaluator as Open60MajorityColorEvaluator,
    RotationPuzzleEvaluator as Open60RotationPuzzleEvaluator,
    SequenceCompletionEvaluator as Open60SequenceCompletionEvaluator,
    SlidingPuzzleEvaluator as Open60SlidingPuzzleEvaluator,
    TrafficLightEvaluator as Open60TrafficLightEvaluator,
    ClockTimeEvaluator as Open60ClockTimeEvaluator,
    RotationEvaluator as Open60RotationEvaluator,
    CommunicatingVesselsEvaluator as Open60CommunicatingVesselsEvaluator,
)


# Task to evaluator mapping
TASK_EVALUATOR_MAP = {
    # Open_60 Tasks (60 tasks)
    'G-3_stable_sort_data-generator': Open60StableSortEvaluator,
    'G-13_grid_number_sequence_data-generator': Open60GridNumberSequenceEvaluator,
    'G-15_grid_avoid_obstacles_data-generator': Open60GridAvoidObstaclesEvaluator,
    'G-16_grid_go_through_block_data-generator': Open60GridGoThroughBlockEvaluator,
    'G-18_grid_shortest_path_data-generator': Open60GridShortestPathEvaluator,
    'G-31_directed_graph_navigation_data-generator': Open60DirectedGraphNavigationEvaluator,
    'G-41_grid_highest_cost_data-generator': Open60GridHighestCostEvaluator,
    'G-131_select_next_figure_increasing_size_sequence_data-generator': Open60SelectNextFigureIncreasingEvaluator,
    'G-134_select_next_figure_large_small_alternating_sequence_data-generator': Open60SelectNextFigureLargeSmallEvaluator,
    'G-51_predict_next_color_data-generator': Open60PredictNextColorEvaluator,
    'G-138_spot_unique_non_repeated_color_data-generator': Open60SpotUniqueColorEvaluator,
    'G-54_connecting_color_data-generator': Open60ConnectingColorEvaluator,
    'G-250_color_triple_intersection_red_data-generator': Open60ColorTripleIntersectionEvaluator,
    'G-158_identify_all_hollow_points_data-generator': Open60IdentifyAllHollowPointsEvaluator,
    'G-168_identify_nearest_to_square_rectangle_data-generator': Open60IdentifyNearestSquareRectangleEvaluator,
    'G-169_locate_intersection_of_segments_data-generator': Open60LocateSegmentIntersectionEvaluator,
    'G-189_draw_midpoint_perpendicular_line_data-generator': Open60DrawMidpointPerpendicularEvaluator,
    'G-194_construct_concentric_ring_data-generator': Open60ConstructConcentricRingEvaluator,
    'G-206_identify_pentagons_data-generator': Open60IdentifyPentagonsEvaluator,
    'G-21_multiple_occlusions_vertical_data-generator': Open60MultipleOcclusionsEvaluator,
    'G-222_mark_tangent_point_of_circles_data-generator': Open60MarkTangentPointEvaluator,
    'G-223_highlight_horizontal_lines_data-generator': Open60HighlightHorizontalLinesEvaluator,
    'G-24_separate_objects_no_spin_data-generator': Open60SeparateObjectsNoSpinEvaluator,
    'G-25_seperate_object_spinning_data-generator': Open60SeparateObjectsSpinningEvaluator,
    'G-29_chart_extreme_with_data_data-generator': Open60ChartExtremeEvaluator,
    'G-39_attention_shift_different_data-generator': Open60AttentionShiftEvaluator,
    'G-43_understand_scene_structure_data-generator': Open60UnderstandSceneStructureEvaluator,
    'G-45_key_door_matching_data-generator': Open60KeyDoorMatchingEvaluator,
    'G-5_multi_object_placement_data-generator': Open60MultiObjectPlacementEvaluator,
    'G-8_track_object_movement_data-generator': Open60TrackObjectMovementEvaluator,
    'G-9_identify_objects_in_region_data-generator': Open60IdentifyObjectsInRegionEvaluator,
    'O-10_shape_outline_fill_data-generator': Open60ShapeOutlineFillEvaluator,
    'O-12_shape_color_then_scale_data-generator': Open60ShapeColorThenScaleEvaluator,
    'O-13_shape_outline_then_move_data-generator': Open60ShapeOutlineThenMoveEvaluator,
    'O-14_shape_scale_then_outline_data-generator': Open60ShapeScaleThenOutlineEvaluator,
    'O-15_ball_bounces_given_time_data-generator': Open60BallBounceEvaluator,
    'O-31_ball_eating_data-generator': Open60BallEatingEvaluator,
    'O-32_rolling_ball_data-generator': Open60RollingBallEvaluator,
    'O-75_communicating_vessels_data-generator': Open60CommunicatingVesselsEvaluator,
    'O-16_color_addition_data-generator': Open60ColorAdditionEvaluator,
    'O-29_ballcolor_data-generator': Open60BallColorEvaluator,
    'O-38_majority_color_data-generator': Open60MajorityColorEvaluator,
    'O-18_glass_refraction_data-generator': Open60GlassRefractionEvaluator,
    'O-19_mirror_reflection_data-generator': Open60MirrorReflectionEvaluator,
    'O-21_construction_blueprint_data-generator': Open60ConstructionBlueprintEvaluator,
    'O-23_domino_chain_branch_path_prediction_data-generator': Open60DominoChainBranchEvaluator,
    'O-24_domino_chain_gap_analysis_data-generator': Open60DominoChainGapEvaluator,
    'O-25_LEGO_construction_assembly_data-generator': Open60LEGOConstructionEvaluator,
    'O-27_move_2_object_to_2_target_data-generator': Open60MoveObjectsToTargetEvaluator,
    'O-30_bookshelf_data-generator': Open60BookshelfEvaluator,
    'O-33_counting_object_data-generator': Open60CountingObjectEvaluator,
    'O-34_dot_to_dot_task_data-generator': Open60DotToDotEvaluator,
    'O-36_grid_shift_data-generator': Open60GridShiftEvaluator,
    'O-37_light_sequence_data-generator': Open60LightSequenceEvaluator,
    'O-44_rotation_puzzle_data-generator': Open60RotationPuzzleEvaluator,
    'O-45_sequence_completion_data-generator': Open60SequenceCompletionEvaluator,
    'O-47_sliding_puzzle_data-generator': Open60SlidingPuzzleEvaluator,
    'O-52_traffic_light_data-generator': Open60TrafficLightEvaluator,
    'O-53_clock_data-generator': Open60ClockTimeEvaluator,
    'O-55_rotation_data-generator': Open60RotationEvaluator,
    
    # Hidden_40 Tasks (40 tasks)
    'G-135_select_next_figure_small_large_alternating_sequence_data-generator': SelectNextFigureAlternatingEvaluator,
    'G-193_draw_next_sized_shape_data-generator': Hidden40DrawNextShapeEvaluator,
    'G-136_locate_point_in_overlapping_area_data-generator': LocatePointInOverlappingAreaEvaluator,
    'G-140_locate_topmost_unobscured_figure_data-generator': Hidden40LocateTopmostFigureEvaluator,
    'G-147_identify_unique_figure_in_uniform_set_data-generator': Hidden40IdentifyUniqueFigureEvaluator,
    'G-160_circle_largest_numerical_value_data-generator': Hidden40CircleLargestEvaluator,
    'G-161_mark_second_largest_shape_data-generator': Hidden40MarkSecondLargestEvaluator,
    'G-167_select_longest_polygon_side_data-generator': Hidden40SelectLongestSideEvaluator,
    'G-202_mark_wave_peaks_data-generator': MarkWavePeaksEvaluator,
    'G-212_find_incorrect_arrow_direction_data-generator': FindIncorrectArrowDirectionEvaluator,
    'G-217_circle_central_dot_data-generator': CircleCentralDotEvaluator,
    'G-218_identify_largest_angle_in_triangle_data-generator': IdentifyLargestAngleEvaluator,
    'G-219_select_leftmost_shape_data-generator': SelectLeftmostShapeEvaluator,
    'G-221_outline_innermost_square_data-generator': OutlineInnermostSquareEvaluator,
    'G-240_add_borders_to_unbordered_shapes_data-generator': AddBordersToUnborderedEvaluator,
    'G-247_identify_chinese_character_data-generator': IdentifyChineseCharacterEvaluator,
    'G-248_mark_asymmetrical_shape_data-generator': MarkAsymmetricalShapeEvaluator,
    'G-174_arrange_circles_by_circumference_data-generator': Hidden40ArrangeCirclesEvaluator,
    'G-273_high_density_liquid_data-generator': HighDensityLiquidEvaluator,
    'G-47_multiple_keys_for_one_door_data-generator': MultipleKeysForOneDoorEvaluator,
    'O-11_shape_color_then_move_data-generator': Hidden40ShapeColorMoveEvaluator,
    'O-56_raven_data-generator': Hidden40RavenMatrixEvaluator,
    'O-22_construction_stack_data-generator': Hidden40ConstructionStackEvaluator,
    'O-2_pigment_color_mixing_subtractive_data-generator': Hidden40PigmentMixingEvaluator,
    'O-39_maze_data-generator': Hidden40MazeEvaluator,
    'O-43_object_subtraction_data-generator': Hidden40ObjectSubtractionEvaluator,
    'O-46_shape_sorter_data-generator': Hidden40ShapeSorterEvaluator,
    'O-49_symmetry_completion_data-generator': Hidden40SymmetryEvaluator,
    'O-5_symbol_deletion_data-generator': Hidden40SymbolDeletionEvaluator,
    'O-54_control_panel_data-generator': Hidden40ControlPanelEvaluator,
    'O-58_symbol_delete_data-generator': Hidden40SymbolDeleteEvaluator,
    'O-59_symbol_insert_data-generator': Hidden40SymbolInsertEvaluator,
    'O-60_symbol_substitute_data-generator': Hidden40SymbolSubstituteEvaluator,
    'O-61_symbol_edit_data-generator': Hidden40SymbolEditEvaluator,
    'O-62_gravity_physics_data-generator': Hidden40GravityPhysicsEvaluator,
    'O-64_animal_matching_data-generator': Hidden40AnimalMatchingEvaluator,
    'O-65_animal_size_sorting_data-generator': Hidden40AnimalSizeSortingEvaluator,
    'O-6_2d_geometric_transformation_data-generator': Hidden40GeometricTransformEvaluator,
    'O-85_2d_object_rotation_data-generator': Hidden40ObjectRotation2DEvaluator,
    'O-9_shape_scaling_data-generator': Hidden40ShapeScalingEvaluator,
}


def get_evaluator(task_name: str, device: str = 'cuda') -> BaseEvaluator:
    """Get the appropriate evaluator for a given task."""
    evaluator_class = TASK_EVALUATOR_MAP.get(task_name, BaseEvaluator)
    return evaluator_class(device=device, task_name=task_name)


def list_all_tasks():
    """List all 100 task names."""
    return list(TASK_EVALUATOR_MAP.keys())


# Task category mapping - based on user specification
TASK_CATEGORY_MAP = {
    # Abstraction (17 tasks)
    'G-13_grid_number_sequence_data-generator': 'Abstraction',
    'G-51_predict_next_color_data-generator': 'Abstraction',
    'G-131_select_next_figure_increasing_size_sequence_data-generator': 'Abstraction',
    'G-134_select_next_figure_large_small_alternating_sequence_data-generator': 'Abstraction',
    'O-37_light_sequence_data-generator': 'Abstraction',
    'O-45_sequence_completion_data-generator': 'Abstraction',
    'O-52_traffic_light_data-generator': 'Abstraction',
    'O-53_clock_data-generator': 'Abstraction',
    'O-10_shape_outline_fill_data-generator': 'Abstraction',
    'G-43_understand_scene_structure_data-generator': 'Abstraction',
    'O-21_construction_blueprint_data-generator': 'Abstraction',
    
    # Categorization (12 tasks)
    'G-3_stable_sort_data-generator': 'Categorization',
    'O-30_bookshelf_data-generator': 'Categorization',
    'O-29_ballcolor_data-generator': 'Categorization',
    'O-38_majority_color_data-generator': 'Categorization',
    'G-39_attention_shift_different_data-generator': 'Categorization',
    'G-29_chart_extreme_with_data_data-generator': 'Categorization',
    
    # Navigation (15 tasks)
    'G-5_multi_object_placement_data-generator': 'Navigation',
    'G-15_grid_avoid_obstacles_data-generator': 'Navigation',
    'G-16_grid_go_through_block_data-generator': 'Navigation',
    'G-18_grid_shortest_path_data-generator': 'Navigation',
    'G-31_directed_graph_navigation_data-generator': 'Navigation',
    'G-41_grid_highest_cost_data-generator': 'Navigation',
    'G-45_key_door_matching_data-generator': 'Navigation',
    'O-34_dot_to_dot_task_data-generator': 'Navigation',
    'O-47_sliding_puzzle_data-generator': 'Navigation',
    'G-47_multiple_keys_for_one_door_data-generator': 'Navigation',
    
    # Perception (many tasks)
    'O-33_counting_object_data-generator': 'Perception',
    'G-8_track_object_movement_data-generator': 'Perception',
    'G-9_identify_objects_in_region_data-generator': 'Perception',
    'G-21_multiple_occlusions_vertical_data-generator': 'Perception',
    'G-135_select_next_figure_small_large_alternating_sequence_data-generator': 'Perception',
    'G-136_locate_point_in_overlapping_area_data-generator': 'Perception',
    'G-138_spot_unique_non_repeated_color_data-generator': 'Perception',
    'G-140_locate_topmost_unobscured_figure_data-generator': 'Perception',
    'G-147_identify_unique_figure_in_uniform_set_data-generator': 'Perception',
    'G-158_identify_all_hollow_points_data-generator': 'Perception',
    'G-160_circle_largest_numerical_value_data-generator': 'Perception',
    'G-161_mark_second_largest_shape_data-generator': 'Perception',
    'G-167_select_longest_polygon_side_data-generator': 'Perception',
    'G-168_identify_nearest_to_square_rectangle_data-generator': 'Perception',
    'G-169_locate_intersection_of_segments_data-generator': 'Perception',
    'G-174_arrange_circles_by_circumference_data-generator': 'Perception',
    'G-189_draw_midpoint_perpendicular_line_data-generator': 'Perception',
    'G-193_draw_next_sized_shape_data-generator': 'Perception',
    'G-194_construct_concentric_ring_data-generator': 'Perception',
    'G-202_mark_wave_peaks_data-generator': 'Perception',
    'G-206_identify_pentagons_data-generator': 'Perception',
    'G-212_find_incorrect_arrow_direction_data-generator': 'Perception',
    'G-217_circle_central_dot_data-generator': 'Perception',
    'G-218_identify_largest_angle_in_triangle_data-generator': 'Perception',
    'G-219_select_leftmost_shape_data-generator': 'Perception',
    'G-221_outline_innermost_square_data-generator': 'Perception',
    'G-222_mark_tangent_point_of_circles_data-generator': 'Perception',
    'G-223_highlight_horizontal_lines_data-generator': 'Perception',
    'G-240_add_borders_to_unbordered_shapes_data-generator': 'Perception',
    'G-247_identify_chinese_character_data-generator': 'Perception',
    'G-248_mark_asymmetrical_shape_data-generator': 'Perception',
    'G-250_color_triple_intersection_red_data-generator': 'Perception',
    'G-54_connecting_color_data-generator': 'Perception',
    
    # Physics (many tasks)
    'G-273_high_density_liquid_data-generator': 'Physics',
    'O-15_ball_bounces_given_time_data-generator': 'Physics',
    'O-18_glass_refraction_data-generator': 'Physics',
    'O-19_mirror_reflection_data-generator': 'Physics',
    'O-23_domino_chain_branch_path_prediction_data-generator': 'Physics',
    'O-24_domino_chain_gap_analysis_data-generator': 'Physics',
    'O-25_LEGO_construction_assembly_data-generator': 'Physics',
    'O-27_move_2_object_to_2_target_data-generator': 'Physics',
    'O-31_ball_eating_data-generator': 'Physics',
    'O-32_rolling_ball_data-generator': 'Physics',
    'O-36_grid_shift_data-generator': 'Physics',
    'O-44_rotation_puzzle_data-generator': 'Physics',
    'O-55_rotation_data-generator': 'Physics',
    'O-62_gravity_physics_data-generator': 'Physics',
    'O-75_communicating_vessels_data-generator': 'Physics',
    'O-85_2d_object_rotation_data-generator': 'Physics',
    
    # Transformation (many tasks)
    'O-2_pigment_color_mixing_subtractive_data-generator': 'Transformation',
    'O-5_symbol_deletion_data-generator': 'Transformation',
    'O-6_2d_geometric_transformation_data-generator': 'Transformation',
    'O-9_shape_scaling_data-generator': 'Transformation',
    'O-11_shape_color_then_move_data-generator': 'Transformation',
    'O-12_shape_color_then_scale_data-generator': 'Transformation',
    'O-13_shape_outline_then_move_data-generator': 'Transformation',
    'O-14_shape_scale_then_outline_data-generator': 'Transformation',
    'O-16_color_addition_data-generator': 'Transformation',
    'O-22_construction_stack_data-generator': 'Transformation',
    'O-39_maze_data-generator': 'Transformation',
    'O-43_object_subtraction_data-generator': 'Transformation',
    'O-46_shape_sorter_data-generator': 'Transformation',
    'O-49_symmetry_completion_data-generator': 'Transformation',
    'O-54_control_panel_data-generator': 'Transformation',
    'O-56_raven_data-generator': 'Transformation',
    'O-58_symbol_delete_data-generator': 'Transformation',
    'O-59_symbol_insert_data-generator': 'Transformation',
    'O-60_symbol_substitute_data-generator': 'Transformation',
    'O-61_symbol_edit_data-generator': 'Transformation',
    'O-64_animal_matching_data-generator': 'Transformation',
    'O-65_animal_size_sorting_data-generator': 'Transformation',
    'G-24_separate_objects_no_spin_data-generator': 'Transformation',
    'G-25_seperate_object_spinning_data-generator': 'Transformation',
}


def get_task_category(task_name: str) -> str:
    """Get the category for a task."""
    return TASK_CATEGORY_MAP.get(task_name, 'Unknown')


def get_tasks_by_category():
    """Get tasks organized by category."""
    categories = {}
    for task_name in TASK_EVALUATOR_MAP.keys():
        category = get_task_category(task_name)
        if category not in categories:
            categories[category] = []
        categories[category].append(task_name)
    return categories


def get_tasks_by_split():
    """
    Get tasks organized by split (In-Domain vs Out-of-Domain).

    Out-of-Domain = Original Hidden_40 + 10 additional tasks from Open_60:
        G-24, G-54, G-168, G-169, G-189, G-206, G-222, G-223, G-250, O-27

    In-Domain = Remaining 50 tasks from original Open_60
    """
    from .. import is_out_of_domain

    out_of_domain = [t for t in TASK_EVALUATOR_MAP if is_out_of_domain(t)]
    in_domain = [t for t in TASK_EVALUATOR_MAP if not is_out_of_domain(t)]

    return {'In_Domain': in_domain, 'Out_of_Domain': out_of_domain}


__all__ = [
    'BaseEvaluator',
    'get_evaluator',
    'list_all_tasks',
    'get_tasks_by_split',
    'get_tasks_by_category',
    'get_task_category',
    'TASK_EVALUATOR_MAP',
    'TASK_CATEGORY_MAP',
]
