"""
VBVR-Bench Evaluators Module

This module provides task-specific evaluators for all 100 VBVR-Bench tasks.
Each evaluator implements rule-based evaluation following documented criteria.
"""

from typing import Dict, Any, Optional
from .base_evaluator import BaseEvaluator

from .add_borders_to_unbordered import AddBordersToUnborderedEvaluator
from .animal_matching import AnimalMatchingEvaluator
from .animal_size_sorting import AnimalSizeSortingEvaluator
from .arrange_circles_by_circumference import ArrangeCirclesByCircumferenceEvaluator
from .attention_shift import AttentionShiftEvaluator
from .ball_bounce import BallBounceEvaluator
from .ball_color import BallColorEvaluator
from .ball_eating import BallEatingEvaluator
from .bookshelf import BookshelfEvaluator
from .chart_extreme import ChartExtremeEvaluator
from .circle_central_dot import CircleCentralDotEvaluator
from .circle_largest_numerical_value import CircleLargestNumericalValueEvaluator
from .clock_time import ClockTimeEvaluator
from .color_addition import ColorAdditionEvaluator
from .color_triple_intersection import ColorTripleIntersectionEvaluator
from .communicating_vessels import CommunicatingVesselsEvaluator
from .connecting_color import ConnectingColorEvaluator
from .construct_concentric_ring import ConstructConcentricRingEvaluator
from .construction_blueprint import ConstructionBlueprintEvaluator
from .construction_stack import ConstructionStackEvaluator
from .control_panel import ControlPanelEvaluator
from .counting_object import CountingObjectEvaluator
from .directed_graph_navigation import DirectedGraphNavigationEvaluator
from .domino_chain_branch import DominoChainBranchEvaluator
from .domino_chain_gap import DominoChainGapEvaluator
from .dot_to_dot import DotToDotEvaluator
from .draw_midpoint_perpendicular import DrawMidpointPerpendicularEvaluator
from .draw_next_sized_shape import DrawNextSizedShapeEvaluator
from .find_incorrect_arrow_direction import FindIncorrectArrowDirectionEvaluator
from .geometric_transformation import GeometricTransformationEvaluator
from .glass_refraction import GlassRefractionEvaluator
from .gravity_physics import GravityPhysicsEvaluator
from .grid_avoid_obstacles import GridAvoidObstaclesEvaluator
from .grid_go_through_block import GridGoThroughBlockEvaluator
from .grid_highest_cost import GridHighestCostEvaluator
from .grid_number_sequence import GridNumberSequenceEvaluator
from .grid_shift import GridShiftEvaluator
from .grid_shortest_path import GridShortestPathEvaluator
from .high_density_liquid import HighDensityLiquidEvaluator
from .highlight_horizontal_lines import HighlightHorizontalLinesEvaluator
from .identify_all_hollow_points import IdentifyAllHollowPointsEvaluator
from .identify_chinese_character import IdentifyChineseCharacterEvaluator
from .identify_largest_angle import IdentifyLargestAngleEvaluator
from .identify_nearest_square_rectangle import IdentifyNearestSquareRectangleEvaluator
from .identify_objects_in_region import IdentifyObjectsInRegionEvaluator
from .identify_pentagons import IdentifyPentagonsEvaluator
from .identify_unique_figure import IdentifyUniqueFigureEvaluator
from .key_door_matching import KeyDoorMatchingEvaluator
from .lego_construction import LEGOConstructionEvaluator
from .light_sequence import LightSequenceEvaluator
from .locate_point_in_overlapping_area import LocatePointInOverlappingAreaEvaluator
from .locate_segment_intersection import LocateSegmentIntersectionEvaluator
from .locate_topmost_figure import LocateTopmostFigureEvaluator
from .majority_color import MajorityColorEvaluator
from .mark_asymmetrical_shape import MarkAsymmetricalShapeEvaluator
from .mark_second_largest_shape import MarkSecondLargestShapeEvaluator
from .mark_tangent_point import MarkTangentPointEvaluator
from .mark_wave_peaks import MarkWavePeaksEvaluator
from .maze_pathfinding import MazePathfindingEvaluator
from .mirror_reflection import MirrorReflectionEvaluator
from .move_objects_to_target import MoveObjectsToTargetEvaluator
from .multi_object_placement import MultiObjectPlacementEvaluator
from .multiple_keys_for_one_door import MultipleKeysForOneDoorEvaluator
from .multiple_occlusions_vertical import MultipleOcclusionsVerticalEvaluator
from .object_rotation2_d import ObjectRotation2DEvaluator
from .object_subtraction import ObjectSubtractionEvaluator
from .outline_innermost_square import OutlineInnermostSquareEvaluator
from .pigment_color_mixing import PigmentColorMixingEvaluator
from .predict_next_color import PredictNextColorEvaluator
from .raven_matrix import RavenMatrixEvaluator
from .rolling_ball import RollingBallEvaluator
from .rotation import RotationEvaluator
from .rotation_puzzle import RotationPuzzleEvaluator
from .select_leftmost_shape import SelectLeftmostShapeEvaluator
from .select_longest_polygon_side import SelectLongestPolygonSideEvaluator
from .select_next_figure_alternating import SelectNextFigureAlternatingEvaluator
from .select_next_figure_increasing import SelectNextFigureIncreasingEvaluator
from .select_next_figure_large_small import SelectNextFigureLargeSmallEvaluator
from .separate_objects_no_spin import SeparateObjectsNoSpinEvaluator
from .separate_objects_spinning import SeparateObjectsSpinningEvaluator
from .sequence_completion import SequenceCompletionEvaluator
from .shape_color_then_move import ShapeColorThenMoveEvaluator
from .shape_color_then_scale import ShapeColorThenScaleEvaluator
from .shape_outline_fill import ShapeOutlineFillEvaluator
from .shape_outline_then_move import ShapeOutlineThenMoveEvaluator
from .shape_scale_then_outline import ShapeScaleThenOutlineEvaluator
from .shape_scaling_analogy import ShapeScalingAnalogyEvaluator
from .shape_sorter import ShapeSorterEvaluator
from .sliding_puzzle import SlidingPuzzleEvaluator
from .spot_unique_color import SpotUniqueColorEvaluator
from .stable_sort import StableSortEvaluator
from .symbol_delete import SymbolDeleteEvaluator
from .symbol_deletion import SymbolDeletionEvaluator
from .symbol_edit_constraint import SymbolEditConstraintEvaluator
from .symbol_insert import SymbolInsertEvaluator
from .symbol_substitute import SymbolSubstituteEvaluator
from .symmetry_completion import SymmetryCompletionEvaluator
from .track_object_movement import TrackObjectMovementEvaluator
from .traffic_light import TrafficLightEvaluator
from .understand_scene_structure import UnderstandSceneStructureEvaluator


TASK_EVALUATOR_MAP = {
    # Open_60 Tasks
    'G-3_stable_sort_data-generator': StableSortEvaluator,
    'G-13_grid_number_sequence_data-generator': GridNumberSequenceEvaluator,
    'G-15_grid_avoid_obstacles_data-generator': GridAvoidObstaclesEvaluator,
    'G-16_grid_go_through_block_data-generator': GridGoThroughBlockEvaluator,
    'G-18_grid_shortest_path_data-generator': GridShortestPathEvaluator,
    'G-31_directed_graph_navigation_data-generator': DirectedGraphNavigationEvaluator,
    'G-41_grid_highest_cost_data-generator': GridHighestCostEvaluator,
    'G-131_select_next_figure_increasing_size_sequence_data-generator': SelectNextFigureIncreasingEvaluator,
    'G-134_select_next_figure_large_small_alternating_sequence_data-generator': SelectNextFigureLargeSmallEvaluator,
    'G-51_predict_next_color_data-generator': PredictNextColorEvaluator,
    'G-138_spot_unique_non_repeated_color_data-generator': SpotUniqueColorEvaluator,
    'G-54_connecting_color_data-generator': ConnectingColorEvaluator,
    'G-250_color_triple_intersection_red_data-generator': ColorTripleIntersectionEvaluator,
    'G-158_identify_all_hollow_points_data-generator': IdentifyAllHollowPointsEvaluator,
    'G-168_identify_nearest_to_square_rectangle_data-generator': IdentifyNearestSquareRectangleEvaluator,
    'G-169_locate_intersection_of_segments_data-generator': LocateSegmentIntersectionEvaluator,
    'G-189_draw_midpoint_perpendicular_line_data-generator': DrawMidpointPerpendicularEvaluator,
    'G-194_construct_concentric_ring_data-generator': ConstructConcentricRingEvaluator,
    'G-206_identify_pentagons_data-generator': IdentifyPentagonsEvaluator,
    'G-21_multiple_occlusions_vertical_data-generator': MultipleOcclusionsVerticalEvaluator,
    'G-222_mark_tangent_point_of_circles_data-generator': MarkTangentPointEvaluator,
    'G-223_highlight_horizontal_lines_data-generator': HighlightHorizontalLinesEvaluator,
    'G-24_separate_objects_no_spin_data-generator': SeparateObjectsNoSpinEvaluator,
    'G-25_seperate_object_spinning_data-generator': SeparateObjectsSpinningEvaluator,
    'G-29_chart_extreme_with_data_data-generator': ChartExtremeEvaluator,
    'G-39_attention_shift_different_data-generator': AttentionShiftEvaluator,
    'G-43_understand_scene_structure_data-generator': UnderstandSceneStructureEvaluator,
    'G-45_key_door_matching_data-generator': KeyDoorMatchingEvaluator,
    'G-5_multi_object_placement_data-generator': MultiObjectPlacementEvaluator,
    'G-8_track_object_movement_data-generator': TrackObjectMovementEvaluator,
    'G-9_identify_objects_in_region_data-generator': IdentifyObjectsInRegionEvaluator,
    'O-10_shape_outline_fill_data-generator': ShapeOutlineFillEvaluator,
    'O-12_shape_color_then_scale_data-generator': ShapeColorThenScaleEvaluator,
    'O-13_shape_outline_then_move_data-generator': ShapeOutlineThenMoveEvaluator,
    'O-14_shape_scale_then_outline_data-generator': ShapeScaleThenOutlineEvaluator,
    'O-15_ball_bounces_given_time_data-generator': BallBounceEvaluator,
    'O-31_ball_eating_data-generator': BallEatingEvaluator,
    'O-32_rolling_ball_data-generator': RollingBallEvaluator,
    'O-75_communicating_vessels_data-generator': CommunicatingVesselsEvaluator,
    'O-16_color_addition_data-generator': ColorAdditionEvaluator,
    'O-29_ballcolor_data-generator': BallColorEvaluator,
    'O-38_majority_color_data-generator': MajorityColorEvaluator,
    'O-18_glass_refraction_data-generator': GlassRefractionEvaluator,
    'O-19_mirror_reflection_data-generator': MirrorReflectionEvaluator,
    'O-21_construction_blueprint_data-generator': ConstructionBlueprintEvaluator,
    'O-23_domino_chain_branch_path_prediction_data-generator': DominoChainBranchEvaluator,
    'O-24_domino_chain_gap_analysis_data-generator': DominoChainGapEvaluator,
    'O-25_LEGO_construction_assembly_data-generator': LEGOConstructionEvaluator,
    'O-27_move_2_object_to_2_target_data-generator': MoveObjectsToTargetEvaluator,
    'O-30_bookshelf_data-generator': BookshelfEvaluator,
    'O-33_counting_object_data-generator': CountingObjectEvaluator,
    'O-34_dot_to_dot_task_data-generator': DotToDotEvaluator,
    'O-36_grid_shift_data-generator': GridShiftEvaluator,
    'O-37_light_sequence_data-generator': LightSequenceEvaluator,
    'O-44_rotation_puzzle_data-generator': RotationPuzzleEvaluator,
    'O-45_sequence_completion_data-generator': SequenceCompletionEvaluator,
    'O-47_sliding_puzzle_data-generator': SlidingPuzzleEvaluator,
    'O-52_traffic_light_data-generator': TrafficLightEvaluator,
    'O-53_clock_data-generator': ClockTimeEvaluator,
    'O-55_rotation_data-generator': RotationEvaluator,

    # Hidden_40 Tasks
    'G-135_select_next_figure_small_large_alternating_sequence_data-generator': SelectNextFigureAlternatingEvaluator,
    'G-193_draw_next_sized_shape_data-generator': DrawNextSizedShapeEvaluator,
    'G-136_locate_point_in_overlapping_area_data-generator': LocatePointInOverlappingAreaEvaluator,
    'G-140_locate_topmost_unobscured_figure_data-generator': LocateTopmostFigureEvaluator,
    'G-147_identify_unique_figure_in_uniform_set_data-generator': IdentifyUniqueFigureEvaluator,
    'G-160_circle_largest_numerical_value_data-generator': CircleLargestNumericalValueEvaluator,
    'G-161_mark_second_largest_shape_data-generator': MarkSecondLargestShapeEvaluator,
    'G-167_select_longest_polygon_side_data-generator': SelectLongestPolygonSideEvaluator,
    'G-202_mark_wave_peaks_data-generator': MarkWavePeaksEvaluator,
    'G-212_find_incorrect_arrow_direction_data-generator': FindIncorrectArrowDirectionEvaluator,
    'G-217_circle_central_dot_data-generator': CircleCentralDotEvaluator,
    'G-218_identify_largest_angle_in_triangle_data-generator': IdentifyLargestAngleEvaluator,
    'G-219_select_leftmost_shape_data-generator': SelectLeftmostShapeEvaluator,
    'G-221_outline_innermost_square_data-generator': OutlineInnermostSquareEvaluator,
    'G-240_add_borders_to_unbordered_shapes_data-generator': AddBordersToUnborderedEvaluator,
    'G-247_identify_chinese_character_data-generator': IdentifyChineseCharacterEvaluator,
    'G-248_mark_asymmetrical_shape_data-generator': MarkAsymmetricalShapeEvaluator,
    'G-174_arrange_circles_by_circumference_data-generator': ArrangeCirclesByCircumferenceEvaluator,
    'G-273_high_density_liquid_data-generator': HighDensityLiquidEvaluator,
    'G-47_multiple_keys_for_one_door_data-generator': MultipleKeysForOneDoorEvaluator,
    'O-11_shape_color_then_move_data-generator': ShapeColorThenMoveEvaluator,
    'O-56_raven_data-generator': RavenMatrixEvaluator,
    'O-22_construction_stack_data-generator': ConstructionStackEvaluator,
    'O-2_pigment_color_mixing_subtractive_data-generator': PigmentColorMixingEvaluator,
    'O-39_maze_data-generator': MazePathfindingEvaluator,
    'O-43_object_subtraction_data-generator': ObjectSubtractionEvaluator,
    'O-46_shape_sorter_data-generator': ShapeSorterEvaluator,
    'O-49_symmetry_completion_data-generator': SymmetryCompletionEvaluator,
    'O-5_symbol_deletion_data-generator': SymbolDeletionEvaluator,
    'O-54_control_panel_data-generator': ControlPanelEvaluator,
    'O-58_symbol_delete_data-generator': SymbolDeleteEvaluator,
    'O-59_symbol_insert_data-generator': SymbolInsertEvaluator,
    'O-60_symbol_substitute_data-generator': SymbolSubstituteEvaluator,
    'O-61_symbol_edit_data-generator': SymbolEditConstraintEvaluator,
    'O-62_gravity_physics_data-generator': GravityPhysicsEvaluator,
    'O-64_animal_matching_data-generator': AnimalMatchingEvaluator,
    'O-65_animal_size_sorting_data-generator': AnimalSizeSortingEvaluator,
    'O-6_2d_geometric_transformation_data-generator': GeometricTransformationEvaluator,
    'O-85_2d_object_rotation_data-generator': ObjectRotation2DEvaluator,
    'O-9_shape_scaling_data-generator': ShapeScalingAnalogyEvaluator,
}


def get_evaluator(task_name: str, device: str = 'cuda') -> BaseEvaluator:
    """Get the appropriate evaluator for a given task."""
    evaluator_class = TASK_EVALUATOR_MAP.get(task_name, BaseEvaluator)
    return evaluator_class(device=device, task_name=task_name)


def list_all_tasks():
    """List all 100 task names."""
    return list(TASK_EVALUATOR_MAP.keys())


TASK_CATEGORY_MAP = {
    # Abstraction
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

    # Categorization
    'G-3_stable_sort_data-generator': 'Categorization',
    'O-30_bookshelf_data-generator': 'Categorization',
    'O-29_ballcolor_data-generator': 'Categorization',
    'O-38_majority_color_data-generator': 'Categorization',
    'G-39_attention_shift_different_data-generator': 'Categorization',
    'G-29_chart_extreme_with_data_data-generator': 'Categorization',

    # Navigation
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

    # Perception
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

    # Physics
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

    # Transformation
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
