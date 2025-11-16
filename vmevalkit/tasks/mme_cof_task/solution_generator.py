"""
MME-CoF Solution Generator

Generates synthetic solution images for MME-CoF tasks using Gemini Vision API.
Uses LLM-based reasoning to understand the task and generate appropriate solutions.
"""

from typing import Optional, Tuple, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import os
import base64
import time


# All categories use LLM-based solution generation
SOLUTION_STRATEGIES = {
    "2D_geometry_reasoning": "llm_vision",
    "3D_geometry_reasoning": "llm_vision",
    "abstract_reasoning": "llm_vision",
    "chess": "llm_vision",
    "common_sense_reasoning": "llm_vision",
    "counting_reasoning": "llm_vision",
    "logical_reasoning": "llm_vision",
    "physics_reasoning": "llm_vision",
    "physics_based_reasoning": "llm_vision",
    "practical_reasoning": "llm_vision",
    "visual_analogy_reasoning": "llm_vision",
    "visual_arithmetic_reasoning": "llm_vision",
    "visual_trace_reasoning": "llm_vision",
    "rotation_reasoning": "llm_vision",
    "real_world_spatial_reasoning": "llm_vision",
    "table_and_chart_reasoning": "llm_vision",
    "visual_detail_reasoning": "llm_vision",
}


def can_generate_solution(category: str) -> bool:
    """
    Check if we can generate a ground truth solution for this category.
    
    Args:
        category: The reasoning category
    
    Returns:
        True - all categories can use LLM-based solution generation
    """
    # Accept any category - we'll generate prompts dynamically
    return True


def get_evaluation_mode(category: str) -> str:
    """
    Get the appropriate evaluation mode for a category.
    
    Args:
        category: The reasoning category
    
    Returns:
        "final_frame" for all categories (using LLM-generated solutions)
    """
    return "final_frame"


def _get_gemini_client():
    """Initialize Gemini API client."""
    import google.generativeai as genai
    
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError(
            "Gemini API key not found. Please set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
        )
    
    genai.configure(api_key=api_key)
    return genai


def _encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def _get_solution_prompt(category: str) -> str:
    """
    Get category-specific prompt for solution generation.
    
    Args:
        category: The reasoning category
    
    Returns:
        Detailed prompt for Gemini to understand and solve the task
    """
    prompts = {
        "2D_geometry_reasoning": """
            Analyze this 2D geometry reasoning puzzle. 
            Describe what geometric transformation or pattern completion is required.
            Explain step-by-step what the final solved state should look like.
            Be very specific about shapes, positions, angles, and spatial relationships.
        """,
        "3D_geometry_reasoning": """
            Analyze this 3D geometry reasoning puzzle.
            Describe the 3D transformation, rotation, or spatial reasoning required.
            Explain what the final solved state should look like from this perspective.
            Be specific about depth, perspective, and 3D relationships.
        """,
        "abstract_reasoning": """
            Analyze this abstract reasoning pattern puzzle.
            Identify the logical rule or pattern governing the sequence.
            Describe what the completed pattern or next element should be.
            Be specific about shapes, colors, arrangements, and transformations.
        """,
        "chess": """
            Analyze this chess position and puzzle.
            Identify what needs to be achieved (checkmate, find best move, etc.).
            Describe the final board position after the solution move(s).
            List the exact piece positions in the solved state.
        """,
        "common_sense_reasoning": """
            Analyze this common sense reasoning scenario.
            Identify what real-world outcome or change should occur.
            Describe what the final state should look like after the logical sequence.
            Be specific about object positions, states, and changes.
        """,
        "counting_reasoning": """
            Analyze this counting or quantity reasoning task.
            Identify what needs to be counted or how quantities change.
            Describe the final state showing the correct count or result.
            Be specific about numbers, groupings, or quantity representations.
        """,
        "logical_reasoning": """
            Analyze this logical reasoning puzzle.
            Identify the logical rules, constraints, or deductions required.
            Describe the final solved state that satisfies all logic constraints.
            Be specific about the arrangement, selections, or outcomes.
        """,
        "physics_reasoning": """
            Analyze this physics-based reasoning scenario.
            Identify what physical process or outcome should occur (gravity, motion, forces, etc.).
            Describe the final state after the physics simulation completes.
            Be specific about object positions, states, and physical changes.
        """,
        "physics_based_reasoning": """
            Analyze this physics-based reasoning scenario.
            Identify what physical process or outcome should occur (gravity, motion, forces, etc.).
            Describe the final state after the physics simulation completes.
            Be specific about object positions, states, and physical changes.
        """,
        "practical_reasoning": """
            Analyze this practical problem-solving scenario.
            Identify what practical action or solution is required.
            Describe the final state after the problem is solved.
            Be specific about changes, arrangements, or outcomes.
        """,
        "visual_analogy_reasoning": """
            Analyze this visual analogy puzzle (A:B :: C:?).
            Identify the transformation relationship between the pairs.
            Describe what the analogous result should be.
            Be specific about how the pattern transforms.
        """,
        "visual_arithmetic_reasoning": """
            Analyze this visual arithmetic or mathematical reasoning task.
            Identify the mathematical operation or calculation required.
            Describe the final state showing the correct result.
            Be specific about numbers, operations, and visual representations.
        """,
        "visual_trace_reasoning": """
            Analyze this path tracing or navigation puzzle.
            Identify the correct path, route, or trace to follow.
            Describe the final state with the completed path drawn or highlighted.
            Be specific about the path trajectory and connections.
        """,
        "rotation_reasoning": """
            Analyze this rotation or orientation puzzle.
            Identify how the object should be rotated or transformed.
            Describe the final state after the rotation is complete.
            Be specific about angles, orientations, and spatial positions.
        """,
        "real_world_spatial_reasoning": """
            Analyze this real-world spatial reasoning scenario.
            Identify the spatial relationships and transformations required.
            Describe the final state with correct spatial arrangement.
            Be specific about positions, distances, and real-world context.
        """,
        "table_and_chart_reasoning": """
            Analyze this table or chart reasoning task.
            Identify what data needs to be extracted, compared, or computed.
            Describe the final state showing the answer or completed analysis.
            Be specific about values, trends, and data relationships.
        """,
        "visual_detail_reasoning": """
            Analyze this visual detail reasoning puzzle.
            Identify the critical visual details that lead to the solution.
            Describe the final state highlighting or showing the key details.
            Be specific about colors, shapes, patterns, and visual features.
        """,
    }
    
    # Dynamic fallback for unknown categories
    if category not in prompts:
        category_readable = category.replace('_', ' ')
        return f"""
            Analyze this {category_readable} puzzle.
            Identify what needs to be solved or determined.
            Describe the final solved state in detail.
            Be very specific about visual elements, positions, values, or arrangements in the solution.
        """
    
    return prompts[category]


def generate_solution_image(
    image: Image.Image, 
    category: str, 
    metadata: dict = None,
    use_imagen: bool = True,
    cache_dir: Optional[str] = None
) -> Optional[Image.Image]:
    """
    Generate a solution image for MME-CoF task using Gemini Vision API.
    
    Strategy:
    1. Use Gemini to analyze the puzzle and describe the solution
    2. Either:
       a) Use Imagen 3 to generate the solution image (if use_imagen=True)
       b) Create an annotated version with the solution description
    
    Args:
        image: Input puzzle image
        category: Reasoning category
        metadata: Additional task metadata
        use_imagen: If True, use Imagen 3 to generate solution image
        cache_dir: Optional directory to cache generated solutions
    
    Returns:
        Generated solution image, or None if generation fails
    """
    
    try:
        # Step 1: Use Gemini to analyze and solve the puzzle
        solution_description = _analyze_puzzle_with_gemini(image, category)
        
        if not solution_description:
            print(f"⚠️  Failed to generate solution description for {category}")
            return None
        
        # Step 2: Generate solution image
        if use_imagen:
            # Use Imagen 3 to generate the solution image
            solution_image = _generate_with_imagen(image, solution_description, category)
        else:
            # Create annotated image with solution overlay
            solution_image = _create_annotated_solution(image, solution_description)
        
        return solution_image
        
    except Exception as e:
        print(f"⚠️  Error generating solution for {category}: {e}")
        return None


def _analyze_puzzle_with_gemini(image: Image.Image, category: str) -> Optional[str]:
    """
    Use Gemini Vision to analyze the puzzle and describe the solution.
    
    Args:
        image: Input puzzle image
        category: Reasoning category
    
    Returns:
        Detailed description of the solved state
    """
    
    genai = _get_gemini_client()
    
    # Use Gemini 2.0 Flash for vision tasks
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    # Get category-specific analysis prompt
    analysis_prompt = _get_solution_prompt(category)
    
    full_prompt = f"""
    {analysis_prompt}
    
    Provide a detailed description of the FINAL SOLVED STATE only.
    Be extremely specific about visual elements, positions, colors, shapes, and arrangements.
    Do not explain the steps - only describe what the final image should look like.
    
    Format your response as:
    SOLUTION: [detailed visual description of the solved state]
    """
    
    # Call Gemini with the image
    response = model.generate_content([full_prompt, image])
    
    if response and response.text:
        return response.text.strip()
    
    return None


def _generate_with_imagen(
    original_image: Image.Image, 
    solution_description: str, 
    category: str
) -> Optional[Image.Image]:
    """
    Generate solution image using Imagen 3 via Vertex AI.
    
    Requires:
    - GOOGLE_CLOUD_PROJECT environment variable
    - GOOGLE_CLOUD_LOCATION environment variable (default: us-central1)
    - Authenticated via: gcloud auth application-default login
    
    Args:
        original_image: Original puzzle image
        solution_description: Description of the solved state
        category: Reasoning category
    
    Returns:
        Generated solution image
    """
    
    try:
        from vertexai.preview.vision_models import ImageGenerationModel
        import vertexai
        
        # Initialize Vertex AI
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
        
        if not project_id:
            print(f"⚠️  GOOGLE_CLOUD_PROJECT not set, falling back to annotated solution")
            return _create_annotated_solution(original_image, solution_description)
        
        vertexai.init(project=project_id, location=location)
        
        # Load Imagen 3
        imagen = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
        
        # Create a generation prompt
        generation_prompt = f"""
        Create an image showing the SOLVED STATE of this {category.replace('_', ' ')} puzzle.
        
        The solution should show:
        {solution_description}
        
        Maintain the same visual style, dimensions, and clarity as a puzzle solution image.
        Show only the final solved state, not the process.
        """
        
        # Generate the image
        response = imagen.generate_images(
            prompt=generation_prompt,
            number_of_images=1,
            aspect_ratio="1:1",
            safety_filter_level="block_few",
        )
        
        if response and len(response.images) > 0:
            # Convert to PIL Image
            generated_image = response.images[0]._pil_image
            
            # Resize to match original dimensions
            if generated_image.size != original_image.size:
                generated_image = generated_image.resize(
                    original_image.size, 
                    Image.Resampling.LANCZOS
                )
            
            return generated_image
            
    except ImportError:
        print(f"⚠️  Vertex AI SDK not installed. Install with: pip install google-cloud-aiplatform")
        print(f"⚠️  Falling back to annotated solution")
        return _create_annotated_solution(original_image, solution_description)
    except Exception as e:
        print(f"⚠️  Imagen generation failed: {e}")
        print(f"⚠️  Falling back to annotated solution")
        return _create_annotated_solution(original_image, solution_description)
    
    return None


def _create_annotated_solution(
    image: Image.Image, 
    solution_description: str
) -> Image.Image:
    """
    Create an annotated version of the image with solution description.
    
    This is a fallback when Imagen is not available or fails.
    
    Args:
        image: Original image
        solution_description: Text description of solution
    
    Returns:
        Image with solution annotation overlay
    """
    
    # Create a copy of the image
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    
    # Add a semi-transparent overlay at the bottom
    width, height = annotated.size
    overlay_height = min(150, height // 4)
    
    # Draw semi-transparent rectangle
    overlay = Image.new('RGBA', (width, overlay_height), (0, 0, 0, 180))
    annotated.paste(overlay, (0, height - overlay_height), overlay)
    
    # Add solution text
    try:
        # Try to use a decent font
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font = ImageFont.load_default()
    
    # Truncate and wrap text
    max_chars = 120
    if len(solution_description) > max_chars:
        solution_text = solution_description[:max_chars] + "..."
    else:
        solution_text = solution_description
    
    # Draw text
    text_y = height - overlay_height + 10
    draw.text(
        (10, text_y), 
        f"SOLUTION:\n{solution_text}", 
        fill=(255, 255, 255), 
        font=font
    )
    
    return annotated


def get_solution_metadata(category: str) -> dict:
    """
    Get metadata about solution generation for a category.
    
    Returns:
        Dictionary with solution strategy info
    """
    strategy = SOLUTION_STRATEGIES.get(category, "llm_vision")
    
    return {
        "strategy": strategy,
        "has_ground_truth": can_generate_solution(category),
        "evaluation_mode": get_evaluation_mode(category),
        "uses_llm": True,
        "model": "Gemini 2.0 Flash + Imagen 3"
    }


def generate_solutions_batch(
    tasks: list[Dict[str, Any]], 
    use_imagen: bool = True,
    progress_callback=None
) -> list[Dict[str, Any]]:
    """
    Generate solution images for a batch of MME-CoF tasks.
    
    Args:
        tasks: List of task dictionaries with 'image', 'category', etc.
        use_imagen: Whether to use Imagen for generation
        progress_callback: Optional callback function for progress updates
    
    Returns:
        List of tasks with 'solution_image' added
    """
    
    results = []
    total = len(tasks)
    
    for idx, task in enumerate(tasks):
        if progress_callback:
            progress_callback(idx, total)
        
        image = task.get('image')
        category = task.get('category')
        
        if not image or not category:
            print(f"⚠️  Skipping task {idx}: missing image or category")
            results.append(task)
            continue
        
        # Generate solution
        solution_image = generate_solution_image(
            image, 
            category, 
            metadata=task,
            use_imagen=use_imagen
        )
        
        # Add to task
        task_result = task.copy()
        task_result['solution_image'] = solution_image
        task_result['solution_generated'] = solution_image is not None
        
        results.append(task_result)
        
        # Rate limiting - be respectful to API
        time.sleep(1)
    
    return results

