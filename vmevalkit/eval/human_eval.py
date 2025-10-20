"""Human evaluation interface for VMEvalKit using Gradio."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import gradio as gr
from datetime import datetime

logger = logging.getLogger(__name__)


class HumanEvaluator:
    """Gradio-based interface for human evaluation of generated videos."""
    
    def __init__(self, output_dir: str = "data/evaluations", 
                 experiment_name: str = "pilot_experiment"):
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.experiment_dir = Path("data/outputs") / experiment_name
        self.annotator_name = "Anonymous"  # Default, will be set in interface
        self.evaluation_queue = []
        self.current_index = 0
        self._load_evaluation_queue()
    
    def _load_evaluation_queue(self):
        """Load tasks that need evaluation, checking evaluations folder to avoid repetition."""
        self.evaluation_queue = []
        skipped_count = 0
        
        for model_dir in self.experiment_dir.iterdir():
            if not model_dir.is_dir(): continue
            for task_type_dir in model_dir.iterdir():
                if not task_type_dir.is_dir(): continue
                for task_dir in task_type_dir.iterdir():
                    if not task_dir.is_dir(): continue
                    
                    # Check if already evaluated
                    eval_path = self.output_dir / self.experiment_name / model_dir.name / task_type_dir.name / task_dir.name
                    
                    # Look for any existing evaluation files
                    has_evaluation = False
                    if eval_path.exists():
                        eval_files = list(eval_path.glob("*-eval.json"))
                        if eval_files:
                            has_evaluation = True
                            skipped_count += 1
                            logger.debug(f"Skipping {model_dir.name}/{task_type_dir.name}/{task_dir.name} - already evaluated")
                    
                    if not has_evaluation:
                        self.evaluation_queue.append({
                            "model_name": model_dir.name,
                            "task_type": task_type_dir.name,
                            "task_id": task_dir.name
                        })
        
        logger.info(f"Loaded {len(self.evaluation_queue)} tasks to evaluate ({skipped_count} already evaluated, skipped)")
    
    def _get_task_data(self, model_name: str, task_type: str, task_id: str) -> Optional[Dict[str, Any]]:
        """Get data for a specific task."""
        task_dir = self.experiment_dir / model_name / task_type / task_id
        output_dirs = list(task_dir.iterdir())
        if not output_dirs: return None
        
        output_dir = output_dirs[0]
        prompt_path = output_dir / "question" / "prompt.txt"
        first_frame = output_dir / "question" / "first_frame.png"
        final_frame = output_dir / "question" / "final_frame.png"
        video_files = list((output_dir / "video").glob("*.mp4"))
        
        prompt = prompt_path.read_text() if prompt_path.exists() else ""
        return {
            "model_name": model_name, "task_type": task_type, "task_id": task_id,
            "prompt": prompt,
            "first_frame": str(first_frame) if first_frame.exists() else None,
            "final_frame": str(final_frame) if final_frame.exists() else None,
            "video_path": str(video_files[0]) if video_files else None
        }
    
    def _save_evaluation(self, model_name: str, task_type: str, task_id: str, evaluation: Dict[str, Any]):
        """Save evaluation result."""
        output_dir = self.output_dir / self.experiment_name / model_name / task_type / task_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result = {
            "metadata": {
                "evaluator": "human-eval",
                "annotator": self.annotator_name,
                "timestamp": datetime.now().isoformat(),
                "model_name": model_name,
                "task_type": task_type,
                "task_id": task_id
            },
            "result": evaluation
        }
        
        with open(output_dir / "human-eval.json", 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved evaluation for {model_name}/{task_type}/{task_id}")
    
    def launch_interface(self, share: bool = False, port: int = 7860):
        """Launch Gradio interface."""
        with gr.Blocks(title="VMEvalKit Human Evaluation") as interface:
            gr.Markdown(f"# VMEvalKit Human Evaluation\n**Experiment:** {self.experiment_name}")
            
            # Annotator name input
            with gr.Row():
                annotator_input = gr.Textbox(
                    label="Annotator Name", 
                    value=self.annotator_name,
                    placeholder="Enter your name",
                    interactive=True
                )
                annotator_btn = gr.Button("Set Annotator", variant="primary")
            
            annotator_status = gr.Markdown(f"Current annotator: **{self.annotator_name}**")
            progress_text = gr.Markdown(f"Progress: 0/{len(self.evaluation_queue)} tasks")
            
            with gr.Row():
                model_info = gr.Textbox(label="Model", interactive=False)
                task_info = gr.Textbox(label="Task", interactive=False)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Task Input")
                    input_image = gr.Image(label="Input Image", type="filepath")
                    prompt_text = gr.Textbox(label="Prompt", lines=3, interactive=False)
                    if_has_final = gr.Checkbox(label="Has Expected Output", value=False, interactive=False)
                    expected_output = gr.Image(label="Expected Output", type="filepath")
                
                with gr.Column():
                    gr.Markdown("### Generated Video")
                    video_player = gr.Video(label="Generated Video")
            
            gr.Markdown("### Evaluation")
            correctness_score = gr.Radio(
                choices=[1, 2, 3, 4, 5],
                label="Solution Correctness (1=Wrong, 5=Perfect)"
            )
            comments = gr.Textbox(label="Comments (Optional)", lines=3)
            
            with gr.Row():
                prev_btn = gr.Button("← Previous", variant="secondary")
                submit_btn = gr.Button("Submit Evaluation", variant="primary")
                next_btn = gr.Button("Next →", variant="secondary")
            
            status_text = gr.Textbox(label="Status", interactive=False)
            
            # Define UI outputs list once
            ui_outputs = [model_info, task_info, input_image, prompt_text, if_has_final, 
                         expected_output, video_player, progress_text, status_text,
                         correctness_score, comments]
            
            def update_display(index):
                """Update display with current task."""
                if index < 0 or index >= len(self.evaluation_queue):
                    return {
                        model_info: "", task_info: "", input_image: None, prompt_text: "",
                        expected_output: None, video_player: None, correctness_score: None, comments: "",
                        progress_text: f"Progress: {index}/{len(self.evaluation_queue)} tasks",
                        status_text: "No more tasks to evaluate!"
                    }
                
                task = self.evaluation_queue[index]
                task_data = self._get_task_data(task["model_name"], task["task_type"], task["task_id"])
                
                if not task_data:
                    return {status_text: "Error loading task data"}
                
                has_final = task_data["final_frame"] is not None
                return {
                    model_info: task_data["model_name"],
                    task_info: f"{task_data['task_type']} - {task_data['task_id']}",
                    input_image: task_data["first_frame"],
                    prompt_text: task_data["prompt"],
                    if_has_final: has_final,
                    expected_output: task_data["final_frame"] if has_final else None,
                    video_player: task_data["video_path"],
                    progress_text: f"Progress: {index}/{len(self.evaluation_queue)} tasks",
                    status_text: "Task loaded successfully",
                    correctness_score: None,
                    comments: ""
                }
            
            def navigate(direction):
                """Navigate tasks."""
                self.current_index = max(0, min(len(self.evaluation_queue) - 1, 
                                               self.current_index + direction))
                return update_display(self.current_index)
            
            def set_annotator(name):
                """Update annotator name."""
                if not name or name.strip() == "":
                    return {annotator_status: "⚠️ Please enter a valid name"}
                self.annotator_name = name.strip()
                return {annotator_status: f"Current annotator: **{self.annotator_name}**"}
            
            def submit_evaluation(correctness, comments_text):
                """Submit evaluation."""
                if correctness is None:
                    return {status_text: "Please select a correctness score!"}
                
                if self.current_index >= len(self.evaluation_queue):
                    return {status_text: "No task to evaluate!"}
                
                task = self.evaluation_queue[self.current_index]
                self._save_evaluation(
                    task["model_name"], task["task_type"], task["task_id"],
                    {"solution_correctness_score": correctness, "comments": comments_text}
                )
                
                self.current_index += 1
                updates = update_display(self.current_index)
                updates[status_text] = "Evaluation saved successfully!"
                return updates
            
            # Connect buttons
            annotator_btn.click(set_annotator, inputs=[annotator_input], outputs=[annotator_status])
            prev_btn.click(lambda: navigate(-1), outputs=ui_outputs)
            next_btn.click(lambda: navigate(1), outputs=ui_outputs)
            submit_btn.click(submit_evaluation, inputs=[correctness_score, comments], outputs=ui_outputs)
            interface.load(lambda: update_display(0), outputs=ui_outputs)
        
        interface.launch(share=share, server_port=port)