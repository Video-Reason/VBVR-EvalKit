#!/usr/bin/env python3
"""
VMEvalKit Web Dashboard
A web application to visualize and explore video generation results.
"""

import os
import json
import logging
from pathlib import Path
from flask import Flask, render_template, jsonify, send_from_directory, request, abort
from datetime import datetime
from typing import Dict, List, Any, Optional
from werkzeug.security import safe_join

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configure for subpath deployment
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Security: Use environment variable for secret key or generate one
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24))
app.config['APPLICATION_ROOT'] = '/video-reason'
app.config['OUTPUT_DIR'] = Path(__file__).parent.parent / 'data' / 'outputs' / 'pilot_experiment'
app.config['QUESTIONS_DIR'] = Path(__file__).parent.parent / 'data' / 'questions'
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Import data loader utilities
from utils.data_loader import (
    scan_all_outputs,
    get_hierarchical_data,
    load_run_information
)


@app.route('/')
def index():
    """Main dashboard page showing hierarchical view: Models → Domains → Tasks."""
    try:
        logger.info("Loading dashboard data...")
        
        # Scan all outputs
        all_results = scan_all_outputs(app.config['OUTPUT_DIR'])
        logger.info(f"Found {len(all_results)} inference results")
        
        # Get hierarchical organization
        hierarchy = get_hierarchical_data(all_results)
        
        # Load run information from inference log
        run_info = load_run_information(app.config['OUTPUT_DIR'])
        if not run_info:
            logger.warning("Could not load run information from inference log")
        
        # Calculate overview stats
        total_inferences = len(all_results)
        total_models = len(hierarchy)
        all_domains = set()
        for model_domains in hierarchy.values():
            all_domains.update(model_domains.keys())
        total_domains = len(all_domains)
        
        overview = {
            'total_inferences': total_inferences,
            'total_models': total_models,
            'total_domains': total_domains
        }
        
        # Get deployment info
        host = request.host
        deployment_info = {
            'local_address': f"http://{host}",
            'github_repo': "https://github.com/hokindeng/VMEvalKit"
        }
        
        logger.info(f"Dashboard loaded: {total_models} models, {total_domains} domains, {total_inferences} inferences")
        
        return render_template(
            'index.html',
            overview=overview,
            hierarchy=hierarchy,
            run_info=run_info,
            deployment_info=deployment_info,
            output_dir=str(app.config['OUTPUT_DIR'])
        )
    except FileNotFoundError as e:
        logger.error(f"Output directory not found: {e}")
        return render_template('error.html', 
                             error="Output directory not found. Please run some experiments first."), 404
    except Exception as e:
        logger.exception("Unexpected error in dashboard")
        return render_template('error.html', 
                             error="An unexpected error occurred while loading the dashboard."), 500


@app.route('/api/results')
def api_results():
    """API endpoint to get all results as JSON."""
    try:
        all_results = scan_all_outputs(app.config['OUTPUT_DIR'])
        
        # Get and validate query parameters
        model = request.args.get('model', '').strip()
        domain = request.args.get('domain', '').strip()
        task_id = request.args.get('task_id', '').strip()
        
        # Input validation: ensure no path traversal attempts
        for param_name, param_value in [('model', model), ('domain', domain), ('task_id', task_id)]:
            if param_value and ('/' in param_value or '\\' in param_value or '..' in param_value):
                logger.warning(f"Invalid character in {param_name} parameter: {param_value}")
                return jsonify({'error': f'Invalid {param_name} parameter'}), 400
        
        filtered_results = all_results
        if model:
            filtered_results = [r for r in filtered_results if r.get('model') == model]
        if domain:
            filtered_results = [r for r in filtered_results if r.get('domain') == domain]
        if task_id:
            filtered_results = [r for r in filtered_results if r.get('task_id') == task_id]
        
        logger.info(f"API request: {len(filtered_results)}/{len(all_results)} results returned")
        
        return jsonify({
            'total': len(filtered_results),
            'results': filtered_results
        })
    except Exception as e:
        logger.exception("Error in API results endpoint")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/video/<path:video_path>')
def serve_video(video_path):
    """Serve video files from the output directory."""
    try:
        # Input validation: prevent path traversal
        if '..' in video_path or video_path.startswith('/'):
            logger.warning(f"Potential path traversal attempt: {video_path}")
            abort(403)
        
        # Use safe_join to prevent path traversal
        safe_path = safe_join(str(app.config['OUTPUT_DIR']), video_path)
        if not safe_path:
            logger.warning(f"Invalid path detected: {video_path}")
            abort(403)
        
        full_path = Path(safe_path)
        if not full_path.exists() or not full_path.is_file():
            logger.info(f"Video file not found: {video_path}")
            abort(404)
        
        # Verify file is within allowed directory
        if not str(full_path).startswith(str(app.config['OUTPUT_DIR'])):
            logger.warning(f"Attempt to access file outside output directory: {video_path}")
            abort(403)
        
        # Verify file extension and determine MIME type
        video_ext = video_path.lower()
        if video_ext.endswith('.mp4'):
            mimetype = 'video/mp4'
        elif video_ext.endswith('.webm'):
            mimetype = 'video/webm'
        elif video_ext.endswith('.avi'):
            mimetype = 'video/x-msvideo'
        elif video_ext.endswith('.mov'):
            mimetype = 'video/quicktime'
        else:
            logger.warning(f"Unsupported video file extension: {video_path}")
            abort(403)
        
        # Get directory and filename
        directory = full_path.parent
        filename = full_path.name
        
        # Send with appropriate headers for video streaming
        return send_from_directory(
            directory, 
            filename, 
            mimetype=mimetype,
            as_attachment=False,
            conditional=True  # Enable conditional requests for video streaming
        )
    except Exception as e:
        logger.exception(f"Error serving video {video_path}")
        abort(500)


@app.route('/image/<path:image_path>')
def serve_image(image_path):
    """Serve image files from the output directory."""
    try:
        # Input validation: prevent path traversal
        if '..' in image_path or image_path.startswith('/'):
            logger.warning(f"Potential path traversal attempt in image: {image_path}")
            abort(403)
        
        # Verify file extension
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp')):
            logger.warning(f"Invalid image file extension: {image_path}")
            abort(403)
        
        # Try output directory first
        safe_path = safe_join(str(app.config['OUTPUT_DIR']), image_path)
        full_path = Path(safe_path) if safe_path else None
        
        if not (full_path and full_path.exists() and full_path.is_file()):
            # Try questions directory as fallback
            safe_path = safe_join(str(app.config['QUESTIONS_DIR']), image_path)
            full_path = Path(safe_path) if safe_path else None
            
            if not (full_path and full_path.exists() and full_path.is_file()):
                logger.info(f"Image file not found: {image_path}")
                abort(404)
        
        # Verify file is within allowed directories
        output_dir_str = str(app.config['OUTPUT_DIR'])
        questions_dir_str = str(app.config['QUESTIONS_DIR'])
        full_path_str = str(full_path)
        
        if not (full_path_str.startswith(output_dir_str) or full_path_str.startswith(questions_dir_str)):
            logger.warning(f"Attempt to access file outside allowed directories: {image_path}")
            abort(403)
        
        # Get directory and filename
        directory = full_path.parent
        filename = full_path.name
        
        return send_from_directory(directory, filename)
    except Exception as e:
        logger.exception(f"Error serving image {image_path}")
        abort(500)


@app.template_filter('datetime_format')
def datetime_format(value, format='%Y-%m-%d %H:%M:%S'):
    """Format datetime strings."""
    if not value:
        return 'N/A'
    try:
        if isinstance(value, str):
            # Try to parse ISO format
            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
        else:
            dt = value
        return dt.strftime(format)
    except (ValueError, TypeError, AttributeError) as e:
        logger.warning(f"Failed to format datetime value '{value}': {e}")
        return str(value) if value else 'N/A'


@app.template_filter('duration_format')
def duration_format(seconds):
    """Format duration in seconds to human-readable format."""
    if not seconds:
        return 'N/A'
    try:
        seconds = float(seconds)
        if seconds < 0:
            return 'Invalid'
        elif seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to format duration value '{seconds}': {e}")
        return str(seconds) if seconds else 'N/A'


@app.template_filter('file_size')
def file_size_format(path):
    """Get and format file size."""
    try:
        if not path:
            return 'N/A'
        
        path_obj = Path(path)
        if not path_obj.exists() or not path_obj.is_file():
            return 'N/A'
            
        size_bytes = path_obj.stat().st_size
        if size_bytes < 0:
            return 'Invalid'
        elif size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    except (OSError, ValueError, TypeError) as e:
        logger.warning(f"Failed to get file size for '{path}': {e}")
        return 'N/A'


if __name__ == '__main__':
    print("=" * 60)
    print("🎥 VMEvalKit Web Dashboard")
    print("=" * 60)
    print(f"Output directory: {app.config['OUTPUT_DIR']}")
    print(f"Questions directory: {app.config['QUESTIONS_DIR']}")
    
    # Check if directories exist
    if not app.config['OUTPUT_DIR'].exists():
        print(f"⚠️  Warning: Output directory does not exist: {app.config['OUTPUT_DIR']}")
    if not app.config['QUESTIONS_DIR'].exists():
        print(f"⚠️  Warning: Questions directory does not exist: {app.config['QUESTIONS_DIR']}")
    
    # Get configuration from environment
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 'on')
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    
    print(f"Starting server at: http://{host}:{port}")
    print(f"Debug mode: {debug_mode}")
    print("=" * 60)
    
    try:
        app.run(debug=debug_mode, host=host, port=port)
    except KeyboardInterrupt:
        print("\n👋 Dashboard shutdown gracefully")
    except Exception as e:
        logger.exception("Failed to start dashboard")
        print(f"❌ Error starting dashboard: {e}")

