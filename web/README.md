# VMEvalKit Web Dashboard 🎥

A modern web interface to visualize and explore video generation results from VMEvalKit experiments.

## Features

- 📊 **Hierarchical Dashboard**: View results organized as Models → Domains → Tasks
- 🤖 **Model Performance**: Detailed analysis per model with collapsible sections
- 🧠 **Domain Analysis**: Results grouped by reasoning domain (Chess, Maze, Raven, Rotation, Sudoku)
- 🎬 **Video Playback**: View generated videos directly in the browser with lazy loading
- 📷 **Image Display**: Input images (first frame) and prompts for each task
- 🚀 **Quick Navigation**: Jump buttons to quickly access any model section
- 🔍 **API Access**: REST endpoints to programmatically access results
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile
- ♿ **Accessibility**: Keyboard navigation, screen reader support, focus management
- 🔒 **Security**: Path traversal protection, input validation, secure file serving

## Quick Start

### 1. Navigate to the web directory
```bash
cd web
```

### 2. Install dependencies
```bash
source ../venv/bin/activate
pip install -r ../requirements.txt
```

### 3. Start the dashboard
```bash
python app.py
```

The dashboard will be available at: **http://localhost:5000**

## Configuration

The dashboard automatically reads from `../data/outputs/pilot_experiment` directory. You can customize paths by modifying `app.py`:

```python
app.config['OUTPUT_DIR'] = Path('/custom/path/to/outputs')
app.config['QUESTIONS_DIR'] = Path('/custom/path/to/questions')
```

## How It Works

### Data Structure
The dashboard scans the following directory structure:
```
data/outputs/pilot_experiment/
├── {model}/
│   ├── {domain}_task/
│   │   ├── {task_id}/
│   │   │   └── {run_id}/
│   │   │       ├── question/
│   │   │       │   ├── prompt.txt          # Task prompt
│   │   │       │   ├── first_frame.png     # Input image  
│   │   │       │   └── final_frame.png     # Expected output
│   │   │       └── video/
│   │   │           └── {model}_{hash}.mp4  # Generated video
```

### Deduplication
If multiple runs exist for the same (model, domain, task_id), only the most recent run is displayed.

### Available Models
The dashboard currently supports these models:
- **openai-sora-2** - OpenAI's Sora model
- **wavespeed-wan-2.2-i2v-720p** - Wavespeed image-to-video model  
- **veo-3.1-720p** - Google's Veo 3.1 model
- **veo-3.0-generate** - Google's Veo 3.0 model
- **luma-ray-2** - Luma's Ray model
- **runway-gen4-turbo** - Runway's Gen-4 Turbo model

Each model is tested on 5 reasoning domains with 15 tasks per domain (75 tasks total per model).

## Interface Overview

### Hierarchical Navigation
- **Models**: Top-level sections for each video generation model
- **Domains**: Reasoning categories within each model (Chess ♟️, Maze 🌀, Raven 🧩, Rotation 🔄, Sudoku 🔢)
- **Tasks**: Individual test cases showing input image, prompt, and generated video

### Key UI Features
- **Auto-expand**: First model section opens automatically for immediate access
- **Quick Navigation**: Yellow navigation bar with buttons for each model
- **Visual Indicators**: Hover effects and expand/collapse arrows
- **Lazy Loading**: Videos only load when their sections are expanded
- **Click-to-play**: Click videos to play/pause
- **Keyboard Support**: Full keyboard navigation and shortcuts

## API Endpoints

### Get All Results
```http
GET /api/results
GET /api/results?model=luma-ray-2
GET /api/results?domain=chess  
GET /api/results?task_id=maze_0001
```

Returns JSON with filtered results based on query parameters:
```json
{
  "total": 450,
  "results": [
    {
      "run_id": "wavespeed-wan-2.2-i2v-720p_chess_0001_20251017_071644",
      "model": "wavespeed-wan-2.2-i2v-720p",
      "domain": "chess",
      "task_id": "chess_0001", 
      "timestamp": "2024-10-17T07:16:44",
      "prompt": "Show the next chess move...",
      "video_path": "/path/to/video.mp4",
      "first_frame": "/path/to/input.png",
      "inference_dir": "/path/to/run/directory"
    }
  ]
}
```

## Production Deployment

### Using Gunicorn
```bash
# Ensure dependencies are installed from main requirements.txt
pip install -r ../requirements.txt
gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
```

### Environment Variables
```bash
export FLASK_DEBUG=false
export FLASK_HOST=0.0.0.0
export FLASK_PORT=5000
export SECRET_KEY=your-secret-key-here
```

### Security Headers
The app includes Flask-Talisman for security headers in production.

## File Structure

```
web/
├── app.py                      # Main Flask application
├── README.md                   # This file
├── utils/
│   ├── __init__.py
│   └── data_loader.py         # Data scanning and loading utilities
├── templates/                  # HTML templates
│   ├── base.html              # Base template with navigation
│   ├── index.html             # Dashboard overview
│   └── error.html             # Error page
└── static/                     # Static assets
    ├── css/
    │   └── style.css          # Clean Apple-style dashboard theme
    └── js/
        └── main.js            # Interactive features and video handling
```

## Troubleshooting

### Videos not loading
- Ensure the output directory path is correct in `app.py`
- Check that video files exist in the expected location
- Verify videos are in supported formats (MP4, WebM, AVI, MOV)
- Check browser console for errors

### Performance issues  
- Large datasets may take time to scan on first load
- Use gunicorn with multiple workers for production
- Videos use lazy loading - they only load when sections are expanded

### Port conflicts
Change the port in `app.py`:
```python
port = int(os.environ.get('FLASK_PORT', 5001))
```

## Browser Compatibility

- ✅ Chrome/Edge: Full support with video streaming
- ✅ Firefox: Full support  
- ✅ Safari: Full support
- ✅ Mobile browsers: Responsive design

## Development

### Run in debug mode
```bash
export FLASK_DEBUG=true
python app.py
```

### Add new features
The modular structure makes it easy to extend:
- Add new templates in `templates/`
- Add styles to `static/css/style.css`
- Add JavaScript to `static/js/main.js`
- Add data utilities to `utils/data_loader.py`

## Technologies Used

- **Backend**: Flask 3.0, Python 3.8+
- **Frontend**: HTML5, CSS3 (Apple-style design), Vanilla JavaScript
- **Video**: HTML5 Video API with lazy loading
- **Security**: Flask-Talisman, Werkzeug path protection
- **Production**: Gunicorn WSGI server

---

**Questions?** Check the main VMEvalKit documentation or open an issue on GitHub.