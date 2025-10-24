#!/bin/bash
# VMEvalKit Web Dashboard Startup Script

echo "====================================="
echo "🎥 VMEvalKit Web Dashboard"
echo "====================================="
echo ""

# Check if we're in the web directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Please run this script from the web/ directory."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "../venv" ]; then
    echo "⚠️  Warning: Virtual environment not found at ../venv"
    echo "Please create a virtual environment first:"
    echo "  cd .."
    echo "  python -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source ../venv/bin/activate

# Check if dependencies are installed
echo "🔍 Checking dependencies..."
python -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📥 Installing dependencies..."
    pip install -r ../requirements.txt
fi

# Check if output directory exists
if [ ! -d "../data/outputs" ]; then
    echo "⚠️  Warning: Output directory not found at ../data/outputs"
    echo "The dashboard will still start, but there may be no data to display."
    echo ""
fi

# Start the server
echo "🚀 Starting Flask development server..."
echo "📊 Dashboard will be available at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""
echo "====================================="
echo ""

python app.py

