"""
Script to help generate screenshots for the README.

This script provides instructions and can optionally open the dashboard/API
for you to take screenshots manually.

Usage:
    python scripts/generate_screenshots.py --help
    python scripts/generate_screenshots.py --open-dashboard
    python scripts/generate_screenshots.py --open-api
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
import webbrowser


SCREENSHOTS_DIR = Path("artifacts/screenshots")
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def print_instructions():
    """Print instructions for taking screenshots."""
    print("\n" + "="*70)
    print("📸 SCREENSHOT GENERATION GUIDE")
    print("="*70)
    print("\nYou need to create 3 screenshots for your README:\n")
    
    print("1️⃣  Dashboard Home Page (dashboard_home.png)")
    print("   - Run: streamlit run dashboard.py")
    print("   - Open: http://localhost:8501")
    print("   - Take screenshot of the main prediction page")
    print(f"   - Save to: {SCREENSHOTS_DIR / 'dashboard_home.png'}\n")
    
    print("2️⃣  SHAP Force Plot (shap_force_plot.png)")
    print("   - In Streamlit dashboard, navigate to 'Model Explainability (SHAP)'")
    print("   - Click 'Generate SHAP Explanations'")
    print("   - Take screenshot of the SHAP waterfall plot")
    print(f"   - Save to: {SCREENSHOTS_DIR / 'shap_force_plot.png'}\n")
    
    print("3️⃣  API Swagger Documentation (api_swagger.png)")
    print("   - Run: docker compose up (or uvicorn src.api.main:app --reload)")
    print("   - Open: http://localhost:8000/docs")
    print("   - Take screenshot of the Swagger UI")
    print(f"   - Save to: {SCREENSHOTS_DIR / 'api_swagger.png'}\n")
    
    print("💡 TIPS:")
    print("   - Use browser developer tools (F12) to hide UI elements if needed")
    print("   - Use Windows Snipping Tool or ShareX for better screenshots")
    print("   - Make sure screenshots are clear and show key features")
    print("   - Recommended size: 1200-1600px width\n")
    print("="*70 + "\n")


def open_dashboard():
    """Open Streamlit dashboard in browser."""
    print("🚀 Starting Streamlit dashboard...")
    print("   Dashboard will open at http://localhost:8501")
    print("   Press Ctrl+C to stop the server\n")
    
    try:
        # Start streamlit in background
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "dashboard.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Open browser
        webbrowser.open("http://localhost:8501")
        
        print("✅ Dashboard opened! Take your screenshots.")
        print("   Press Enter when done to stop the server...")
        input()
        
        process.terminate()
        print("✅ Dashboard stopped.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Try running manually: streamlit run dashboard.py")


def open_api():
    """Open FastAPI docs in browser."""
    print("🚀 Starting FastAPI server...")
    print("   API docs will open at http://localhost:8000/docs")
    print("   Press Ctrl+C to stop the server\n")
    
    try:
        # Try docker compose first
        print("Attempting to start with docker compose...")
        process = subprocess.Popen(
            ["docker", "compose", "up"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        time.sleep(5)
        webbrowser.open("http://localhost:8000/docs")
        
        print("✅ API docs opened! Take your screenshot.")
        print("   Press Enter when done to stop the server...")
        input()
        
        process.terminate()
        print("✅ Server stopped.")
        
    except Exception as e:
        print(f"❌ Docker compose failed: {e}")
        print("\n💡 Try running manually:")
        print("   Option 1: docker compose up")
        print("   Option 2: uvicorn src.api.main:app --reload")


def main():
    parser = argparse.ArgumentParser(
        description="Generate screenshots for README"
    )
    parser.add_argument(
        "--open-dashboard",
        action="store_true",
        help="Open Streamlit dashboard in browser"
    )
    parser.add_argument(
        "--open-api",
        action="store_true",
        help="Open FastAPI Swagger docs in browser"
    )
    
    args = parser.parse_args()
    
    if args.open_dashboard:
        open_dashboard()
    elif args.open_api:
        open_api()
    else:
        print_instructions()


if __name__ == "__main__":
    main()
