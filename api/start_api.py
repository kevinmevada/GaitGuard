#!/usr/bin/env python3
"""
Startup script for GaitGuard API
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def main():
    # CORS origins are loaded from .env file in main.py
    # No need to override them here
    
    api_host = os.getenv("API_HOST", "0.0.0.0")
    api_port = os.getenv("API_PORT", "8001")
    
    print("Starting GaitGuard API...")
    print(f"API will be available at: http://{api_host}:{api_port}")
    print(f"API Documentation: http://{api_host}:{api_port}/docs")
    print(f"Health Check: http://{api_host}:{api_port}/health")
    print("\nMake sure your trained models are in:")
    print("   fall_risk_pipeline/results/checkpoints/")
    print("\nStarting server...\n")
    
    import uvicorn
    uvicorn.run("main:app", host=api_host, port=int(api_port), reload=False)

if __name__ == "__main__":
    main()
