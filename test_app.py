#!/usr/bin/env python3
"""
Simple test script to verify the Streamlit app works correctly.
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✓ Streamlit imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Streamlit: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ Pandas imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Pandas: {e}")
        return False
    
    try:
        import json
        print("✓ JSON imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import JSON: {e}")
        return False
    
    try:
        from dotenv import load_dotenv
        print("✓ python-dotenv imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import python-dotenv: {e}")
        return False
    
    try:
        from pydantic import BaseModel, Field
        print("✓ Pydantic imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Pydantic: {e}")
        return False
    
    # Test optional imports
    try:
        from agno.agent import Agent, RunResponse
        print("✓ Agno imported successfully")
    except ImportError as e:
        print(f"⚠ Agno not available: {e}")
    
    try:
        from reddit import RedditService
        print("✓ RedditService imported successfully")
    except ImportError as e:
        print(f"⚠ RedditService not available: {e}")
    
    return True

def test_streamlit_app():
    """Test if the Streamlit app can be imported."""
    print("\nTesting Streamlit app import...")
    
    try:
        import streamlit_app
        print("✓ Streamlit app imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import Streamlit app: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Trading LLM Analysis App")
    print("=" * 50)
    
    # Test basic imports
    if not test_imports():
        print("\n❌ Basic imports failed. Please install required dependencies.")
        return False
    
    # Test Streamlit app
    if not test_streamlit_app():
        print("\n❌ Streamlit app import failed.")
        return False
    
    print("\n✅ All tests passed!")
    print("\n🚀 To run the app, use:")
    print("   streamlit run streamlit_app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 