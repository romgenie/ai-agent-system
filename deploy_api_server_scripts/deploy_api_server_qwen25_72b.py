#!/usr/bin/env python3
"""
API Server for Qwen2.5-72B-Instruct model.
This server handles external requests to interact with the language model.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to sys.path for importing from src
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Import FastAPI components
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel

# Import project modules
from src.llm import QwenModel
from src.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Qwen2.5-72B-Instruct API Server")

# Model instance
model = None

class QueryRequest(BaseModel):
    """Request schema for query endpoint"""
    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9

@app.on_event("startup")
async def startup_event():
    """Load the model during startup"""
    global model
    try:
        logger.info("Initializing Qwen2.5-72B-Instruct model...")
        model_path = os.path.join(project_root, "local_model_weights")
        model = QwenModel(model_path=model_path)
        logger.info("Model initialization complete")
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        # We'll continue running so the health endpoint still works,
        # but query endpoint will return errors

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        return {"status": "warning", "message": "Server is running but model is not loaded"}
    return {"status": "ok", "model": "Qwen2.5-72B-Instruct"}

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process a query with the language model"""
    if model is None:
        logger.error("Model not initialized but received query request")
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        logger.info(f"Processing query with {len(request.prompt)} characters")
        response = model.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred"}
    )

if __name__ == "__main__":
    # Default port is 8760, but can be overridden with environment variable
    port = int(os.environ.get("API_PORT", 8760))
    logger.info(f"Starting API server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)