import uvicorn
from qwenc import QwenChat
from fastapi import FastAPI, Request 
from sse_starlette.sse import EventSourceResponse
from typing import List, Optional, Tuple
import asyncio
from pydantic import BaseModel


class ChatCompletionRequest(BaseModel):
    question: str
    history: List[str] = []
    stream: Optional[bool] = False


def create_app() -> FastAPI:
    app = FastAPI()
    qwen = QwenChat(
        devid=0, 
        bmodel_path="./qwen-7b_int4_fast.bmodel", 
        tokenizer_path="./qwen.tiktoken"
    )
    
    @app.post("/chat/completions")
    async def create_chat_completion(request: ChatCompletionRequest):
        generate = qwen.predict_no_state(request.question, request.history)
        if request.stream:
            return EventSourceResponse(generate, media_type="text/event-stream")
        else:
            result = None
            for r in generate:
                result = r
            
            return result
        
    return app

if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
