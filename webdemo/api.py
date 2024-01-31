import uvicorn
from qwenc import QwenChat
from fastapi import FastAPI, Request 
from sse_starlette.sse import EventSourceResponse
from typing import List, Optional, Tuple
from pydantic import BaseModel
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--bmodel_path', type=str, default = "/workspace/aa/qwen-7b_int4.bmodel")
parser.add_argument('--tokenizer_path', type=str, default= "/workspace/aa/Qwen-TPU/webdemo/qwen.tiktoken")
args = parser.parse_args()


class ChatCompletionRequest(BaseModel):
    question: str
    history: List[str] = []
    stream: Optional[bool] = False



def create_app() -> FastAPI:
    app = FastAPI()
    qwen = QwenChat(bmodel_path=args.bmodel_path, tokenizer_path=args.tokenizer_path)
    

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