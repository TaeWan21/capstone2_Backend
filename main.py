from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.images import router as image_router

app = FastAPI(title="Otkki Dokki Yo App", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 모든 도메인 허용(개발용, 실제 배포시에는 제한 권장)
    allow_methods=["*"],      # 모든 HTTP 메소드 허용
    allow_headers=["*"],      # 모든 헤더 허용
)

app.include_router(image_router)

@app.get("/")
async def root():
    return {"message": "Hello World"}