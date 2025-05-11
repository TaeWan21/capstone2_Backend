from fastapi import APIRouter, UploadFile, File
from services import json_to_natural_text
import json
router = APIRouter(prefix="/images", tags=["Image Processing"])

@router.post("/predict")
async def predict_image(image: UploadFile = File(...)):
    """이미지 분석 엔드포인트"""
    image_data = await image.read()
    
    # AI 모델 추론 로직 (output : json 파일)
    json_data = '''
{
  "파일 번호": 1,
  "파일 이름": "REIGN_001_04.jpg",
  "스타일": [
    {
      "스타일": "스트리트"
    }
  ],
  "라벨링": {
    "아우터": [
      {
        "기장": "롱",
        "색상": "베이지",
        "카테고리": "점퍼",
        "디테일": ["스트링", "지퍼"],
        "소매기장": "긴팔",
        "프린트": ["무지"],
        "핏": "오버사이즈"
      }
    ],
    "하의": [
      {
        "기장": "발목",
        "색상": "스카이블루",
        "카테고리": "청바지",
        "디테일": ["롤업"],
        "소재": ["데님"],
        "핏": "노멀"
      }
    ],
    "원피스": [{}],
    "상의": [
      {
        "색상": "화이트",
        "카테고리": "티셔츠",
        "소매기장": "없음",
        "소재": ["저지"],
        "프린트": ["무지"],
        "넥라인": "라운드넥",
        "핏": "루즈"
      }
    ]
  }
}

'''
    json_data = json.loads(json_data)
    # LLM 으로 json 파일을 string 설명으로 변환
    text_from_llm = json_to_natural_text(json_data)
    
    return {'result' : text_from_llm}