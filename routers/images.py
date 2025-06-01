from fastapi import APIRouter, UploadFile, File, HTTPException
from services import json_to_natural_text
import openai
import json
import base64
import io
from PIL import Image
import os
from typing import Dict
router = APIRouter(prefix="/images", tags=["Image Processing"])

def encode_image_to_base64(image_bytes: bytes) -> str:
    """이미지 바이트를 base64로 인코딩"""
    return base64.b64encode(image_bytes).decode('utf-8')

def validate_image(image_bytes: bytes) -> bool:
    """이미지 파일 유효성 검사"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # 지원하는 이미지 형식 확인
        if image.format not in ['JPEG', 'PNG', 'GIF', 'WEBP']:
            return False
        return True
    except Exception:
        return False

@router.post("/predict")
async def predict_image(image: UploadFile = File(...)) -> Dict[str, str]:
    try:
        # 파일 타입 검증
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="업로드된 파일이 이미지가 아닙니다.")
        
        # 이미지 파일 읽기
        image_bytes = await image.read()
        
        # 파일 크기 확인 (20MB 제한)
        if len(image_bytes) > 20 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="파일 크기가 20MB를 초과합니다.")
        
        # 이미지 유효성 검사
        if not validate_image(image_bytes):
            raise HTTPException(status_code=400, detail="유효하지 않은 이미지 파일입니다.")
        
        # 이미지를 base64로 인코딩
        base64_image = encode_image_to_base64(image_bytes)
        
        # OpenAI Vision API 호출
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """당신은 시각 장애인을 위한 의류 분석 전문가입니다. 이미지를 자세히 관찰하고 다음 내용을 포함하여 분석해주세요: 

1) 이미지의 분류는 다음과 같습니다. 이 특성 이외에 분석은 할 수 없습니다.
{
  "색상": [
    "블랙", "화이트", "그레이", "레드", "핑크",
    "오렌지", "베이지", "브라운", "옐로우", "그린",
    "카키", "민트", "블루", "네이비", "스카이블루",
    "퍼플", "라벤더", "와인", "네온", "골드", "실버"
  ],
  "디테일": [
    "비즈", "단추", "니트꽈배기", "체인", "컷오프",
    "더블브레스티드", "드롭숄더", "자수", "프릴", "프린지",
    "퍼프", "퀼팅", "태슬", "리본", "집업",
    "롤업", "띠", "러플", "디스트로이드", "셔링",
    "드롭웨이스트", "슬릿", "버클", "스팽글",
    "컷아웃", "스티치", "X스트랩", "스터드", "비대칭"
  ],
  "프린트": [
    "체크", "플로럴", "스트라이프", "레터링", "지그재그",
    "해골", "호피", "타이다이", "지브라", "그라데이션",
    "도트", "무지", "카무플라쥬", "그래픽", "페이즐리",
    "하운즈투스", "아가일", "깅엄"
  ],
  "소재": [
    "퍼", "니트", "무스탕", "레이스", "스웨이드",
    "린넨", "앙고라", "메시", "코듀로이", "플리스",
    "시퀸/글리터", "네오프렌", "데님", "실크", "저지",
    "스판덱스", "트위드", "자카드", "벨벳", "가죽",
    "비닐/PVC", "면", "울/캐시미어", "시폰", "합성섬유"
  ],
  "기장": [
    "크롭", "노멀", "롱",
    "미니", "니렝스", "미디", "발목", "맥시",
    "하프"
  ],
  "소매기장": [
    "민소매", "반팔", "캡소매", "7부소매", "긴팔"
  ],
  "넥라인": [
    "라운드넥", "유넥", "브이넥", "홀터넥", "오프숄더",
    "원숄더", "스퀘어넥", "노카라", "후드", "스위트하트"
  ],
  "칼라": [
    "셔츠칼라", "보우칼라", "세일러칼라", "숄칼라",
    "폴로칼라", "피터팬칼라", "노치드칼라", "밴드칼라"
  ],
  "핏": [
    "타이트", "노멀", "루즈", "오버사이즈",
    "스키니", "와이드", "벨보텀"
  ]
}

2) 이미지에 포함된 의류의 종류는 아우터, 상의, 하의, 원피스 입니다.
3) 이미지에는 의류의 종류중 1개 이상 포함되어 있으며, 1개만 포함되어 있을 수 있습니다. 아우터를 입고있으면 상의는 분석하지 않습니다.
4) 시각장애인이 알아들을수 있게 자연스러운 한국어 문장으로 간결하게 설명해주세요.

예시: 이 옷은 스트리트 스타일로, 베이지색의 롱 기장의 오버사이즈 점퍼가 특징입니다. 점퍼는 스트링과 지퍼 디테일이 있으며, 긴팔로 되어 있습니다. 하의는 스카이블루 색상의 발목 길이 청바지로, 데님 소재에 롤업 디테일이 있는 노멀 핏입니다. 상의는 화이트 색상의 티셔츠로, 소매가 없고 저지 소재로 만들어졌으며, 라운드넥과 루즈 핏을 가지고 있습니다. 전체적으로 무지 프린트로 깔끔한 느낌을 줍니다."""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "이 이미지를 분석해주세요."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        # 응답에서 텍스트 추출
        analysis_text = response.choices[0].message.content
        
        return {
            "status": "success",
            "analysis": analysis_text,
            "filename": image.filename
        }
        
    except openai.APIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API 오류: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")