import os
import json
from dotenv import load_dotenv
import openai
# .env 파일에서 환경변수 읽어오기
load_dotenv()

# 환경변수에서 API 키 가져오기
api_key = os.getenv("OPENAI_API_KEY")

def json_to_natural_text(data: dict) -> str:
    """
    JSON 데이터를 사람이 읽기 쉬운 자연스러운 한국어 문장으로 변환합니다.
    """
    system_prompt = (
        f'''당신은 옷에 대한 설명이 담긴 json데이터로 부터 이해하기 쉬운 자연스러운 한국어 문장으로 변환하주는 변환기 입니다.
        답변은 한국어 문장으로 변환한 값만 제공해주세요. 그외에 다른 내용은 포함하지 마세요. 이 옷은 ~ 으로 답변을 시작해주세요.
        '''
    )
    user_prompt = json.dumps(data, ensure_ascii=False, indent=2)
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",  # 또는 "gpt-4" 사용 가능
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt + '\n답변 :'}
        ],
        temperature=0
    )
    # GPT-4o API의 응답에서 결과 추출
    return response.choices[0].message.content