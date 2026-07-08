import os
import json
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import BaseModel, Field
from .prompt_templates import SYSTEM_INSTRUCTION, USER_PROMPT_TEMPLATE

# 1. 환경 변수 로드
load_dotenv()

# Pydantic을 이용한 구조화된 출력 스키마 정의
class WeightRecommendation(BaseModel):
    지역여건: float = Field(description="지역여건 가중치 비율 (0.0 ~ 1.0)")
    물류여건: float = Field(description="물류여건 가중치 비율 (0.0 ~ 1.0)")
    산업혁신여건: float = Field(description="산업혁신여건 가중치 비율 (0.0 ~ 1.0)")
    생활정주여건: float = Field(description="생활정주여건 가중치 비율 (0.0 ~ 1.0)")
    근로자이동여건: float = Field(description="근로자이동여건 가중치 비율 (0.0 ~ 1.0)")
    reason: str = Field(description="추천 사유 요약 (반드시 1~2문장의 간결한 요약형 한글 문장)")

class ComplexDetail(BaseModel):
    dan_name: str = Field(description="산업단지 이름")
    lat: float = Field(description="산업단지의 대략적인 실제 위도 (latitude) 좌표. 반드시 부산, 울산, 경상남도 지리 범위 내의 유효값(35.0 ~ 35.6)을 실수형(float)으로 제시하십시오. 누락하거나 0.0을 주면 안 됩니다.")
    lon: float = Field(description="산업단지의 대략적인 실제 경도 (longitude) 좌표. 반드시 부산, 울산, 경상남도 지리 범위 내의 유효값(128.5 ~ 129.4)을 실수형(float)으로 제시하십시오. 누락하거나 0.0을 주면 안 됩니다.")

    short_desc: str = Field(description="산업단지의 주요 특성 및 장점에 대한 한국어 한 줄 요약 (최대 60자)")
    detail_desc: str = Field(description="산업단지의 산업군 구성, 교통/정주 편의성, 향후 발전 방향 등을 담은 상세한 한국어 설명 (3~4문장)")

class TopComplexesResponse(BaseModel):
    complexes: list[ComplexDetail] = Field(description="상위 산업단지 상세 정보 리스트")

class GeminiLLMClient:
    """
    Gemini API 연동을 통해 사용자의 요구사항을 분석하여
    평가 항목별 적정 가중치 비율을 구조화된 형태로 추천받고,
    추천된 산업단지의 상세 정보와 좌표를 검색 및 요약하는 클라이언트.
    """
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.is_configured = False
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.is_configured = True
            except Exception as e:
                logging.error(f"Gemini API configure 실패: {e}")
                
    def get_weight_recommendation(self, user_input: str) -> dict:
        """
        사용자의 요구사항을 파싱하여 가중치를 제안받습니다.
        """
        if not self.is_configured:
            raise ValueError("GEMINI_API_KEY가 설정되어 있지 않거나 설정에 실패했습니다.")
            
        try:
            model = genai.GenerativeModel(
                model_name="gemini-3.1-flash-lite",
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": WeightRecommendation,
                    "temperature": 0.2
                },
                system_instruction=SYSTEM_INSTRUCTION
            )
            
            prompt = USER_PROMPT_TEMPLATE.format(user_input=user_input)
            response = model.generate_content(prompt)
            
            result = json.loads(response.text)
            return result
        except Exception as e:
            logging.error(f"Gemini 호출 중 오류 발생: {e}")
            raise e

    def get_top_complexes_details(self, complexes: list[str]) -> dict:
        """
        상위 5개 산업단지명 리스트에 대한 정보(위도, 경도, 한줄 특성, 상세 특성)를
        구조화된 형태로 받아옵니다.
        """
        if not self.is_configured:
            raise ValueError("GEMINI_API_KEY가 설정되어 있지 않거나 설정에 실패했습니다.")
            
        try:
            model = genai.GenerativeModel(
                model_name="gemini-3.1-flash-lite",
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": TopComplexesResponse,
                    "temperature": 0.2
                },
                system_instruction="당신은 대한민국 부울경(부산, 울산, 경상남도) 지역의 산업단지 분석 전문가입니다. 주어진 산업단지들의 정확한 지리적 위치(위도, 경도)를 파악하고, 각 산업단지의 고유 특성, 장점, 주력 산업군, 물류 여건 등을 분석하여 요약 정보와 상세 정보를 구조화된 JSON으로 반환해 주세요."
            )
            
            prompt = f"다음 산업단지들의 지리적 위도/경도 좌표 및 주요 특성을 분석해 주세요:\n{', '.join(complexes)}"
            response = model.generate_content(prompt)
            
            result = json.loads(response.text)
            return result
        except Exception as e:
            logging.error(f"산업단지 상세 정보 조회 중 오류 발생: {e}")
            raise e

