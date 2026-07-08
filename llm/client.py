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
        import streamlit as st
        self.api_key_source = "None"
        # 오직 Streamlit secrets에서만 가져오도록 고정
        if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"]:
            self.api_key = st.secrets["GEMINI_API_KEY"]
            self.api_key_source = "Streamlit Secrets Only"
        else:
            self.api_key = None
            self.api_key_source = "Not Found in Secrets"
        self.is_configured = False
        
        if self.api_key:
            try:
                # SDK 내부에서 os.environ 환경 변수를 직접 다시 읽어와 사용하는 것을 대비해
                # 시스템 환경 변수 값 자체를 새 API 키로 강제 업데이트합니다.
                os.environ["GEMINI_API_KEY"] = self.api_key
                genai.configure(api_key=self.api_key)
                self.is_configured = True
            except Exception as e:
                logging.error(f"Gemini API configure 실패: {e}")
                
    def get_masked_api_key(self) -> str:
        """
        API Key 보안을 위해 마스킹된 문자열을 반환합니다.
        """
        if not self.api_key:
            return "None"
        key_str = str(self.api_key)
        if len(key_str) <= 10:
            return "*****"
        return f"{key_str[:8]}...{key_str[-4:]}"
                
    def get_weight_recommendation(self, user_input: str) -> dict:
        """
        사용자의 요구사항을 바탕으로 구글 검색(Grounding)을 돌려 
        실제 뉴스 기사나 논문 출처 링크를 포함한 가중치 결과를 반환합니다.
        """
        if not self.is_configured:
            raise ValueError("GEMINI_API_KEY가 설정되어 있지 않거나 설정에 실패했습니다.")
            
        try:
            # 1단계: Google Search Grounding을 통해 실재하는 기사/논문 출처 수집 (토큰 최적화용 단문 요약)
            search_model = genai.GenerativeModel(
                model_name="gemini-3.1-flash-lite",
                tools=[{"google_search_retrieval": {}}]
            )
            search_prompt = (
                f"사용자의 산업단지 입지 요구사항인 '{user_input}'과 관련된 국내외 입지 기준, 정부 정책 뉴스 기사, "
                "또는 학술 연구 자료를 구글에서 찾은 뒤, 가장 대표성 있는 기사/논문의 [제목](URL 링크) 1~2개와 핵심 논거를 "
                "최대 150자 내외로 매우 압축하여 한글로 기술해 주세요."
            )
            search_response = search_model.generate_content(search_prompt)
            search_grounding = search_response.text if search_response.text else "관련 논거 기사를 찾지 못했습니다."

            # 2단계: 수집된 실재 기사/논문 링크 정보를 컨텍스트로 주입하여 최종 가중치 JSON 구조화 출력 생성
            struct_model = genai.GenerativeModel(
                model_name="gemini-3.1-flash-lite",
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": WeightRecommendation,
                    "temperature": 0.2
                },
                system_instruction=SYSTEM_INSTRUCTION + f"\n\n[실시간 구글 검색 근거 및 링크]\n{search_grounding}"
            )
            
            prompt = (
                f"사용자의 요구사항 '{user_input}'을 바탕으로 5대 지표별 가중치(합산 1.0)를 결정하고, "
                "위 제공된 [실시간 구글 검색 근거 및 링크]의 마크다운 포맷(기사/논문 제목 및 파란색 링크 URL)을 "
                "반드시 포함하여 추천 사유(reason)를 작성해 주세요."
            )
            response = struct_model.generate_content(prompt)
            
            result = json.loads(response.text)
            return result
        except Exception as e:
            logging.error(f"Gemini 가중치 추천 호출 중 오류 발생: {e}")
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

