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
    price_per_pyeong: str = Field(description="해당 산업단지 또는 인근 지역(시군구/읍면동)의 최근 공장/토지 평당 평균 실거래가 (예: '평당 약 150만 원'). 구체적인 금액과 단위를 명시하세요.")
    recent_transaction_info: str = Field(description="최근 팩토리온/국토부 실거래 내역 요약 (예: '2025년 8월, 3,200㎡ 매매 15.5억 원, 평당 약 160만 원'). 구체적인 년월, 거래 규모, 총액을 포함하십시오.")

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
        # 오직 Streamlit secrets에서만 가져오도록 고정 (공백 문자 제거 포함)
        if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"]:
            self.api_key = st.secrets["GEMINI_API_KEY"].strip()
            self.api_key_source = "Streamlit Secrets Only"
        else:
            self.api_key = None
            self.api_key_source = "Not Found in Secrets"
            
        # Streamlit secrets에서 사용할 Gemini 모델명을 설정 가능하도록 함 (기본값은 gemini-3.1-flash-lite)
        if "GEMINI_MODEL" in st.secrets and st.secrets["GEMINI_MODEL"]:
            self.model_name = st.secrets["GEMINI_MODEL"].strip()
        else:
            self.model_name = "gemini-3.1-flash-lite"
            
        # 한도 초과(429, Quota Exceeded) 발생 시 자동으로 폴백할 이전 세대 안정 모델
        self.fallback_model_name = "gemini-2.5-flash"
            
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
        기본 모델 한도 초과 시 gemini-2.5-flash 모델로 자동 폴백 재시도합니다.
        """
        if not self.is_configured:
            raise ValueError("GEMINI_API_KEY가 설정되어 있지 않거나 설정에 실패했습니다.")
            
        try:
            return self._execute_weight_recommendation(user_input, self.model_name)
        except Exception as e:
            err_msg = str(e).lower()
            # 쿼타 초과(429, Quota Exceeded, ResourceExhausted) 감지 시
            if "429" in err_msg or "quota" in err_msg or "exhausted" in err_msg or "resource_exhausted" in err_msg:
                logging.warning(f"기본 모델 {self.model_name} 쿼타 초과로 {self.fallback_model_name} 모델로 자동 전환 재시도합니다. 에러: {e}")
                try:
                    return self._execute_weight_recommendation(user_input, self.fallback_model_name)
                except Exception as ex:
                    logging.error(f"폴백 모델 {self.fallback_model_name} 실행 중에도 오류 발생: {ex}")
                    raise ex
            else:
                raise e

    def _execute_weight_recommendation(self, user_input: str, model_name: str) -> dict:
        """
        실제 가중치 추천 요청을 전송하는 핵심 로직.
        """
        # 1단계: Google Search Grounding을 통해 실재하는 기사/논문 출처 수집 (토큰 최적화용 단문 요약)
        # 구형 SDK 버전 계열을 사용하는 한, 모델명에 관계없이 구글 검색 도구는 항상 [{"google_search_retrieval": {}}] 로 단일 지정해야 에러가 발생하지 않습니다.
        search_grounding = "관련 논거 기사를 찾지 못했습니다."
        try:
            search_model = genai.GenerativeModel(
                model_name=model_name,
                tools=[{"google_search_retrieval": {}}]
            )
            search_prompt = (
                f"사용자의 산업단지 입지 요구사항인 '{user_input}'과 관련된 국내외 입지 기준, 정부 정책 뉴스 기사, "
                "또는 학술 연구 자료를 구글에서 찾은 뒤, 가장 대표성 있는 기사/논문의 [제목](URL 링크) 1~2개와 핵심 논거를 "
                "최대 150자 내외로 매우 압축하여 한글로 기술해 주세요."
            )
            search_response = search_model.generate_content(search_prompt)
            if search_response and search_response.text:
                search_grounding = search_response.text
        except Exception as e:
            logging.warning(f"Google Search Grounding 비활성화 혹은 실패 (일반 LLM 추론으로 폴백): {e}")

        # 2단계: 수집된 실재 기사/논문 링크 정보를 컨텍스트로 주입하여 최종 가중치 JSON 구조화 출력 생성
        struct_model = genai.GenerativeModel(
            model_name=model_name,
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

    def get_top_complexes_details(self, complexes: list[str]) -> dict:
        """
        상위 5개 산업단지명 리스트에 대한 정보(위도, 경도, 한줄 특성, 상세 특성)를
        구조화된 형태로 받아옵니다.
        기본 모델 한도 초과 시 gemini-2.5-flash 모델로 자동 폴백 재시도합니다.
        """
        if not self.is_configured:
            raise ValueError("GEMINI_API_KEY가 설정되어 있지 않거나 설정에 실패했습니다.")
            
        try:
            return self._execute_top_complexes_details(complexes, self.model_name)
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "quota" in err_msg or "exhausted" in err_msg or "resource_exhausted" in err_msg:
                logging.warning(f"기본 모델 {self.model_name} 쿼타 초과로 {self.fallback_model_name} 모델로 자동 전환 재시도합니다. 에러: {e}")
                try:
                    return self._execute_top_complexes_details(complexes, self.fallback_model_name)
                except Exception as ex:
                    logging.error(f"폴백 모델 {self.fallback_model_name} 실행 중에도 오류 발생: {ex}")
                    raise ex
            else:
                raise e

    def _execute_top_complexes_details(self, complexes: list[str], model_name: str) -> dict:
        """
        실제 단지 상세 정보를 요청하는 핵심 로직.
        실시간 구글 검색을 활용하여 팩토리온(factoryon.go.kr) 및 최근 실거래가 데이터를 긁어와
        평당 실거래 가격과 최근 계약 정보를 포함해 분석합니다.
        """
        # 1단계: 각 산업단지에 대해 팩토리온 실거래가 실시간 검색 수행
        search_grounding_list = []
        for dan in complexes:
            try:
                search_model = genai.GenerativeModel(
                    model_name=model_name,
                    tools=[{"google_search_retrieval": {}}]
                )
                search_prompt = (
                    f"대한민국 산업단지 '{dan}'의 최근 부동산 실거래가, 팩토리온(factoryon.go.kr) 공장 실거래가, "
                    f"혹은 해당 행정구역(시군구/읍면동) 인근 공업용지/공장 매매 평당 가격과 최근 실거래가 정보를 구글 검색을 통해 찾아주세요. "
                    f"구체적인 거래 금액(만원, 억원 단위), 거래 면적(㎡), 계약 년월 정보를 포함하여 150자 내로 간결하게 한글로 서술하십시오."
                )
                search_response = search_model.generate_content(search_prompt)
                if search_response and search_response.text:
                    search_grounding_list.append(f"[{dan} 실거래 검색결과]\n{search_response.text}")
            except Exception as e:
                logging.warning(f"{dan} 실거래가 실시간 검색 실패 (일반 지식 기반 추론): {e}")
                search_grounding_list.append(f"[{dan} 실거래 검색결과]\n검색 실패. 기존 데이터를 토대로 대략적인 시세를 추정하십시오.")

        combined_search_context = "\n\n".join(search_grounding_list)

        # 2단계: 수집된 실거래가 정보를 주입하여 구조화된 JSON 응답 생성
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": TopComplexesResponse,
                "temperature": 0.2
            },
            system_instruction=(
                "당신은 대한민국 부울경(부산, 울산, 경상남도) 지역의 산업단지 및 토지/공장 부동산 실거래가 분석 전문가입니다. "
                "주어진 산업단지들의 지리적 위도/경도 좌표를 제시하고, 각 산업단지의 고유 특성, 장점, 주력 산업군을 요약하십시오. "
                "특히, 제공된 [실시간 실거래 검색결과]를 바탕으로 해당 산업단지 또는 인근 지역의 공장/토지 평당 실거래가(만원 단위)를 "
                "수학적으로 정확하게 계산하여 'price_per_pyeong'과 'recent_transaction_info' 필드에 기입해 주세요. "
                "계산 공식: 평당 가격 = 거래금액 / (면적㎡ * 0.3025) 또는 (1㎡당 가격 * 3.3058)을 적용하십시오."
                f"\n\n[실시간 실거래 검색결과]\n{combined_search_context}"
            )
        )
        
        prompt = f"다음 산업단지들의 지리적 위도/경도 좌표, 주요 특성, 그리고 평당 실거래가 분석 결과를 JSON으로 반환해 주세요:\n{', '.join(complexes)}"
        response = model.generate_content(prompt)
        result = json.loads(response.text)
        return result

