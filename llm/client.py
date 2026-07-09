import os
import json
import logging
import urllib.parse
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import BaseModel, Field
from .prompt_templates import SYSTEM_INSTRUCTION, USER_PROMPT_TEMPLATE
from .factory_api import get_factory_cluster_density, get_land_price_stats, get_nearest_transit

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
    lat: float = Field(description="산업단지 내부(임야, 바다 등)가 아닌, 구글 스트리트뷰(로드뷰) 파노라마가 실제로 촬영되어 존재하는 가장 가까운 인접 도로변 또는 진입로의 실제 위도 (latitude) 좌표. 반드시 부산, 울산, 경상남도 지리 범위 내의 유효값(35.0 ~ 35.6)을 실수형(float)으로 제시하십시오. 누락하거나 0.0을 주면 안 됩니다.")
    lon: float = Field(description="산업단지 내부(임야, 바다 등)가 아닌, 구글 스트리트뷰(로드뷰) 파노라마가 실제로 촬영되어 존재하는 가장 가까운 인접 도로변 또는 진입로의 실제 경도 (longitude) 좌표. 반드시 부산, 울산, 경상남도 지리 범위 내의 유효값(128.5 ~ 129.4)을 실수형(float)으로 제시하십시오. 누락하거나 0.0을 주면 안 됩니다.")

    short_desc: str = Field(description="산업단지의 주요 특성 및 장점에 대한 한국어 한 줄 요약 (최대 60자)")
    detail_desc: str = Field(description="산업단지의 산업군 구성, 교통/정주 편의성, 향후 발전 방향 등을 담은 상세한 한국어 설명 (3~4문장)")
    price_per_pyeong: str = Field(description="해당 산업단지 또는 인근 지역(시군구/읍면동/특정 동)의 최근 공장/토지 평당 평균 실거래가 (예: '평당 약 150만 원'). 팩토리온 정보를 구체적으로 기입하며, 찾기 어려운 경우 인근 유사 공업지역의 실거래가를 기준으로 유추하십시오.")
    recent_transaction_info: str = Field(description="최근 팩토리온/국토부 실거래 내역 요약 (예: '2025년 8월, 3,200㎡ 매매 15.5억 원, 평당 약 160만 원'). 년월, 거래 규모, 거래액을 포함하며 정보가 없으면 '정보 없음'으로 처리하십시오.")
    subway_distance: str = Field(description="해당 산업단지 경계 또는 중심에서 가장 가까운 지하철역 명칭 및 도보 거리/시간 (예: '벡스코역 (도보 5분, 350m)'). 지하철역이 멀거나 없는 경우 버스 연계 시간과 거리 혹은 가장 가까운 주정차역 정보를 성실히 제공하십시오.")
    bus_distance: str = Field(description="해당 산업단지 경계 또는 중심에서 가장 가까운 버스정류장 명칭 및 도보 거리/시간 (예: '센텀시티역.벡스코 정류소 (도보 2분, 120m)')")

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
        
        # 1. 시스템 환경 변수 및 .env에서 최우선 로드 시도
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            self.api_key = api_key.strip()
            self.api_key_source = "Environment Variable / .env"
        # 2. 없거나 비어있는 경우 st.secrets에서 로드 시도
        else:
            try:
                if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"]:
                    self.api_key = st.secrets["GEMINI_API_KEY"].strip()
                    self.api_key_source = "Streamlit Secrets Only"
                else:
                    self.api_key = None
                    self.api_key_source = "Not Found in Environment or Secrets"
            except Exception:
                self.api_key = None
                self.api_key_source = "Not Found (Streamlit Secrets Error)"
            
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
        # 1단계: 직접 구글 검색 페이지를 백엔드에서 스크레이핑하여 기사/논문 출처 수집 (API 툴 차단 및 400 에러 영구 차단)
        search_prompt = f"산업단지 입지 선정 조건 '{user_input}' 관련 국내 언론 뉴스 보도자료 및 정책 연구"
        search_grounding = google_search_fallback(search_prompt)

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

    def extract_industry_keyword(self, user_input: str) -> str:
        """
        사용자의 요구사항(자연어)에서 한국산업단지공단 공장등록 업종명(indutyNm) 매칭에
        가장 적합한 단 하나의 대표 업종 키워드(예: 반도체, 자동차, 화학, 금속, 선박, 식품 등)를 추출합니다.
        """
        if not self.is_configured:
            return "기계"
            
        model = genai.GenerativeModel(self.model_name)
        prompt = (
            "사용자의 산업단지 추천 요구사항에서 공장등록정보 업종 매칭을 위한 "
            "단 하나의 핵심 한국어 명사 업종 키워드(예: 자동차, 반도체, 선박, 화학, 식품, 금속, 섬유 등)만 추출해 주세요.\n"
            "출력은 아무런 부연 설명 없이 딱 단어 한 개만 반환해야 합니다.\n"
            f"요구사항: {user_input}"
        )
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception:
            return "기계"

    def get_top_complexes_details(self, complexes: list[dict], industry_keyword: str = "기계") -> dict:
        """
        상위 5개 산업단지명 리스트에 대한 정보(위도, 경도, 한줄 특성, 상세 특성)를
        구조화된 형태로 받아옵니다.
        기본 모델 한도 초과 시 gemini-2.5-flash 모델로 자동 폴백 재시도합니다.
        """
        if not self.is_configured:
            raise ValueError("GEMINI_API_KEY가 설정되어 있지 않거나 설정에 실패했습니다.")
            
        try:
            return self._execute_top_complexes_details(complexes, industry_keyword, self.model_name)
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "quota" in err_msg or "exhausted" in err_msg or "resource_exhausted" in err_msg:
                logging.warning(f"기본 모델 {self.model_name} 쿼타 초과로 {self.fallback_model_name} 모델로 자동 전환 재시도합니다. 에러: {e}")
                try:
                    return self._execute_top_complexes_details(complexes, industry_keyword, self.fallback_model_name)
                except Exception as ex:
                    logging.error(f"폴백 모델 {self.fallback_model_name} 실행 중에도 오류 발생: {ex}")
                    raise ex
            else:
                raise e

    def _execute_top_complexes_details(self, complexes: list[dict], industry_keyword: str, model_name: str) -> dict:
        """
        실제 단지 상세 정보를 요청하는 핵심 로직.
        공공데이터포털 공장등록정보 API 및 로컬 실거래가 통계 데이터를 결합하여 프롬프트 컨텍스트를 구성합니다.
        """
        search_grounding_list = []
        for dan in complexes:
            dan_name = dan.get('dan_name', '')
            sigungu = dan.get('sigungu', '')
            lat = dan.get('lat')
            lon = dan.get('lon')
            
            # 동 이름 매핑 힌트 (5대 산단 중심)
            dong_map = {
                "센텀": "우동",
                "센텀시티": "우동",
                "신평장림": "신평동",
                "금곡": "금곡동",
                "회동석대": "회동동",
                "서김해": "주촌면"
            }
            dong_name = next((v for k, v in dong_map.items() if k in dan_name), "우동")
            
            # 1) 공공데이터포털 API를 통한 특정 업종 밀집도 데이터 분석
            cluster_info = get_factory_cluster_density(dan_name, industry_keyword)
            
            # 2) 8354건의 로컬 2025 실거래가 데이터 기반 직관적 평당가 통계 획득
            price_info = get_land_price_stats(sigungu, dong_name)
            
            # 3) Neon DB에 마이그레이션된 버스정류장/지하철역 최단거리 쿼리
            transit_info = get_nearest_transit(lat, lon) if lat and lon else {"subway_text": "정보 없음", "bus_text": "정보 없음"}
            
            companies_text = ", ".join([c["name"] for c in cluster_info["companies"]]) if cluster_info["companies"] else "정보 없음"
            
            grounding_data = (
                f"[{dan_name} 실데이터 그라운딩]\n"
                f"- 타겟 유사 업종: '{industry_keyword}'\n"
                f"- 해당 업종 입주 공장 수: {cluster_info['matched_count']}개 (분석 대상 {cluster_info['total_count']}개사 중 약 {cluster_info['density']}% 점유)\n"
                f"- 입주 유사 기업명 예시: {companies_text}\n"
                f"- 실제 평당 실거래가 통계: {price_info['text']}\n"
                f"- DB 기반 가장 가까운 지하철역: {transit_info['subway_text']}\n"
                f"- DB 기반 가장 가까운 버스정류장: {transit_info['bus_text']}\n"
            )
            search_grounding_list.append(grounding_data)

        combined_search_context = "\n\n".join(search_grounding_list)
        complexes_names = [d.get('dan_name', '') for d in complexes]

        # 2단계: 수집된 정량적 실데이터 정보를 시스템 인스트럭션으로 주입하여 JSON 응답 생성
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
                "특히 다음 사항을 준수하십시오:\n"
                "1. 주어지는 위도/경도 좌표는 절대 산업단지 중심점(임야, 바다, 공장 지붕, 건물 내부 등)으로 지정하지 마십시오. "
                "구글 스트리트뷰(로드뷰) 서비스가 정상적으로 지원되는 가장 가까운 인접 공도(도로변/진입로)의 실제 위도/경도 좌표로 보정하여 제공해야 합니다.\n"
                "2. 제공된 [실시간 정량 실데이터 분석 결과]를 바탕으로 해당 산업단지 또는 인근 지역의 공장/토지 평당 실거래가(만원 단위)와 최근 실거래 내역을 "
                "그대로 인용 또는 수학적으로 계산하여 'price_per_pyeong'과 'recent_transaction_info' 필드에 기입해 주세요. "
                "또한 'detail_desc' 필드 내에 해당 업종 입주 공장 수 및 집적도(%) 실데이터를 반드시 직접 인용하여 입지 적합성의 정량적 근거로 제시하십시오.\n"
                "3. 각 산업단지에서 가장 가까운 대중교통(지하철역 및 버스정류장)까지의 명칭과 거리(도보 시간 및 m/km 거리 단위) 정보를 정확하게 분석하여 "
                "'subway_distance'와 'bus_distance' 필드에 적시하십시오. 예: '벡스코역 (도보 5분, 350m)', '동해선 신해운대역 (도보 15분, 1km)'"
                f"\n\n[실시간 정량 실데이터 분석 결과]\n{combined_search_context}"
            )
        )
        
        prompt = f"다음 산업단지들의 지리적 위도/경도 좌표, 주요 특성, 그리고 평당 실거래가 분석 결과를 JSON으로 반환해 주세요:\n{', '.join(complexes_names)}"
        response = model.generate_content(prompt)
        result = json.loads(response.text)
        return result

    def check_multiple_complexes_eligibility(self, complexes_list: list[dict], user_business_description: str) -> dict:
        """
        제시된 복수의 산업단지(각 이름과 허용 업종 목록)에 대하여,
        Gemini 1회 호출(배치 처리)로 입주 자격 판정(가능/조건부 가능/불가)을 심사하여 결과를 반환합니다.
        """
        if not self.is_configured:
            # 한국어 주석: API 키 누락 시 모든 단지에 대해 안전한 기본값 반환
            return {
                "results": [
                    {
                        "dan_name": comp.get("dan_name", ""),
                        "status": "조건부 가능",
                        "matched_industry": "확인 불가 (API 키 누락)",
                        "analysis": "Gemini API 키가 설정되지 않아 자격 심사를 진행할 수 없습니다. 개별 확인이 필요합니다."
                    } for comp in complexes_list
                ]
            }

        # COMPLEXES CONTEXT 구성
        complexes_context_parts = []
        for comp in complexes_list:
            name = comp.get("dan_name", "")
            allowed = comp.get("allowed_industries", [])
            if not allowed:
                industries_text = "브이월드 API 조회 결과 허용 업종 정보 없음 (데이터 누락)"
            else:
                industries_text = ", ".join([
                    f"{ind.get('induty_nm')}({ind.get('category_nm', '제조업')})"
                    for ind in allowed
                ])
            complexes_context_parts.append(
                f"### 산업단지명: {name}\n- 허용 유치업종 목록: {industries_text}\n"
            )
        complexes_context = "\n".join(complexes_context_parts)

        class SingleEligibility(BaseModel):
            dan_name: str = Field(description="산업단지 이름")
            status: str = Field(description="판정 결과. 반드시 '가능', '조건부 가능', '불가' 중 하나만 출력할 것.")
            matched_industry: str = Field(description="허용 업종 목록 중 사용자의 사업 내용과 매칭된 업종명. 없으면 'N/A'.")
            analysis: str = Field(description="판정 근거 요약 (반드시 2~3문장의 간결한 요약형 한글 문장)")

        class MultipleEligibilityResult(BaseModel):
            results: list[SingleEligibility] = Field(description="각 산업단지별 입주 자격 심사 결과 리스트")

        system_inst = (
            "당신은 대한민국 산업단지 입주 자격 및 법률 행정 심사 에이전트입니다. "
            "주어진 여러 산업단지 각각의 유치 업종 허용 목록과 사용자의 사업 설명을 대비하여, "
            "각 단지별로 입주 자격 적합성을 의미론적 유사도(Semantic Match)를 바탕으로 엄격히 심사하십시오.\n\n"
            "**[의미론적 유사도 심사 지침]**\n"
            "- 허용 유치업종명은 주로 거시적인 대분류/중분류(예: '기타 기계 및 장비', '금속가공제품', '자동차', '전기장비', '제조업')로 표현됩니다.\n"
            "- 사용자의 구체적인 사업 설명(예: '자동차용 기계 부품 및 금속 가공 제조업') 내 키워드(자동차, 기계, 금속, 가공 등)가 허용 유치업종 목록 내 단어들과 의미론적으로 연관성이 있다면(예: '자동차' -> '자동차', '기계' -> '기타 기계 및 장비', '금속/가공' -> '금속가공제품') **절대 '불가'로 판정하지 마십시오. 반드시 '가능' 또는 '조건부 가능'으로 합격 판정해야 합니다.**\n"
            "- '불가' 판정은 사용자의 사업이 해당 단지의 유치 업종과 아예 하등의 관련이 없는 경우(예: 허용 업종이 '기술서비스' 뿐인데 전통 '주물 공장'을 하려 하거나, 허용 업종이 '가구 제조업' 뿐인데 '정밀 반도체 연구소'를 지으려 하는 등)에만 매우 극단적이고 보수적으로 적용해야 합니다.\n\n"
            "**[판정 기준 규칙]**\n"
            "1. 가능: 사용자의 사업 설명이 허용 업종 목록 중 하나와 의미론적으로 부합하며, 환경오염 요인이 전혀 없는 경우.\n"
            "2. 조건부 가능 (지자체 환경 협의 필요):\n"
            "   - 사용자의 사업 설명에 '염색', '도금', '주물', '폐수 다량 배출', '폐수', '유독물질', '도장', '화학물질', '배출업소' 등 환경오염 유발 키워드가 명시적으로 포함되어 있거나,\n"
            "   - 사업 설명 내용 상 화학물질 처리, 고열 가열 금속 주조, 폐수 대량 배출 등이 논리적으로 유추되는 경우,\n"
            "   - 허용 업종 코드/명칭에 부합하더라도 반드시 '조건부 가능'으로 판정해야 합니다.\n"
            "3. 불가: 사용자의 사업 설명이 허용 업종 목록의 어떤 업종과도 의미론적으로 전혀 관련이 없거나 매칭되지 않는 경우.\n\n"
            "**[출력 규칙]**\n"
            "- 반드시 입력으로 주어지는 각각의 'dan_name' 명칭을 임의로 축약하거나 생략하지 말고 **그대로 정확히 전달받은 문자열로** JSON 결과의 'dan_name' 필드에 기입해 주십시오."
        )

        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": MultipleEligibilityResult,
                    "temperature": 0.1
                },
                system_instruction=system_inst
            )
            
            prompt = (
                f"사용자 사업 설명: {user_business_description}\n\n"
                f"대상 산업단지 목록 정보:\n{complexes_context}\n\n"
                "위 정보를 토대로 제시된 모든 산업단지 각각에 대한 입주 적합성을 심사한 후 정의된 JSON 스키마로 응답해 주세요."
            )
            
            try:
                response = model.generate_content(prompt)
                return json.loads(response.text)
            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "quota" in err_msg or "exhausted" in err_msg or "resource_exhausted" in err_msg:
                    fallback_model = genai.GenerativeModel(
                        model_name=self.fallback_model_name,
                        generation_config={
                            "response_mime_type": "application/json",
                            "response_schema": MultipleEligibilityResult,
                            "temperature": 0.1
                        },
                        system_instruction=system_inst
                    )
                    response = fallback_model.generate_content(prompt)
                    return json.loads(response.text)
                else:
                    raise e
        except Exception as e:
            # 오류 발생 시 폴백 반환
            return {
                "results": [
                    {
                        "dan_name": comp.get("dan_name", ""),
                        "status": "조건부 가능",
                        "matched_industry": "N/A",
                        "analysis": f"AI 분석 진행 중 예외가 발생했습니다: {str(e)}. 안전을 위해 조건부 가능으로 판정합니다."
                    } for comp in complexes_list
                ]
            }


def google_search_fallback(query: str) -> str:
    """
    구형 SDK의 Google Search Grounding 도구 에러 문제를 해결하기 위해,
    직접 구글 검색 결과를 백엔드에서 스크레이핑하여 텍스트로 반환합니다.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
        res = requests.get(url, headers=headers, timeout=6)
        if res.status_code != 200:
            return "최근 실거래 정보를 획득하지 못했습니다. (네트워크 연결 문제)"
            
        soup = BeautifulSoup(res.text, "html.parser")
        
        search_results = []
        for g in soup.select('div.g')[:5]:
            title_el = g.select_one('h3')
            snippet_el = g.select_one('div[style*="webkit-line-clamp"]') or g.select_one('div.VwiC3b') or g.select_one('span.aCOp2e')
            
            title = title_el.text.strip() if title_el else ""
            snippet = snippet_el.text.strip() if snippet_el else ""
            
            if title:
                search_results.append(f"- 제목: {title}\n  내용: {snippet}")
                
        if not search_results:
            spans = [span.text.strip() for span in soup.find_all('span') if len(span.text.strip()) > 30]
            if spans:
                return "검색 스니펫 정보:\n" + "\n".join(spans[:3])
            return "최근 관련 실거래 정보가 없습니다."
            
        return "\n\n".join(search_results)
    except Exception as e:
        return f"실거래가 검색 요약 (검색 중 에러): {str(e)}"

