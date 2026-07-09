# -*- coding: utf-8 -*-
"""
한국산업단지공단 공장등록정보조회 Open API 연동 모듈
설명: 사용자의 타겟 업종 키워드를 기반으로 5대 추천 산단 내 유사 업종 집적율(Cluster Density)을 분석합니다.
"""

import os
import requests
import urllib.parse
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# 공공데이터포털 API Key (디코딩된 키 사용 권장)
API_KEY = os.getenv("DATA_PORTAL_API_KEY", "6d50ea661cd8af8ff1a63bcba54f1e53d1047232c0d1c20d9ae0f1b14b4dffa4")
BASE_URL = "http://apis.data.go.kr/B550624/fctryRegistInfo/getFctryListInIrsttService_v2"

# 5대 입지 명칭 ➡️ 공장등록 API용 산업단지 한글 정명 매핑 테이블
IR_NAME_MAP = {
    "센텀": "센텀시티일반산업단지",
    "센텀시티": "센텀시티일반산업단지",
    "신평장림": "신평장림일반산업단지",
    "금곡": "부산금곡일반산업단지",
    "회동석대": "회동석대일반산업단지",
    "서김해": "서김해일반산업단지"
}

def get_factory_cluster_density(dan_name_keyword, industry_keyword):
    """
    특정 산업단지에서 사용자가 요구하는 업종 키워드에 해당하는 입주 공장의 비중 및 회사 샘플을 조회합니다.
    """
    # 1. 단지명 매핑 변환
    target_ir_name = None
    for k, v in IR_NAME_MAP.items():
        if k in dan_name_keyword:
            target_ir_name = v
            break
            
    if not target_ir_name:
        # 매핑에 없을 시 예외 처리로 원본 키워드를 시도하되, 없으면 기본값 세팅
        target_ir_name = f"{dan_name_keyword}일반산업단지"

    print(f"[API 호출] 산업단지명: {target_ir_name}, 업종키워드: {industry_keyword}")

    params = {
        "serviceKey": API_KEY,
        "pageNo": "1",
        "numOfRows": "300",  # 집적율 계산을 위해 대량 로드
        "irsttNm": target_ir_name,
        "type": "JSON"
    }

    try:
        # URL 디코딩 오류 방지를 위해 직접 쿼리스트링 조합 호출
        query_string = urllib.parse.urlencode(params)
        full_url = f"{BASE_URL}?{query_string}"
        
        # 타임아웃 8초 설정하여 API 지연 시 플랫폼이 뻗지 않도록 방어
        response = requests.get(full_url, timeout=8.0)
        
        if response.status_code != 200:
            print(f"⚠️ 공장등록 API 호출 실패 (Status: {response.status_code})")
            return {"density": 0.0, "matched_count": 0, "total_count": 0, "companies": []}
            
        res_data = response.json()
        
        # 공공데이터 API 응답 바디 구조 파싱
        body = res_data.get("response", {}).get("body", {})
        total_count = int(body.get("totalCount", 0))
        items_wrap = body.get("items", {})
        
        if not items_wrap or "item" not in items_wrap:
            return {"density": 0.0, "matched_count": 0, "total_count": 0, "companies": []}
            
        item_list = items_wrap["item"]
        if isinstance(item_list, dict):
            item_list = [item_list]  # 단일 항목인 경우 리스트 치환
            
        # 유사 업종 매칭 개수 및 회사 목록 수집
        matched_companies = []
        clean_industry = str(industry_keyword).strip().lower()
        
        for item in item_list:
            induty_nm = str(item.get("indutyNm", "")).strip().lower()
            induty_code = str(item.get("indutyCode", "")).strip().lower()
            cmpny_nm = str(item.get("cmpnyNm", "")).strip()
            
            # 업종명 또는 업종코드에 키워드가 매칭되는지 판정 (예: '반도체', 'C26')
            if clean_industry in induty_nm or clean_industry in induty_code:
                # 중복 배제하여 리스트 기재
                if cmpny_nm not in [c["name"] for c in matched_companies]:
                    matched_companies.append({
                        "name": cmpny_nm,
                        "industry": item.get("indutyNm", ""),
                        "code": item.get("indutyCode", "")
                    })

        total_analyzed = len(item_list)
        matched_cnt = len(matched_companies)
        density = (matched_cnt / total_analyzed * 100) if total_analyzed > 0 else 0.0
        
        return {
            "density": round(density, 1),
            "matched_count": matched_cnt,
            "total_count": total_analyzed,
            "companies": matched_companies[:8]  # 최대 8개 회사 샘플만 반환하여 프롬프트 토큰 최적화
        }

    except Exception as e:
        print(f"⚠️ 공장등록 API 데이터 수집 중 예외 발생: {e}")
        return {"density": 0.0, "matched_count": 0, "total_count": 0, "companies": []}

def get_land_price_stats(sigungu_name, dong_name):
    """
    factoryon_landprice_2025.csv 데이터를 분석하여,
    해당 동네의 공장 관련 지목(공장용지, 대, 잡종지, 창고용지 등)의 평당가 통계(최저, 평균, 최고)를 산출합니다.
    """
    csv_path = "./data/factoryon_landprice_2025.csv"
    if not os.path.exists(csv_path):
        return {"min": 0, "avg": 0, "max": 0, "count": 0, "text": "실거래 데이터 없음"}
        
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        clean_sigungu = str(sigungu_name).strip().split()[-1] if sigungu_name else ""
        clean_dong = str(dong_name).strip()
        
        mask = (df["시군구"].str.contains(clean_sigungu)) & (df["읍면동"].str.contains(clean_dong))
        sub = df[mask].copy()
        
        # 공장 관련 주요 지목 필터링
        factory_jimok = ["공장용지", "대", "잡종지", "창고용지"]
        sub = sub[sub["지목"].isin(factory_jimok)]
        
        if len(sub) == 0:
            # 동 단위가 없을 시 구 단위로 광역화
            mask_gu = df["시군구"].str.contains(clean_sigungu)
            sub = df[mask_gu].copy()
            sub = sub[sub["지목"].isin(factory_jimok)]
            
        if len(sub) == 0:
            return {"min": 0, "avg": 0, "max": 0, "count": 0, "text": "해당 지역 실거래 정보 부족"}
            
        # 평당가(만원) 계산 (1평 = 3.3058 m2)
        sub["pyeong_price"] = sub["거래금(만원)"] / (sub["거래면적(m2)"] / 3.3058)
        
        # 이상치 제거
        sub = sub[(sub["pyeong_price"] >= 5) & (sub["pyeong_price"] <= 10000)]
        
        if len(sub) == 0:
            return {"min": 0, "avg": 0, "max": 0, "count": 0, "text": "유효 시세 데이터 없음"}
            
        p_min = int(sub["pyeong_price"].min())
        p_avg = int(sub["pyeong_price"].mean())
        p_max = int(sub["pyeong_price"].max())
        count = len(sub)
        
        # 직관적으로 알기 쉬운 자연어 텍스트로 치환하여 챗봇 가독성 극대화
        text = f"평당 평균 {p_avg:,}만 원 선 (최근 거래 시세: 평당 {p_min:,}만 ~ {p_max:,}만 원)"
        
        return {
            "min": p_min,
            "avg": p_avg,
            "max": p_max,
            "count": count,
            "text": text
        }
    except Exception as e:
        print(f"⚠️ 실거래가 통계 산출 중 오류: {e}")
        return {"min": 0, "avg": 0, "max": 0, "count": 0, "text": "시세 분석 실패"}
