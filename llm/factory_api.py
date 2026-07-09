# -*- coding: utf-8 -*-
"""
한국산업단지공단 공장등록정보조회 Open API 연동 모듈
설명: 사용자의 타겟 업종 키워드를 기반으로 5대 추천 산단 내 유사 업종 집적율(Cluster Density)을 분석합니다.
"""

import os
import requests
import urllib.parse
import pandas as pd
from sqlalchemy import create_engine, text
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
    로컬 CSV 캐시 파일(data/factory_registry_cache.csv)에서 
    해당 산업단지의 유사 업종 기업들의 통계를 분석하여 오프라인으로 반환합니다.
    """
    target_ir_name = None
    for k, v in IR_NAME_MAP.items():
        if k in dan_name_keyword:
            target_ir_name = v
            break
            
    if not target_ir_name:
        target_ir_name = f"{dan_name_keyword}일반산업단지"

    csv_path = "./data/factory_registry_cache.csv"
    
    # 캐시 파일이 존재하지 않는 비상 상황에만 API 조회로 동적 폴백
    if not os.path.exists(csv_path):
        print(f"⚠️ [폴백] 로컬 캐시가 없어 온라인 API를 호출합니다: {target_ir_name}")
        return get_factory_cluster_density_online(target_ir_name, industry_keyword)

    print(f"[OFFLINE 조회] 산업단지명: {target_ir_name}, 업종키워드: {industry_keyword}")

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        
        # 1. irsttNm 컬럼 매칭 필터링
        sub = df[df["irsttNm"] == target_ir_name].copy()
        
        # 2. 매칭된 행이 전혀 없을 시 부분 유사 명칭 검색으로 유연성 확보
        if len(sub) == 0:
            sub = df[df["irsttNm"].str.contains(dan_name_keyword, na=False)].copy()
            
        if len(sub) == 0:
            return {"density": 0.0, "matched_count": 0, "total_count": 0, "companies": []}

        # 3. 유사 업종 필터링
        clean_industry = str(industry_keyword).strip().lower()
        
        # indutyNm(업종명) 또는 indutyCode(업종코드)에 해당 검색 키워드가 포함되어 있는지 대조
        mask = (sub["indutyNm"].str.lower().str.contains(clean_industry, na=False)) | \
               (sub["indutyCode"].str.lower().str.contains(clean_industry, na=False))
               
        matched_df = sub[mask].copy()
        
        # 회사명 기준 중복 제거
        matched_df = matched_df.drop_duplicates(subset=["cmpnyNm"])
        total_analyzed_df = sub.drop_duplicates(subset=["cmpnyNm"])
        
        total_analyzed = len(total_analyzed_df)
        matched_cnt = len(matched_df)
        density = (matched_cnt / total_analyzed * 100) if total_analyzed > 0 else 0.0
        
        # 결과 리스트 딕셔너리 정규화
        matched_companies = []
        for _, row in matched_df.iterrows():
            matched_companies.append({
                "name": row["cmpnyNm"],
                "industry": row["indutyNm"],
                "code": row["indutyCode"]
            })

        return {
            "density": round(density, 1),
            "matched_count": matched_cnt,
            "total_count": total_analyzed,
            "companies": matched_companies[:8]  # 최대 8개 샘플 반환 (Gemini 토큰 절약 최적화)
        }
        
    except Exception as e:
        print(f"⚠️ 로컬 캐시 조회 중 오류 발생: {e}")
        return {"density": 0.0, "matched_count": 0, "total_count": 0, "companies": []}


def get_factory_cluster_density_online(target_ir_name, industry_keyword):
    """
    [비상 폴백] 실시간 Open API를 통해 단지 등록 정보를 조회하는 온라인 오리지널 함수.
    """
    params = {
        "serviceKey": API_KEY,
        "pageNo": "1",
        "numOfRows": "300",
        "irsttNm": target_ir_name,
        "type": "JSON"
    }

    try:
        query_string = urllib.parse.urlencode(params)
        full_url = f"{BASE_URL}?{query_string}"
        response = requests.get(full_url, timeout=8.0)
        
        if response.status_code != 200:
            return {"density": 0.0, "matched_count": 0, "total_count": 0, "companies": []}
            
        res_data = response.json()
        body = res_data.get("response", {}).get("body", {})
        items_wrap = body.get("items", {})
        
        if not items_wrap or "item" not in items_wrap:
            return {"density": 0.0, "matched_count": 0, "total_count": 0, "companies": []}
            
        item_list = items_wrap["item"]
        if isinstance(item_list, dict):
            item_list = [item_list]
            
        matched_companies = []
        clean_industry = str(industry_keyword).strip().lower()
        
        for item in item_list:
            induty_nm = str(item.get("indutyNm", "")).strip().lower()
            induty_code = str(item.get("indutyCode", "")).strip().lower()
            cmpny_nm = str(item.get("cmpnyNm", "")).strip()
            
            if clean_industry in induty_nm or clean_industry in induty_code:
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
            "companies": matched_companies[:8]
        }
    except Exception:
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

def get_nearest_transit(lat, lon):
    """
    Neon DB(PostgreSQL)의 subway_stations, bus_stations 테이블을 대상으로,
    주어진 위경도(lat, lon)에서 가장 가까운 지하철역 및 버스정류장까지의 최단거리(km)와 명칭을 구합니다.
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return {
            "subway_text": "정보 없음 (DB 설정 누락)",
            "bus_text": "정보 없음 (DB 설정 누락)"
        }
        
    try:
        engine = create_engine(db_url)
        
        # SQL 레벨 하버사인 구면 소팅 쿼리 (제한 1개)
        subway_query = text("""
            SELECT station_name, line_name,
                   (6371 * acos(cos(radians(:lat)) * cos(radians(latitude)) * cos(radians(longitude) - radians(:lon)) + sin(radians(:lat)) * sin(radians(latitude)))) AS distance_km
            FROM subway_stations
            ORDER BY distance_km ASC
            LIMIT 1
        """)
        
        bus_query = text("""
            SELECT station_name,
                   (6371 * acos(cos(radians(:lat)) * cos(radians(latitude)) * cos(radians(longitude) - radians(:lon)) + sin(radians(:lat)) * sin(radians(latitude)))) AS distance_km
            FROM bus_stations
            ORDER BY distance_km ASC
            LIMIT 1
        """)
        
        with engine.connect() as conn:
            # 1) 지하철역 쿼리
            sub_res = conn.execute(subway_query, {"lat": float(lat), "lon": float(lon)}).fetchone()
            # 2) 버스 정류장 쿼리
            bus_res = conn.execute(bus_query, {"lat": float(lat), "lon": float(lon)}).fetchone()
            
        subway_text = "정보 없음"
        if sub_res:
            s_name, s_line, s_dist = sub_res[0], sub_res[1], float(sub_res[2])
            if s_dist < 1.0:
                dist_str = f"{int(s_dist * 1000)}m"
            else:
                dist_str = f"{s_dist:.2f}km"
            subway_text = f"{s_name} ({s_line}, {dist_str})"
            
        bus_text = "정보 없음"
        if bus_res:
            b_name, b_dist = bus_res[0], float(bus_res[1])
            if b_dist < 1.0:
                dist_str = f"{int(b_dist * 1000)}m"
            else:
                dist_str = f"{b_dist:.2f}km"
            bus_text = f"{b_name} ({dist_str})"
            
        return {
            "subway_text": subway_text,
            "bus_text": bus_text
        }
    except Exception as e:
        print(f"⚠️ 대중교통 DB 쿼리 중 오류 발생: {e}")
        return {
            "subway_text": "정보 없음",
            "bus_text": "정보 없음"
        }
