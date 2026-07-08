import pandas as pd
import geopandas as gpd
from .interface import BaseSpatialPipeline

def calculate_regional_score(df: pd.DataFrame) -> dict[str, float]:
    """
    지역여건 지표 점수를 계산합니다.
    - 비율: 환경오염물질_배출사업장수_norm (0.6), 낙후도지수_norm (0.4)
    """
    scores = {}
    if df is None or df.empty:
        return scores

    for _, row in df.iterrows():
        dan_id = str(row['DAN_ID'])
        
        val_env = float(row.get('환경오염물질_배출사업장수_norm', 0.0))
        val_env = 0.0 if pd.isna(val_env) else val_env
        
        val_dev = float(row.get('낙후도지수_norm', 0.0))
        val_dev = 0.0 if pd.isna(val_dev) else val_dev
        
        scores[dan_id] = round(val_env * 0.6 + val_dev * 0.4, 4)
            
    return scores

def calculate_logistics_score(df: pd.DataFrame) -> dict[str, float]:
    """
    물류여건 지표 점수를 계산합니다.
    - 비율: 항만거리_norm (0.4), 고속도로거리_norm (0.35), 화물역거리_norm (0.15), 공항거리_norm (0.1)
    """
    scores = {}
    if df is None or df.empty:
        return scores

    for _, row in df.iterrows():
        dan_id = str(row['DAN_ID'])
        
        val_port = float(row.get('항만거리_norm', 0.0))
        val_port = 0.0 if pd.isna(val_port) else val_port
        
        val_hwy = float(row.get('고속도로거리_norm', 0.0))
        val_hwy = 0.0 if pd.isna(val_hwy) else val_hwy
        
        val_train = float(row.get('화물역거리_norm', 0.0))
        val_train = 0.0 if pd.isna(val_train) else val_train
        
        val_air = float(row.get('공항거리_norm', 0.0))
        val_air = 0.0 if pd.isna(val_air) else val_air
        
        scores[dan_id] = round(val_port * 0.4 + val_hwy * 0.35 + val_train * 0.15 + val_air * 0.1, 4)

    return scores

def calculate_innovation_score(df: pd.DataFrame) -> dict[str, float]:
    """
    산업혁신여건 지표 점수를 계산합니다.
    - 비율: 연구소전담부서_norm (0.6), univ_count_norm (0.4)
    """
    scores = {}
    if df is None or df.empty:
        return scores

    for _, row in df.iterrows():
        dan_id = str(row['DAN_ID'])
        
        val_lab = float(row.get('연구소전담부서_norm', 0.0))
        val_lab = 0.0 if pd.isna(val_lab) else val_lab
        
        val_univ = float(row.get('univ_count_norm', 0.0))
        val_univ = 0.0 if pd.isna(val_univ) else val_univ
        
        scores[dan_id] = round(val_lab * 0.6 + val_univ * 0.4, 4)

    return scores

def calculate_living_score(df: pd.DataFrame) -> dict[str, float]:
    """
    생활정주여건 지표 점수를 계산합니다.
    - 비율: hospital_count_norm (0.35), pharmacy_count_norm (0.25), mart_count_norm (0.25), store_count_norm (0.15)
    """
    scores = {}
    if df is None or df.empty:
        return scores

    for _, row in df.iterrows():
        dan_id = str(row['DAN_ID'])
        
        val_hosp = float(row.get('hospital_count_norm', 0.0))
        val_hosp = 0.0 if pd.isna(val_hosp) else val_hosp
        
        val_pharm = float(row.get('pharmacy_count_norm', 0.0))
        val_pharm = 0.0 if pd.isna(val_pharm) else val_pharm
        
        val_mart = float(row.get('mart_count_norm', 0.0))
        val_mart = 0.0 if pd.isna(val_mart) else val_mart
        
        val_store = float(row.get('store_count_norm', 0.0))
        val_store = 0.0 if pd.isna(val_store) else val_store
        
        scores[dan_id] = round(val_hosp * 0.35 + val_pharm * 0.25 + val_mart * 0.25 + val_store * 0.15, 4)

    return scores

def calculate_mobility_score(df: pd.DataFrame) -> dict[str, float]:
    """
    근로자이동여건 지표 점수를 계산합니다.
    - 비율: bus_count_norm (0.8), subway_count_norm (0.2)
    """
    scores = {}
    if df is None or df.empty:
        return scores

    for _, row in df.iterrows():
        dan_id = str(row['DAN_ID'])
        
        val_bus = float(row.get('bus_count_norm', 0.0))
        val_bus = 0.0 if pd.isna(val_bus) else val_bus
        
        val_sub = float(row.get('subway_count_norm', 0.0))
        val_sub = 0.0 if pd.isna(val_sub) else val_sub
        
        scores[dan_id] = round(val_bus * 0.8 + val_sub * 0.2, 4)

    return scores


class DefaultSpatialPipeline(BaseSpatialPipeline):
    """
    5대 지표별 스코어를 계산하고 사용자가 임의 설정한 지표별 만점(가중치)을 반영하여
    최종 랭킹 및 종합점수를 도출하는 분석 파이프라인.
    """

    def run_analysis(
        self, 
        candidate_gdf: gpd.GeoDataFrame, 
        infrastructure_gdf: pd.DataFrame, 
        weights: dict[str, float]
    ) -> gpd.GeoDataFrame:
        """
        5대 지표 데이터를 바탕으로 최종 가중합 점수 및 랭킹을 도출합니다.
        
        Args:
            candidate_gdf: 후보지 GeoDataFrame (단지 기본 마스터 정보 및 위치 좌표)
            infrastructure_gdf: 정규화 데이터 단일 pd.DataFrame
            weights: 사용자가 수동 설정 혹은 LLM이 추천한 5대 지표별 가중치 (만점 배점)
        """
        # 1. 5대 지표 점수 산출 (각 0.0 ~ 1.0 범위)
        regional_scores = calculate_regional_score(infrastructure_gdf)
        logistics_scores = calculate_logistics_score(infrastructure_gdf)
        innovation_scores = calculate_innovation_score(infrastructure_gdf)
        living_scores = calculate_living_score(infrastructure_gdf)
        mobility_scores = calculate_mobility_score(infrastructure_gdf)

        # 2. 결과 데이터 복제
        result_gdf = candidate_gdf.copy()
        
        # 3. 5대 지표별 가중치(만점) 추출 (누락 시 기본값 20.0점)
        w_reg = float(weights.get("지역여건", 20.0))
        w_log = float(weights.get("물류여건", 20.0))
        w_inn = float(weights.get("산업혁신여건", 20.0))
        w_liv = float(weights.get("생활정주여건", 20.0))
        w_mob = float(weights.get("근로자이동여건", 20.0))

        scores_list = []
        regional_list = []
        logistics_list = []
        innovation_list = []
        living_list = []
        mobility_list = []

        # 4. 각 산업단지별 최종 종합 스코어 계산
        for _, row in result_gdf.iterrows():
            dan_id = str(row['DAN_ID'])
            
            # 개별 지표 정규화 가중합 (0.0 ~ 1.0)
            s_reg = regional_scores.get(dan_id, 0.0)
            s_log = logistics_scores.get(dan_id, 0.0)
            s_inn = innovation_scores.get(dan_id, 0.0)
            s_liv = living_scores.get(dan_id, 0.0)
            s_mob = mobility_scores.get(dan_id, 0.0)

            # 가변 만점(가중치)을 반영한 각 지표 점수 및 종합 점수 연산
            # 지표 점수 = 해당 지표 가중치 * ∑(해당 지표 세부 칼럼 정규화값 * 세부 칼럼 고정 비율)
            score_reg = round(s_reg * w_reg, 2)
            score_log = round(s_log * w_log, 2)
            score_inn = round(s_inn * w_inn, 2)
            score_liv = round(s_liv * w_liv, 2)
            score_mob = round(s_mob * w_mob, 2)
            
            final_score = round(score_reg + score_log + score_inn + score_liv + score_mob, 2)
            
            scores_list.append(final_score)
            regional_list.append(score_reg)
            logistics_list.append(score_log)
            innovation_list.append(score_inn)
            living_list.append(score_liv)
            mobility_list.append(score_mob)

        # 5. 결과 데이터프레임에 필드 주입
        result_gdf['score'] = scores_list
        result_gdf['지역여건_점수'] = regional_list
        result_gdf['물류여건_점수'] = logistics_list
        result_gdf['산업혁신여건_점수'] = innovation_list
        result_gdf['생활정주여건_점수'] = living_list
        result_gdf['근로자이동여건_점수'] = mobility_list
        
        # 6. 종합 점수를 내림차순으로 랭킹 부여
        result_gdf['rank'] = result_gdf['score'].rank(ascending=False, method='min').astype(int)
        
        return result_gdf.sort_values(by='rank')

