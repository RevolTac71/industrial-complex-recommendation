import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import folium
from streamlit_folium import st_folium
import os
import glob
from sqlalchemy import create_engine

from llm.client import GeminiLLMClient
from analysis.default_pipeline import DefaultSpatialPipeline

# 페이지 기본 설정
st.set_page_config(page_title="산업단지 입지 추천 서비스", layout="wide")
st.title("🗺️ 지능형 산업단지 입지 추천 플랫폼")

# 1. 초기 세션 상태 설정
keys = ["지역여건", "물류여건", "산업혁신여건", "생활정주여건", "근로자이동여건"]
if "current_weights" not in st.session_state:
    st.session_state.current_weights = {k: 20.0 for k in keys}

# 분석 실행 시점에 실제 사용되었던 가중치 정보 (초기값은 각각 20.0점)
if "analyzed_weights" not in st.session_state:
    st.session_state.analyzed_weights = {k: 20.0 for k in keys}

# 슬라이더 및 숫자 입력 세션 상태 초기값 할당
for k in keys:
    slider_key = f"slider_{k}"
    num_key = f"num_{k}"
    if slider_key not in st.session_state:
        st.session_state[slider_key] = 20.0
    if num_key not in st.session_state:
        st.session_state[num_key] = 20.0

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

llm_client = GeminiLLMClient()
spatial_pipeline = DefaultSpatialPipeline()

def normalize_name(name):
    if not name:
        return ""
    # 공백, 특수문자 및 접미사 제거
    for word in ["일반산업단지", "국가산업단지", "농공단지", "산업단지", "일반산단", "국가산단", "산단", "특별", "전문"]:
        name = name.replace(word, "")
    for char in [" ", "·", ".", "-", "(", ")", " 기존", " 협업"]:
        name = name.replace(char, "")
    return name.strip()

# 2. 상호 반응형 가중치 슬라이더 및 숫자 입력 동기화 콜백 함수 정의
def update_weights(changed_key, widget_type):
    # 한국어 주석: 슬라이더와 숫자 입력 칸 간의 양방향 동기화만 처리하고, 다른 지표를 자동으로 깎는 로직은 삭제합니다.
    new_val = st.session_state[f"{widget_type}_{changed_key}"]
    st.session_state.current_weights[changed_key] = new_val
    
    # 동일 지표의 다른 입력 위젯을 동기화합니다.
    other_widget_type = "num" if widget_type == "slider" else "slider"
    st.session_state[f"{other_widget_type}_{changed_key}"] = new_val


# ==================== 사이드바: LLM 챗봇 및 가중치 조정 ====================
st.sidebar.header("💬 가중치 추천 챗봇")




chat_container = st.sidebar.container(height=320)
with chat_container:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

if user_input := st.sidebar.chat_input("원하는 산업단지 요건을 자연어로 설명하세요."):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    try:
        response = llm_client.get_weight_recommendation(user_input)
        recommendation_reason = response.get("reason", "가중치 비율을 추천했습니다.")
        
        # 추천된 가중치를 별도 보존
        st.session_state.recommended_weights = {
            "지역여건": round(response.get("지역여건", 0.20) * 100.0, 1),
            "물류여건": round(response.get("물류여건", 0.20) * 100.0, 1),
            "산업혁신여건": round(response.get("산업혁신여건", 0.20) * 100.0, 1),
            "생활정주여건": round(response.get("생활정주여건", 0.20) * 100.0, 1),
            "근로자이동여건": round(response.get("근로자이동여건", 0.20) * 100.0, 1)
        }
        
        # 현재 활성화된 가중치 및 UI 연동도 자동 적용
        for k in keys:
            st.session_state.current_weights[k] = st.session_state.recommended_weights[k]
            st.session_state[f"slider_{k}"] = st.session_state.current_weights[k]
            st.session_state[f"num_{k}"] = st.session_state.current_weights[k]
        
        st.session_state.recommendation_reason = recommendation_reason
        
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": f"{recommendation_reason}\n\n추천 분배 결과(총 100점 기준): {st.session_state.recommended_weights}"
        })
    except Exception as e:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"오류가 발생했습니다: {e}"
        })
    st.rerun()

if "recommendation_reason" in st.session_state and st.session_state.recommendation_reason:
    st.sidebar.markdown("#### 📊 최근 추천 가중치 분배")
    # 사용자가 직접 슬라이더를 건드려도 추천되었던 고정 수치가 바뀌지 않도록 st.session_state.recommended_weights 사용
    rec_w = st.session_state.get("recommended_weights", st.session_state.current_weights)
    weights_str = " | ".join([f"**{k[:2]}**: {v}%" for k, v in rec_w.items()])
    st.sidebar.info(weights_str)
    with st.sidebar.expander("💡 가중치 추천 근거 보기"):
        st.write(st.session_state.recommendation_reason)

st.sidebar.write("---")


# ==================== 단일 정규화 데이터 로드 (DB 연동) ====================
@st.cache_data
def load_data():
    db_url = os.getenv("DATABASE_URL")
    if not db_url and "DATABASE_URL" in st.secrets:
        db_url = st.secrets["DATABASE_URL"]
        
    normalized_df = None
    location_df = None
    
    if db_url:
        try:
            engine = create_engine(db_url)
            # Neon DB에서 dan_normalized 데이터 쿼리
            normalized_df = pd.read_sql("SELECT * FROM dan_normalized", con=engine)
            # Neon DB에서 dan_integrated 데이터 쿼리
            location_df = pd.read_sql("SELECT * FROM dan_integrated", con=engine)
        except Exception as e:
            st.warning(f"⚠️ DB 연결 실패, 로컬 백업 로드를 시도합니다: {e}")
            
    # 로컬 fallback
    if normalized_df is None:
        data_dir = "./data"
        
        # 정규화 데이터 탐색 및 로드
        norm_path = None
        for f in os.listdir(data_dir):
            if "정규화" in f:
                norm_path = os.path.join(data_dir, f)
                break
        if norm_path and os.path.exists(norm_path):
            try:
                normalized_df = pd.read_csv(norm_path, encoding="utf-8")
            except Exception:
                normalized_df = pd.read_csv(norm_path, encoding="cp949")
                
        # 통합(위치) 데이터 탐색 및 로드
        integrated_path = None
        for f in os.listdir(data_dir):
            if "통합" in f:
                integrated_path = os.path.join(data_dir, f)
                break
        if integrated_path and os.path.exists(integrated_path):
            try:
                location_df = pd.read_csv(integrated_path, encoding="cp949")
            except Exception:
                location_df = pd.read_csv(integrated_path, encoding="utf-8")

    return normalized_df, location_df

normalized_df, location_df = load_data()

candidate_master = None
if normalized_df is not None:
    candidate_master = normalized_df[['DAN_ID', 'DAN_NAME']].drop_duplicates()

# ==================== 사이드바: 가중치 조정 UI ====================
st.sidebar.subheader("🎛️ 입지 분석 가중치 세부 조정")

for k in keys:
    st.sidebar.caption(f"**{k} 가중치 (만점)**")
    col_slider, col_num = st.sidebar.columns([3, 1])
    with col_slider:
        st.slider(
            label=k,
            min_value=0.0,
            max_value=100.0,
            key=f"slider_{k}",
            on_change=update_weights,
            args=(k, "slider"),
            step=0.5,
            label_visibility="collapsed"
        )
    with col_num:
        st.number_input(
            label=k,
            min_value=0.0,
            max_value=100.0,
            key=f"num_{k}",
            on_change=update_weights,
            args=(k, "num"),
            step=0.5,
            label_visibility="collapsed"
        )

st.session_state.current_weights = {k: st.session_state[f"slider_{k}"] for k in keys}

# 한국어 주석: 현재 배정된 가중치 총합을 계산하고, 총합 정보 및 100점 초과 시 경고 메시지를 표시합니다.
total_weight = sum(st.session_state.current_weights.values())

st.sidebar.markdown(f"**현재 배정된 총점: {total_weight:.1f} / 100.0점**")
if total_weight > 100.0:
    st.sidebar.warning(f"⚠️ 총합이 100점을 초과했습니다! (현재 {total_weight:.1f}점)")
elif total_weight == 100.0:
    st.sidebar.success(f"✅ 총합 100점 충족")
else:
    st.sidebar.info(f"💡 총합이 100점 미만입니다. (현재 {total_weight:.1f}점)")

run_analysis_button = False
if candidate_master is not None:
    if st.sidebar.button("⚙️ 추천 입지 분석 실행", width="stretch"):
        run_analysis_button = True


# ==================== 메인 화면: 분석 결과 및 지도 시각화 ====================
col1, col2 = st.columns([1.1, 0.9])

if run_analysis_button and candidate_master is not None and normalized_df is not None:
    with st.spinner("지표별 데이터 매칭 및 가중치 반영 연산 중..."):
        if location_df is not None and 'lat' in location_df.columns and 'lon' in location_df.columns:
            candidate_master['DAN_ID'] = candidate_master['DAN_ID'].astype(str)
            location_df['DAN_ID'] = location_df['DAN_ID'].astype(str)
            
            merged_master = pd.merge(candidate_master, location_df[['DAN_ID', 'lat', 'lon']], on='DAN_ID', how='left')
            merged_master = merged_master.dropna(subset=['lat', 'lon'])
            
            geometry = [Point(xy) for xy in zip(merged_master['lon'], merged_master['lat'])]
            candidate_gdf = gpd.GeoDataFrame(merged_master, geometry=geometry, crs="EPSG:4326")
        else:
            candidate_gdf = gpd.GeoDataFrame(candidate_master, geometry=[Point(0,0)]*len(candidate_master), crs="EPSG:4326")

        result_gdf = spatial_pipeline.run_analysis(
            candidate_gdf,
            normalized_df,
            st.session_state.current_weights
        )
        st.session_state.result_gdf = result_gdf
        # 분석 실행에 실제 사용된 가중치 정보 동기화
        st.session_state.analyzed_weights = st.session_state.current_weights.copy()
        
        top_5_complexes = result_gdf.head(5)
        top_5_names = top_5_complexes['DAN_NAME'].tolist()
        
        try:
            if "top_5_details" not in st.session_state or st.session_state.top_5_details_key != tuple(top_5_names):
                # 단지명과 시군구명을 묶어 검색 힌트를 보강해 LLM 클라이언트에 전송
                complexes_info = [
                    {"dan_name": row['DAN_NAME'], "sigungu": row.get('SIGUNGU_NM', '')} 
                    for _, row in top_5_complexes.iterrows()
                ]
                response = llm_client.get_top_complexes_details(complexes_info)
                st.session_state.top_5_details = response.get("complexes", [])
                st.session_state.top_5_details_key = tuple(top_5_names)
        except Exception as e:
            st.error(f"❌ 2차 LLM 상세 정보 검색 중 오류가 발생했습니다: {e}")
            st.session_state.top_5_details = []

with col1:
    st.subheader("🗺️ 입지 공간 시각화")
    
    map_center = [35.25, 128.9]
    zoom_level = 9
    
    m = folium.Map(location=map_center, zoom_start=zoom_level, control_scale=True)
    
    if "result_gdf" in st.session_state and st.session_state.result_gdf is not None:
        result_gdf = st.session_state.result_gdf
        top_5_complexes = result_gdf.head(5)
        
        candidates_group = folium.FeatureGroup(name="추천 상위 5대 산업단지").add_to(m)
        
        llm_info_map = {}
        if "top_5_details" in st.session_state and st.session_state.top_5_details:
            for detail in st.session_state.top_5_details:
                norm_llm_name = normalize_name(detail.get('dan_name', ''))
                llm_info_map[norm_llm_name] = detail
        
        has_marked = False
        for idx, (_, row) in enumerate(top_5_complexes.iterrows()):
            dan_name = row['DAN_NAME']
            dan_id = str(row['DAN_ID'])
            rank = idx + 1
            score = row.get('score', 0.0)
            
            norm_target_name = normalize_name(dan_name)
            detail_info = llm_info_map.get(norm_target_name)
            
            if not detail_info:
                for norm_key, info in llm_info_map.items():
                    if norm_key in norm_target_name or norm_target_name in norm_key:
                        detail_info = info
                        break
            
            if not detail_info and len(st.session_state.top_5_details) > idx:
                detail_info = st.session_state.top_5_details[idx]
            
            # 1. 로컬 CSV 좌표 우선 사용 -> 2. 2차 LLM 추론 좌표 사용 -> 3. geometry 좌표 사용 순서로 매핑
            lat, lon = None, None
            if location_df is not None:
                match_df = location_df[location_df['DAN_ID'].astype(str) == dan_id]
                if not match_df.empty and 'lat' in match_df.columns:
                    lat = float(match_df.iloc[0]['lat'])
                    lon = float(match_df.iloc[0]['lon'])
            
            if (lat is None or lon is None) and detail_info:
                lat = detail_info.get('lat')
                lon = detail_info.get('lon')
            
            if lat is None or lon is None:
                geom = row.geometry
                if geom and geom.x != 0 and geom.y != 0:
                    lat, lon = geom.y, geom.x
            
            if lat is not None and lon is not None:
                has_marked = True
                short_desc = detail_info.get('short_desc', '상세 분석 정보가 로드되었습니다.') if detail_info else '상세 분석 정보가 로드되었습니다.'
                
                kakao_roadview_url = f"https://map.kakao.com/link/roadview/{lat},{lon}"
                kakao_map_url = f"https://map.kakao.com/link/map/{dan_name},{lat},{lon}"
                popup_html = f"""
                <div style="font-family: Arial; width: 220px; padding: 5px;">
                    <h4 style="margin: 0 0 5px 0; color: #1f77b4; font-size: 14px;">🏆 {dan_name} ({rank}위)</h4>
                    <b style="font-size: 12px; color: #2ca02c;">종합 평가 점수: {score}점</b>
                    <p style="font-size: 11px; margin: 5px 0 8px 0; color: #555; line-height: 1.4;"><i>"{short_desc}"</i></p>
                    <div style="margin-top: 8px; display: flex; gap: 5px; justify-content: center;">
                        <a href="{kakao_roadview_url}" target="_blank" style="display: inline-block; padding: 6px 10px; background-color: #FFEB3B; color: #3E2723; text-decoration: none; border-radius: 4px; font-size: 10px; font-weight: bold; border: 1px solid #FBC02D; text-align: center;">🛣️ 카카오 로드뷰</a>
                        <a href="{kakao_map_url}" target="_blank" style="display: inline-block; padding: 6px 10px; background-color: #FF9800; color: white; text-decoration: none; border-radius: 4px; font-size: 10px; font-weight: bold; text-align: center;">📍 카카오 지도</a>
                    </div>
                </div>
                """
                
                marker_color = "red" if rank == 1 else "orange" if rank <= 3 else "blue"
                
                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup_html, max_width=250),
                    icon=folium.Icon(color=marker_color, icon="star" if rank <= 3 else "info-sign")
                ).add_to(candidates_group)
                
        if not has_marked:
            st.warning("⚠️ 상위 5개 산업단지의 위치 좌표 정보를 획득하지 못해 지도에 표시할 수 없습니다.")
            
    folium.LayerControl().add_to(m)
    st_folium(m, width="100%", height=650, returned_objects=[])

with col2:
    st.subheader("📊 추천 분석 및 랭킹 결과")
    
    if "result_gdf" in st.session_state and st.session_state.result_gdf is not None:
        result_gdf = st.session_state.result_gdf
        
        st.markdown("### 🏆 추천 상위 5대 산업단지 상세 분석")
        
        llm_info_map = {}
        if "top_5_details" in st.session_state and st.session_state.top_5_details:
            for detail in st.session_state.top_5_details:
                norm_llm_name = normalize_name(detail.get('dan_name', ''))
                llm_info_map[norm_llm_name] = detail
        
        top_5_complexes = result_gdf.head(5)
        for idx, (_, row) in enumerate(top_5_complexes.iterrows()):
            dan_name = row['DAN_NAME']
            dan_id = str(row['DAN_ID'])
            rank = idx + 1
            score = row.get('score', 0.0)
            
            norm_target_name = normalize_name(dan_name)
            detail_info = llm_info_map.get(norm_target_name)
            
            if not detail_info:
                for norm_key, info in llm_info_map.items():
                    if norm_key in norm_target_name or norm_target_name in norm_key:
                        detail_info = info
                        break
                        
            if not detail_info and len(st.session_state.top_5_details) > idx:
                detail_info = st.session_state.top_5_details[idx]
            
            st.markdown(f"#### **{rank}위: {dan_name}** `({score}점)`")
            
            short_desc = detail_info.get('short_desc', '상세 특성을 로드하는 데 실패했거나 검색 중입니다.') if detail_info else '상세 특성을 로드하는 데 실패했거나 검색 중입니다.'
            st.markdown(f"*{short_desc}*")
            
            with st.expander("더 자세한 상세정보 보기"):
                # 지하철역 및 버스정류장 개수 (이미 데이터셋에 존재하는 3km 반경 내 개수 데이터) 표기
                subway_cnt = row.get('subway_count', 0)
                bus_cnt = row.get('bus_count', 0)
                
                import math
                subway_display = int(subway_cnt) if (pd.notnull(subway_cnt) and not math.isnan(subway_cnt)) else 0
                bus_display = int(bus_cnt) if (pd.notnull(bus_cnt) and not math.isnan(bus_cnt)) else 0
                
                st.markdown("##### 🚇 인근 대중교통 인프라 현황 (3km 반경)")
                col_sub, col_bus = st.columns(2)
                with col_sub:
                    st.markdown(
                        f"<div style='background-color: #f7f9fc; padding: 10px; border-radius: 6px; border-left: 4px solid #4CAF50;'>"
                        f"<span style='font-size: 11px; color: #666;'>지하철역 수</span><br>"
                        f"<strong style='font-size: 15px; color: #2E7D32;'>{subway_display} 개역</strong>"
                        f"</div>", 
                        unsafe_allow_html=True
                    )
                with col_bus:
                    st.markdown(
                        f"<div style='background-color: #f7f9fc; padding: 10px; border-radius: 6px; border-left: 4px solid #FF9800;'>"
                        f"<span style='font-size: 11px; color: #666;'>버스 정류장 수</span><br>"
                        f"<strong style='font-size: 15px; color: #E65100;'>{bus_display} 개소</strong>"
                        f"</div>", 
                        unsafe_allow_html=True
                    )
                st.write("") # 마진 확보
                
                detail_desc = detail_info.get('detail_desc', '상세 설명 정보를 불러오지 못했습니다.') if detail_info else '상세 설명 정보를 불러오지 못했습니다.'
                st.write(detail_desc)
                
                if detail_info:
                    price_per_pyeong = str(detail_info.get('price_per_pyeong', '')).strip()
                    recent_transaction_info = str(detail_info.get('recent_transaction_info', '')).strip()
                    
                    # 결측치 체크 ("정보 없음", "N/A" 등 예외 처리)
                    invalid_keywords = ["정보 없음", "정보없음", "n/a", "N/A", "확인 불가", "확인불가", "None", "none", "null"]
                    is_price_valid = price_per_pyeong and not any(kw in price_per_pyeong.lower() for kw in invalid_keywords)
                    is_info_valid = recent_transaction_info and not any(kw in recent_transaction_info.lower() for kw in invalid_keywords)
                    
                    st.markdown("---")
                    st.markdown("##### 💰 FactoryON 실거래가 및 평당 시세 분석")
                    
                    if is_price_valid or is_info_valid:
                        col_p1, col_p2 = st.columns([1, 2])
                        with col_p1:
                            # st.metric 대신 일반 텍스트 카드로 표현하여 텍스트 잘림("...") 방지
                            st.markdown(
                                f"<div style='background-color: #f0f2f6; padding: 12px; border-radius: 8px; border-left: 5px solid #ff4b4b; margin-bottom: 10px;'>"
                                f"<span style='font-size: 12px; color: #555;'>평당 평균 시세</span><br>"
                                f"<strong style='font-size: 16px; color: #111;'>{price_per_pyeong if is_price_valid else '검색 실패 (인근 시세 참고)'}</strong>"
                                f"</div>", 
                                unsafe_allow_html=True
                            )
                        with col_p2:
                            st.markdown(f"**최근 실거래 내역 요약**\n\n{recent_transaction_info if is_info_valid else '최근 거래 이력 정보가 존재하지 않습니다.'}")
                    else:
                        st.warning("⚠️ 해당 산업단지 혹은 인근 행정구역의 최근 팩토리온 실거래 정보를 찾을 수 없습니다. (데이터 부족)")
                    st.markdown("---")
                
                score_df = pd.DataFrame({
                    "평가 지표": ["🏭 지역여건", "🚚 물류여건", "💡 산업혁신여건", "🏡 생활정주여건", "🚶 근로자이동여건"],
                    "배정 점수 (만점)": [
                        st.session_state.analyzed_weights["지역여건"],
                        st.session_state.analyzed_weights["물류여건"],
                        st.session_state.analyzed_weights["산업혁신여건"],
                        st.session_state.analyzed_weights["생활정주여건"],
                        st.session_state.analyzed_weights["근로자이동여건"]
                    ],
                    "반영 점수": [
                        row.get('지역여건_점수', 0.0),
                        row.get('물류여건_점수', 0.0),
                        row.get('산업혁신여건_점수', 0.0),
                        row.get('생활정주여건_점수', 0.0),
                        row.get('근로자이동여건_점수', 0.0)
                    ]
                })
                st.table(score_df)
                
            st.markdown("---")
            
        st.markdown("### 📋 전체 랭킹 테이블")
        output_df = pd.DataFrame(result_gdf.drop(columns="geometry"))
        display_cols = [
            'rank', 'DAN_NAME', 'score', 
            '지역여건_점수', '물류여건_점수', 
            '산업혁신여건_점수', '생활정주여건_점수', '근로자이동여건_점수'
        ]
        valid_display_cols = [c for c in display_cols if c in output_df.columns]
        st.dataframe(output_df[valid_display_cols].sort_values(by="rank"), width="stretch")
        
    else:
        st.info("사이드바의 '추천 입지 분석 실행' 버튼을 누르시면 분석 결과 랭킹 테이블이 표시됩니다.")
