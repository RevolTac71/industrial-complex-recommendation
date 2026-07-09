# -*- coding: utf-8 -*-
"""
산업단지 WGS84 좌표 수집 스크립트
카카오 로컬 API(장소검색)를 활용하여 데이터에 존재하는 모든 산업단지의
정확한 WGS84 위경도 좌표를 수집하고 data/dan_coords.csv 로 저장합니다.

실행 방법: python build_dan_coords.py
"""

import os
import time
import pandas as pd
import requests

KAKAO_API_KEY = "3c2fe59c43808fd63005e28977a423f1"
KAKAO_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"
HEADERS = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}

# 부울경 지역 경계 박스 (WGS84): 대략적인 범위로 결과 필터링
LAT_MIN, LAT_MAX = 34.5, 36.0
LON_MIN, LON_MAX = 127.5, 130.0

# DAN_ID → (lat, lon) 하드코딩 백업 좌표 (검증된 값)
FALLBACK_COORDS = {
    "326050": (35.2398, 128.9990),   # 금곡일반산업단지
    "226040": (35.1690, 129.1281),   # 센텀시티
    "226030": (35.1056, 128.9802),   # 신평·장림일반산업단지
    "226031": (35.1056, 128.9802),   # 신평.장림(기존)
    "226032": (35.1056, 128.9802),   # 신평.장림(협업)
    "326010": (35.2066, 129.1118),   # 회동·석대일반산업단지
    "248850": (35.2092, 128.8407),   # 서김해일반산업단지 (카카오 API 실측 풍유동 좌표 반영)
}


def kakao_search_complex(dan_name: str, sigungu: str = "") -> tuple[float, float] | None:
    """
    카카오 로컬 API로 산업단지 좌표를 검색합니다.
    시군구명을 region 파라미터로 추가해 지역 범위를 좁힙니다.
    부울경 범위(lat 34.5~36.0, lon 127.5~130.0) 내에 있는 결과만 반환합니다.
    """
    # 검색 쿼리 조합: 단지명 + 산업단지 접미사
    queries = [
        f"{dan_name}",
        f"{sigungu} {dan_name}" if sigungu else f"{dan_name}",
        f"{dan_name} 산업단지",
    ]

    for query in queries:
        try:
            params = {"query": query, "size": 5}
            resp = requests.get(KAKAO_URL, headers=HEADERS, params=params, timeout=8)
            if resp.status_code != 200:
                continue

            docs = resp.json().get("documents", [])
            for doc in docs:
                lat = float(doc["y"])
                lon = float(doc["x"])
                # 부울경 범위 필터링
                if LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX:
                    return lat, lon
        except Exception as e:
            print(f"  [검색오류] {query}: {e}")
        time.sleep(0.05)  # API 요청 간 50ms 간격

    return None


def main():
    # 1. 산업단지 목록 로드
    data_dir = "./data"
    norm_path = None
    for f in os.listdir(data_dir):
        if "정규화" in f:
            norm_path = os.path.join(data_dir, f)
            break

    if not norm_path:
        print("❌ 산업단지_정규화데이터.csv 파일을 찾을 수 없습니다.")
        return

    try:
        df = pd.read_csv(norm_path, encoding="utf-8")
    except Exception:
        df = pd.read_csv(norm_path, encoding="cp949")

    # DAN_ID, DAN_NAME, SIGUNGU_NM 컬럼 추출
    required = ["DAN_ID", "DAN_NAME"]
    for col in required:
        if col not in df.columns:
            print(f"❌ 필수 컬럼 '{col}' 없음")
            return

    complexes = df[required + (["SIGUNGU_NM"] if "SIGUNGU_NM" in df.columns else [])].drop_duplicates(subset="DAN_ID")
    print(f"✅ 총 {len(complexes)}개 산업단지 좌표 수집 시작\n")

    # 2. 기존 dan_coords.csv가 있으면 이미 처리된 DAN_ID 스킵 (재시작 내성)
    output_path = os.path.join(data_dir, "dan_coords.csv")
    already_done = set()
    if os.path.exists(output_path):
        done_df = pd.read_csv(output_path, dtype={"DAN_ID": str})
        already_done = set(done_df["DAN_ID"].tolist())
        print(f"   ↩️  이미 수집된 {len(already_done)}개 건너뜀\n")

    results = []
    total = len(complexes)
    for i, (_, row) in enumerate(complexes.iterrows()):
        dan_id = str(row["DAN_ID"])
        dan_name = str(row["DAN_NAME"])
        sigungu = str(row.get("SIGUNGU_NM", "")) if "SIGUNGU_NM" in row else ""

        if dan_id in already_done:
            continue

        # 하드코딩 백업 좌표 우선 확인
        if dan_id in FALLBACK_COORDS:
            lat, lon = FALLBACK_COORDS[dan_id]
            source = "hardcoded"
        else:
            coord = kakao_search_complex(dan_name, sigungu)
            if coord:
                lat, lon = coord
                source = "kakao"
            else:
                lat, lon = None, None
                source = "not_found"

        results.append({
            "DAN_ID": dan_id,
            "DAN_NAME": dan_name,
            "lat": lat,
            "lon": lon,
            "source": source
        })

        status = f"✅ ({source})" if lat else "❌ 미발견"
        print(f"[{i+1:3d}/{total}] {dan_name:25s} ({dan_id}) → lat={lat}, lon={lon}  {status}")

        # 100건마다 중간 저장 (네트워크 오류 대비)
        if len(results) % 100 == 0:
            _flush(results, output_path, already_done)
            already_done.update(r["DAN_ID"] for r in results)
            results = []
        
        time.sleep(0.1)  # API 호출 간격

    # 3. 최종 저장
    if results:
        _flush(results, output_path, already_done)

    # 4. 결과 요약
    final_df = pd.read_csv(output_path, dtype={"DAN_ID": str})
    found = final_df["lat"].notna().sum()
    print(f"\n📊 수집 완료: {found}/{len(final_df)}개 좌표 확보")
    print(f"   저장 위치: {output_path}")
    not_found = final_df[final_df["lat"].isna()]["DAN_NAME"].tolist()
    if not_found:
        print(f"   ⚠️ 미발견 단지 ({len(not_found)}개): {not_found[:10]}")


def _flush(results: list, output_path: str, already_done: set):
    """결과를 CSV에 누적 저장합니다."""
    new_df = pd.DataFrame(results)
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path, dtype={"DAN_ID": str})
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="DAN_ID", keep="last")
        combined.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        new_df.to_csv(output_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
