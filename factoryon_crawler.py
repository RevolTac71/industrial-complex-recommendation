# -*- coding: utf-8 -*-
"""
FactoryON(팩토리온) 실거래가 조회 크롤러 스크립트
설명: Selenium을 활용하여 factoryon.go.kr/geon/map/map.do 지도 페이지의 좌측 실거래가 조회 기능을 자동화하고,
      조회된 공장/토지 실거래가를 평당 가격으로 환산하여 CSV 파일로 저장합니다.
요구사항: pip install selenium webdriver-manager pandas
"""

import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager

def run_factoryon_crawler(target_sido="부산광역시", target_sigungu="해운대구", target_dong="우동", target_year="2025"):
    print(f"=== FactoryON 실거래가 크롤러 시작 ({target_sido} {target_sigungu} {target_dong} - {target_year}년) ===")
    
    # 1. 크롬 드라이버 옵션 설정
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless') # 디버깅을 위해 로컬에서 브라우저가 직접 보이도록 설정
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

    # 2. 드라이버 실행
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.implicitly_wait(10)
    
    try:
        # 3. 팩토리온 GIS 지도 페이지 접속
        url = "https://www.factoryon.go.kr/geon/map/map.do"
        driver.get(url)
        time.sleep(5) # 페이지 및 지도 엔진 완전 로드 대기
        
        # 4. 좌측 실거래가 탭 버튼 클릭 (클릭 장애 방지를 위해 JS 강제 클릭 적용)
        print("실거래가 탭 진입 시도...")
        real_price_tab = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "button.price.layerOpen"))
        )
        # 일반 클릭 대신 JavaScript 강제 클릭으로 로딩 가림막 가로막힘 우회
        driver.execute_script("arguments[0].click();", real_price_tab)
        print("실거래가 탭 진입 성공! 레이어 팝업 로딩 중...")
        time.sleep(2)
        
        # 5. 검색 조건 선택 (기준연도, 시도, 시군구, 읍면동)
        # 동적 생성되는 select 드롭다운 요소 대기 및 선택
        print("검색 조건 설정 중...")
        
        # 기준연도 선택 (태그를 select로 명시)
        year_select_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//select[contains(@id, 'Year') or contains(@id, 'year') or contains(@id, 'Yy') or contains(@id, 'yy') or contains(@id, 'Date') or contains(@id, 'date') or contains(@class, 'year') or @name='year']"))
        )
        Select(year_select_element).select_by_visible_text(target_year)
        
        # 시도 선택
        sido_select_element = driver.find_element(By.XPATH, "//select[contains(@id, 'Sido') or contains(@id, 'sido') or contains(@id, 'sd') or contains(@class, 'sido') or @name='sido']")
        Select(sido_select_element).select_by_visible_text(target_sido)
        time.sleep(1) # 시군구 리스트 동적 렌더링 대기
        
        # 시군구 선택
        sigungu_select_element = driver.find_element(By.XPATH, "//select[contains(@id, 'Sigungu') or contains(@id, 'sigungu') or contains(@id, 'sgg') or contains(@class, 'sigungu') or @name='sigungu']")
        Select(sigungu_select_element).select_by_visible_text(target_sigungu)
        time.sleep(1) # 읍면동 리스트 동적 렌더링 대기
        
        # 읍면동 선택
        dong_select_element = driver.find_element(By.XPATH, "//select[contains(@id, 'Dong') or contains(@id, 'dong') or contains(@id, 'umd') or contains(@class, 'dong') or @name='dong']")
        Select(dong_select_element).select_by_visible_text(target_dong)
        
        # 6. 조회 버튼 클릭
        print("조회 실행...")
        search_btn = driver.find_element(By.XPATH, "//button[contains(text(), '조회')] | //a[contains(text(), '조회')] | //*[contains(@id, 'btnSearch') or contains(@id, 'searchBtn') or contains(@class, 'search')]")
        driver.execute_script("arguments[0].click();", search_btn)
        time.sleep(4) # 데이터 비동기 조회 완료 대기
        
        # 7. 테이블 데이터 파싱
        print("데이터 파싱 중...")
        # 동적으로 생성된 테이블 행(tr) 탐색
        rows = driver.find_elements(By.XPATH, "//table[contains(@class, 'list')]//tr | //tbody[contains(@id, 'List') or contains(@id, 'Result') or contains(@id, 'tbody')]//tr | //*[contains(@class, 'grid')]//tr")
        
        data_list = []
        for idx, row in enumerate(rows):
            cols = row.find_elements(By.TAG_NAME, "td")
            if not cols:
                continue # 헤더행 등 건너뜀
                
            col_text = [col.text.strip() for col in cols]
            
            # 테이블 컬럼이 최소 4개 이상 매핑되어 들어왔는지 체크
            if len(col_text) >= 4:
                dan_info = col_text[0] # 단지명 또는 지번
                contract_date = col_text[1] # 계약일
                amount_str = col_text[2].replace(",", "") # 거래금액 (만원단위)
                area_str = col_text[3] # 면적 (㎡)
                
                try:
                    amount = float(amount_str)
                    area = float(area_str.replace(",", ""))
                    
                    # 8. 평당 시세 환산 연산
                    pyeong = area * 0.3025
                    price_per_pyeong = amount / pyeong if pyeong > 0 else 0
                    
                    data_list.append({
                        "행정구역": f"{target_sido} {target_sigungu} {target_dong}",
                        "단지명/지번": dan_info,
                        "계약일자": contract_date,
                        "거래금액(만원)": amount,
                        "면적(㎡)": area,
                        "평수": round(pyeong, 2),
                        "평당단가(만원)": round(price_per_pyeong, 1)
                    })
                except Exception as ex:
                    # 파싱 예외 발생 시 원본 데이터 보존 기록
                    data_list.append({
                        "행정구역": f"{target_sido} {target_sigungu} {target_dong}",
                        "단지명/지번": dan_info,
                        "계약일자": contract_date,
                        "원본데이터": str(col_text),
                        "파싱에러": str(ex)
                    })
        
        # 9. CSV 파일 내보내기
        if data_list:
            df = pd.DataFrame(data_list)
            output_file = f"factoryon_real_estate_{target_sigungu}_{target_dong}.csv"
            df.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"\n📊 [성공] {len(data_list)}건의 실거래 정보가 '{output_file}'에 저장되었습니다.")
            print(df.head(5))
        else:
            # 테이블 구조가 예상과 다른 경우 전체 HTML 구조 진단을 위해 텍스트 로그 출력
            print("⚠️ 조회 완료하였으나 테이블에 표시된 실거래 데이터를 파싱하지 못했습니다.")
            # 페이지에 나타난 첫 몇개의 td 요소 내용만 출력해 구조 진단 유도
            tds = driver.find_elements(By.TAG_NAME, "td")
            if tds:
                print("발견된 td 샘플 데이터:")
                for td in tds[:10]:
                    print(f" - {td.text.strip()}")
            
    except Exception as e:
        print(f"❌ 크롤러 구동 중 오류 발생: {e}")
    finally:
        driver.quit()
        print("=== 크롤러 종료 ===")

if __name__ == "__main__":
    # 기본값으로 해운대구 우동의 2025년 실거래 조회를 시연합니다.
    run_factoryon_crawler(
        target_sido="부산광역시",
        target_sigungu="해운대구",
        target_dong="우동",
        target_year="2025"
    )
