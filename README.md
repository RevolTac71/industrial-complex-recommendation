# 산업단지 입지 추천

부산·울산·경남 지역 산업단지 관련 데이터와 공간정보를 이용해 입지 후보를 살펴보는 Streamlit 프로젝트입니다. 사용자 조건에 따른 가중치 설정과 지도 기반 결과 확인을 지원합니다.

## 구성

- `app.py`: Streamlit 화면 및 분석 흐름
- `analysis/`: 공간 분석 인터페이스와 기본 파이프라인
- `llm/`: Gemini 연동 및 관련 API 코드
- `data/`: 산업단지·주변 시설 등 데이터
- `eda.ipynb`: 탐색적 분석 노트북
- `docs/blueprint.md`: 프로젝트 설계 문서
- `build_dan_coords.py`, `migrate_productivity_to_db.py`: 데이터 준비 및 이전 스크립트

## 실행

Python 의존성은 `requirements.txt`에 있습니다. 일부 기능은 데이터베이스, 외부 API 및 환경 설정에 의존합니다. 저장소의 데이터·설정과 관련 문서를 확인한 후 실행하세요.

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

실행 환경 및 외부 서비스 연결은 이 문서 작성 시 검증하지 않았습니다. 추천 결과는 입력 데이터와 설정한 가중치에 영향을 받으므로 실제 입지 의사결정 전에 별도 확인이 필요합니다.
