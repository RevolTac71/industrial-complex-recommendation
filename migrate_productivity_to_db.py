import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
db_url = os.getenv("DATABASE_URL")
if not db_url:
    raise ValueError("DATABASE_URL 환경 변수가 없습니다.")

engine = create_engine(db_url)

print("=== dan_productivity 마이그레이션 ===")
df = pd.read_csv("./data/industrial_park_productivity_prediction.csv", dtype={"DAN_ID": str}, encoding="utf-8-sig")
print(f"  로컬 CSV 행수: {len(df)}")

# DB 적재 (덮어쓰기)
df.to_sql("dan_productivity", engine, if_exists="replace", index=False)
print("  dan_productivity 테이블 업로드 완료")

with engine.connect() as conn:
    cnt = conn.execute(text("SELECT COUNT(*) FROM dan_productivity")).scalar()
    print(f"  DB 확인: 총 {cnt}행 적재 완료")
