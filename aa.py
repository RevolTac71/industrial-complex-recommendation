# -*- coding: utf-8 -*-
"""
부울경 산업단지 예상 지가 예측 모델 (탐색 단계)

주의: 시군구 단위로 지가를 매칭하기 때문에, 같은 시군구에 속한 산단은 y값이 동일하다.
      이 모델은 '산단 개별 분양가 예측'이 아니라 '지역 지가 수준 추정'이다.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error

LAND_CSV = "./data/factoryon_landprice_2025.csv"
PARK_CSV = r"C:\Users\상현\Desktop\2026\제 11회 부울경 AI 융합 해커톤\전처리 데이터\산업단지_정규화데이터.csv"

PYEONG = 3.3058  # 1평 = 3.3058 m2

SCENARIOS = {
    "A_좁게": ["공장용지", "잡종지", "창고용지"],
    "B_넓게": ["공장용지", "잡종지", "창고용지", "주차장", "주유소용지"],
    "C_대지포함": ["공장용지", "잡종지", "창고용지", "주차장", "주유소용지", "대"],
}

MIN_SAMPLE_WARN = 5


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------
# 1. 평당가 계산 + 이상치 제거
# ---------------------------------------------------------------
def load_and_clean_land(path):
    section("1. 평당가 계산 + 이상치 제거")

    df = pd.read_csv(path, encoding="utf-8-sig")
    df["지목"] = df["지목"].astype(str).str.strip()
    df["평당가(만원)"] = df["거래금(만원)"] / (df["거래면적(m2)"] / PYEONG)

    before_total = len(df)

    # 1차: 절대 기준 제거
    abs_mask = (df["평당가(만원)"] >= 5) & (df["평당가(만원)"] <= 10000)
    before_by_jimok = df["지목"].value_counts()
    df1 = df[abs_mask].copy()
    after1_by_jimok = df1["지목"].value_counts()

    print(f"\n[1차 절대 제거: 평당가 5만원 미만 또는 10,000만원 초과]")
    print(f"전체: {before_total}건 -> {len(df1)}건 (제거 {before_total - len(df1)}건)")
    cmp1 = pd.DataFrame({
        "제거전": before_by_jimok,
        "제거후": after1_by_jimok,
    }).fillna(0).astype(int)
    cmp1["제거건수"] = cmp1["제거전"] - cmp1["제거후"]
    print(cmp1.sort_values("제거전", ascending=False).to_string())

    # 2차: 지목별 1~99 분위 밖 값 제거
    before2_by_jimok = df1["지목"].value_counts()
    grp = df1.groupby("지목")["평당가(만원)"]
    lo = grp.transform(lambda s: s.quantile(0.01))
    hi = grp.transform(lambda s: s.quantile(0.99))
    df2 = df1[(df1["평당가(만원)"] >= lo) & (df1["평당가(만원)"] <= hi)].copy()
    after2_by_jimok = df2["지목"].value_counts()

    print(f"\n[2차 지목별 1~99분위 제거]")
    print(f"전체: {len(df1)}건 -> {len(df2)}건 (제거 {len(df1) - len(df2)}건)")
    cmp2 = pd.DataFrame({
        "제거전": before2_by_jimok,
        "제거후": after2_by_jimok,
    }).fillna(0).astype(int)
    cmp2["제거건수"] = cmp2["제거전"] - cmp2["제거후"]
    print(cmp2.sort_values("제거전", ascending=False).to_string())

    print(f"\n최종: {before_total}건 -> {len(df2)}건 (총 제거 {before_total - len(df2)}건, "
          f"{(before_total - len(df2)) / before_total * 100:.1f}%)")

    return df2


# ---------------------------------------------------------------
# 2. 지목 범위 3가지 시나리오로 시군구별 평당가 중앙값 집계
# ---------------------------------------------------------------
def aggregate_scenarios(df_clean):
    section("2. 시나리오별 시군구 평당가 중앙값")

    results = {}
    for name, jimok_list in SCENARIOS.items():
        sub = df_clean[df_clean["지목"].isin(jimok_list)]
        agg = sub.groupby(["시도", "시군구"])["평당가(만원)"].agg(
            중앙값="median", 표본수="count"
        ).reset_index()
        agg["신뢰도낮음"] = agg["표본수"] < MIN_SAMPLE_WARN
        agg = agg.sort_values(["시도", "시군구"]).reset_index(drop=True)

        print(f"\n--- 시나리오 {name}: {jimok_list} ---")
        print(f"대상 거래 {len(sub)}건 / 시군구 {len(agg)}곳 "
              f"(표본 5건 미만: {agg['신뢰도낮음'].sum()}곳)")
        print(agg.to_string(index=False))

        results[name] = agg

    return results


# ---------------------------------------------------------------
# 3. 산단에 지가 매칭
# ---------------------------------------------------------------
def normalize_sigungu(name):
    """'부산광역시 중구' 같은 표기를 '중구'로 정규화"""
    if pd.isna(name):
        return name
    parts = str(name).strip().split()
    return parts[-1] if parts else name


def match_parks(park_df, scenario_agg):
    """
    park_df: DAN_ID, DAN_NAME, SIGUNGU_CD, SIGUNGU_NM 포함
    scenario_agg: 시도, 시군구, 중앙값, 표본수 (실거래 집계 결과)

    SIGUNGU_NM만으로는 '북구'(부산/울산 중복) 같은 이름 충돌이 있으므로,
    SIGUNGU_CD의 앞 2자리로 시도를 구분해 (시도, 시군구) 키로 조인한다.
    """
    prefix_to_sido = {"21": "부산광역시", "26": "울산광역시", "38": "경상남도"}

    park = park_df.copy()
    park["시군구_norm"] = park["SIGUNGU_NM"].apply(normalize_sigungu)
    park["시도_추정"] = park["SIGUNGU_CD"].astype(str).str[:2].map(prefix_to_sido)

    agg = scenario_agg.rename(columns={"시군구": "시군구_norm"})

    merged = park.merge(
        agg[["시도", "시군구_norm", "중앙값", "표본수", "신뢰도낮음"]],
        left_on=["시도_추정", "시군구_norm"],
        right_on=["시도", "시군구_norm"],
        how="left",
    )
    return merged


def report_matching(merged, scenario_name):
    unmatched = merged[merged["중앙값"].isna()]
    matched = merged[merged["중앙값"].notna()]

    print(f"\n--- [{scenario_name}] 매칭 결과 ---")
    print(f"전체 산단 {len(merged)}개 중 매칭 {len(matched)}개 / 미매칭 {len(unmatched)}개")

    if len(unmatched):
        print("미매칭 산단 목록 (시군구 자체에 실거래 데이터가 없는 경우 포함, "
              "군 지역은 API 자체가 데이터를 반환하지 않는 구조적 한계임):")
        cols = ["DAN_ID", "DAN_NAME", "SIGUNGU_NM", "시도_추정"]
        print(unmatched[cols].to_string(index=False))

    low_conf = matched[matched["신뢰도낮음"] == True]
    if len(low_conf):
        print(f"\n매칭은 됐으나 표본 5건 미만(신뢰도 낮음): {len(low_conf)}개")
        print(low_conf[["DAN_NAME", "SIGUNGU_NM", "표본수"]].to_string(index=False))

    return matched


# ---------------------------------------------------------------
# 4. 회귀 모델
# ---------------------------------------------------------------
def run_model(matched, norm_cols, scenario_name):
    section(f"4. 회귀 모델 - 시나리오 {scenario_name}")

    data = matched.dropna(subset=norm_cols + ["중앙값"]).copy()
    if len(data) < 20:
        print(f"표본 수가 너무 적습니다 ({len(data)}개). 모델 생략.")
        return None

    # 피처 필터링: 지가 중앙값과의 상관관계 절대값이 0.15 이상인 피처들만 유의미한 변수로 선택 (오버핏 제어 핵심)
    corrs = data[norm_cols].corrwith(data["중앙값"])
    selected_cols = corrs[corrs.abs() >= 0.15].index.tolist()
    if len(selected_cols) == 0:
        selected_cols = norm_cols
        
    print(f"상관관계 0.15 이상 필터링 후 선택된 피처 ({len(selected_cols)}개): {selected_cols}")

    X = data[selected_cols].values
    y_raw = data["중앙값"].values
    y = np.log1p(y_raw)

    # 극소수 샘플의 오버핏 방지를 위해 트리의 max_depth 및 max_features 규제 설정
    model = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=4, max_depth=3, random_state=42
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_log = cross_val_predict(model, X, y, cv=kf)
    y_pred_raw = np.expm1(y_pred_log)

    # R2 연산을 log 스케일이 아닌 '원래 가격 스케일' 기준으로 계산하여 왜곡 보정
    r2 = r2_score(y_raw, y_pred_raw)
    mae_raw = mean_absolute_error(y_raw, y_pred_raw)

    n_groups = data["시군구_norm"].nunique()

    print(f"표본 수: {len(data)}개 (시군구 {n_groups}곳)")
    print(f"5-fold CV R^2 (원래 지가 스케일, 일반 KFold): {r2:.4f}")
    print(f"5-fold CV MAE (원래 지가 스케일, 일반 KFold): {mae_raw:.1f}만원/평")

    n_splits_group = min(5, n_groups)
    r2_group = None
    if n_splits_group >= 2:
        gkf = GroupKFold(n_splits=n_splits_group)
        groups = data["시군구_norm"].values
        y_pred_log_group = cross_val_predict(model, X, y, cv=gkf, groups=groups)
        y_pred_raw_group = np.expm1(y_pred_log_group)
        
        # 원래 가격 스케일 기준으로 R2 계산
        r2_group = r2_score(y_raw, y_pred_raw_group)
        mae_group = mean_absolute_error(y_raw, y_pred_raw_group)
        print(f"{n_splits_group}-fold GroupKFold(시군구 기준) R^2: {r2_group:.4f}  <- 미학습 지역 일반화 성능(더 정직한 지표)")
        print(f"{n_splits_group}-fold GroupKFold MAE: {mae_group:.1f}만원/평")
    else:
        print("시군구 그룹 수가 너무 적어 GroupKFold를 수행할 수 없습니다.")

    # 변수중요도는 전체 데이터로 재학습해서 산출
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=selected_cols)
    top8 = importances.sort_values(ascending=False).head(8)
    print("\n변수중요도 Top8:")
    print(top8.to_string())

    data["예측_평당가(만원)"] = y_pred_raw
    data["실제_평당가(만원)"] = y_raw
    data["예측_평당가_GroupKFold(만원)"] = y_pred_raw_group if r2_group is not None else np.nan

    return {
        "scenario": scenario_name,
        "r2": r2,
        "mae": mae_raw,
        "r2_group": r2_group,
        "n": len(data),
        "top8": top8,
        "data": data,
    }


# ---------------------------------------------------------------
# 5. 결과 해석
# ---------------------------------------------------------------
def interpret(results):
    section("5. 결과 해석")

    valid = {k: v for k, v in results.items() if v is not None}
    if not valid:
        print("모든 시나리오에서 모델을 만들 수 없었습니다.")
        return

    # 일반 KFold R^2는 같은 시군구의 산단이 train/test에 함께 섞여 들어가는
    # 누수(leakage) 때문에 부풀려질 수 있다. GroupKFold(시군구 기준) R^2가
    # '학습에 없던 지역'에 대한 일반화 성능을 보여주는 더 정직한 지표이므로
    # 이를 기준으로 시나리오를 비교/추천한다.
    def honest_r2(res):
        return res["r2_group"] if res["r2_group"] is not None else res["r2"]

    best_name = max(valid, key=lambda k: honest_r2(valid[k]))
    best = valid[best_name]

    print("시나리오별 성능 비교:")
    for name, res in valid.items():
        flag = " <- 추천" if name == best_name else ""
        gap = res["r2"] - res["r2_group"] if res["r2_group"] is not None else None
        gap_str = f", KFold-GroupKFold 격차={gap:.3f}" if gap is not None else ""
        print(f"  {name}: 일반KFold R^2={res['r2']:.4f}, GroupKFold R^2={res['r2_group']:.4f}, "
              f"MAE={res['mae']:.1f}만원/평, n={res['n']}{gap_str}{flag}")

    print(f"\n추천 시나리오: {best_name} (GroupKFold R^2={honest_r2(best):.4f})")

    print("\n[누수(leakage) 경고] y(시군구 지가 중앙값)가 14~15개 값만 존재하는 상태에서 "
          "일반 KFold를 쓰면, 같은 시군구의 다른 산단이 학습/검증에 함께 섞여 들어가 "
          "실제로는 '지역을 맞히는' 것에 가까운 결과가 R^2를 부풀릴 수 있다. "
          "위 표의 '일반KFold R^2'와 'GroupKFold R^2' 격차가 클수록 누수 영향이 큰 것이다.")

    if honest_r2(best) < 0.3:
        print("\n[솔직한 평가] GroupKFold 기준 R^2가 0.3 미만입니다. 14개 입지지표만으로 "
              "'학습에 없던 지역'의 지가 수준을 설명하는 신호가 약합니다. 이 접근을 "
              "'유망 입지 추천'의 근거로 쓰기에는 무리가 있으며, 추가 변수(용도지역, "
              "개발계획, 인구/산업 밀도 등) 없이는 설득력 있는 결론을 내리기 어렵습니다.")
    else:
        print(f"\n[솔직한 평가] GroupKFold 기준 R^2={honest_r2(best):.4f}로 0.3을 넘습니다. \n"
              "다만 표본이 시군구 14~15곳, 산단 146~147개에 불과해 소수 지역의 특성에 \n"
              "과적합되었을 가능성이 있으니 참고용으로만 활용할 것.")

    data = best["data"]
    use_group = best["r2_group"] is not None
    pred_col = "예측_평당가_GroupKFold(만원)" if use_group else "예측_평당가(만원)"
    plotted_r2 = best["r2_group"] if use_group else best["r2"]

    # 산점도 (GroupKFold 예측 기준 - 학습에 없던 지역에 대한 실제 예측력을 보여줌)
    plt.figure(figsize=(7, 7))
    plt.scatter(data["실제_평당가(만원)"], data[pred_col], alpha=0.6, s=25)
    lims = [0, max(data["실제_평당가(만원)"].max(), data[pred_col].max()) * 1.05]
    plt.plot(lims, lims, "r--", linewidth=1, label="y = x")
    plt.xlabel("실제 지가 중앙값 (만원/평)")
    plt.ylabel("예측 지가 (GroupKFold, 만원/평)")
    plt.title(f"예측 vs 실제 지가 (시나리오 {best_name}, GroupKFold R^2={plotted_r2:.3f})")
    plt.legend()
    plt.tight_layout()
    out_png = "predicted_vs_actual.png"
    plt.savefig(out_png, dpi=150)
    print(f"\n산점도 저장: {out_png} (GroupKFold 예측 기준 - R^2={plotted_r2:.3f}로 대각선에서 크게 벗어남에 유의)")

    # 저평가 산단 top10 (실제 < 예측) - GroupKFold 예측 기준
    data["저평가폭(만원)"] = data[pred_col] - data["실제_평당가(만원)"]
    top10 = data.sort_values("저평가폭(만원)", ascending=False).head(10)
    print(f"\n'여건 대비 저평가 산단' Top10 (실제 지가 < 예측 지가, {'GroupKFold' if use_group else 'KFold'} 기준):")
    cols = ["DAN_NAME", "SIGUNGU_NM", "실제_평당가(만원)", pred_col, "저평가폭(만원)"]
    print(top10[cols].round(1).to_string(index=False))

    if use_group and plotted_r2 < 0.3:
        print(f"\n[중요] GroupKFold R^2={plotted_r2:.3f}로 모델이 학습에 없던 지역을 사실상 "
              "설명하지 못합니다(음수면 평균으로 찍는 것보다도 못함). 위 저평가 산단 목록은 "
              "통계적으로 유의미한 신호가 아니라 모델 잡음에 가까우며, '유망 입지 추천'의 "
              "근거로 제시해서는 안 됩니다. 참고용 예시로만 취급할 것.")

    print("\n[주의] 위 결과는 시군구 단위 지가 추정치를 기준으로 한 것이며, "
          "같은 시군구 내 산단은 실제/예측 지가가 동일하게 나타난다. "
          "산단별 실제 분양가 예측이 아니라 '지역 지가 수준' 기반의 참고 지표로만 활용할 것.")


def main():
    pd.set_option("display.max_rows", 300)
    pd.set_option("display.width", 200)

    df_clean = load_and_clean_land(LAND_CSV)
    scenario_aggs = aggregate_scenarios(df_clean)

    park_df = pd.read_csv(PARK_CSV, encoding="utf-8-sig")
    norm_cols = [c for c in park_df.columns if c.endswith("_norm")]
    print(f"\n입지지표(_norm) {len(norm_cols)}개: {norm_cols}")

    section("3. 산단 - 지가 매칭")
    matched_by_scenario = {}
    for name, agg in scenario_aggs.items():
        merged = match_parks(park_df, agg)
        matched = report_matching(merged, name)
        matched_by_scenario[name] = matched

    results = {}
    for name, matched in matched_by_scenario.items():
        results[name] = run_model(matched, norm_cols, name)

    interpret(results)


if __name__ == "__main__":
    main()
