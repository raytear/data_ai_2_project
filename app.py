import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(
    page_title="일별 사고건수 예측 앱",
    page_icon="🚨",
    layout="wide"
)

# 제목
st.title("🚨 일별 사고건수 예측 애플리케이션")
st.markdown("**기상 데이터를 입력하면 예상되는 사고건수를 예측합니다.**")
st.markdown("")
st.info("💡 **사용 방법**: 왼쪽에서 날씨 정보(기온, 강수량, 적설량, 습도 등)를 입력하고 예측 버튼을 클릭하세요.")
st.markdown("---")

# 모델 로드 (캐싱)
@st.cache_resource
def load_model():
    """모델 로드"""
    try:
        return joblib.load('./2_team/accident_model.joblib')
    except FileNotFoundError:
        st.error("❌ 모델 파일을 찾을 수 없습니다. 먼저 model_training.ipynb를 실행하여 모델을 학습하세요.")
        return None

@st.cache_data
def load_model_info():
    """모델 정보 로드"""
    try:
        with open('./2_team/model_info.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# 모델 및 정보 로드
model = load_model()
model_info = load_model_info()

if model is None:
    st.stop()

# 사이드바
st.sidebar.header("⚙️ 설정")

# 모델 정보 표시
st.sidebar.subheader("📊 모델 성능")
if model_info:
    st.sidebar.metric("모델", "다중 선형 회귀 모델")
    st.sidebar.metric("R² Score", f"{model_info.get('r2_test', 0):.3f}")
    st.sidebar.metric("RMSE", f"{model_info.get('rmse', 0):.2f}건")
    st.sidebar.metric("MAE", f"{model_info.get('mae', 0):.2f}건")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 학습 데이터")
    st.sidebar.write(f"훈련: {model_info.get('train_days', 0)}일")
    st.sidebar.write(f"테스트: {model_info.get('test_days', 0)}일")

# 히스토리 초기화
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

# 메인 영역
tab1, tab2, tab3, tab4 = st.tabs(["🔮 예측", "📈 모델 성능", "📊 데이터 분석", "📋 예측 히스토리"])

# 탭 1: 예측
with tab1:
    st.header("일별 사고건수 예측")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🌤️ 기상 데이터 입력")
        st.markdown("**독립변수**: 기상 데이터 (기온, 강수량, 적설량, 습도 등)")
        
        # 날씨 정보 입력
        st.markdown("#### 날씨 조건")
        
        col_temp, col_humidity = st.columns(2)
        with col_temp:
            avg_temp = st.number_input(
                "평균 기온 (°C)",
                min_value=-20.0,
                max_value=40.0,
                value=15.0,
                step=0.1,
                help="평균 기온을 입력하세요"
            )
        
        with col_humidity:
            avg_humidity = st.number_input(
                "평균 습도 (%)",
                min_value=0.0,
                max_value=100.0,
                value=60.0,
                step=1.0,
                help="평균 습도를 입력하세요"
            )
        
        col_rain, col_snow = st.columns(2)
        with col_rain:
            total_rain = st.number_input(
                "총 강수량 (mm)",
                min_value=0.0,
                max_value=500.0,
                value=10.0,
                step=1.0,
                help="총 강수량을 입력하세요"
            )
        
        with col_snow:
            total_snow = st.number_input(
                "총 적설량 (cm)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
                help="총 적설량을 입력하세요"
            )
        
        col_rain_hours, col_snow_hours = st.columns(2)
        with col_rain_hours:
            rain_hours = st.number_input(
                "강수 발생 시간 (시간)",
                min_value=0,
                max_value=24,
                value=2,
                step=1,
                help="강수가 발생한 총 시간 수 (0-24시간)"
            )
        
        with col_snow_hours:
            snow_hours = st.number_input(
                "적설 발생 시간 (시간)",
                min_value=0,
                max_value=24,
                value=0,
                step=1,
                help="적설이 발생한 총 시간 수 (0-24시간)"
            )
        
        # 예측 버튼
        if st.button("🔮 사고건수 예측하기", type="primary", use_container_width=True):
            # 입력 데이터 준비
            input_data = pd.DataFrame({
                'avg_temp': [avg_temp],
                'total_rain': [total_rain],
                'total_snow': [total_snow],
                'rain_hours': [rain_hours],
                'snow_hours': [snow_hours],
                'avg_humidity': [avg_humidity]
            })
            
            # 예측 (스케일링 없이)
            predicted_accident = model.predict(input_data)[0]
            
            # 예측값이 음수가 되지 않도록 조정
            predicted_accident = max(0, predicted_accident)
            
            # 세션 상태에 저장
            prediction_record = {
                'avg_temp': avg_temp,
                'total_rain': total_rain,
                'total_snow': total_snow,
                'rain_hours': rain_hours,
                'snow_hours': snow_hours,
                'avg_humidity': avg_humidity,
                'predicted_accident': predicted_accident
            }
            st.session_state.predictions.append(prediction_record)
    
    with col2:
        # 예측 결과 표시
        if st.session_state.predictions:
            latest = st.session_state.predictions[-1]
            st.subheader("예측 결과")
            st.metric(
                label="예상 사고건수",
                value=f"{latest['predicted_accident']:.0f}건",
                delta=f"{latest['predicted_accident'] - 200:.0f}건" if latest['predicted_accident'] > 200 else None,
                delta_color="inverse"
            )
            
            # 결과 해석
            with st.expander("📖 결과 해석"):
                st.write(f"""
                **입력된 기상 데이터 (독립변수)**:
                - 평균 기온: {latest['avg_temp']:.1f}°C
                - 총 강수량: {latest['total_rain']:.1f}mm
                - 총 적설량: {latest['total_snow']:.1f}cm
                - 강수 발생 시간: {latest['rain_hours']}시간
                - 적설 발생 시간: {latest['snow_hours']}시간
                - 평균 습도: {latest['avg_humidity']:.1f}%
                
                **예측된 사고건수 (종속변수)**: {latest['predicted_accident']:.0f}건
                
                **해석**:
                - 입력하신 기상 데이터를 기반으로 예상 사고건수는 약 {latest['predicted_accident']:.0f}건입니다.
                - 이 값은 학습된 Linear Regression 모델이 기상 데이터와 사고건수 간의 관계를 학습하여 예측한 결과입니다.
                """)
                
                if model_info:
                    st.write(f"""
                    **오차 범위**:
                    - 모델의 RMSE는 {model_info.get('rmse', 0):.2f}건입니다.
                    - 실제 사고건수는 예측값에서 평균적으로 ±{model_info.get('rmse', 0):.2f}건 정도 차이날 수 있습니다.
                    """)
        else:
            st.info("👈 왼쪽에서 날씨 정보를 입력하고 예측 버튼을 클릭하세요.")

# 탭 2: 모델 성능
with tab2:
    st.header("모델 성능 정보")
    
    if model_info:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 성능 지표")
            metrics = {
                "모델명": model_info.get('model_name', 'Unknown'),
                "검증 R²": f"{model_info.get('r2_test', 0):.4f}",
                "평균제곱오차 (MSE)": f"{model_info.get('mse', 0):.2f}",
                "루트평균제곱오차 (RMSE)": f"{model_info.get('rmse', 0):.2f}건",
                "평균절대오차 (MAE)": f"{model_info.get('mae', 0):.2f}건"
            }
            
            for key, value in metrics.items():
                st.metric(key, value)
        
        with col2:
            st.subheader("📐 회귀 계수")
            coefficients = model_info.get('coefficients', [])
            feature_names = model_info.get('feature_names', [])
            intercept = model_info.get('intercept', 0)
            
            st.write("**특성별 계수**:")
            for i, feature in enumerate(feature_names):
                if i < len(coefficients):
                    st.write(f"- {feature}: {coefficients[i]:.4f}")
            
            st.write(f"\n**절편**: {intercept:.4f}")
            
            st.markdown("---")
            st.subheader("📅 데이터 분할")
            st.write(f"**훈련 데이터**: {model_info.get('train_days', 0)}일")
            st.write(f"**테스트 데이터**: {model_info.get('test_days', 0)}일")
        
        # 성능 지표 해석
        st.subheader("📖 성능 지표 해석")
        with st.expander("자세한 해석 보기"):
            r2_test = model_info.get('r2_test', 0)
            rmse = model_info.get('rmse', 0)
            
            st.write(f"""
            **1. 결정계수 (R²) = {r2_test:.4f}**
            - 모델이 사고건수 변동의 약 {r2_test*100:.1f}%를 설명합니다.
            - {r2_test*100:.1f}%는 날씨 조건으로 설명 가능하고, 나머지는 다른 요인에 의해 설명됩니다.
            
            **2. RMSE = {rmse:.2f}건**
            - 예측값이 실제값과 평균적으로 {rmse:.2f}건 정도 차이가 납니다.
            - 이 값이 작을수록 모델의 예측 정확도가 높습니다.
            
            **3. 모델의 한계**
            - 날씨 외에도 교통량, 도로 상태, 계절적 요인 등이 사고건수에 영향을 미칩니다.
            - 더 정확한 예측을 위해서는 추가 변수가 필요할 수 있습니다.
            """)
    else:
        st.info("모델 정보를 불러올 수 없습니다.")

# 탭 3: 데이터 분석
with tab3:
    st.header("데이터 분석")

    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import re

    TIME_ACC_PATH = "./2_team/time_accident.csv"
    WEATHER_PATH  = "./2_team/timedata.csv"


    # 시간대 라벨("0시~2시")에서 시작 시각(0)을 추출해 정렬에 활용
    def start_hour(label: str) -> int:
        m = re.match(r"(\d+)시~", str(label))
        return int(m.group(1)) if m else 999

    @st.cache_data(show_spinner=False)
    def build_analysis_frames(time_acc_path: str, weather_path: str):
        # 2) 사고(시간대별, 연간 집계) 전처리
        df_time_raw = pd.read_csv(time_acc_path, encoding="utf-8")

        header_row = df_time_raw.iloc[0].to_dict()
        rename_map = {}
        for c in df_time_raw.columns:
            if c in ["시도", "연도"]:
                rename_map[c] = c
            else:
                rename_map[c] = str(header_row.get(c, c)).strip()

        df_time = df_time_raw.rename(columns=rename_map).iloc[1:].copy()

        df_time["시도"] = df_time["시도"].astype(str).str.strip()
        df_time["연도"] = df_time["연도"].astype(str).str.strip()

        df_time_seoul = df_time[(df_time["시도"] == "서울") & (df_time["연도"] == "사고[건]")].copy()

        time_band_cols = [c for c in df_time_seoul.columns if re.search(r"시~", str(c))]
        time_band_cols = sorted(time_band_cols, key=start_hour)

        df_acc_band = df_time_seoul.melt(
            id_vars=["시도"],
            value_vars=time_band_cols,
            var_name="시간대",
            value_name="사고건수"
        )
        df_acc_band["사고건수"] = (
            df_acc_band["사고건수"]
            .astype(str).str.replace(",", "", regex=False).str.strip()
            .replace({"": np.nan})
            .astype(float).astype("Int64")
        )
        df_acc_band = df_acc_band[["시간대", "사고건수"]].copy()
        df_acc_band["sort_key"] = df_acc_band["시간대"].apply(start_hour)
        df_acc_band = df_acc_band.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)

        # 3) 날씨 전처리(시간별) -> 2시간 주기, 월 파생
        df_w_raw = pd.read_csv(weather_path, encoding="cp949")
        need_cols = ['일시', '기온(°C)', '강수량(mm)', '습도(%)', '적설(cm)']
        df_w = df_w_raw[need_cols].rename(columns={
            '일시': 'datetime',
            '기온(°C)': 'temp_avg',
            '강수량(mm)': 'rain_mm',
            '습도(%)': 'humidity_pct',
            '적설(cm)': 'snow_cm'
        }).copy()

        df_w["rain_mm"] = df_w["rain_mm"].fillna(0)
        df_w["snow_cm"] = df_w["snow_cm"].fillna(0)

        for c in ["temp_avg", "rain_mm", "humidity_pct", "snow_cm"]:
            df_w[c] = pd.to_numeric(df_w[c], errors="coerce")

        df_w["datetime"] = pd.to_datetime(df_w["datetime"], errors="coerce")
        df_w = df_w.dropna(subset=["datetime"])

        df_w = df_w[(df_w["datetime"] >= "2024-01-01") & (df_w["datetime"] < "2025-01-01")].copy()
        df_w["month"] = df_w["datetime"].dt.month
        df_w["hour"] = df_w["datetime"].dt.hour

        df_w["band_start"] = (df_w["hour"] // 2) * 2
        df_w["시간대"] = df_w["band_start"].apply(lambda h: f"{h}시~{h+2}시" if h < 22 else "22시~24시")

        # 4) 연간(시간대별) 날씨 요약
        df_w_band_annual = (
            df_w.groupby("시간대", as_index=False)
              .agg(
                  avg_temp=("temp_avg", "mean"),
                  total_rain=("rain_mm", "sum"),
                  total_snow=("snow_cm", "sum"),
                  rain_hours=("rain_mm", lambda s: int((s > 0).sum())),
                  snow_hours=("snow_cm", lambda s: int((s > 0).sum())),
              )
        )
        df_w_band_annual["sort_key"] = df_w_band_annual["시간대"].apply(start_hour)
        df_w_band_annual = df_w_band_annual.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)

        # 시간 별 날씨,사고 데이터 병합
        df_band = df_acc_band.merge(df_w_band_annual, on="시간대", how="left")

        # 5) 월별(기상) 가중 사고지수(추정)
        df_w_month = (
            df_w.groupby("month", as_index=False)
              .agg(
                  avg_temp=("temp_avg", "mean"),
                  total_rain=("rain_mm", "sum"),
                  total_snow=("snow_cm", "sum"),
                  rain_hours=("rain_mm", lambda s: int((s > 0).sum())),
                  snow_hours=("snow_cm", lambda s: int((s > 0).sum()))
              )
        )

        df_w_month_band = (
            df_w.groupby(["month", "시간대"], as_index=False)
              .agg(
                  rain_hours=("rain_mm", lambda s: int((s > 0).sum())),
                  snow_hours=("snow_cm", lambda s: int((s > 0).sum()))
              )
        ).merge(df_acc_band, on="시간대", how="left")

        df_w_month_band["precip_hours"] = df_w_month_band["rain_hours"] + df_w_month_band["snow_hours"]

        month_vals = []
        for m, g in df_w_month_band.groupby("month"):
            ph = float(g["precip_hours"].sum())
            if ph > 0:
                wi = float((g["사고건수"].astype(float) * g["precip_hours"]).sum() / ph)
            else:
                wi = np.nan
            month_vals.append((m, ph, wi))

        df_month_index = pd.DataFrame(month_vals, columns=["month", "precip_hours", "weighted_index"])

        month_template = pd.DataFrame({"month": list(range(1, 13))})
        df_month = (
            month_template
            .merge(df_w_month, on="month", how="left")
            .merge(df_month_index, on="month", how="left")
            .sort_values("month")
            .reset_index(drop=True)
        )

        for c in ["total_rain", "total_snow", "rain_hours", "snow_hours", "precip_hours"]:
            df_month[c] = df_month[c].fillna(0)

        df_month["avg_temp"] = df_month["avg_temp"].interpolate(limit_direction="both")
        df_month["no_precip_flag"] = df_month["weighted_index"].isna().astype(int)
        df_month["weighted_index"] = df_month["weighted_index"].fillna(0)
        df_month["month_label"] = df_month["month"].apply(lambda m: f"{int(m):02d}")

        return df_band, df_month

    # ===== 데이터 생성 =====
    try:
        df_band, df_month = build_analysis_frames(TIME_ACC_PATH, WEATHER_PATH)
    except FileNotFoundError as e:
        st.error(f"파일을 찾을 수 없습니다: {e}")
        st.stop()
    except Exception as e:
        st.error(f"데이터 분석 전처리 중 오류가 발생했습니다: {e}")
        st.stop()

    # ===== 시각화 1) 시간대별 사고건수 vs 강수&적설 발생 빈도 =====
    st.subheader("시간대별 사고건수 vs 강수·적설 발생 빈도(시간 수, 2024)")
    
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 막대 그래프 (사고건수)
    fig1.add_trace(
        go.Bar(x=df_band["시간대"], y=df_band["사고건수"], name="사고건수", opacity=0.5, marker_color='lightblue'),
        secondary_y=False,
    )
    
    # 선 그래프 (강수 발생시간)
    fig1.add_trace(
        go.Scatter(x=df_band["시간대"], y=df_band["rain_hours"], name="강수 발생시간", mode='lines+markers', marker=dict(symbol='circle')),
        secondary_y=True,
    )
    
    # 선 그래프 (적설 발생시간)
    fig1.add_trace(
        go.Scatter(x=df_band["시간대"], y=df_band["snow_hours"], name="적설 발생시간", mode='lines+markers', marker=dict(symbol='circle')),
        secondary_y=True,
    )
    
    fig1.update_xaxes(title_text="시간대", tickangle=-45)
    fig1.update_yaxes(title_text="사고건수(건)", secondary_y=False)
    fig1.update_yaxes(title_text="발생 시간 수(시간)", secondary_y=True)
    fig1.update_layout(
        title="시간대별 사고건수 vs 강수·적설 발생 빈도(시간 수, 2024)",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig1, use_container_width=True)

    # ===== 시각화 2) 시간대별 사고건수 vs 강수/적설 '량'(합계) =====
    st.subheader("시간대별 사고건수 vs 강수·적설량(2024)")
    
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 막대 그래프 (사고건수)
    fig2.add_trace(
        go.Bar(x=df_band["시간대"], y=df_band["사고건수"], name="사고건수", opacity=0.5, marker_color='lightblue'),
        secondary_y=False,
    )
    
    # 선 그래프 (강수량 합계)
    fig2.add_trace(
        go.Scatter(x=df_band["시간대"], y=df_band["total_rain"], name="강수량 합계(mm)", mode='lines+markers', marker=dict(symbol='circle')),
        secondary_y=True,
    )
    
    # 선 그래프 (적설량 합계)
    fig2.add_trace(
        go.Scatter(x=df_band["시간대"], y=df_band["total_snow"], name="적설량 합계(cm)", mode='lines+markers', marker=dict(symbol='circle')),
        secondary_y=True,
    )
    
    fig2.update_xaxes(title_text="시간대", tickangle=-45)
    fig2.update_yaxes(title_text="사고건수(건)", secondary_y=False)
    fig2.update_yaxes(title_text="합계 강수/적설 (mm / cm)", secondary_y=True)
    fig2.update_layout(
        title="2024년도 시간대별 사고건수 vs 강수·적설량",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig2, use_container_width=True)

    # ===== 시각화 3) 월별 조건 그래프(3축) =====
    st.subheader("월별 기상 가중 추정 사고지수(3축, 2024)")
    
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 막대 그래프 (추정 사고지수)
    fig3.add_trace(
        go.Bar(x=df_month["month_label"], y=df_month["weighted_index"], name="추정 사고지수", opacity=0.5, marker_color='lightblue'),
        secondary_y=False,
    )
    
    # 선 그래프 (강수량 합계)
    fig3.add_trace(
        go.Scatter(x=df_month["month_label"], y=df_month["total_rain"], name="월 강수량 합계(mm)", mode='lines+markers', marker=dict(symbol='circle')),
        secondary_y=True,
    )
    
    # 선 그래프 (적설량 합계)
    fig3.add_trace(
        go.Scatter(x=df_month["month_label"], y=df_month["total_snow"], name="월 적설량 합계(cm)", mode='lines+markers', marker=dict(symbol='circle')),
        secondary_y=True,
    )
    
    # 선 그래프 (평균기온) - 다른 스타일로 추가
    fig3.add_trace(
        go.Scatter(x=df_month["month_label"], y=df_month["avg_temp"], name="월 평균기온(°C)", mode='lines+markers', 
                  marker=dict(symbol='circle', color='red'), line=dict(dash='dash', color='red')),
        secondary_y=True,
    )
    
    # no_precip 텍스트 추가
    for i, r in df_month.iterrows():
        if int(r["no_precip_flag"]) == 1:
            fig3.add_annotation(
                x=r["month_label"],
                y=r["weighted_index"],
                text="no<br>precip",
                showarrow=False,
                font=dict(size=8),
                yshift=10
            )
    
    fig3.update_xaxes(title_text="월(2024)")
    fig3.update_yaxes(title_text="추정 사고지수(강수·적설 발생시간 가중)", secondary_y=False)
    fig3.update_yaxes(title_text="월 강수/적설 합계 (mm / cm) / 월 평균기온(°C)", secondary_y=True)
    fig3.update_layout(
        title="2024년도 월별 기상 가중 추정 사고지수",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig3, use_container_width=True)

    # ===== 테이블 =====
    with st.expander("월별 요약 테이블(df_month) 보기"):
        st.dataframe(df_month)

    with st.expander("시간대별 요약 테이블(df_band) 보기"):
        st.dataframe(df_band)


# 탭 4: 예측 히스토리
with tab4:
    st.header("예측 히스토리")
    
    if st.session_state.predictions:
        pred_df = pd.DataFrame(st.session_state.predictions)
        
        # 히스토리 테이블
        display_df = pred_df.copy()
        display_df['predicted_accident'] = display_df['predicted_accident'].apply(lambda x: f"{x:.0f}건")
        display_df.insert(0, '번호', range(1, len(display_df) + 1))
        display_df.columns = ['번호', '평균기온(°C)', '총강수량(mm)', '총적설량(cm)', '강수발생시간(시간)', 
                             '적설발생시간(시간)', '평균습도(%)', '예측사고건수']
        
        st.dataframe(display_df, use_container_width=True)
        
        
        # 히스토리 초기화 버튼
        if st.button("🗑️ 히스토리 초기화", type="secondary"):
            st.session_state.predictions = []
            st.rerun()
    else:
        st.info("아직 예측 기록이 없습니다. 예측 탭에서 예측을 수행해보세요.")
