"""RestaurantIQ — screenshot-friendly historical review prediction dashboard."""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="RestaurantIQ", page_icon="🍒", layout="wide", initial_sidebar_state="collapsed")

CUISINES = [
    "cuisine_pizza", "cuisine_chinese", "cuisine_italian", "cuisine_mexican",
    "cuisine_american", "cuisine_asian", "cuisine_japanese", "cuisine_indian",
    "cuisine_thai", "cuisine_vietnamese", "cuisine_korean", "cuisine_spanish",
    "cuisine_french", "cuisine_middle_eastern", "cuisine_burger", "cuisine_seafood",
    "cuisine_vegetarian", "cuisine_vegan", "cuisine_cafe", "cuisine_deli",
]
FEATURES = [
    "week", "month", "year", "prev_week_review_count", "moving_avg_14_day",
    "moving_avg_30_day", "temp_max", "temp_avg", "temp_min", "weather_code",
    "precipitation", "has_precipitation", "heavy_precipitation", "is_freezing",
    "is_holiday", "num_holiday_days", "restaurant_stars", "total_review_count",
    "nearby_restaurant_count", "avg_nearby_stars",
] + CUISINES

st.markdown("""
<style>
:root {color-scheme: light;}
.stApp {background: #fff9f9; color: #361e2a;}
header[data-testid="stHeader"] {background: #fff9f9; height: 2.4rem;}
[data-testid="stSidebar"], [data-testid="collapsedControl"] {display:none;}
.block-container {max-width: 1500px; padding: 1.0rem 2.2rem 2rem;}
h1, h2, h3, p, label {color: #361e2a;}
h1 {letter-spacing: -.055em;}
h2, h3 {letter-spacing: -.025em;}
[data-testid="stVerticalBlock"] {gap: .65rem;}
[data-testid="stMetric"] {background:#ffffff; border:1px solid #f2d9df; border-radius:16px;
    padding:16px 18px; box-shadow:0 5px 18px rgba(129,31,65,.04);}
[data-testid="stMetricLabel"] {color:#805667;}
[data-testid="stMetricValue"] {color:#ae274e; font-weight:750;}
[data-testid="stSelectbox"] > div > div {background:#fff; border-color:#edc7d2;}
[data-testid="stRadio"] label {background: #fff; border:1px solid #f0d8df;
    border-radius:11px; padding: .48rem .7rem; margin: .1rem 0;}
[data-testid="stRadio"] label:has(input:checked) {background:#ffe5eb; border-color:#d9537c;}
[data-testid="stRadio"] label p {font-weight:600;}
[data-testid="stRadio"] [role="radiogroup"] {gap:.55rem; padding-top: .85rem;}
[data-testid="stRadio"] {margin-top:.45rem;}
[data-testid="stSelectbox"] {margin-top:.8rem;}
[data-testid="stSelectbox"] label {margin-bottom:.45rem;}
[data-testid="stVegaLiteChart"] {background:#fff; border:1px solid #f2d9df;
    border-radius:16px; padding: 12px;}
.stAlert {border-radius: 12px;}
div.stButton > button {background:#ad2b51; color:white; border:0;}
.brand {font-size:2.25rem; line-height:1.28; font-weight:850; color:#922341; letter-spacing:-.055em; margin-top:.3rem; margin-bottom:.45rem;}
.eyebrow {font-size:.75rem; font-weight:750; letter-spacing:.17em; color:#c24b6d; text-transform:uppercase;}
.subtle {color:#886a78; font-size:.92rem; line-height:1.65;}
.pill {display:inline-block; padding:.38rem .7rem; border-radius:50px; background:#ffe3e9;
    color:#9b2448; font-size:.8rem; font-weight:700;}
.navtitle {font-size:.72rem; color:#b16a81; font-weight:800; letter-spacing:.13em; margin-top:1.35rem; margin-bottom:.65rem;}
.footnote {font-size:.76rem; color:#9b7c87;}
@media (max-width:800px) {.block-container {padding:1rem;}}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading restaurant profiles…")
def load_restaurants():
    df = pd.read_csv(ROOT / "restaurants.csv")
    df["name"] = df["name"].fillna("Unnamed restaurant")
    return df

@st.cache_data(show_spinner="Preparing historical model predictions…")
def load_predictions():
    df = pd.read_csv(ROOT / "features.csv", parse_dates=["date"])
    df["year"] = df["date"].dt.isocalendar().year.astype(int)
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["num_holiday_days"] = df["is_holiday"]
    agg = {
        "daily_review_count": "sum", "month": "first", "season": lambda x: x.mode().iloc[0],
        "temp_max": "max", "temp_avg": "mean", "temp_min": "min",
        "weather_code": lambda x: x.mode().iloc[0], "precipitation": "sum",
        "has_precipitation": "max", "heavy_precipitation": "max", "is_freezing": "max",
        "is_holiday": "max", "num_holiday_days": "sum",
        "moving_avg_14_day": "mean", "moving_avg_30_day": "mean",
        "restaurant_stars": "first", "total_review_count": "first",
        "nearby_restaurant_count": "first", "avg_nearby_stars": "first",
    }
    agg.update({col: "first" for col in CUISINES})
    weekly = (df.groupby(["business_id", "year", "week"]).agg(agg).reset_index()
              .rename(columns={"daily_review_count": "weekly_review_count"})
              .sort_values(["business_id", "year", "week"]))
    weekly["prev_week_review_count"] = weekly.groupby("business_id")["weekly_review_count"].shift(1)
    weekly = weekly.dropna(subset=["prev_week_review_count"]).dropna().reset_index(drop=True)
    X = pd.concat([weekly[FEATURES], pd.get_dummies(weekly["season"], prefix="season", drop_first=True)], axis=1)
    train, test = weekly["year"] < 2019, weekly["year"] == 2019
    if not train.any() or not test.any():
        raise ValueError("The data needs pre-2019 training and 2019 holdout observations.")
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=.05,
                         subsample=.8, colsample_bytree=.8, random_state=42, n_jobs=-1)
    model.fit(X.loc[train], weekly.loc[train, "weekly_review_count"])
    result = weekly.loc[test, ["business_id", "year", "week", "weekly_review_count"]].copy()
    result["predicted"] = model.predict(X.loc[test])
    result["week_start"] = pd.to_datetime(
        result["year"].astype(str) + "-W" + result["week"].astype(str).str.zfill(2) + "-1",
        format="%G-W%V-%u", errors="coerce")
    return result

try:
    places, pred = load_restaurants(), load_predictions()
except (FileNotFoundError, ValueError, KeyError) as exc:
    st.error(f"Could not load the project data: {exc}")
    st.stop()

options = places[["business_id", "name"]].drop_duplicates("business_id")
options = options[options["business_id"].isin(pred["business_id"])].copy()
counts = pred.groupby("business_id").agg(weeks=("week", "size"), total=("weekly_review_count", "sum"))
options = options.join(counts, on="business_id")
options["display"] = options["name"] + " · " + options["business_id"].str[:8]
options = options.sort_values(["weeks", "total"], ascending=False)
if options.empty:
    st.error("No restaurants have 2019 holdout observations.")
    st.stop()

st.markdown('<div class="eyebrow">Philadelphia restaurant analytics / 2019</div>', unsafe_allow_html=True)
st.markdown('<div class="brand">RestaurantIQ <span style="color:#e56a8c">✦</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subtle" style="margin-bottom:1.15rem">Understand historical restaurant activity through data-driven insights.</div>', unsafe_allow_html=True)

filter_col, meta_col = st.columns([3, 2], vertical_alignment="bottom")
with filter_col:
    chosen = st.selectbox("Explore a restaurant", options["display"].tolist(), index=0)
with meta_col:
    st.markdown('<span class="pill">● Historical model evaluation · 2019</span>', unsafe_allow_html=True)

business_id = options.loc[options["display"] == chosen, "business_id"].iloc[0]
place = places.loc[places["business_id"] == business_id].iloc[0]
subset = pred.loc[pred["business_id"] == business_id].sort_values("week_start").copy()

nav, main = st.columns([1, 4.4], gap="large")
with nav:
    st.markdown('<div class="navtitle" style="margin-bottom:1.1rem">DASHBOARD VIEWS</div>', unsafe_allow_html=True)
    page = st.radio("Choose view", ["Overview", "Weekly comparison", "Model performance", "Restaurant details"],
                    label_visibility="collapsed")
    st.markdown('<div class="navtitle">DATA SCOPE</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">2013–2018 training<br>2019 evaluation<br>Yelp review counts</div>', unsafe_allow_html=True)

with main:
    st.markdown(f"### {place['name']}")
    st.markdown('<div class="subtle">Philadelphia, PA &nbsp; • &nbsp; Historical evaluation, not live demand</div>', unsafe_allow_html=True)
    if page == "Overview":
        a, b, c = st.columns(3)
        a.metric("Weeks with observations", f"{len(subset):,}")
        b.metric("Avg actual reviews / week", f"{subset['weekly_review_count'].mean():.2f}")
        c.metric("Avg predicted reviews / week", f"{subset['predicted'].mean():.2f}")
        st.markdown("#### Weekly activity at a glance")
    elif page == "Weekly comparison":
        st.markdown("#### Historical actual vs. predicted reviews")
        a, b = st.columns(2)
        a.metric("Total actual reviews", f"{subset['weekly_review_count'].sum():,.0f}")
        b.metric("Total predicted reviews", f"{subset['predicted'].sum():,.1f}")
    elif page == "Model performance":
        st.markdown("#### Model accuracy · 2019 holdout")
        a, b, c = st.columns(3)
        a.metric("Overall R²", f"{r2_score(pred['weekly_review_count'], pred['predicted']):.3f}")
        b.metric("Overall MAE", f"{mean_absolute_error(pred['weekly_review_count'], pred['predicted']):.2f}")
        c.metric("Restaurant-weeks evaluated", f"{len(pred):,}")
        st.caption("Overall metrics are calculated across all restaurants, not only the selected restaurant.")
    else:
        st.markdown("#### Restaurant profile")
        a, b, c = st.columns(3)
        a.metric("Yelp rating", str(place.get("stars", "N/A")))
        b.metric("Dataset review count", str(place.get("review_count", "N/A")))
        c.metric("Observed 2019 weeks", str(len(subset)))
        st.markdown(f"**Categories:** {place.get('categories', 'N/A')}")

    if page in ("Overview", "Weekly comparison", "Model performance"):
        long = subset[["week_start", "weekly_review_count", "predicted"]].melt(
            id_vars="week_start", var_name="Series", value_name="Reviews")
        long["Series"] = long["Series"].map({
            "weekly_review_count": "Actual", "predicted": "Predicted"})
        plot = (alt.Chart(long).mark_line(point=alt.OverlayMarkDef(size=38), strokeWidth=3)
                .encode(x=alt.X("week_start:T", title="Week in 2019", axis=alt.Axis(format="%b %d", labelAngle=0)),
                        y=alt.Y("Reviews:Q", title="Yelp reviews per week", scale=alt.Scale(zero=True)),
                        color=alt.Color("Series:N", scale=alt.Scale(domain=["Actual", "Predicted"],
                                       range=["#a5234c", "#ef8fa7"]),
                                        legend=alt.Legend(orient="top", title=None)),
                        tooltip=[alt.Tooltip("week_start:T", title="Week", format="%b %d, %Y"),
                                 "Series:N", alt.Tooltip("Reviews:Q", format=".2f")])
                .properties(height=340).interactive(bind_y=False))
        st.altair_chart(plot, use_container_width=True)
        st.markdown('<div class="footnote">Historical weekly Yelp review counts. Lines connect available observations; missing weeks are not imputed.</div>', unsafe_allow_html=True)
    if page == "Model performance":
        st.markdown("**Interpretation:** R² summarizes variation explained across the 2019 holdout; "
                    "MAE is the average absolute prediction error in weekly Yelp reviews.")
    if page == "Restaurant details":
        st.dataframe(subset[["week", "weekly_review_count", "predicted"]].rename(columns={
            "week": "2019 ISO week", "weekly_review_count": "Actual reviews", "predicted": "Predicted reviews"}),
            hide_index=True, use_container_width=True)

st.markdown('<div class="footnote" style="margin-top:1rem">Prototype based on historical Yelp Open Dataset records. '
            'Review counts are a proxy for activity—not visits, sales, reservations, or a forecast of current demand.</div>',
            unsafe_allow_html=True)
