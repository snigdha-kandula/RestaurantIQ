from pathlib import Path
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='RestaurantIQ', page_icon='🍽️', layout='wide')
CUISINES = ['cuisine_pizza','cuisine_chinese','cuisine_italian','cuisine_mexican','cuisine_american','cuisine_asian','cuisine_japanese','cuisine_indian','cuisine_thai','cuisine_vietnamese','cuisine_korean','cuisine_spanish','cuisine_french','cuisine_middle_eastern','cuisine_burger','cuisine_seafood','cuisine_vegetarian','cuisine_vegan','cuisine_cafe','cuisine_deli']
NUMERIC = ['week','month','year','prev_week_review_count','moving_avg_14_day','moving_avg_30_day','temp_max','temp_avg','temp_min','weather_code','precipitation','has_precipitation','heavy_precipitation','is_freezing','is_holiday','num_holiday_days','restaurant_stars','total_review_count','nearby_restaurant_count','avg_nearby_stars'] + CUISINES

@st.cache_data(show_spinner='Loading restaurants…')
def restaurants():
    df = pd.read_csv(ROOT / 'restaurants.csv')
    df['name'] = df['name'].fillna('Unnamed restaurant')
    return df

@st.cache_data(show_spinner='Preparing weekly data and training the historical model…')
def predictions():
    df = pd.read_csv(ROOT / 'features.csv', parse_dates=['date'])
    df['year'] = df['date'].dt.isocalendar().year.astype(int)
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    df['num_holiday_days'] = df['is_holiday']
    agg = {'daily_review_count':'sum','month':'first','season':lambda s:s.mode().iloc[0], 'temp_max':'max','temp_avg':'mean','temp_min':'min','weather_code':lambda s:s.mode().iloc[0],'precipitation':'sum','has_precipitation':'max','heavy_precipitation':'max','is_freezing':'max','is_holiday':'max','num_holiday_days':'sum','moving_avg_14_day':'mean','moving_avg_30_day':'mean','restaurant_stars':'first','total_review_count':'first','nearby_restaurant_count':'first','avg_nearby_stars':'first'}
    agg.update({c:'first' for c in CUISINES})
    weekly = (df.groupby(['business_id','year','week']).agg(agg).reset_index().rename(columns={'daily_review_count':'weekly_review_count'}).sort_values(['business_id','year','week']))
    weekly['prev_week_review_count'] = weekly.groupby('business_id')['weekly_review_count'].shift(1)
    weekly = weekly.dropna(subset=['prev_week_review_count']).dropna().reset_index(drop=True)
    X = pd.concat([weekly[NUMERIC], pd.get_dummies(weekly['season'], prefix='season', drop_first=True)], axis=1)
    train, test = weekly['year'] < 2019, weekly['year'] == 2019
    if not train.any() or not test.any():
        raise ValueError('Missing training or 2019 evaluation observations')
    model = XGBRegressor(n_estimators=100,max_depth=3,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=-1)
    model.fit(X.loc[train], weekly.loc[train,'weekly_review_count'])
    result = weekly.loc[test,['business_id','year','week','weekly_review_count']].copy()
    result['predicted'] = model.predict(X.loc[test])
    result['week_start'] = pd.to_datetime(result['year'].astype(str)+'-W'+result['week'].astype(str).str.zfill(2)+'-1',format='%G-W%V-%u',errors='coerce')
    return result

st.title('🍽️ RestaurantIQ')
st.caption('Exploring restaurant activity through historical machine learning predictions')
st.info('**Historical prototype:** This model predicts weekly Yelp review counts in 2019 using training data from 2013–2018. Yelp reviews are a proxy for activity, **not** actual customer visits, reservations, sales, or current demand.')
try:
    places, preds = restaurants(), predictions()
except (FileNotFoundError, KeyError, ValueError) as exc:
    st.error(f'Unable to load dashboard: {exc}')
    st.stop()
options = places[['business_id','name']].drop_duplicates('business_id')
options = options[options['business_id'].isin(preds['business_id'])].copy()
options['display'] = options['name']+' · '+options['business_id'].str[:8]
options = options.sort_values('display')
if options.empty:
    st.warning('No restaurants found in 2019 evaluation data.')
    st.stop()
chosen = st.sidebar.selectbox('Select a Philadelphia restaurant', options['display'].tolist())
business_id = options.loc[options['display']==chosen,'business_id'].iloc[0]
place = places.loc[places['business_id']==business_id].iloc[0]
subset = preds.loc[preds['business_id']==business_id].sort_values('week')
st.subheader(str(place['name']))
st.caption('Philadelphia, PA · 2019 evaluation period')
a,b,c = st.columns(3)
a.metric('Weeks evaluated',len(subset))
b.metric('Avg actual reviews / week',f"{subset['weekly_review_count'].mean():.2f}")
c.metric('Avg predicted reviews / week',f"{subset['predicted'].mean():.2f}")
st.subheader('Actual vs. predicted weekly Yelp reviews')
chart = subset.set_index('week_start')[['weekly_review_count','predicted']].rename(columns={'weekly_review_count':'Actual reviews','predicted':'Predicted reviews'})
st.line_chart(chart)
st.caption('Retrospective 2019 test-set predictions; not forecasts for today.')
left,right = st.columns(2)
with left:
    st.subheader('Model evaluation across all 2019 restaurants')
    st.metric('R²',f"{r2_score(preds['weekly_review_count'],preds['predicted']):.3f}")
    st.metric('Mean absolute error',f"{mean_absolute_error(preds['weekly_review_count'],preds['predicted']):.2f} reviews")
with right:
    st.subheader('Restaurant profile')
    st.write(f"**Yelp rating:** {place.get('stars','N/A')}")
    st.write(f"**Review count (dataset):** {place.get('review_count','N/A')}")
    st.write(f"**Categories:** {place.get('categories','N/A')}")
with st.expander('View and download historical predictions'):
    st.dataframe(subset[['week','weekly_review_count','predicted']],hide_index=True,use_container_width=True)
    st.download_button('Download predictions',subset.to_csv(index=False),file_name='restaurantiq_historical_predictions.csv',mime='text/csv')
st.divider()
st.subheader('From model to product')
st.write('RestaurantIQ explores how data-driven insights could eventually help independent restaurant owners plan staffing and inventory. This prototype makes historical predictions understandable to nontechnical users; it does not yet make operational recommendations.')
st.subheader('Product roadmap')
st.markdown('1. Interview restaurant owners to validate planning challenges.\n2. Incorporate permissioned sales, reservation, or foot-traffic data.\n3. Test genuine future forecasts using only features known at prediction time.\n4. Measure whether insights improve decisions and reduce waste.')
st.caption('Academic project extension. Historical Yelp Open Dataset and Philadelphia weather data.')
