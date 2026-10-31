# RestaurantIQ

A Streamlit prototype extending [CS210Project](https://github.com/snigdha-kandula/CS210Project) into a historical restaurant-activity forecasting dashboard.

## Run locally

Copy `app.py` and `requirements.txt` into the RestaurantIQ repository alongside `features.csv` and `restaurants.csv`.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app recreates the original weekly feature aggregation and XGBoost configuration, trains on 2013–2018, and shows historical 2019 holdout predictions. First load can take longer while training; Streamlit caches the result.

## Deploy

On Streamlit Community Cloud, connect the GitHub repository, choose branch `main`, and select `app.py` as the entrypoint.

## Limitations

Predictions concern weekly Yelp review counts, not actual foot traffic, sales, or reservations. The model's 2019 evaluation is retrospective, not live forecasting. A future version should use permissioned operational data and prospective validation.

Data source: [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/). Follow applicable dataset licensing and attribution requirements.
