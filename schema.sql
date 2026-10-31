PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS features;
DROP TABLE IF EXISTS competition;
DROP TABLE IF EXISTS calendar;
DROP TABLE IF EXISTS daily_demand;
DROP TABLE IF EXISTS weather;
DROP TABLE IF EXISTS reviews;
DROP TABLE IF EXISTS restaurants;

CREATE TABLE restaurants(
            business_id VARCHAR(25) PRIMARY KEY,
            name VARCHAR(50),
            latitude FLOAT,
            longitude FLOAT,
            stars FLOAT,
            review_count INT,
            categories TEXT
);

CREATE TABLE reviews(
            review_id VARCHAR(25) PRIMARY KEY,
            business_id VARCHAR(25),
            user_id VARCHAR(25),
            stars INT,
            date CHAR(10),
            useful INT,
            funny INT,
            cool INT,
            FOREIGN KEY (business_id) REFERENCES restaurants(business_id)
);

CREATE TABLE daily_demand(
            business_id VARCHAR(25) NOT NULL,
            date TEXT NOT NULL,
            daily_review_count INTEGER NOT NULL,
            PRIMARY KEY (business_id, date),
            FOREIGN KEY (business_id) REFERENCES restaurants(business_id)
);

CREATE TABLE weather(
            weather_date CHAR(10) PRIMARY KEY,
            temp_max FLOAT,
            temp_avg FLOAT,
            temp_min FLOAT,
            weather_code INT,
            precipitation FLOAT
);

.mode csv
.import --skip 1 restaurants.csv restaurants
.import --skip 1 reviews.csv reviews
.import --skip 1 filtered_pa_weather.csv weather

--helps queries that group reviews by restaurant and date
CREATE INDEX idx_reviews_business_date ON reviews(business_id, date);

--helps queries that group weather by date
CREATE INDEX idx_weather_date ON weather(weather_date);

INSERT INTO daily_demand (business_id, date, daily_review_count)
SELECT business_id, date, COUNT(*) AS daily_review_count
FROM reviews
GROUP BY business_id, date;

CREATE TABLE calendar (
    calendar_date TEXT NOT NULL PRIMARY KEY,
    day_of_week INT,
    day_of_month INT,
    day_name TEXT,
    month INT,
    year INT,
    is_weekend INT,
    season TEXT,
    is_holiday INT
);

INSERT INTO calendar
SELECT
    date AS calendar_date,
    CAST(strftime('%w', date) AS INTEGER) AS day_of_week, --pulls out day of week as an integer
    CAST(strftime('%d', date) AS INTEGER) AS day_of_month, --pulls out day of month as an integer
    CASE CAST(strftime('%w', date) AS INTEGER) --converts day of week integer to day of week name
        WHEN 0 THEN 'Sunday'
        WHEN 1 THEN 'Monday'
        WHEN 2 THEN 'Tuesday'
        WHEN 3 THEN 'Wednesday'
        WHEN 4 THEN 'Thursday'
        WHEN 5 THEN 'Friday'
        WHEN 6 THEN 'Saturday'
    END AS day_name,
    CAST(strftime('%m', date) AS INTEGER) AS month, --pulls out month as an integer
    CAST(strftime('%Y', date) AS INTEGER) AS year, --pulls out year as an integer
    CASE
        WHEN strftime('%w', date) IN ('0', '6') THEN 1 --checks if day of week is Saturday or Sunday
        ELSE 0
    END AS is_weekend,
    CASE -- considers all seasons to see when demand would be highest
        WHEN CAST(strftime('%m', date) AS INTEGER) IN (12, 1, 2) THEN 'winter' 
        WHEN CAST(strftime('%m', date) AS INTEGER) IN (3, 4, 5) THEN 'spring'
        WHEN CAST(strftime('%m', date) AS INTEGER) IN (6, 7, 8) THEN 'summer'
        WHEN CAST(strftime('%m', date) AS INTEGER) IN (9, 10, 11) THEN 'fall'
    END AS season,
    CASE --considers all holidays to see when demand would be highest
        WHEN (CAST(strftime('%m', date) AS INTEGER) = 11 AND CAST(strftime('%d', date) AS INTEGER) = 23) OR
             (CAST(strftime('%m', date) AS INTEGER) = 11 AND CAST(strftime('%d', date) AS INTEGER) = 24) OR
             (CAST(strftime('%m', date) AS INTEGER) = 11 AND CAST(strftime('%d', date) AS INTEGER) = 25) OR
             (CAST(strftime('%m', date) AS INTEGER) = 11 AND CAST(strftime('%d', date) AS INTEGER) = 26) OR
             (CAST(strftime('%m', date) AS INTEGER) = 12 AND CAST(strftime('%d', date) AS INTEGER) = 25) OR
             (CAST(strftime('%m', date) AS INTEGER) = 12 AND CAST(strftime('%d', date) AS INTEGER) = 26) OR
             (CAST(strftime('%m', date) AS INTEGER) = 2 AND CAST(strftime('%d', date) AS INTEGER) = 14) OR
             (CAST(strftime('%m', date) AS INTEGER) = 1 AND CAST(strftime('%d', date) AS INTEGER) = 1) THEN 1

        ELSE 0
    END AS is_holiday
FROM (SELECT DISTINCT date FROM daily_demand);

-- finds nearby restaurants (in the same table) to see if they are competition
--criteria: within 0.00724 degrees of latitude and longitude of the restaurant (about 0.5 miles)
CREATE TABLE competition AS
SELECT r1.business_id, COUNT(r2.business_id) AS nearby_restaurant_count, ROUND(AVG(r2.stars), 3) AS avg_nearby_stars
FROM restaurants r1
LEFT JOIN restaurants r2
    ON r1.business_id != r2.business_id
    AND r2.latitude BETWEEN r1.latitude - 0.00724 AND r1.latitude + 0.00724
    AND r2.longitude BETWEEN r1.longitude - 0.00724 AND r1.longitude + 0.00724
GROUP BY r1.business_id;

--creates the features table that will be used for the machine learning model
CREATE TABLE features AS
SELECT d.business_id, d.date, d.daily_review_count,
    c.day_of_week, c.day_of_month, c.day_name, c.month, c.year, c.is_weekend, c.is_holiday, c.season,

--notes the restaurant's demand (# yelp reviews)from the previous day to help the model understand if demand is increasing or decreasing
    LAG(d.daily_review_count) OVER (  -- assisted by cgpt 
        PARTITION BY d.business_id
        ORDER BY d.date
    ) AS previous_day_review_count,

--aggregation function that calculates the average demand over the previous 7 days for nearby restaurants
    ROUND(AVG(d.daily_review_count) OVER ( -- assisted by cgpt 
            PARTITION BY d.business_id 
            ORDER BY d.date 
            ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING --looks at the previous 7 days
        ),
        3
    ) AS moving_avg_7_day,

--aggregation function that calculates the average demand over the previous 14 days for nearby restaurants
    ROUND(
        AVG(d.daily_review_count) OVER ( -- assisted by cgpt 
            PARTITION BY d.business_id
            ORDER BY d.date
            ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING
        ),
        3
    ) AS moving_avg_14_day,

--aggregation function that calculates the average demand over the previous 30 days for nearby restaurants
    ROUND(
        AVG(d.daily_review_count) OVER (
            PARTITION BY d.business_id
            ORDER BY d.date
            ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
        ),
        3
    ) AS moving_avg_30_day,

    w.temp_max, w.temp_avg, w.temp_min, w.weather_code, w.precipitation, --weather data for features table

-- creates flags for different weather conditions
    CASE
        WHEN w.precipitation > 0 THEN 1
        ELSE 0
    END AS has_precipitation,

    CASE
        WHEN w.precipitation > 0.1 THEN 1
        ELSE 0
    END AS heavy_precipitation,

    CASE
        WHEN w.temp_avg < 32 THEN 1
        ELSE 0
    END AS is_freezing,

    r.name AS restaurant_name, r.stars AS restaurant_stars, r.review_count AS total_review_count, r.latitude, r.longitude, r.categories,

-- creates flags for different cuisines (we picked the most common cuisines)
    CASE WHEN r.categories LIKE '%Pizza%' THEN 1 ELSE 0 END AS cuisine_pizza,
    CASE WHEN r.categories LIKE '%Chinese%' THEN 1 ELSE 0 END AS cuisine_chinese,
    CASE WHEN r.categories LIKE '%Italian%' THEN 1 ELSE 0 END AS cuisine_italian,
    CASE WHEN r.categories LIKE '%Mexican%' THEN 1 ELSE 0 END AS cuisine_mexican,
    CASE WHEN r.categories LIKE '%American%' THEN 1 ELSE 0 END AS cuisine_american,
    CASE WHEN r.categories LIKE '%Asian%' THEN 1 ELSE 0 END AS cuisine_asian,
    CASE WHEN r.categories LIKE '%Japanese%' THEN 1 ELSE 0 END AS cuisine_japanese,
    CASE WHEN r.categories LIKE '%Indian%' THEN 1 ELSE 0 END AS cuisine_indian,
    CASE WHEN r.categories LIKE '%Thai%' THEN 1 ELSE 0 END AS cuisine_thai,
    CASE WHEN r.categories LIKE '%Vietnamese%' THEN 1 ELSE 0 END AS cuisine_vietnamese,
    CASE WHEN r.categories LIKE '%Korean%' THEN 1 ELSE 0 END AS cuisine_korean,
    CASE WHEN r.categories LIKE '%Spanish%' THEN 1 ELSE 0 END AS cuisine_spanish,
    CASE WHEN r.categories LIKE '%French%' THEN 1 ELSE 0 END AS cuisine_french,
    CASE WHEN r.categories LIKE '%Middle%' THEN 1 ELSE 0 END AS cuisine_middle_eastern,
    CASE WHEN r.categories LIKE '%Burger%' THEN 1 ELSE 0 END AS cuisine_burger,
    CASE WHEN r.categories LIKE '%Seafood%' THEN 1 ELSE 0 END AS cuisine_seafood,
    CASE WHEN r.categories LIKE '%Vegetarian%' THEN 1 ELSE 0 END AS cuisine_vegetarian,
    CASE WHEN r.categories LIKE '%Vegan%' THEN 1 ELSE 0 END AS cuisine_vegan,
    CASE WHEN r.categories LIKE '%Cafe%' THEN 1 ELSE 0 END AS cuisine_cafe,
    CASE WHEN r.categories LIKE '%Deli%' THEN 1 ELSE 0 END AS cuisine_deli,

    comp.nearby_restaurant_count, comp.avg_nearby_stars --competition data for features table

FROM daily_demand d
LEFT JOIN calendar c ON d.date = c.calendar_date
LEFT JOIN weather w ON d.date = w.weather_date
LEFT JOIN restaurants r ON d.business_id = r.business_id
LEFT JOIN competition comp ON d.business_id = comp.business_id;

--creates a csv file of the features table
.headers on
.mode csv
.output features.csv
SELECT * FROM features;
.output stdout


--some queries to check the data for features table
SELECT COUNT(*) AS total_restaurants
FROM restaurants;

SELECT COUNT(*) AS total_reviews
FROM reviews;

SELECT COUNT(*) AS total_daily_demand_rows
FROM daily_demand;

SELECT COUNT(*) AS total_feature_rows
FROM features;

SELECT COUNT(DISTINCT business_id) AS restaurants_in_features
FROM features;

SELECT MIN(date) AS start_date, MAX(date) AS end_date
FROM features;

SELECT COUNT(*) AS rows_missing_weather
FROM features
WHERE temp_avg IS NULL;

SELECT COUNT(*) AS rows_with_holidays
FROM features
WHERE is_holiday = 1;
