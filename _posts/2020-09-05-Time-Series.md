---
layout: post
title: Applications of Time Series analysis from public bike usage records
feature-img: assets/img/pexels/triangular.jpeg
thumbnail: "assets/img/bike.jpg"
tags: [data-science, time-series, ARIMA]
author: joey99
excerpt_separator: <!--more-->
---

<p> When we analyze time series statistically, an autoregressive integrated moving average model (ARIMA) can be used for the better understanding of the data. </p>
<p> ARIMA model is also widely utilized to predict future values of the series by training forecast model. Today, we are going to understand and utilize the concept of ARIMA model in order to find out the mean velocity of individual public bikes across one month time period.</p>  
<!--more-->

## Preprocessing <!--more-->
* TOC
{:toc}

Through the preprocessing stage, we process messy original data to the perfect data we can analyze with proper models.

{% highlight ruby %}
{% raw %}
{% bike = pd.read_csv('공공자전거 대여이력 정보_2020.01_056.csv', sep = ',')" %}
{% print(bike.shape) -> (1048575, 11) %}
{% print(bike.shape) -> (1048575, 11) %}
{% endraw %}
{% endhighlight %}

![model.head]({{ "/assets/img/2020-09-24-ARIMA-model-head(5).PNG" | relative_url }})

First, calculate the average velocity by dividing distance with used time. Next, the information at the first row is not that friendly. simplify it by removing the str value of bike's number. Finally, after converting time feature to a time series and indexification, we can conclude preprocessing step and move over to the analysis process.

{% highlight ruby %}
{% raw %}
{% bike['Average Velocity (meters/minutes)'] = bike['이용거리'] / bike['이용시간'] %}
{% bike['bikenum'] = bike['자전거번호'].apply(lambda x: x[-5:]) %}
{% bike['date'] = pd.to_datetime(bike['대여일시']) %}
{% endraw %}
{% endhighlight %}


## Time-series analysis
<p> When given a time series data X with an real number index, ARIMA models can be denoted to ARIMA(p,q,r) where p is the order of the autoregressive model and AR(p) is the extent to which the previous value affects the subsequent values. r is the order of the moving average model and Ma(r) is a trend in which the mean value of Random variables increases or decreases continuously. q is the degree of differencing. </p>

<p> Consider the public bike model follows ARIMA(2,1,2) model. Train the data and evaluate the result both visually using graph and mathematically using the rmse value. </p>

$$ Yt = α1Yt-1 + α2Yt-2 + β1εt-1 + β2εt-2 + εt $$ 

{% highlight ruby %}
{% raw %}
{% print("%d num bike's average velocity is %.2f (m/m)" %(int(bike_number), bike_train_speed)) %}
{% 14251 num bike's average velocity is 181.14 (m/m) %}
{% print("%d num bike's average velocity is %.2f (m/m)." %(int(bike_number), bike_test_speed)) %}
{% 14251 num bike's average velocity is 159.87 (m/m) %}
{% endraw %}
{% endhighlight %}

{% include aligner.html images="2020-09-24-ARIMA-model-10_2.png" %}

## Time-series prediction
학습 데이터를 통해 학습한 모델이 예측한 예측 데이터를 산출하고, 좀 전에 따로 구비해두었던 테스트 데이터와 상호비교하여 모델의 정확성을 확인하였다. 이로써 시계열데이터를 변수로 넣으면 자전거의 평균속도가 산출되는 ARIMA 모델이 학습되었고, 데이터의 시간을 벗어난 다른 시간대의 자전거의 평균속도를 구할 수 있게 되었다. 그래프는 모델이 예상한 최소, 최대, 그리고 평균 속도 그래프와 테스트 데이터를 시각적으로 비교하였고 마지막으로 추정 값 혹은 모델이 예측한 값과 실제 환경에서 관찰되는 값의 차이를 다룬 평균 제곱근 편차의 값을 산출하여 모델의 정확성을 측정하였다.

{% highlight ruby %}
{% raw %}
{% model = ARIMA(bike_train['Average Velocity (meters/minutes)'].values, order=(2,1,2)) %}
{% model_fit = model.fit(trend='c', full_output=True, disp=True) %}
{% fig = model_fit.plot_predict() %}
{% residuals = pd.DataFrame(model_fit.resid) %}
{% endraw %}
{% endhighlight %}

![model.result]({{ "/assets/img/2020-09-24-ARIMA-model-result.PNG" | relative_url }})

![model.graph]({{ "/assets/img/arima2.png" | relative_url }})

{% highlight ruby %}
{% raw %}
{% fore = model_fit.forecast(steps=20) %}
{% rmse = sqrt(mean_squared_error(pred_y, test_y)) %}
{% print("model's rmse is %.2f." %(rmse)) %}
{% model's rmse is 55.91. %}
{% endraw %}
{% endhighlight %}

{% include aligner.html images="2020-09-24-ARIMA-model_12_1.png" %}

{% include aligner.html images="arima4.png" %}
