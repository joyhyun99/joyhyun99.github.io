---
layout: post
title: Applications of Time Series analysis from public bike usage records
feature-img: assets/img/pexels/triangular.jpeg
thumbnail: "assets/img/bike.jpg"
tags: [data-science, time-series, ARIMA]
author: joey99
excerpt_separator: <!--more-->
---

<p> When we analyze time series statistically, an autoregressive integrated moving average(model) can be used for the better understanding of the data. </p>
<p> ARIMA model is also widely utilized to predict future values of the series by training forecast model. Today, we are going to understand and utilize the concept of ARIMA model in order to find out the mean velocity of individual public bikes across one month time period.</p>  
<!--more-->

## Preprocessing <!--more-->
* TOC
{:toc}

데이터전처리 단계를 통해 결측치를 처리하고 분석이 가능한 상태로 데이터를 변모시킨다. 자전거의 이용거리와 이용시간을 나누어 평균 이동속도를 산출하고, 자전거번호만을 입력해 원하는 자료를 찾을 수 있도록 앞의 영어를 없애 간단화했으며 대여일시 피처를 시계열 피처로 변환한 다음, 인덱스로 지정한다.
이전의 값이 이후의 값에 영향을 미치는 정도인 AR(p) 모형이 2차시이고, RV의 평균값이 지속적으로 증가하거나 감소하는 추세인 MA(r) 모형이 2차시이고, 1차 차분이 완료되어 있는 파라미터인 ARIMA(2,1,2) Autogressive Integrated Moving Average 모델을 2020년도 1월 1일부터 2월 12일 까지의 따릉이 대여이력을 중심으로 학습한다.

{% highlight ruby %}
{% raw %}
{% bike = pd.read_csv('공공자전거 대여이력 정보_2020.01_056.csv', sep = ',')" %}
{% print(bike.shape) -> (1048575, 11) %}
{% print(bike.shape) -> (1048575, 11) %}
{% endraw %}
{% endhighlight %}

{% highlight ruby %}
{% raw %}
{% bike['Average Velocity (meters/minutes)'] = bike['이용거리'] / bike['이용시간'] %}
{% bike['bikenum'] = bike['자전거번호'].apply(lambda x: x[-5:]) %}
{% bike['date'] = pd.to_datetime(bike['대여일시']) %}
{% endraw %}
{% endhighlight %}

![model.head]({{ "/assets/img/2020-09-24-ARIMA-model-head(5).PNG" | relative_url }})

## Time-series analysis
Yt = α1Yt-1 + α2Yt-2 + β1εt-1 + β2εt-2 + εt 학습 데이터의 예측 결과와 실제 데이터를 비교한 그래프와 잔차의 변동을 시각화한 그래프를 첨부하였다.

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

모델이 20일 후 예측한 값.

{% include aligner.html images="arima4.png" %}
