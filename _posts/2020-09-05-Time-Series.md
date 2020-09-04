---
layout: post
title: Applications of Time Series analysis from public bike usage records
feature-img: "assets/img/bike.jpg"
thumbnail: "assets/img/bike.jpg"
tags: [beginning]
author: joey99
excerpt_separator: <!--more-->
---
TBA!

<!--more-->

{% include aligner.html images="2020-09-24-ARIMA-model-10_2.png" %}

{% include aligner.html images="pexels/arima2.png,arima3.png" column = 2 %}


{% include aligner.html images="2020-09-24-ARIMA-model_12_1.png" %}

<p> "2020년도 1월 1일부터 2월 12일 까지의 따릉이 대여이력을 중심으로 구축" </p>

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

"/assets/img/2020-09-24-ARIMA-model-head(5).PNG"

{% highlight ruby %}
{% raw %}
{% print("%d num bike's average velocity is %.2f (m/m)" %(int(bike_number), bike_train_speed)) %}
{% 14251 num bike's average velocity is 181.14 (m/m) %}
{% print("%d num bike's average velocity is %.2f (m/m)." %(int(bike_number), bike_test_speed)) %}
{% 14251 num bike's average velocity is 159.87 (m/m) %}
{% endraw %}
{% endhighlight %}

"/assets/img/2020-09-24-ARIMA-model-result.PNG"

{% highlight ruby %}
{% raw %}
{% model = ARIMA(bike_train['Average Velocity (meters/minutes)'].values, order=(2,1,2)) %}
{% model_fit = model.fit(trend='c', full_output=True, disp=True) %}
{% fig = model_fit.plot_predict() %}
{% residuals = pd.DataFrame(model_fit.resid) %}
{% endraw %}
{% endhighlight %}

{% highlight ruby %}
{% raw %}
{% fore = model_fit.forecast(steps=20) %}
{% rmse = sqrt(mean_squared_error(pred_y, test_y)) %}
{% print("model's rmse is %.2f." %(rmse)) %}
{% model's rmse is 55.91. %}
{% endraw %}
{% endhighlight %}
