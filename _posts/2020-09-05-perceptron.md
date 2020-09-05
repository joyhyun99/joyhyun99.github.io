---
layout: post
title: "The Rise of Machine Learning: Perceptron Learning Algorithms and its Fallacies"
feature-img: assets/img/pexels/triangular.jpeg
thumbnail: "assets/img/perceptron_image.jpg"
tags: [deep-learning, perceptron, Adaline, personal-review]
author: joey99
excerpt_separator: <!--more-->
---

<p> 최초의 인공지능 알고리즘 퍼셉트론은 기계가 주어진 데이터를 가지고 스스로 가중치를 학습해 두 개의 클래스로 분류하는 이진 분류 알고리즘이다. </p>
<p> 퍼셉트론은 수동적인 computing 수용자였던 기계에서 능동적으로 사고하는 기계로의 첫 걸음이라 많은 기대를 받았지만, 방대한 데이터를 다루는 상황을 고려했을 때 적은 선형적 데이터만 다룰 수 있었던 알고리즘의 한계가 명확히 드러났고 퍼셉트론의 문제점을 개선한 각 훈련 샘플마다 진짜 클래스 레이블과 선형 활성화 함수의 실수 출력 값을 비교하여 모델의 오차를 계산하고 가중치를 업데이트하는 Adaline 모델도 방대한 데이터를 다루지는 못했다. </p>  

<!--more-->

인공지능 알고리즘이 나오기 전, Dr. W.S. McCulloch와 Dr. Walter Pitts는 생물학적 뇌가 동작하는 방식을 이해하기 위해
뇌의 신경 세포를 이진 출력을 내는 간단한 논리 회로로 표현한 MCP 뉴런을 창안해내었다.
MCP 뉴런의 원리는 간단하다. 신경세포가 정보를 받아들이는 수상 돌기에 여러 신호가 도착하면, 그 정보들이 세포체에서 합쳐지게 되는데,
이 합쳐진 신호가 특정 임계값을 넘으면 1의 값을 부여해 다음 신경세포로 정보를 전달하고,
특정 임계값을 넘지 못하면 -1의 값을 부여해 다음 신경세포로 정보를 전달하지 않는다.

최초의 인공지능 알고리즘인 퍼셉트론도 MCP 뉴런과 마찬가지로 데이터를 -1와 1 두 개의 클래스로 분류하는 이진 분류 작업이다.
퍼셉트론은 0보다 작은 실수에 대해서 0, 0보다 큰 실수에 대해서 1의 값을 부여하는 단위 계단 함수를 약간 변형하여
최종 입력 z가 사전에 정의된 임계값 θ보다 크면 클래스 1로 예측하고, 그렇지 않으면 클래스 -1로 예측하는 결정 함수 φ(z) 를 정의한다.

![6]({{ "/assets/img/perceptrongraph.png" | relative_url }})

퍼셉트론은 이렇게 주어진 데이터를 가지고 스스로 양성 클래스 1, 그리고 음성 클래스 -1으로 분류하는 이진 분류 작업을 수행하기 위해 
샘플의 최적가중치 ω를 학습하고, 이 가중치는 뉴런의 출력 신호를 낼지 말지를 결정하기 위해 입력 특성에 곱하는 계수이다.
다시 말해 최종 입력 z는 z = w1x1 + w2x2 +...+ wnxn = wTx으로 정의할 수 있고,
z 가 0보다 작으면 음성 클래스로 분류하고 0보다 크면 양성 클래스로 분류한다.

퍼셉트론의 가중치를 학습하기 전, 가중치를 0 또는 랜덤한 작은 값으로 초기화 해준다.
가중치를 0으로 초기화하지 않는 이유는 가중치가 0이 아니어야 학습률 η(eta)가 분류 결과에 영향을 주기 때문인데,
가중치가 0으로 초기화되어 있다면 학습률 파라미터 eta는 가중치 벡터의 방향이 아니라 크기에만 영향을 미치기 때문이다.
Perceptron class에서는 표준편차가 0.01인 정규분포에서 뽑은 랜덤한 작은 수를 뽑는 것으로 구현이 되어 있다.

{% highlight ruby %}
{% raw %}
self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
{% endraw %}
{% endhighlight %}


그 다음에 각 훈련 샘플 xi에서 출력 값 yhat을 계산하고 가중치를 업데이트한다
η = 학습률. yi는 i번째 훈련 샘플의 진짜 클래스 레이블. yhati는 예측 클래스 레이블.
퍼셉트론은 모든 가중치 wj를 업데이트하기 전에 yhati를 다시 계산하지 않는다

{% highlight ruby %}
{% raw %}
for xi, target in zip(X, y):
update = self.eta * (target - self.predict(xi))
self.w_[1:] += update * xi
self.w_[0] += update
{% endraw %}
{% endhighlight %}

![5]({{ "/assets/img/perceptron_algo.jpeg" | relative_url }})

학습률 eta와 에포크 횟수(훈련 데이터를 반복하는 횟수) n_iter로 새로운 Perceptron 객체를 초기화한다
fit 메서드에서 self.w_ 가중치를 벡터 Rm+1로 초기화한다
여기서 m은 데이터셋에 있는 차원 개수이다
self.w_[0]은 앞서 언급한 절편이다

그 다음 입력 값 x와 이에 상응하는 가중치 벡터 w의 선형 조합으로 결정 함수 φ(z)를 정의한다.

{% highlight ruby %}
{% raw %}
np.dot(X, self.w_[1:]) + self.w_[0]
{% endraw %}
{% endhighlight %}

특정 샘플 xi의 최종 입력이 사전에 정의된 임계값 θ보다 크면 클래스 1로 예측하고, 그렇지 않으면 클래스 -1로 예측한다.

{% highlight ruby %}
{% raw %}
np.where(self.net_input(X) >= 0.0, 1, -1)
{% endraw %}
{% endhighlight %}
 
 
퍼셉트론은 두 클래스가 선형적으로 구분되고 학습률이 충분히 작을 때만 수렴이 보장된다.

IMDB 데이터셋

인터넷 영화 데이터베이스로부터 가져온 양극단의 리뷰 50,000개로 이루어진 IMDB 데이터셋을 사용하겠습니다. 이 데이터셋은 훈련 데이터 25,000개와 테스트 데이터 25,000개로 나뉘어 있고 각각 50%는 부정, 50%는 긍정 리뷰로 구성되어 있습니다.
이 데이터는 전처리되어 있어 각 리뷰(단어 시퀀스)가 숫자 시퀀스로 변환되어 있습니다. 여기서 각 숫자는 사전에 있는 고유한 단어를 나타냅니다.

{% highlight ruby %}
{% raw %}
{% def vectorize_sequences(sequences, dimension = 10000): %}
{% train_data = vectorize_sequences(train_data) %}
{% train_data = np.where(train_data == 1, 1, -1) %}
{% train_labels = np.asarray(train_labels).astype('float32') %}
{% endraw %}
{% endhighlight %}

{% highlight ruby %}
{% raw %}
{% train_data.shape -> (25000, 10000) %}
{% train_labels.shape 0> (25000, ) %}
{% endraw %}
{% endhighlight %}

{% highlight ruby %}
{% raw %}
{% ppn = Perceptron(eta=0.01, n_iter=40) %}
{% ppn.fit(train_data, train_labels) %}
{% endraw %}
{% endhighlight %}

![4]({{ "/assets/img/perceptron.png" | relative_url }})

{% highlight ruby %}
{% raw %}
{% y_pred = ppn.predict(test_data) %}
{% print('Perceptron Algorithm Learning Reports') %}
{% Number of samples incorrectly classified : 20581 %}
{% Accuracy = 17.68% (Total sample size = 25000) %}
{% endraw %}
{% endhighlight %}

적응형 선형 뉴런 Adaline
아달린은 연속 함수로 비용 함수를 정의하고 최소화하는 핵심 개념
가중치를 업데이트하는 데 퍼셉트론처럼 단위 계단 함수 대신 선형 활성화 함수를 사용
아달린에서 선형 활성화 함수 φ(z)는 최종 입력과 동일한 함수
φ(wTx) = wTx
선형 활성화 함수가 가중치 학습에 사용되지만 최종 예측을 만드는 데 여전히 임계 함수를 사용. 
아달린 알고리즘은 진짜 클래스 레이블과 선형 활성화 함수의 실수 출력 값을 비교하여 모델의 오차를 계산하고 가중치를 업데이트

지도학습 알고리즘의 핵심 구성 요소는 학습 과정 동안 최적화하기 위해 정의한 목적 함수이다
아달린은 계산된 출력과 진짜 클래스 레이블 사이의 제곱오차합으로 가중치를 학습할 비용 함수 J를 정의한다

함수 J

경사 하강법을 사용하면 비용함수 J(w)의 그래디언트 ΔJ(w) 반대 방향으로 조금씩 가중치를 업데이트
가중치 변화량 w은 음수의 그래디언트에 학습률 eta를 곱한 것으로 정의
비용 함수의 그래디언트를 계산하려면 각 가중치 wj에 대한 편도 함수를 계산
가중치 wj의 업데이트 공식
 
zi는 정수 클래스 레이블이 아니라 실수
훈련 세트에 있는 모든 샘플을 기반으로 가중치 업데이트를 계산한다. 이 방식을 배치 경사 하강법 (batch gradient dexcent)이라고도 한다
 
 
퍼셉트론처럼 개별 훈련 샘플마다 평가한 후 가중치를 업데이트하지 않고 전체 훈련 데이터셋을 기반으로 그래디언트를 계산
절편(0번째 가중치)은 self.eta * errors.sum()이고 가중치 1에서 m까지는 self.eta * X.T.dot(errors이다
X.T.dot(errors)는 특성 행렬과 오차 벡터간의 행렬-벡터 곱셈이다
 
이 코드의 activation 메서드는 단순한 항등 함수(identity function)이기 때문에 아무런 영향을 미치지 않는다

![1]({{ "/assets/img/adaline.png" | relative_url }})
 
![2]({{ "/assets/img/adaline_algo.png" | relative_url }})
 
![3]({{ "/assets/img/adalinegraph.png" | relative_url }})
 

 
 
 
 
 
 
 
 
 
 
