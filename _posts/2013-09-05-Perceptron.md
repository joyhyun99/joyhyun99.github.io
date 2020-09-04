---
layout: post
title: The Rise of Machine Learning: Perceptron Learning Algorithm and its fallacies
feature-img: assets/img/pexels/triangular.jpeg
thumbnail: "assets/img/perceptron_image.jpg"
tags: [deep-learning, perceptron, Adaline, personal-review]
author: joey99
excerpt_separator: <!--more-->
---

<p> When we analyze time series statistically, an autoregressive integrated moving average model (ARIMA) can be used for the better understanding of the data. </p>
<p> ARIMA model is also widely utilized to predict future values of the series by training forecast model. Today, we are going to understand and utilize the concept of ARIMA model in order to find out the mean velocity of individual public bikes across one month time period.</p>  
<!--more-->

퍼셉트론 규칙에서 로젠블라트는 자동으로 최적의 가중치를 학습하는 알고리즘을 제안했습니다.
이 가중치는 뉴런의 출력 신호를 낼지 말지를 결정하기 위해 입력 특성에 곱하는 계수입니다.
퍼셉트론은 인공 뉴런 아이디어를 두 개의 클래스가 있는 이진 분류 작업으로 볼 수 있다.
두 클래스는 간단하게 1(양성 클래스)과 -1(음성 클래스)로 나타낸다

eta = float 학습률 (0.0과 1.0 사이)
n_iter = 훈련 데이터셋 반복 횟수
random_state = 가중치 무작위 초기화를 위한 난수 생성기 시드
w_ = 학습된 가중치
errors_ = 에포크마다 누적된 분류 오류

매개변수
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          n_samples개의 샘플과 n_features개의 특성으로 이루어진 훈련 데이터
        y : array-like, shape = [n_samples]
          타깃값

        반환값
        -------
        self : object
        
 errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            
 return np.dot(X, self.w_[1:]) + self.w_[0]
 
 return np.where(self.net_input(X) >= 0.0, 1, -1)
 
 학습률 eta와 에포크 횟수(훈련 데이터를 반복하는 횟수) n_iter로 새로운 Perceptron 객체를 초기화한다
 fit 메서드에서 self.w_ 가중치를 벡터 Rm+1로 초기화한다
 여기서 m은 데이터셋에 있는 차원 개수이다
 self.w_[0]은 앞서 언급한 절편이다
 
 rgen.normal(loc = 0..0, scale = 0.01, size1 = 1 + X.shape[1])을 사용해 표준편차가 0.01인 정규분포에서 뽑은 랜덤한 작은 수
 
 가중치를 0으로 초기화하지 않는 이유는 가중치가 0이 아니어야 학습률 η(eta)가 분류 결과에 영향을 주기 때문
 가중치가 0으로 초기화되어 있다면 학습률 파라미터 eta는 가중치 벡터의 방향이 아니라 크기에만 영향을 미친다
 
 fit 메서드는 가중치를 초기화한 후 훈련 세트에 있는 모든 개개의 샘플을 반복 순회하면서 퍼셉트론 학습 규칙에 따라 가중체를 업데이트한다
 클래스 레이블은 predict 메서드에서 예측한다
 fit 메서드에서 가중치를 여ㅓㅂ데이트하기 위해 predict 메서드를 호출하여 클래스 레이블에 대한 예측을 얻는다
 predict 메서드는 모델이 학습되고 난 수 새로운 데이터의 클래스 레이블을 예측하는 데도 사용할 수 있다
 
 에포크마다 self.errors_ 리스트에 잘못 분류된 횟수를 기록한다. 나중에 훈련하는 동안 얼마나 퍼셉트론을 잘 수행했는지 분석할 수 있다
 
 np.dot 함수는 벡터 점곱 wTx를 계산한다
 
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
 
 
 
 
 
 
 
 
 
 
 
 
