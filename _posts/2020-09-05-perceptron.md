---
layout: post
title: "The Rise of Machine Learning: Perceptron Learning Algorithms and its Fallacies"
feature-img: assets/img/pexels/triangular.jpeg
thumbnail: "assets/img/perceptron_image.jpg"
tags: [deep-learning, perceptron, Adaline, personal-review]
author: joey99
excerpt_separator: <!--more-->
---


<p> Perceptron Learning Algorithm is among the first prototypes of Artificial Intelligence trained to obtain the optimum weight in order to classify binary datas solely by its own. </p>
<p> Perceptron was highly aspired in the 60s since it was the first step for a passive computing machine to actively utilize self-trained algorithms to analyze given datas. However, Perceptron algorithm was limited to only handle small linear data with high accuracy. So when given huge and vast amount of datas, Perceptron algorithm's accuracy had fallen to a point where it could not be utilized. </p>  

<!--more-->
![0]({{ "/assets/img/perceptron_image.jpg" | relative_url }})

<p> Just before the era of Artificial Intelligence, Dr. W.S. McCulloch and Dr. Walter Pitts came up with simple logical circuit that produces binary outputs to express how biological transmission of data works in our nerve cells.</p>
<p> MCP neural network's principles are simple. When multiple different signals arrive at the dendrite where our nerve cells accept information, these signals are merged to produce a single big signal. If the combined signal exceeds a certain threshold, the MCP hands over a positive one value to next MCP. And when the combined signal does not exceed a certain threshold, the MCP does not hand over signal, thus a negative one value is produced. </p>

<p> Like MCP neural network, Perceptron learning algorithm classifies binary datas into positive one value and negative one value.</p>
<p> Perceptron employs a slightly modified version of unit step function. Unit step function assigns number zero to values less than zero, and assigns number one to values more than zero. Perceptron's net input also assigns number one to input greater than the predifined threshold. However, if not, it assigns number negative one for those datas </p>

![6]({{ "/assets/img/perceptrongraph.png" | relative_url }})

<br>

<p> In order to classify binary datas, Perceptron algorithm trains to obtain the obtimum weight of the sample, and this factor is multiplied with input datas to determine whether to hand over a signal or not to. </p>
<p> In other words, the net input of the function z can be defined into the function below and if net input is less than zero, the data is classified into the negative class and if not, the data is classified into the positive class.</p>

$$ z = w1x1 + w2x2 +...+ wnxn = wTx $$

<br>

<p> Before training the weight of algorithm, initialize the first weight to random minimal number. </p>
<p> The reason why the first weight is not initialized into zero is because the learning rate of algorithm η(eta) can only affect classification process when the first weight is not zero. </p>
<p> If the first weight is initialized to zero, the parameter η would only affect scalar value, not the vector direction of the weight </p>
<p> This feature is implemented in the Perceptron class to choose a random small number from a normal distribution with standard deviation of 0.01 </p>
<p> Then, calcuate the predicted values from each train data xi, and update individual weight by multiplying learning rate with difference between predicted values and actual values. </p>

{% highlight ruby %}
{% raw %}
self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
for xi, target in zip(X, y):
   update = self.eta * (target - self.predict(xi))
{% endraw %}
{% endhighlight %}

<br>

<p> Perceptron Algorithm does not recalculate the predicted values before updating all weights of the data. Take sample x as input and connect it with weight w to calculate a predicted error. Then updated the weight using the predicted error and repeat the process to obtain the optimum weight. Therefore, we can simply the Perceptron Algorithm learning process to the diagram listed below </p>

![8]({{ "/assets/img/perceptron_algo.png" | relative_url }})

<br>

<p> After training Perceptron Algorithm, the algorithm can determine weight for each sample self.w_[1], and the intercept of the equation self.w_[0]. Now, to complete the classification process, define the decision function φ(z) by a linear combination of input value x and corresponding weight value w. If the net input is greater than the predifined threshold θ, predict class 1. Otherwise, predict class -1/ </p>

{% highlight ruby %}
{% raw %}
np.dot(X, self.w_[1:]) + self.w_[0]
np.where(self.net_input(X) >= 0.0, 1, -1)
{% endraw %}
{% endhighlight %}


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
