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

{% include aligner.html images="2020-09-24-ARIMA-model_10_1.png" %}

{% include aligner.html images="2020-09-24-ARIMA-model_11_2.png, 2020-09-24-ARIMA-model_11_3.png" column = 2 %}


{% include aligner.html images="2020-09-24-ARIMA-model_12_1.png" %}

<p> "2020년도 1월 1일부터 2월 12일 까지의 따릉이 대여이력을 중심으로 구축" </p>

{% highlight ruby %}
{% raw %}
{% bike = pd.read_csv('공공자전거 대여이력 정보_2020.01_056.csv', sep = ',')" %}
{% print(bike.shape) -> (1048575, 11) %}
{% print(bike.shape) -> (1048575, 11) %}
{% endraw %}
{% endhighlight %}

{
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>자전거번호</th>\n",
       "      <th>대여일시</th>\n",
       "      <th>대여 대여소번호</th>\n",
       "      <th>대여 대여소명</th>\n",
       "      <th>대여거치대</th>\n",
       "      <th>반납일시</th>\n",
       "      <th>반납대여소번호</th>\n",
       "      <th>반납대여소명</th>\n",
       "      <th>반납거치대</th>\n",
       "      <th>이용시간</th>\n",
       "      <th>이용거리</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>SPB-04061</td>\n",
       "      <td>2020-01-01 0:01</td>\n",
       "      <td>429</td>\n",
       "      <td>송도병원</td>\n",
       "      <td>2</td>\n",
       "      <td>2020-01-01 0:04</td>\n",
       "      <td>372</td>\n",
       "      <td>약수역 3번출구 뒤</td>\n",
       "      <td>8</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>SPB-06686</td>\n",
       "      <td>2020-01-01 0:02</td>\n",
       "      <td>1637</td>\n",
       "      <td>KT 노원점 건물 앞</td>\n",
       "      <td>14</td>\n",
       "      <td>2020-01-01 0:04</td>\n",
       "      <td>1656</td>\n",
       "      <td>중앙하이츠 아파트 입구</td>\n",
       "      <td>9</td>\n",
       "      <td>1</td>\n",
       "      <td>350</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>SPB-15937</td>\n",
       "      <td>2020-01-01 0:01</td>\n",
       "      <td>1924</td>\n",
       "      <td>삼부르네상스파크빌</td>\n",
       "      <td>10</td>\n",
       "      <td>2020-01-01 0:05</td>\n",
       "      <td>1955</td>\n",
       "      <td>디지털입구 교차로</td>\n",
       "      <td>7</td>\n",
       "      <td>4</td>\n",
       "      <td>800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>SPB-14805</td>\n",
       "      <td>2020-01-01 0:03</td>\n",
       "      <td>437</td>\n",
       "      <td>대흥역 1번출구</td>\n",
       "      <td>1</td>\n",
       "      <td>2020-01-01 0:05</td>\n",
       "      <td>126</td>\n",
       "      <td>서강대 후문 옆</td>\n",
       "      <td>18</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>SPB-09038</td>\n",
       "      <td>2020-01-01 0:01</td>\n",
       "      <td>1168</td>\n",
       "      <td>마곡엠밸리10단지 앞</td>\n",
       "      <td>5</td>\n",
       "      <td>2020-01-01 0:05</td>\n",
       "      <td>1152</td>\n",
       "      <td>마곡역교차로</td>\n",
       "      <td>2</td>\n",
       "      <td>4</td>\n",
       "      <td>660</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       자전거번호             대여일시  대여 대여소번호      대여 대여소명  대여거치대             반납일시  \\\n",
       "0  SPB-04061  2020-01-01 0:01       429         송도병원      2  2020-01-01 0:04   \n",
       "1  SPB-06686  2020-01-01 0:02      1637  KT 노원점 건물 앞     14  2020-01-01 0:04   \n",
       "2  SPB-15937  2020-01-01 0:01      1924    삼부르네상스파크빌     10  2020-01-01 0:05   \n",
       "3  SPB-14805  2020-01-01 0:03       437     대흥역 1번출구      1  2020-01-01 0:05   \n",
       "4  SPB-09038  2020-01-01 0:01      1168  마곡엠밸리10단지 앞      5  2020-01-01 0:05   \n",
       "\n",
       "   반납대여소번호        반납대여소명  반납거치대  이용시간  이용거리  \n",
       "0      372    약수역 3번출구 뒤      8     2     0  \n",
       "1     1656  중앙하이츠 아파트 입구      9     1   350  \n",
       "2     1955     디지털입구 교차로      7     4   800  \n",
       "3      126      서강대 후문 옆     18     2     0  \n",
       "4     1152        마곡역교차로      2     4   660  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bike.head(5)"
   ]
  }
  
