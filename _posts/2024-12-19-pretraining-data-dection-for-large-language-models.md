---
layout: post
title: "Pretraining Data Detection for Large Language Models: A Divergence-based Calibration Method 설명"
date: 2024-12-18 00:00:00 +0900
description:
categories: [paper, llm]
tags: [paper, llm]
giscus_comments: true
related_posts: true
---

# Introduction  

- 많은 model 개발자들은 LLM을 개발할 때 사용한 corpus를 비공개로 처리한다.  
    - 저작권, 윤리적 문제가 존재하기 때문이다.  
- 저자는 black-box LLM과 text가 주어졌을 때 해당 text가 training data에 포함되어있는지 확인하는 방법론을 제시한다.  

## 아이디어  

- Divergence-from-randomness에서 영감을 받는다.  
- ***특정 단어가 한 문서 내에서 사용되는 빈도(Within-document term-frequency)***와 ***그 단어가 전체 문서 컬렉션 내에서 사용되는 빈도(frequency of a word within the collection)***가 얼마나 차이가 나는지를 측정하면, 그 단어가 해당 문서에서 얼마나 중요한 정보를 담고 있는지 알 수 있다.  
    - Within-document term-frequency  
        - LLM predicted token의 probability  
        - Token probability distribution  
    - Frequency of a word within the collection  
        - Corpus에서 해당 token의 빈도수  
        - Token frequency distribution  
- Token probability distribution와 Token frequency distribution의 divergence가 높으면 해당 text가 모델의 training corpus에 있다는 의미다.  

## 방법론  

1. **Within-document term-frequency**  
    - 특정 text에서 probability distribution을 계산한다.  
2. **Frequency within the collection**  
    - 전체 corpus에서 해당 token이 평균적으로 얼마나 자주 등장하는지를 나타낸다.  
3. **Divergence**  
    - Within-document term-frequency와 Frequency within the collection을 비교한다.  

# Problem Statement  

- Text $$x$$, LLM $$\mathcal{M}$$, 정보가 없는 pretraining corpus $$D$$, pretraining data detection task $$A$$에 대해서  
    - $$\mathcal{A}(x,\mathcal{M})\rightarrow\{0,1\}$$  

<p align="center"><img src="/assets/post/image/2024-12-19-pretraining-data-dection-for-large-language-models/image.png" width="80%"></p>

1. Token probability distribution computation  
    - $$\mathcal{M}$$에 텍스트 $$x$$를 query하여 각 token probability를 계산한다.  
2. Token probability distribution computation  
    - 접근 가능한 대규모 참조 말뭉치 $$\mathcal{D}^\prime$$를 사용하여 토큰 빈도를 추정한다.  
3. Score calculation via *comparison*  
    - 두 분포를 비교하여 각 token의 probability을 calibration하고, calibration된 probability를 기반으로 pretraining data인지 판별하는 점수를 계산한다.  
4. Binary decision  
    - Score에 threshold를 적용하여 $$x$$가 모델 $$\mathcal{M}$$의 pretraining corpus에 있는지 예측한다.  

## 3.2 Token Probability Distribution Computation  

- $$x_0$$: start-of-sentence token  

$$x^\prime=x_0x_1x_2...x_n$$  

- $$\mathcal{M}$$에 $$x$$를 query한다.  

$$\{p(x_i|x_{\lt i};\mathcal{M}):0\lt i \le n \}$$  

## 3.3 Frequency of a word within the collection  

- 다음의 term으로 계산한다.  

$$p(x_i,\mathcal{D}^\prime)=\frac{\text{count}(x_i)}{N^\prime}$$  

- $$x_i$$가 없는 경우를 위해 Laplace smoothing을 추가한다. $$|V|$$는 vocabulary size다.  

$$p(x_i;D^\prime)=\frac{\text{count}(x_i)+1}{N^\prime+|V|}$$  

## 3.4 Score Calculation through Compression  

- $$p(x_i;\mathcal{M})$$과 $$p(x_i;D^\prime)$$의 cross-entropy를 계산한다.  

$$\alpha_i = -p(x_i; \mathcal{M}) \cdot \log p(x_i; D^\prime).$$  

- 특정 token이 우세한 영향을 미치지 않도록 upper bound를 정의한다.  

$$  
\begin{equation}  
\alpha_i =  
\begin{cases}  
\alpha_i, & \text{if } \alpha_i < a \\  
a, & \text{if } \alpha_i \geq a  
\end{cases}  
\end{equation}  
$$  

- Text $$x$$에 대해서 token $$x_i$$가 여러 개 존재할 수 있다.  
    - 이럴 때는 첫 번째 토큰의 결과를 가져온다.  

$$\beta=\frac{1}{|\text{FOS}(x)|}\sum_{x_j \in \text{FOS(x)}}\alpha_j$$  

<p align="center"><img src="/assets/post/image/2024-12-19-pretraining-data-dection-for-large-language-models/image%201.png" width="80%"></p>

## 3.5 Binary Decision  

- Threshold로 pretraining corpus $$D$$에 있는지 결정  

$$  
\text{Decision}(x, \mathcal{M}) =  
\begin{cases}  
0 \quad (x \notin \mathcal{D}), & \text{if } \beta < \tau, \\  
1 \quad (x \in \mathcal{D}), & \text{if } \beta \geq \tau.  
\end{cases}  
$$  

<p align="center"><img src="/assets/post/image/2024-12-19-pretraining-data-dection-for-large-language-models/image%202.png" width="80%"></p>
