# 1. Introduction
SASRec은 Sequential Recommendation을 위한 Self-Attention 기반의 모델입니다. 이 모델은 user의 behavior 데이터를 분석하여 다음 item을 예측하는 task를 해결합니다. 이 모델의 특징은 RNN 없이 sequence를 모델링하며, 적응적으로 과거의 행동 중
중요한 행동에 가중치를 부여하여 예측을 수행합니다. RNN처럼 장기적인 패턴을 학습할 수 있으며, MC(Markov Chain)처럼 최근 행동을 효율적으로 반영할 수 있는 구조입니다.

## 1.1 SASRec Architecture

<p align="center"><img src="https://github.com/user-attachments/assets/c685c3c3-5d70-451f-a930-447f97a97150" width="40%" height="40%"></p>


SASRec은 아래의 3개의 주요 컴포넌트로 구성됩니다.

- Embedding Layer
- Self-Attention Block
- Prediction Layer

### 1.1.1 Embedding Layer
user의 과거 행동 데이터 (즉, item 클릭 여부)를 고정된 길이의 벡터로 변환하여 self-attention module이 처리할 수 있도록 만듦니다. 

먼저, Item Embedding은 각 item $i$를 $d$-차원 벡터로 변환하여 Embedding Matrix $M (|I| \times d)$를 생성하고 user의 behavior data $S_u$(item sequence)를 item embedding vector로 매핑하여 입력 데이터로 사용합니다.

두 번째로, Self-Attention은 순서를 고려하지 않는 모델이므로, sequence의 시퀀스의 순서 정보를 유지하기 위해 Positional Embedding 추가합니다. 기존 Transformer에서 사용된 sin/cosine 기반의 Positional Embedding 대신, trainable position embedding $P(n \times d)$를 사용합니다.

마지막으로, 최종 입력 벡터는 $\hat E = M + P$의 형태가 되며, 여기서 $M$은 item embedding, $P$는 positional embedding 입니다.

### 1.1.2 Self-Attention Layer
SASRec의 핵심 컴포넌트로, user의 행동 데이터 중 중요한 아이템에 대한 attention weight를 반영하는 역할을 수행합니다.

기본적인 Self-Attention 연산은 아래와 같이 정의됩니다.

<p align="center"><img src="https://github.com/user-attachments/assets/dcdd487c-8d4b-439c-939d-919f4a20521d" width="40%" height="40%"></p>

여기서, $Q, \ K, \ V$는 입력 벡터를 선형 변환한 행렬을 의미하며, softmax 연산을 통해 각 item간의 중요도를 계산하고, 가장 중요한 item에 높은 가중치를 부여합니다. $\sqrt d$로 나누는 이유는 큰 값이 softmax 연산에서 gradient explosion을 일으키는 것을 방지하기 위함입니다.

SASRec은 미래 정보가 포함되면 안되는 sequence 정보를 처리하므로, 미래의 item을 참조하는 것을 방지하는 masking을 적용해야 합니다. 즉, 현재 시점 $t$에서 미래 시점인 $t+1, \ t+2, ...$의 정보가 보이지 않도록 softmax 연산 전에 masking 처리를 수행합니다.

또한, Self-Attention의 결과는 여전히 선형 변환이므로, 비선형성을 추가하기 위해 두 개의 Fully Connected Layer (FFN) 적용을 적용합니다.

<p align="center"><img src="https://github.com/user-attachments/assets/4ca0631a-70b1-4b81-a812-92cd7240704f" width="35%" height="35%"></p>

마지막으로, Self-Attention Blocks을 여러개 쌓아 심층적인 representation을 학습할 수 있도록 하며, Residual Connection + Layer Normalization + Dropout의 조합을 사용하여 학습의 안정성을 높이고 overfitting을 방지합니다.

### 1.1.3 Prediction Layer
Self-Attention 블록을 통과한 최종 벡터 $F_t$를 사용하여 다음 item을 예측합니다. Matrix Factorization 방식으로 추천 점수 계산합니다.

<p align="center"><img src="https://github.com/user-attachments/assets/30c60718-8ff4-4ff7-a595-b84dc4a4be17" width="10%" height="10%"></p>

$M_i$는 shared item embedding layer이며 $r_{i,t}$의 값이 클수록 해당 item이 추천될 확률이 높음을 의미합니다. user의 sequence에 없는 랜덤한 item에 대하여 negative sampling을 수행하고 Binary Cross-Entropy Loss를 통해 모델을 최적화합니다.

### 1.1.4 SASRec의 학습 방식
SASRec은 Negative Sampling을 활용한 Binary Cross-Entropy Loss를 사용하여 학습됩니다. user의 item sequence 데이터에서 다음 item을 예측하는 방식으로 학습하며 각 user의 sequence에 없는 item을 negative sample로 추가하여 추천 성능을 극대화합니다.

<p align="center"><img src="https://github.com/user-attachments/assets/bcd03115-7cf4-4c3c-9ab4-45fc8dc4cff8" width="45%" height="45%"></p>

# 2. Dataset Preparation
이 저장소에서 활용하는 dataset은 아래와 같습니다.

- Amazon Games
- Amazon Beauty
- Steam
- MovieLens 1M

다운로드 링크는 아래와 같습니다.

- https://github.com/pmixer/SASRec.pytorch/tree/master/data

다운로드 완료 후 모델 학습을 위하여 데이터 전처리를 수행해야 합니다. data_preprocessing.py 코드를 이용하여 데이터 전처리를 수행해주세요.

```bash
python data_preprocessing.py --[args]
```

# 3. Train
데이터 전처리 완료 후 모델 학습을 수행해야 합니다. train.py를 참고하여 학습을 수행하여주세요. args에 대한 자세한 내용은 train.py를 참고해주세요.

```bash
python train.py --[args]
```

# 4. Evaluate
학습이 완료되면 evaluate.py 코드를 이용하여 각 모델에 대한 testset performance를 측정할 수 있습니다. precision, recall, hit_ratio, ndcg score를 보실수 있습니다. 아래와 같은 명령어를 실행하여 주세요. args에 대한 자세한 내용은 evaluate.py를 참고하세요.
```bash
python evaluate.py --[args]
```

# 5. 학습 결과

## Learning Curve

<p align="center"><img src="https://github.com/user-attachments/assets/f8737aad-3040-4568-8a3c-66d5024556b7" width="60%" height="60%"></p>

위 그림은 4개의 dataset에 대하여 SASRec의 epoch 별 HR@10 및 NDCG@10의 평균 성능을 보여줍니다. Amazon Beauty dataset에 대해서는 약간의 불안정한 learning curve를 보이지만, 나머지 3개의 dataset에 대해서는 모델이 점차 학습되어 평균 성능이 점차 상승하는 것을 보여줍니다.

## Testset Performance
|Dataset|HR@10|NDCG@10|
|------|---|---|
|Games|0.7269|0.5171|
|Beauty|0.3795|0.2223|
|Steam|0.8589|0.6045|
|MovieLens 1M|0.8190|0.5860|
