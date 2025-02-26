# 1. Introduction

<p align="center"><img src="https://github.com/user-attachments/assets/6d578dc8-b639-4ed9-8559-cd96ee6fd766" width="35%" height="35%"></p>

- Sequential Recommendation은 user의 과거 행동 시퀀스(예를 들어, 제품 구매 이력)을 기반으로 미래 행동(다음에 어떤 제품을 구매 할지)을 예측하는 것을 목표로 함
- 기존의 Markov Chains 및 RNN 기반 모델은 일정 수준의 성능을 보였지만, user의 구매 이력 내에서 복잡한 관계를 완전히 capture하지 못함
- CosRec은 2D CNN 기반의 모델을 제안하며, 이는 item sequence를 3D tensor로 변환하고, 2D Convolution filter를 사용하여 local feature를 학습하며, Feed-forward 방식으로 high-order interaction을 aggregation함
- 두 개의 public dataset (MovieLens 1M, Gowalla)으로 실험을 수행하며, 기존 모델에 비해 좋은 성능을 달성함

## 1.1 Architecture

<p align="center"><img src="https://github.com/user-attachments/assets/b54917ed-e90d-47e7-ba51-2675a91e4c29" width="70%" height="70%"></p>

### 1.1.1 Embedding Look-up Layer
- item과 user를 각각 embedding matrix로 변환하여 feature 추출 수행

### 1.1.2 Pairwise Encoding
- 기존의 CNN 기반 추천 모델은 sequence를 단순히 item vector로 변환하여 사용
- CosRec에서는 pairwise interactions을 활용하여 item을 3차원 tensor 형태로 변환
- 이런 방법을 통해 비 연속적인 item간의 관계도 학습이 가능해짐 (예를 들어, 사진 관련 제품을 연속적으로 구매하지만 중간에 다른 제품이 끼어드는 경우도 고려 가능)
  
### 1.1.3 2D Convolution Module
- CosRec의 핵심 모듈로써, 2D filter를 사용하여 더 깊고 복잡한 item relation을 학습할 수 있음
- Convolution network는 아래와 같은 구조로 설계됨:
  - 두 개의 convolution block을 사용
  - 첫 번째 layer는 $1 \times 1$ kernel을 사용하여 feature를 확장
  - 두 번째 layer는 $3 \times 3$ kernel을 사용하여 복잡한 관계를 학습

## 1.2 모델 학습

<p align="center"><img src="https://github.com/user-attachments/assets/bfe90d6e-420f-483d-ae0d-941d6659dea4" width="40%" height="40%"></p>

- 목적 함수로 binary cross-entropy를 사용
- Adam Optimizer를 사용하여 네트워크 최적화
- 각 training sample마다 3개의 negative sample을 랜덤하게 추출하여 사용

## 1.3 기존 CNN 기반 모델과의 비교
- 기존 CNN 기반 추천 모델(Caser 등)의 한계를 극복
  - 비연속적인 item간의 관계 고려 가능 (skip behavior 반영)
  - 더 깊은 neural network 구조 지원 가능 (단순 가중합이 아닌 다양한 패턴 학습 가능)

# 2. Dataset Preparation
- 이 저장소에서 활용되는 데이터셋은 아래와 같음
  - MovieLens 1M
  - Gowalla
- 다운로드 링크 : https://github.com/zzxslp/CosRec/tree/master/data

- 다운로드 완료 후 data 폴더에 위치

# 3. Train
- 데이터 다운로드 후 train.py를 이용하여 학습 수행
- args에 대한 자세한 내용은 train.py 참조

```bash
python train.py --[args]
```

# 4. Evaluate
- 학습 완료 후, 각 모델에 대한 testset performance 산출 가능
- precision@1, precision@5, precision@10, recall@1, recall@5, recall@10을 metric로 사용
- evaluate.py를 사용하며, args에 대한 자세한 내용은 코드 참조

```bash
python evaluate.py --[args]
```

# 5. 학습 결과

## Learning Curve

## Testset Performance
|Dataset|모델|Precision@1|Precision@5|Precision@10|Recall@1|Recall@5|Recall10|MAP|
|------|---|---|---|---|---|---|---|---|
|MovieLens 1M|CosRec CNN|0.3288|0.2800|0.2501|0.0206|0.0835|0.1557|0.1898|
|MovieLens 1M|CosRec MLP|0.3097|0.2625|0.2317|0.0195|0.0783|0.1332|0.1758|
|Gowalla|CosRec CNN|0.000|0.000|0.000|0.000|0.000|0.000|0.000|
|Gowalla|CosRec MLP|0.000|0.000|0.000|0.000|0.000|0.000|0.000|
