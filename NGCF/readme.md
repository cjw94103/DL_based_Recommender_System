# 1. Introduction
기존의 추천 시스템의 핵심은 user의 선호도를 정확히 예측하는 것이며, 이를 위해 대부분의 model-based collaborative filtering(CF) 기법은 user 및 item을 vector embedding을 변환한 후, 이를 기반으로 추천을 수행합니다.
하지만 기존의 방법들은 user-item간의 collaborative signal을 명시적으로 인코딩하지 않아 최적의 embedding을 학습하는데 한계가 있습니다.


NGCF는 user-item interaction을 graph structure로 모델링하여, high-order connectivity을 활용한 embedding propagation을 수행합니다. 이 방법은 collaborative signal을 명확하게 embedding 과정에 반영함으로써 보다 정교한 user 및 item representation을
학습할 수 있도록 합니다.

논문에서는 NGCF가 기존의 SOTA 모델(HOP-Rec, Collaborative Memory Network 등)보다 우수한 성능을 보이며, 특히 spase dataset에서도 효과적임을 실험적으로 검증합니다.

## 1.1 High-order Connectivity

<p align="center"><img src="https://github.com/user-attachments/assets/6c24ca7f-8d05-4fd6-8158-92758585e18f" width="50%" height="50%"></p>

Figure 1에서 오른쪽 그림은 왼쪽의 user-item interaction graph를 기반으로 user $u_1$의 high-order connectivity를 표현한 것입니다. high-order connectivity는 user-item interaction graph를 기반으로 하기 때문에 노드의 연결 순서는 
user-item-user-item을 반복하게 됩니다. 해당 high-order connectivity의 표현은 각 user 혹은 item에 따라 다르게 구성됩니다.

high-order connectivity는 collabarative signal을 나타내는 두 가지 의미를 포함합니다. 먼저, 오른쪽 그림의 경로 $u_1 ← i_2 ← u_2 ← i_4$는 user $u_1, \ u_2$ 모두 item $i_2$와 interaction했기 때문에 $u_1$과 $u_2$의 행동 유사성을 나타내고
유사한 user $u_2$가 item $i_4$를 채택했기 때문에 user $u_1$ 또한 item $i_4$를 채택할 가능성이 높음을 알 수 있습니다.

오른쪽 그림 $l=3$의 범위에서 item $i_4$와 user $u_1$을 연결하는 경로가 두 개인 반면, item $i_5$와 user $u_1$을 연결하는 경로는 한 개이기 때문에 이는 item $i_4$가 item $i_5$보다 user $u_1$의 관심을 끌 가능성이 높다는 것을 암시합니다.
NGCF는 embedding function 내에서 high-order connectivity을 모델링하는 방법을 제안하고, 이는 이 논문의 main contribution입니다.

## 1.2 Architecture

<p align="center"><img src="https://github.com/user-attachments/assets/4ec7b569-fffb-41e4-82f1-5f341c60f11d" width="50%" height="50%"></p>

NGCF는 Graph Convolution Network(GCN)을 이용하여 user-item간의 collaborative signal을 반영하는 구조를 가집니다. 이를 위해 Embedding layer, Embedding Propagation Layer, Prediction Layer의 세 가지 구성요소를 갖습니다.

### 1.2.1 Embedding Layer
user $u$와 item $i$를 각각 vector embedding $e_u, \ e_i$로 변환합니다. 초기 embedding을 trainable parameter로 설정하며, 기존 CF 기법과 동일한 방식으로 시작됩니다.
하지만 NGCF에서는 이 초기 embedding을 그대로 사용하지 않고, graph structure를 활용한 propagation 과정을 통해 지속적으로 개선합니다.

### 1.2.2 Embedding Propagation Layer
user-item간의 연결 정보를 활용하여 embedding을 단계적으로 업데이트하는 핵심 모듈입니다. 각 user는 본인이 소비한 item의 정보를 받아오며, 반대도 item도 소비한 user들의 정보를 전달받습니다.
propagation 과정은 multi-layer 구조로 쌓을 수 있으며, depth $L$이 커질수록 high-order connectivity를 더욱 많이 반영할 수 있습니다.
각 propagation step에서 user 및 item embedding을 아래와 같이 업데이트 합니다.

<p align="center"><img src="https://github.com/user-attachments/assets/fe6785c0-d9e1-4916-80ec-6021ca2152a6" width="40%" height="40%"></p>

여기서 $W_1, \ W_2$는 trainable weight이며, ⊙는 element-wise product을 의미합니다.

### 1.2.3 Prediction Layer
최종적인 user-item interaction score는 모든 propagation stage의 출력을 concatenate하여 계산됩니다.

<p align="center"><img src="https://github.com/user-attachments/assets/fb963eae-c8da-4d2b-8c35-d60b308b8287" width="30%" height="30%"></p>

즉, 여러 단계의 embedding을 concatenate하여 user 및 item의 최종 표현을 만들고, 내적으로 user의 선호도를 예측합니다.

## 1.3 Optimization
모델 학습을 위해 BPR (Bayesian Personalized Ranking) Loss를 사용하여 학습을 수행합니다. 이는 observed user-item interaction이 unobserved user-item interaction보다 높은 score를 가지도록 학습하는 방식입니다.

# 2. Dataset Preparation
이 저장소의 dataset은 Amazon Book dataset을 활용합니다. 다운로드 링크는 아래와 같습니다.

- https://github.com/huangtinglin/NGCF-PyTorch/tree/master/Data

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
