# 1. Introduction

<p align="center"><img src="https://github.com/user-attachments/assets/c3cb94e6-7e24-478b-b095-9cce5f4c8f19" width="50%" height="50%"></p>

Neural Collaborative Filtering은 2017년에 발표된 논문으로, 딥러닝을 활용하여 추천 시스템의 핵심 문제인 Collaborative Filtering(CF)을 개선하고자 제안된 모델입니다.
기존의 Matrix Factorization(MF) 기법은 user와 item의 latent factor를 내적(inner-product)하여 선형적으로 interaction을 모델링하지만, 이는 복잡한 user-item relation을 충분히 capture하지 못하는 한계가 있습니다. NCF는 MF과 Multi-Layer Perceptron(MLP)을 결합하여 leave-one-out evlaution에서 MovieLens, Pinterst 데이터셋에 대하여 좋은 성능을 산출하였습니다.

NCF의 주요 특징은 아래와 같습니다.

### General Framework
MF는 NCF의 특수한 경우로 해석될 수 있습니다. input layer의 user 또는 item의 ID에 대한 one-hot encoding으로 인해 얻어진 embedding vector는 user(item)의 latent vector로 볼 수 있습니다. user latent vector를 $\textbf P^T \textbf v_u^U$로, item latent vector를 $\textbf Q^T \textbf v_i^I$로 하면 첫 번째 neural CF layer의 mapping 함수를 아래와 같이 정의할 수 있습니다.

<p align="center"><img src="https://github.com/user-attachments/assets/985839ee-60cf-44dc-8c0d-7a9ac4e2caf7" width="20%" height="20%"></p>

여기서, circle operation은 element-wise multiplication을 의미하며, 그런 다음 벡터를 output layer에 projection을 수행합니다.

<p align="center"><img src="https://github.com/user-attachments/assets/a79f8b88-2fb4-4ebe-a76d-4d9d1451ce57" width="20%" height="20%"></p>

여기서, $a_{out}$과 $\textbf h$는 각각 output layer의 activation과 edge weight를 의미합니다. $a_{out}$을 identity function으로, $\textbf h$를 1을 원소로 가진 uniform vector로 강제하면 정확히 MF 모델을 복원할 수 있습니다. 따라서 MF는 NCF의 특수한 경우로 해석될 수 있습니다.

### MLP
NCF의 핵심은 MLP을 활용하여 user와 item간의 non-linear interaction을 효율적으로 학습하는 것입니다. MLP은 여러 개의 hidden layer로 구성되며, 각 layer는 ReLU activation을 이용하여 non-linearity를 부여합니다. 이러한 구조를 통해 모델은 user와 item간의 복잡한 non-linear relationship을 효과적으로 모델링할 수 있습니다.

### NeuMF
NCF는 일반화된 MF(GMF)와 MLP를 결합한 NeuMF(Neural Matrix Factorization) 모델을 제안합니다. GMF는 MF의 구조를 일반화하여 표현하며, MLP는 non-linear interaction을 학습합니다. NeuMF는 이 두 모델의 출력을 결합하여 최종 예측을 수행합니다.
이를 통해 MF, MLP을 모두 활용하여 user-item interaction을 더욱 정교하게 모델링할 수 있습니다.

### Implicit Feedback 기반의 학습
NCF는 explicit feedback (예 : star ratings) 대신 implicit feedback (예 : click, view 등)를 활용하여 모델을 학습합니다. implicit feedback은 user의 명시적 선호도를 나타내지 않지만, user의 행동을 통해 선호도를 추정할 수 있습니다. 이를 위해 NCF는 Binary Cross-Entropy Loss를 사용하며, 모델의 출력은 sigmoid를 활용한 [0, 1] 범위의 확률로 해석합니다. 또한, 학습 과정에서 Negative Sampling을 통해 unobservable item을 negative feedback으로 간주하여 학습을 수행합니다.

# 2. Dataset Preparation
이 저장소의 데이터셋은 MovieLens 1M을 활용합니다. 데이터셋의 다운로드 링크는 아래와 같습니다.

- https://www.kaggle.com/datasets/odedgolden/movielens-1m-dataset

다운로드 완료 후 모델 학습을 위하여 데이터 전처리를 수행해야 합니다. data_preprocessing.py를 참고하여 데이터 전처리를 수행하여 주세요.

# 3. Train
### 1st stage : Pretrain
Neumf는 MF(GMF)와 MLP를 결합한 아키텍처를 사용합니다. Neumf 학습 전 GMF와 MLP에 대하여 Pretraining을 수행해야합니다. 아래와 같은 명령어를 실행하여 pretraining을 수행하여 주세요. args에 대한 자세한 내용은 pretrain_*.py를 참고하세요
```bash
python pretrain_gmf.py --[args]
python pretrain_mlp.py --[args]
```
### 2nd stage : Neumf Train
Pretrain 완료 후 pretrained GMF 및 MLP의 모델 파라미터를 Neumf 모델의 초기화로 사용합니다. weight 로드 후 Neumf 학습을 수행합니다. 아래와 같은 명령어를 실행하여 학습을 수행하여 주세요. args에 대한 자세한 내용은 train_neumf.py를 참고해주세요. 
```bash
python train_neumf.py --[args]
```

# 4. Evaluate
학습이 완료되면 evaluate.py 코드를 이용하여 각 모델에 대한 testset performance를 측정할 수 있습니다. hit_ratio, ndcg score를 보실수 있습니다. 아래와 같은 명령어를 실행하여 주세요. args에 대한 자세한 내용은 evaluate.py를 참고하세요.
```bash
python evaluate.py --[args]
```

# 5. 학습 결과
### Learning Curve

<p align="center"><img src="https://github.com/user-attachments/assets/2826e179-67d9-410f-bd26-51bddabbf86a" width="50%" height="50%"></p>

위 그래프는 validation set에 대한 각 epoch별 모델의 HR@10, NDCG@10 Score의 변화를 보여줍니다. Neumf 모델은 좋은 초기화를 제공하여 학습 초기부터 성능이 높고 4개의 모델 중 가장 좋은 성능을 보입니다.

### Testset Score

|모델|HR@10|NDCG@10|
|------|---|---|
|gmf|0.6471|0.3767|
|mlp|0.6628|0.3896|
|neumf (not pretrained)|0.6777|0.4072|
|neumf|0.6779|0.4082|
