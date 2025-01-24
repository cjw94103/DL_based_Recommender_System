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
NCF는 일반화된 MF(GMF)와 MLP를 결합한 NeuMF(Neural Matrix Factorization) 모델을 제안합니다. 
