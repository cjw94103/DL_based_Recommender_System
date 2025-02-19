# 1. Introduction
NARM은 Session-based Recommendation을 개선하기 위한 아키텍처를 제안합니다. 기존의 Session-based Recommendation method들은 user 행동의 순차적인 특성만 고려하였으나, user의 main purpose을 명확히 반영하지 못하는 한계가 있었습니다.
이 문제를 해결하기 위해 NARM은 attention mechanism을 활용한 하이브리드 인코더를 도입하여 아래와 같은 모델링을 수행합니다.

- GRU 기반의 Global Encoder를 통해 user의 sequential behavior을 모델링하고
- GRU 기반의 Local Encoder를 통해 user의 main purpose를 capture하여 추천의 정확도를 높임

이를 통해 다양한 benchmark dataset에서 기존의 SOTA 대비 높은 성능을 달성했음을 실험적으로 검증하였습니다.

## 1.1 Global Encoder

<p align="center"><img src="https://github.com/user-attachments/assets/ba31f62c-9597-4623-98ab-e1ff7649e0d6" width="50%" height="50%"></p>

global encoder의 입력인 이전의 모든 클릭이며, 출력은 현재 session에서 user의 sequential behavior feature입니다.

- user의 sequential behavior를 모델링하는 역할 수행
- GRU를 기반으로 session 내 item의 click 순서를 학습
- 최종 GRU hidden state를 $c_g$로 정의하여 user의 behavior pattern을 요약하는 벡터로 사용함
  
## 1.2 Local Encoder

<p align="center"><img src="https://github.com/user-attachments/assets/01f78adf-517d-4f1f-ac42-f07bdcbe4551" width="50%" height="50%"></p>

local encoder 역시 입력은 이전의 모든 클릭이며, 출력은 모든 hidden state의 vector를 활용하여 attention weight를 계산합니다. 여기서 Global, Local encoder는 2개가 아닌 하나의 GRU에서 파생됩니다.

- user의 main purpose를 capture하는 역할 수행
- attention mechanism을 통해 session 내 중요도가 높은 item에 가중치를 부여함
- 최종 목적 벡터 $c_l$은 중요한 item들의 weighted sum으로 계산됨

<p align="center"><img src="https://github.com/user-attachments/assets/e9444619-9f51-4258-b2c9-4200ced59ef2" width="15%" height="15%"></p>

- 여기서 $\alpha_{tj}$는 item $j$가 얼마나 중요한지를 결정하는 attention score를 의미함

## 1.3 Bi-linear Matching based Decoder
- item과 session representation간의 similarity를 계산하여 추천 점수를 생성
- Bi-linear Similarity Function을 사용하여 아이템 임베딩과 세션 벡터를 비교

<p align="center"><img src="https://github.com/user-attachments/assets/d025b40c-fdfe-4c61-832d-26816e35add0" width="15%" height="15%"></p>

- 여기서, $B$는 학습 가능한 weight matrix를 의미함

# 2. Dataset Preparation
이 저장소에서 활용되는 dataset은 아래와 같습니다.

- Diginetica
  - 다운로드 링크 : https://drive.google.com/drive/folders/0B7XZSACQf0KdXzZFS21DblRxQ3c?resourcekey=0-3k4O5YlwnZf0cNeTZ5Y_Uw&usp=sharing
- Yoochoose 1/64
  - Yoochoose dataset의 item sequence를 최신순으로 정렬하고 원래 데이터 양의 1/64만큼 샘플링
  - 다운로드 링크 : https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015
- Yoochoose 1/4
  - Yoochoose dataset의 item sequence를 최신순으로 정렬하고 원래 데이터 양의 1/4만큼 샘플링
  - 다운로드 링크 : https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015

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
학습이 완료되면 evaluate.py 코드를 이용하여 각 모델에 대한 testset performance를 측정할 수 있습니다. recall, mrr score를 보실수 있습니다. 아래와 같은 명령어를 실행하여 주세요. args에 대한 자세한 내용은 evaluate.py를 참고하세요.
```bash
python evaluate.py --[args]
```

# 5. 학습 결과

## Learning Curve

<p align="center"><img src="https://github.com/user-attachments/assets/52dc21d5-85dc-4404-b727-a3f36b0932ec" width="60%" height="60%"></p>

위 그림은 각 dataset에 대한 NARM의 epoch별 recall@20 및 mrr@20 score의 변화 추이를 보여줍니다. Diginetica의 경우 논문의 성능보다 조금 낮은 성능을 보이지만, Yoochoose dataset의 경우 논문의 명시된 만큼의 성능을 보여줍니다.
모든 dataset에 대하여 validation score가 모델이 점차 학습되어 점차 안정적으로 상승하는 것을 보여줍니다.

## Testset Performance
|Dataset|Recall@20|MRR@20|
|------|---|---|
|Diginetica|0.5319|0.1849|
|Yoochoose 1/64|0.6909|0.2959|
|Yoochoose 1/4|0.7051|0.3016|
