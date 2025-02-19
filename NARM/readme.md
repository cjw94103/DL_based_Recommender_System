# 1. Introduction
NARM은 Session-based Recommendation을 개선하기 위한 아키텍처를 제안합니다. 기존의 Session-based Recommendation method들은 user 행동의 순차적인 특성만 고려하였으나, user의 main purpose을 명확히 반영하지 못하는 한계가 있었습니다.
이 문제를 해결하기 위해 NARM은 attention mechanism을 활용한 하이브리드 인코더를 도입하여 아래와 같은 모델링을 수행합니다.

- GRU 기반의 Global Encoder를 통해 user의 sequential behavior을 모델링하고
- GRU 기반의 Local Encoder를 통해 user의 main purpose를 capture하여 추천의 정확도를 높임

이를 통해 다양한 benchmark dataset에서 기존의 SOTA 대비 높은 성능을 달성했음을 실험적으로 검증하였습니다.

## 1.1 Global Encoder

<p align="center"><img src="https://github.com/user-attachments/assets/ba31f62c-9597-4623-98ab-e1ff7649e0d6" width="50%" height="50%"></p>

global encoder의 입력인 이전의 모든 클릭이며, 출력은 현재 session에서 user의 sequential behavior feature입니다. 이를 모델링하기 위하여 GRU를 사용하며, 최종 GRU의 hidden vector를 $c_g$로 정의하고 
이 feature는 user의 behavior pattern을 요약하는 벡터로서 사용합니다.

## 1.2 Local Encoder
