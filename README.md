# Numpy 공부 내용 summary

## 1. N차원 배열 생성 (Array Creation)

리스트보다 빠르고 효율적인 `ndarray` 생성법과, 딥러닝 데이터 생성에 필수적인 난수(Random) 활용법 정리.

### 1️⃣ 기본 생성 & 초기화 (Initialization)
데이터 전처리나 마스킹(Masking) 작업 시 자주 사용되는 초기화 함수들.

```python
import numpy as np

# 1. 기본 생성 (List -> Array)
arr = np.array([1, 2, 3])  # dtype 통일됨 (정수+실수 혼용 시 실수로 변환)

# 2. 초기화 함수
np.zeros((3, 3))       # 0으로 채움
np.ones((2, 5))        # 1로 채움
np.full((2, 2), 7)     # 특정 값(7)으로 채움
np.eye(3)              # 3x3 단위 행렬 (대각선만 1)

# 3. Shape 복사 (매우 유용)
# 기존 배열(arr)과 동일한 shape을 가지되, 값만 0으로 채움
np.zeros_like(arr)
```

### 2️⃣ 수열 생성 (Sequence)
그래프의 x축을 그리거나, 하이퍼파라미터 탐색 시 사용.

```python
# 1. 간격(Step) 기준: 0부터 2씩 증가 (끝값 포함 X)
np.arange(0, 10, 2)    # [0, 2, 4, 6, 8]

# 2. 개수(Num) 기준: 0부터 10까지 딱 5개로 등분 (끝값 포함 O)
np.linspace(0, 10, 5)  # [0., 2.5, 5., 7.5, 10.]

# 3. 로그 스케일: Learning Rate 튜닝 시 사용 (10^-3 ~ 10^-1)
np.logspace(-3, -1, 3) # [0.001, 0.01, 0.1]
```

### 3️⃣ 난수 생성 (Random) - ⭐AI 데이터 생성 필수
가중치 초기화(Weight Initialization)나 노이즈(Noise) 추가 시 필수 사용.

```python
# 1. 균등 분포 (Uniform): 0 ~ 1 사이 실수
np.random.rand(3, 3)

# 2. 표준 정규 분포 (Normal): 평균 0, 표준편차 1
np.random.randn(3, 3)

# ⭐ 응용: 원하는 평균과 표준편차를 가진 난수 만들기
# 공식: (표준편차 * randn) + 평균
# 예: 평균 3, 표준편차 2인 데이터 100개
noise_data = 2 * np.random.randn(100) + 3

# 3. 정수 뽑기: 랜덤 인덱싱 등에 사용
np.random.randint(0, 10, (5, 5)) # 0 이상 10 미만

# 4. 시드 고정 (재현성 확보)
np.random.seed(42)
```
