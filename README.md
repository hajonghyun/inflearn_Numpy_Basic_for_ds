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

## 2. N차원 배열 인덱싱 (Array Indexing)

데이터의 특정 부분을 선택(Selection), 필터링(Filtering), 재배열(Shuffling)하는 다양한 인덱싱 기법 정리.

### 1️⃣ 기본 인덱싱 & 슬라이싱 (Basic & Slicing)
리스트와 달리 `[행, 열]` 표기법을 지원하며, 이를 통해 **열(Column)** 단위 추출이 가능함.

```python
import numpy as np
arr = np.array([[1, 2, 3], [4, 5, 6]])

# 1. 요소 접근 (행, 열)
print(arr[0, 1])   # 2 (0행 1열) -> arr[0][1]보다 권장됨

# 2. 슬라이싱 (Slicing)
print(arr[0, :])   # [1, 2, 3] (0행 전체)
print(arr[:, 1])   # [2, 5]    (모든 행의 1열 -> 열 추출 ⭐)
print(arr[:2, 1:]) # 0~1행, 1열~끝열 부분 추출
```

### 2️⃣ 팬시 인덱싱 (Fancy Indexing)
인덱스로 **리스트(List)**를 전달하여, 특정 순서대로 데이터를 추출하거나 섞을 때 사용.

```python
matrix = np.array([[10, 10], [20, 20], [30, 30]])

# 1. 원하는 행만 콕 집어 가져오기
print(matrix[[0, 2]]) 
# 결과: [[10, 10], [30, 30]] (0번, 2번 행 추출)

# 2. 순서 섞기 (Shuffling)
print(matrix[[2, 1, 0]]) 
# 결과: 2번 -> 1번 -> 0번 행 순서로 재배열
```

### 3️⃣ 불리언 인덱싱 (Boolean Indexing) - ⭐데이터 필터링 필수
조건문(True/False)을 인덱스로 사용하여, 조건에 맞는 데이터만 추출.

```python
data = np.array([1, 2, 3, 4, 5])

# 1. 짝수만 추출 (Filtering)
evens = data[data % 2 == 0] # [2, 4]

# 2. 특정 기준 이상인 값만 살리기
high_val = data[data > 3]   # [4, 5]
```
