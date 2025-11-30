# Numpy 공부 내용 summary

---

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

---

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

---

## 3. N차원 배열 연산 (Array Operations)

반복문 없이 데이터 전체를 한 번에 계산하는 **벡터 연산(Vectorization)**과, 딥러닝 레이어 연산의 핵심인 **브로드캐스팅 & 행렬 곱** 정복.

### 1️⃣ 요소별 연산 vs 행렬 곱 (Element-wise vs Dot Product)
가장 많이 하는 실수! `*`는 행렬 곱이 아닙니다.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[1, 0], [0, 1]])

# 1. 요소별 연산 (Element-wise)
# 같은 위치끼리만 곱함. 크기(Shape)가 같거나 브로드캐스팅 가능해야 함.
print(A * B) 

# 2. 행렬 곱 (Matrix Multiplication) - ⭐딥러닝 필수
# 앞 행렬의 열과 뒤 행렬의 행 개수가 같아야 함. ((N, M) @ (M, K) -> (N, K))
print(A @ B)      # Python 3.5+ 권장 문법
print(np.dot(A, B)) # 같은 기능
```

### 2️⃣ 브로드캐스팅 (Broadcasting)
크기(Shape)가 다른 배열끼리 연산할 때, **작은 쪽을 자동으로 확장(Stretch)**해주는 기능.
* **규칙:** 뒤에서부터 차원을 비교했을 때, 축의 크기가 **같거나 1이어야** 함.

```python
data = np.array([[1, 2, 3], [4, 5, 6]]) # (2, 3)
bias = np.array([10, 20, 30])           # (3,) -> (1, 3)으로 자동 확장

# (2, 3) + (1, 3) => 각 행마다 bias가 더해짐
result = data + bias 
```

### 3️⃣ 집계 함수와 축 (Aggregation & Axis)
데이터를 압축(Reduction)할 때 사용. 방향 헷갈림 주의!

```python
arr = np.array([[1, 2, 3], 
                [4, 5, 6]]) # (2, 3)

# axis=None (기본값): 전체 합
print(arr.sum()) # 21

# axis=0 (행을 따라가며 = 세로 방향 압축) -> 열별 통계
print(arr.sum(axis=0)) # [5, 7, 9] -> 결과 shape: (3,)

# axis=1 (열을 따라가며 = 가로 방향 압축) -> 행별(샘플별) 통계
print(arr.sum(axis=1)) # [6, 15] -> 결과 shape: (2,)
```

### 🔄 [심화] 그림으로 보는 행렬 곱의 흐름 (Shape 맞추기)

딥러닝 선형 레이어(`Y = XW + b`)의 차원 변화 이해하기.
**`3`은 중간에서 연결만 해주고 사라지는 것**이 핵심입니다!

**[상황]**
학생 **5**명이, 과목 **3**개를 쳐서, 적성 **2**개를 알아냄.

$$
\begin{bmatrix} 
\text{학생1} \\ \text{학생2} \\ \text{학생3} \\ \text{학생4} \\ \text{학생5} 
\end{bmatrix}
\times
\begin{bmatrix} \text{국어} & \text{영어} & \text{수학} \end{bmatrix}
\quad
\xrightarrow{\text{변환}}
\quad
\begin{bmatrix} \text{문과} & \text{이과} \end{bmatrix}
$$

**[행렬 크기 변화]**

$$
(\mathbf{5} \times \mathbf{3}) \quad @ \quad (\mathbf{3} \times \mathbf{2}) \quad \rightarrow \quad (\mathbf{5} \times \mathbf{2})
$$

1. 앞의 **5 (학생 수)**는 결과까지 그대로 갑니다. (학생이 5명이면 결과도 5명분이어야 하니까요)
2. 가운데 **3 (과목 수)**은 **서로 만나서 계산되고 사라집니다.** (국/영/수 점수는 적성 점수로 변환되어 흡수됨)
3. 뒤의 **2 (적성 종류)**가 결과의 새로운 정보가 됩니다.

---

## 4. N차원 배열 정렬 (Array Sorting)

데이터를 순서대로 나열하거나(Sort), 딥러닝 모델의 예측 결과에서 **상위 확률 클래스(Top-k)**를 추출할 때 사용하는 핵심 기능.

### 1️⃣ 값 정렬 (Sort)
데이터의 **'값(Value)'** 자체를 정렬할 때 사용.

```python
import numpy as np
arr = np.array([3, 1, 9, 5])

# 1. np.sort(): 원본 유지 (복사본 반환) -> 권장 ✅
sorted_arr = np.sort(arr)  # [1, 3, 5, 9]

# 2. arr.sort(): 원본 변경 (In-place) -> 주의 ⚠️
arr.sort() 
```

### 2️⃣ 인덱스 정렬 (ArgSort) - ⭐딥러닝 필수 (Top-k)
값이 아니라, **"값이 있는 위치(인덱스)"**를 정렬된 순서대로 반환.
> **활용:** "확률이 가장 높은 클래스는 몇 번인가?" (`argmax`, `top-k`)

```python
probs = np.array([0.1, 0.7, 0.2]) # 1번 클래스가 0.7로 1등

# 값(0.1...)이 아니라 인덱스(0, 1, 2)를 정렬해서 반환
indices = np.argsort(probs) 
print(indices) # [0, 2, 1] (0.1 < 0.2 < 0.7 순서)

# 가장 확률 높은 클래스 (ArgMax와 동일)
print(indices[-1]) # 1
```

### 3️⃣ 🚨 [주의] 내림차순 정렬 (Descending Sort)
NumPy는 `ascending=False` 옵션이 없음. **슬라이싱(`[::-1]`)**으로 직접 뒤집어야 함.

#### **(1) 1차원 배열**
단순히 전체를 뒤집으면 됨.
```python
arr = np.array([1, 2, 3])
print(np.sort(arr)[::-1]) # [3, 2, 1]
```

#### **(2) 2차원 배열 (⭐실수 주의)**
행(데이터 순서)은 유지하고, **열(값의 순서)**만 뒤집어야 함!

```python
matrix = np.array([[1, 2, 3], 
                   [4, 5, 6]])

# ❌ 틀린 예: 행(Row) 자체를 거꾸로 뒤집음 (데이터 섞임)
# matrix[::-1] -> [[4, 5, 6], [1, 2, 3]] 

# ✅ 옳은 예: [:, ::-1] (모든 행에 대해, 열만 거꾸로)
desc_matrix = np.sort(matrix, axis=1)[:, ::-1]
# 결과: [[3, 2, 1], 
#        [6, 5, 4]]
```

### 4️⃣ 실전 예제: Top-3 클래스 뽑기
모델 예측값(`pred`)에서 확률이 높은 상위 3개 클래스 인덱스 추출하기.

```python
# (데이터 5개, 클래스 10개)
pred = np.random.rand(5, 10) 

# 1. argsort로 인덱스 정렬 (오름차순)
# 2. [:, ::-1]로 내림차순 뒤집기 (확률 높은 순)
# 3. [:, :3]로 상위 3개 자르기
top3_indices = np.argsort(pred, axis=1)[:, ::-1][:, :3]
```
---

## 5. N차원 배열 형태 변경 (Reshaping)

딥러닝 모델의 입력 규격(Shape)에 맞춰 데이터를 변환하는 전처리 핵심 스킬.

### 1️⃣ 형태 변경 (Reshape)
데이터의 개수(Size)는 유지한 채 차원 구조만 변경.
> **핵심 꿀팁 (`-1`):** 남은 차원을 자동으로 계산. 배치 사이즈가 가변적일 때 필수 사용.

```python
import numpy as np
arr = np.arange(12)

# 1. (3, 4)로 변경
print(arr.reshape(3, 4))

# 2. -1 활용 (자동 계산)
# 전체 12개 중 행을 3으로 고정 -> 열은 자동으로 4가 됨
print(arr.reshape(3, -1)) 
```

### 2️⃣ 차원 교환 (Transpose) - ⭐CV 데이터 처리 필수
축(Axis)의 순서를 바꿀 때 사용. 특히 이미지 데이터 포맷 변환에 필수적임.
* **OpenCV/Matplotlib:** `(H, W, C)` (높이, 너비, 채널)
* **PyTorch:** `(C, H, W)` (채널, 높이, 너비)

```python
# (Height, Width, Channel) -> (Channel, Height, Width)
# 0번축, 1번축, 2번축 -> 2번, 0번, 1번 순서로 재배치
img_pytorch = img.transpose(2, 0, 1)
```

### 3️⃣ 차원 추가 및 제거 (Expand_dims & Squeeze)
모델 입력(Batch)을 위해 가짜 차원을 추가하거나 제거할 때 사용.

```python
arr = np.zeros((28, 28)) # (H, W)

# 1. 차원 추가 (Batch Dimension)
# (28, 28) -> (1, 28, 28)
arr_expanded = np.expand_dims(arr, axis=0)

# 2. 차원 제거 (불필요한 1차원 삭제)
# (1, 28, 28) -> (28, 28)
arr_squeezed = np.squeeze(arr_expanded)
```

### 4️⃣ 평탄화 (Flatten vs Ravel)
다차원 배열을 1차원 벡터로 펼칠 때 사용 (CNN Feature Map -> FC Layer).
* `flatten()`: 복사본 생성 (메모리 사용)
* `ravel()`: 원본 참조 (메모리 절약, 빠름) -> **권장**
