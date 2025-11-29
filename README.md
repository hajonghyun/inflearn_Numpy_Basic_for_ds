# inflearn_Numpy_Basic_for_ds


## 1. N차원 배열 생성 (Array Creation)

리스트보다 빠르고 효율적인 `ndarray` 생성법과, 딥러닝 데이터 생성에 필수적인 난수(Random) 활용법 정리.

### 1️⃣ 기본 생성 & 초기화 (Initialization)
데이터 전처리나 마스킹(Masking) 작업 시 자주 사용되는 초기화 함수들.

```python
import numpy as np
```

# 1. 기본 생성 (List -> Array)
```python
arr = np.array([1, 2, 3])  # dtype 통일됨 (정수+실수 혼용 시 실수로 변환)
```
# 2. 초기화 함수
```python
np.zeros((3, 3))       # 0으로 채움
np.ones((2, 5))        # 1로 채움
np.full((2, 2), 7)     # 특정 값(7)으로 채움
np.eye(3)              # 3x3 단위 행렬 (대각선만 1)
```
# 3. Shape 복사 (매우 유용)
# 기존 배열(arr)과 동일한 shape을 가지되, 값만 0으로 채움
```python
np.zeros_like(arr)
```
