# 20250709-titanic-mlflow

Titanic 생존 예측 모델을 MLflow를 사용하여 학습하고 관리하는 프로젝트입니다.

## 📋 프로젝트 개요

이 프로젝트는 Kaggle의 Titanic 데이터셋을 사용하여 승객 생존을 예측하는 머신러닝 모델을 MLflow를 통해 학습하고 관리합니다.

### 주요 기능
- 데이터 전처리 및 특성 공학
- 로지스틱 회귀 모델 학습
- K-Fold 교차 검증
- MLflow를 통한 실험 추적 및 모델 관리
- 모델 성능 메트릭 로깅

## 🚀 핸즈온 가이드

### 1. 환경 설정

#### 1.1 Conda 환경 생성 및 활성화

```bash
# Conda 환경 생성
conda env create -f environment.yml

# 환경 활성화
conda activate titanic_mlflow
```

#### 1.2 필요한 패키지 확인

환경이 제대로 설정되었는지 확인:

```bash
# Python 버전 확인
python --version

# MLflow 설치 확인
python -c "import mlflow; print(mlflow.__version__)"
```

### 2. MLflow 서버 시작

#### 2.1 MLflow 서버 실행 (포트 5050)

```bash
# MLflow 서버 실행 (포트 5050)
mlflow server --port 5050
```

서버가 성공적으로 시작되면 다음과 같은 메시지가 표시됩니다:
```
[2024-01-XX XX:XX:XX] INFO: Starting MLflow server on port 5050
[2024-01-XX XX:XX:XX] INFO: MLflow server is running on http://localhost:5050
```

#### 2.2 MLflow UI 접속

웹 브라우저에서 다음 URL로 접속:
```
http://localhost:5050
```

MLflow UI가 정상적으로 로드되면 실험 목록이 표시됩니다.

### 3. 모델 학습 및 실험 실행

#### 3.1 train_main_with_mlflow.py 스크립트 소개

`train_main_with_mlflow.py`는 Titanic 생존 예측 모델을 학습하는 메인 스크립트입니다.

**주요 기능:**
- **데이터 전처리**: 결측치 처리, 인코딩, 특성 공학
- **모델 학습**: 로지스틱 회귀 모델 학습
- **성능 평가**: 검증 데이터셋과 K-Fold 교차 검증
- **MLflow 로깅**: 실험 파라미터, 메트릭, 모델 아티팩트 저장
- **예측 생성**: 테스트 데이터에 대한 예측 결과 생성

**전처리 과정:**
1. Age, Fare 결측치를 중앙값으로 대체
2. Embarked 결측치를 최빈값으로 대체
3. Sex를 숫자로 인코딩 (male: 0, female: 1)
4. Embarked를 원-핫 인코딩
5. FamilySize 특성 생성 (SibSp + Parch + 1)
6. Age와 Fare를 구간화하여 범주형 변수로 변환
7. 불필요한 컬럼 제거 (Name, Ticket, Cabin 등)

#### 3.2 학습 스크립트 실행

새로운 터미널 창을 열고 다음 명령어를 실행합니다:

```bash
# 모델 학습 실행
python train_main_with_mlflow.py
```

#### 3.3 학습 과정 모니터링

실행 중 다음과 같은 로그를 확인할 수 있습니다:

```
INFO:__main__:MLflow 실험 시작 (간소화 버전)
INFO:__main__:훈련 데이터셋 크기: (712, 7)
INFO:__main__:검증 데이터셋 크기: (179, 7)
INFO:__main__:Accuracy (Validation): 0.8268
INFO:__main__:AUC (Validation): 0.8476
INFO:__main__:K-Fold Cross-Validation 시작...
INFO:__main__:K-Fold Cross-Validation 완료.
INFO:__main__:모델 학습 및 MLflow 로깅 완료 (간소화 버전).
INFO:__main__:모델 레지스트리에 'Titanic_Simple_Logistic_Model' 이름으로 모델 등록.
INFO:__main__:제출 파일 'submission_main.csv' 생성 완료.
```

학습이 완료되면 `submission_main.csv` 파일이 생성됩니다.

### 4. MLflow UI에서 결과 확인

MLflow UI를 통해 실험 결과를 시각적으로 확인할 수 있습니다.

#### 4.1 MLflow UI 접속 및 기본 화면

1. **웹 브라우저에서 접속**: `http://localhost:5050`
2. **기본 화면 구성**:
   - 상단: Experiments, Models, Model Registry 탭
   - 중앙: 실험 목록 또는 선택된 실험의 런 목록
   - 좌측: 필터 및 정렬 옵션

#### 4.2 실험 목록 확인

1. **Experiments 탭 클릭**
2. **"Titanic_Simple_Logistic_Regression_Training" 실험 찾기**
3. **실험 클릭**하여 해당 실험의 런 목록으로 이동

#### 4.3 실험 런 상세 정보 확인

실험을 클릭하면 다음과 같은 정보들을 확인할 수 있습니다:

**📊 Parameters (파라미터)**
- `solver`: liblinear (최적화 알고리즘)
- `random_state`: 42 (랜덤 시드)
- `test_split_ratio`: 0.2 (검증 데이터 비율)
- `stratify_by_target`: True (층화 샘플링 사용)

**📈 Metrics (성능 지표)**
- `accuracy`: 검증 정확도 (~82-85%)
- `precision`: 정밀도 (양성 예측의 정확도)
- `recall`: 재현율 (실제 양성 중 예측된 비율)
- `f1_score`: F1 점수 (정밀도와 재현율의 조화평균)
- `roc_auc`: ROC AUC 점수 (~84-87%)
- `kfold_mean_*`: K-Fold 교차 검증 평균 메트릭
- `kfold_std_*`: K-Fold 교차 검증 표준편차

**📁 Artifacts (아티팩트)**
- `model/`: 학습된 모델 파일 (다운로드 가능)
- `simple_preprocessor_params.json`: 전처리 파라미터 (재사용 가능)

#### 4.4 모델 레지스트리 확인

1. **Models 탭 클릭**
2. **"Titanic_Simple_Logistic_Model" 모델 찾기**
3. **모델 클릭**하여 버전 정보 확인:
   - 모델 버전 번호
   - 스테이징 상태 (None, Staging, Production)
   - 생성 시간 및 설명

#### 4.5 실험 비교하기

여러 실험을 비교하려면:
1. **비교할 런들을 체크박스로 선택**
2. **"Compare" 버튼 클릭**
3. **파라미터와 메트릭을 나란히 비교**하여 성능 차이 확인

#### 4.6 모델 다운로드

학습된 모델을 다운로드하려면:
1. **실험 런 클릭**
2. **Artifacts 섹션에서 "model" 폴더 클릭**
3. **모델 파일 다운로드**하여 다른 환경에서 사용 가능

### 5. 생성된 파일들

학습 완료 후 다음 파일들이 생성됩니다:

- `submission_main.csv`: Kaggle 제출용 예측 결과
- `mlruns/`: MLflow 실험 데이터
- `mlartifacts/`: MLflow 아티팩트 저장소
- `mlflow.db`: MLflow 메타데이터 데이터베이스

### 6. 추가 분석 및 실험

#### 6.1 다른 하이퍼파라미터로 실험
`train_main_with_mlflow.py` 파일을 수정하여 다양한 하이퍼파라미터로 실험할 수 있습니다:

```python
# 예시: 다른 solver 사용
model = LogisticRegression(solver="saga", random_state=42)

# 예시: 다른 C 값 사용
model = LogisticRegression(solver="liblinear", C=0.1, random_state=42)
```

#### 6.2 새로운 특성 추가
`preprocess_data` 함수를 수정하여 새로운 특성을 추가할 수 있습니다:

```python
# 예시: 새로운 특성 추가
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["Title"] = df["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
```

## 📊 모델 성능

현재 로지스틱 회귀 모델의 예상 성능:
- **Accuracy**: ~82-85%
- **AUC**: ~84-87%
- **F1 Score**: ~80-83%

## 🔧 문제 해결

### MLflow 서버 연결 문제
```bash
# 포트가 이미 사용 중인 경우
lsof -ti:5050 | xargs kill -9

# 또는 다른 포트 사용
mlflow server --host 0.0.0.0 --port 5051 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```

### Conda 환경 문제
```bash
# 환경 재생성
conda env remove -n titanic_mlflow
conda env create -f environment.yml
```

### 패키지 설치 문제
```bash
# pip로 직접 설치
pip install mlflow pandas scikit-learn matplotlib seaborn jupyter
```

## 📝 참고 자료

- [MLflow 공식 문서](https://mlflow.org/docs/latest/index.html)
- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [Scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.