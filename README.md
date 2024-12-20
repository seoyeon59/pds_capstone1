# 중고차 가격 예측 프로젝트_캡스톤 연계 실습 (10조)

### 🚗프로젝트 소개
중고차의 기본 정보와 추가적인 이력 정보를 포함하여 중고차 가격 예측에 적합한 요인들을 선정하여 중고차 거래에서 발생하는 정보 비대칭성 문제를 완화하고 합리적인 중고차 가격 결정 요인 분석의 방향성을 제시하고자 프로젝트를 진행하였다. 사이트에서 차종 별 200대씩 총 6종의 차종(SUV, 세단, 트럭, 스포츠카, 쿠페, 크로스오버)을 수집하여 하나의 데이터 테이블을 만들어 독립변수와 종속변수의 상관관계 분석 및 모델링을 진행하였다.

- 중고차 기본 정보 : 연비, 엔진, 차종, 중량, 연식, 연료, 토크, 치수, 주행 거리
- 중고차 이력 정보 : 서비스 및 수리 이력, 보험 이력, 사고 이력
- 종속변수 : 중고차 가격


### ⚙️개발 환경
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> 

### 🙋‍♀️10조 멤버
김지윤 (서울여대 수학과 2020110798)

전서연 (서울여대 데이터사이언스학과 2023111810)

전민경 (서울여대 데이터사이언스학과 2022111159)


### 💻발전 내용
기존에는 heatmap을 이용하여 상관관계만 나타내었다. 이에 더 발전하여 상관계수가 높은 것끼리 산점도로 나타냈으며, 여러 모델링(선형회귀모델(LineerRegression,Ridge,Lasso), 회귀모델 성능향상을 위한 앙상블(XGBRegressor, LGBMRegressor))을 진행하였다.

---
### clawler 요약 설명
크롤링 사이트 : Carmax <https://www.carmax.com/cars>

이 사이트는 미국의 유명 중고차 사이트로 현재는 사이트에서 미국 이외의 지역에서 접속하지 못하게 막았다.

현재 resource에 업로드 된 크로링 한 데이터는 2023년 11월 ~ 12월에 실시된 자료이다. 
크롤링 중 코드도 중간 수정을 하여 csv 파일들이 약간 다를 수 있다.

---
### processing 요약 설명
#### preprocessing_code_1 : 첫 번째 단계 (자동차 기본 정보에 대한 전처리)
1. 우선 필요없는 열([horsepower', 'front_tire_size', 'prior_use', 'keys')을 삭제 시킨다. 삭제시킨 열의 기준은 행에 결측치가 많거나 값이 모두 같은 등 중고차 가격에 영향을 많이 미치지 않을 열들이다.
2. 한 열에 2개 이상의 정보를 가지고 있어 나누어 새로운 열을 만들었으며, 나눈 기존 열들은 삭제시켜 준다.
3. toque 열의 경우 단위가 통일되어 있지 않으므로 단위를 통일시키기 위해 divide_columns라는 함수를 정의하여 사용한다.
4. 1차 전처리를 했음을 나타내기 위해 전처리 한 파일을 '차종_prepro_1.csv' 파일로 저장한다.

#### preprocessing_code_2 : 두 번째 단계 (자동차 이력 정보에 대한 전처리 + 값들을 수치형으로 변경)
1. 1차 전처리를 완료한 csv 파일들을 들고오는데 인덱스가 있는 파일들만 line 6 을 실행한다.(suv, coupes 차종만 실행을 하지 않음)
2. 이력 정보 뿐만 아니라 수치로 값을 알 수 있는 데이터를 숫자만 뽑아 수치형 데이터로 데이터형 변환을 한다.
3. 2차 전처리를 했음을 나타내기 위해 전처리 한 파일을 '차종_prepro_2.csv' 파일로 저장한다.

#### preprocessing_code_3 : 세 번째 단계 (이전 단계까지 전처리를 완료한 파일들을 힙침)
1. csv 파일에 프로젝트와 필요없는 컬럼을 제거하고 각 파일에 차종 열을 추가해 준 후 모든 파일들을 합쳐 'all_cars.csv' 파일로 저장한다.

---
### resouse 요약 설명
crawling_resource는 크롤링한 파일로 차종별로 차이가 있으니 확인하여야한다.

preprocessing_resource는 전처리를 단계별로 한 파일들이 다 존재한다.
최종 파일을 'all_cars.csv' 이다.

---
### visualization_code 요약 설명
1. 수치형 변수들만 이용하여 상관관계 수치를 heatmap으로 이용하여 확인한다.
2. 가격과 상관계수가 가장 큰 것들을 따로 시각화한다.
3. 독립변수들 중에서 상관계수가 큰 것들끼리의 분포를 확인한다.

---
### machinlearning 요약 설명
#### <모델 및 주요 함수 설명>
#### LinearRegression
- LinearRgression은 어떠한 독립 변수들과 종속 변수간의 관계를 예측할 때, 그 사이 관계를 선형 관계(1차 함수)로 가정하고, 모델링하는 지도 학습 알고리즘이다.
- LinearRgression은 보통, 인자와 결과 간의 대략적인 관계 해석이나 예측에 활용된다.

<image src='https://github.com/user-attachments/assets/f126daa1-7257-4a22-8a82-ea60a83232ab'>

- 다중 회귀 모형은 과적합(overfitting)되는 경향이 존재한다.

#### Ridgi와 Lasso
<image src='https://github.com/user-attachments/assets/e49cee02-d785-4d86-a6f9-e40925eac050'>
  
- lasso와 ridge는 선형 회귀의 단점을 보완해 범용성을 제공한다.

  
#### 하이퍼파라미터 튜닝

```python
# 라이브러리 불러오기
from sklearn.model_selection import GridSearchCV

# 하이퍼파라미터 튜닝
def print_best_params(model, params):
    grid_model = GridSearchCV(model, param_grid=params, 
                              scoring='neg_mean_squared_error', cv=5)
    grid_model.fit(X_2, y_target_log)
    rmse = np.sqrt(-1* grid_model.best_score_)
    print('{0} 5 CV 시 최적 평균 RMSE 값: {1}, 최적 alpha:{2}'.format(model.__class__.__name__,
                                        np.round(rmse, 4), grid_model.best_params_))
    return grid_model.best_estimator_
```


#### XGBRegressor
- 여러 개의 결정 트리를 임의적으로 학습하는 부스팅 계열의 트리 모델이다.
- 약한 분류기를 세트로 묶어서 정확도를 예측하는 기법으로 Greedy Algorithm을 사용하여 분류기를 발견하고 분산처리를 사용하여 빠른 속도로 적합한 비중 파라미터를 찾는 알고리즘이다.
- 병렬 처리를 사용하기에 학습과 분류가 빠르고 다른 알고리즘과 연계하여 앙상블 학습이 가능하다.
- Y = w1 * M(x)+ w2 * G(x)+ w3 * H(x) + error

```python
xgb_params = {'n_estimators':[1000]}
xgb_reg = XGBRegressor(n_estimators=1000, learning_rate=0.05,
                       colsample_bytree=0.5, subsample=0.8)
best_xgb = print_best_params(xgb_reg, xgb_params)
```



#### LGBMRegressor
- 데이터에 가중치를 부여하여 모델을 학습시키는 모델을 학습시키는 부스팅 계열의 트리 모델이다.
- 과적합 규제 기능(Regularization), 자체 교차 검증 알고리즘, 결측치 처리 기능의 이점이 있다.
- 리프 중심 트리 분할 방식으로 비대칭적인 트리를 형성하여 모델을 학습하고, 예측 오류 손실을 최소화 한다.

```python
lgbm_params = {'n_estimators':[1000]}
lgbm_reg = LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=4,
                         subsample=0.6, colsample_bytree=0.4, reg_lambda=10, n_jobs=-1)
best_lgbm = print_best_params(lgbm_reg, lgbm_params)
```



#### <추가 설명>
#### RMSE : 평균 제곱근 오차
- 값이 작을수록 모델의 예측이 실제값에 더 가깝다는 것을 의미한다.
- RMSE는 오차를 제곱하므로 큰 오차에 더 큰 페널티를 부여한다.


#### R² : 결정 계수
- R² = 1 :  완벽한 예측.
- R² = 0 :  모델이 실제값의 평균만큼도 예측하지 못함.
- R² < 0 : 모델이 실제값의 평균보다 예측값이 나쁨.
- Overfitting 확인: 훈련 데이터 R² 값이 높고, 테스트 데이터 R² 값이 낮으면 모델이 과적합(overfitting) 되었을 가능성이 있다.
- Underfitting 확인: 두 값 모두 낮으면 모델이 과소적합(underfitting) 되었을 가능성이 있다.


