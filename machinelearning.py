# 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
df=pd.read_csv("all_cars.csv")

### 사이킷런은 파이썬에서 머신러닝 분석을 할 때 유용하게 사용할 수 있는 라이브러리 입니다. 여러가지 머신러닝 모듈로 구성되어있습니다.
# StandardScaler을 진행하기 위한 라이브러리 불러오기
from sklearn.preprocessing import StandardScaler
#sklearn: 머신러닝 알고리즘 zip
#sklearn의 preprocessing(전처리)의 StandardScaler

# feature standardization
#범주형 변수 빼고 진행
x_2_cols=['Mile','Ages','Service_repairs','fuel_capacity','dirver_leg_room','dirver_head_room',
         'curb_weight','dimension_L','dimension_W','dimension_H','mile_per_gallon_city',
         'engine_cly','torque']

scaler = StandardScaler() #scaler object 만들기 : 평균 0, 분산 1
df[x_2_cols] = scaler.fit_transform(df[x_2_cols]) #_를 총해 한번에 fit=run=compile
print(df[x_2_cols].head()) # 결과의 일부(앞부분) 확인

X_2 = df[x_2_cols]
y = df['Price']

# 선형회귀 모델의 학습/예측/평가
# RMSE 계산을 위한 함수
def get_rmse(model):
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test , pred)
    rmse = np.sqrt(mse)
    print('{0} 로그 변환된 RMSE: {1}'.format(model.__class__.__name__,np.round(rmse, 3)))
    return rmse

# RMSE 계산을 위한 함수
def get_rmses(models):
    rmses = [ ]
    for model in models:
        rmse = get_rmse(model)
        rmses.append(rmse)
    return rmses

# 모델링 진행에 필요한 라이브러리 불러오기
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

y_target_log = np.log1p(y)

# train set과 test set 나누기 (8:2)
X_train, X_test, y_train, y_test = train_test_split(X_2, y_target_log, test_size=0.2, random_state=156)

# LinearRegression 학습
lr_reg = LinearRegression().fit(X_train, y_train)

# Ridge 학습
ridge_reg = Ridge().fit(X_train, y_train)

# Lasso 학습
lasso_reg = Lasso().fit(X_train, y_train)

# RMSE : 평균 제곱근 오차
models = [lr_reg, ridge_reg, lasso_reg]
print(get_rmses(models))

# 모델의 R² (결정 계수) 값
print('LinearRegression train score :',lr_reg.score(X_train, y_train))  # training set
print('LinearRegression test score :',lr_reg.score(X_test, y_test))

print('Ridge train score :',ridge_reg.score(X_train, y_train))  # training set
print('Ridge test score :', ridge_reg.score(X_test, y_test))

print('Lasso train score',lasso_reg.score(X_train, y_train))  # training set
print('Lasso test score', lasso_reg.score(X_test, y_test))


# 교차 검증을 위한 라이브러리 불러오기기
from sklearn.model_selection import cross_val_score

#5개의 교차 검증 폴드 세트
def get_avg_rmse_cv(models):
    for model in models:
        # 분할하지 않고 전체 데이터로 cross_val_score( ) 수행. 모델별 CV RMSE값과 평균 RMSE 출력
        rmse_list = np.sqrt(-cross_val_score(model,X_2, y_target_log,
                                             scoring="neg_mean_squared_error", cv = 5))
        rmse_avg = np.mean(rmse_list)
        print('\n{0} CV RMSE 값 리스트: {1}'.format( model.__class__.__name__, np.round(rmse_list, 3)))
        print('{0} CV 평균 RMSE 값: {1}'.format( model.__class__.__name__, np.round(rmse_avg, 3)))

# 앞 예제에서 학습한 lr_reg, ridge_reg, lasso_reg 모델의 CV RMSE값 출력           
models = [lr_reg, ridge_reg, lasso_reg]
get_avg_rmse_cv(models)


# 하이퍼파라미터 튜닝을 위한 라이브러리 불러오기기
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

# 최적의 파라미터 찾기
ridge_params = { 'alpha':[0.05, 0.1, 1, 5, 8, 10, 12, 15, 20] }
lasso_params = { 'alpha':[0.001, 0.005, 0.008, 0.05, 0.03, 0.1, 0.5, 1,5, 10] }
best_rige = print_best_params(ridge_reg, ridge_params)
best_lasso = print_best_params(lasso_reg, lasso_params)

# coef 영향이 큰 이상치 제거거
def get_top_bottom_coef(model):
    # coef_ 속성을 기반으로 Series 객체를 생성. index는 컬럼명. 
    coef = pd.Series(model.coef_, index=X_2.columns)
    
    # + 상위 10개 , - 하위 10개 coefficient 추출하여 반환.
    coef_high = coef.sort_values(ascending=False).head(10)
    coef_low = coef.sort_values(ascending=False).tail(10)
    return coef_high, coef_low


def visualize_coefficient(models):
    # 3개 회귀 모델의 시각화를 위해 3개의 컬럼을 가지는 subplot 생성
    fig, axs = plt.subplots(figsize=(24,10),nrows=1, ncols=3)
    fig.tight_layout() 
    # 입력인자로 받은 list객체인 models에서 차례로 model을 추출하여 회귀 계수 시각화. 
    for i_num, model in enumerate(models):
        # 상위 10개, 하위 10개 회귀 계수를 구하고, 이를 판다스 concat으로 결합. 
        coef_high, coef_low = get_top_bottom_coef(model)
        coef_concat = pd.concat( [coef_high , coef_low] )
        # 순차적으로 ax subplot에 barchar로 표현. 한 화면에 표현하기 위해 tick label 위치와 font 크기 조정. 
        axs[i_num].set_title(model.__class__.__name__+' Coeffiecents', size=25)
        axs[i_num].tick_params(axis="y",direction="in", pad=-120)
        for label in (axs[i_num].get_xticklabels() + axs[i_num].get_yticklabels()):
            label.set_fontsize(22)
        sns.barplot(x=coef_concat.values, y=coef_concat.index , ax=axs[i_num])


# 앞의 최적화 alpha값으로 학습데이터로 학습, 테스트 데이터로 예측 및 평가 수행
# print_best_params 함수의 return 값 바탕으로 파라미터를 찾아서 수정 필요
lr_reg = LinearRegression().fit(X_train, y_train)
ridge_reg = Ridge(alpha=20).fit(X_train, y_train)
lasso_reg = Lasso(alpha=0.005).fit(X_train, y_train)

# 모든 모델의 RMSE 출력
models = [lr_reg, ridge_reg, lasso_reg]
get_rmses(models)

# 모든 모델의 회귀 계수 시각화 
models = [lr_reg, ridge_reg, lasso_reg]
visualize_coefficient(models)

# LinearRegression 학습 점수 확인
print('LinearRegression train score',lr_reg.score(X_train, y_train))  # training set
print('LinearRegression test score', lr_reg.score(X_test, y_test))

# Ridge 학습 점수 확인
print('Ridge train score', ridge_reg.score(X_train, y_train))  # training set
print('Ridge test score', ridge_reg.score(X_test, y_test))

# Lasso 학습 점수 확인
print('Lasso train score', lasso_reg.score(X_train, y_train))  # training set
print('Lasso test score', lasso_reg.score(X_test, y_test))


# 왜곡 정도 확인하기 위한 라이브러리 불러오기
from scipy.stats import skew

# object가 아닌 숫자형 피처의 칼럼 index 객체 추출.
features_index = df.dtypes[df.dtypes != 'object'].index
# house_df에 칼럼 index를 [ ]로 입력하면 해당하는 칼럼 데이터 세트 반환. apply lambda로 skew( ) 호출
skew_features = df[features_index].apply(lambda x : skew(x))
# skew(왜곡) 정도가 1 이상인 칼럼만 추출.
skew_features_top = skew_features[skew_features > 1]
print(skew_features_top.sort_values(ascending=False))

df[skew_features_top.index] = np.log1p(df[skew_features_top.index])

# train set과 test set 나누기 (8:2)
X_train, X_test, y_train, y_test = train_test_split(X_2, y_target_log, test_size=0.2, random_state=156)

ridge_params = { 'alpha':[0.05, 0.1, 1, 5, 8, 10, 12, 15, 20] }
lasso_params = { 'alpha':[0.001, 0.005, 0.008, 0.05, 0.03, 0.1, 0.5, 1,5, 10] }
best_rige = print_best_params(ridge_reg, ridge_params)
best_lasso = print_best_params(lasso_reg, lasso_params)

# 앞의 최적화 alpha값으로 학습데이터로 학습, 테스트 데이터로 예측 및 평가 수행
# print_best_params 함수의 return 값 바탕으로 파라미터를 찾아서 수정 필요
# LinearRegression 학습
lr_reg = LinearRegression().fit(X_train, y_train)

# Ridge 학습
ridge_reg = Ridge(alpha=20).fit(X_train, y_train)

# Lasso 학습
lasso_reg = Lasso(alpha=0.005).fit(X_train, y_train)

# 모든 모델의 RMSE 출력
models = [lr_reg, ridge_reg, lasso_reg]
get_rmses(models)

# 모든 모델의 회귀 계수 시각화 
models = [lr_reg, ridge_reg, lasso_reg]
visualize_coefficient(models)


# 회귀트리 모델링을 진행하기 위한 라이브러리 불러오기
from xgboost import XGBRegressor

xgb_params = {'n_estimators':[1000]}
xgb_reg = XGBRegressor(n_estimators=1000, learning_rate=0.05,
                       colsample_bytree=0.5, subsample=0.8)
best_xgb = print_best_params(xgb_reg, xgb_params)


# 앙상블 모델링을 위한 라이브러리 불러오기
from lightgbm import LGBMRegressor

lgbm_params = {'n_estimators':[1000]}
lgbm_reg = LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=4,
                         subsample=0.6, colsample_bytree=0.4, reg_lambda=10, n_jobs=-1)
best_lgbm = print_best_params(lgbm_reg, lgbm_params)

# RMSE 계산을 위한 함수
def get_rmse_pred(preds):
    for key in preds.keys():
        pred_value = preds[key]
        mse = mean_squared_error(y_test , pred_value)
        rmse = np.sqrt(mse)
        print('{0} 모델의 RMSE: {1}'.format(key, rmse))

# 개별 모델의 학습
# alpha 파라미트 수정
ridge_reg = Ridge(alpha=8).fit(X_train, y_train)
lasso_reg = Lasso(alpha=0.001).fit(X_train, y_train)

# 개별 모델 예측
ridge_pred = ridge_reg.predict(X_test)
lasso_pred = lasso_reg.predict(X_test)

# 개별 모델 예측값 혼합으로 최종 예측값 도출
pred = 0.4 * ridge_pred + 0.6 * lasso_pred
preds = {'최종 혼합': pred,
         'Ridge': ridge_pred,
         'Lasso': lasso_pred}

#최종 혼합 모델, 개별모델의 RMSE 값 출력
get_rmse_pred(preds)

xgb_reg = XGBRegressor(n_estimators=1000, learning_rate=0.05,
                       colsample_bytree=0.5, subsample=0.8)
lgbm_reg = LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=4,
                         subsample=0.6, colsample_bytree=0.4, reg_lambda=10, n_jobs=-1)
# 학습 시키기
xgb_reg.fit(X_train, y_train)
lgbm_reg.fit(X_train, y_train)

# 예측값 할당
xgb_pred = xgb_reg.predict(X_test)
lgbm_pred = lgbm_reg.predict(X_test)

pred = 0.5 * xgb_pred + 0.5 * lgbm_pred
preds = {'최종 혼합': pred,
         'XGBM': xgb_pred,
         'LGBM': lgbm_pred}

get_rmse_pred(preds)

# XGBRegressor 학습 점수 확인
print('XGBRegressor train score :',xgb_reg.score(X_train, y_train))  # training set
print('XGBRegressor test score :', xgb_reg.score(X_test, y_test))

# LGBMRegressor 학습 점수 확인
print('LGBMRegressor train score :',lgbm_reg.score(X_train, y_train))  # training set
print('LGBMRegressor test score :',lgbm_reg.score(X_test, y_test))

