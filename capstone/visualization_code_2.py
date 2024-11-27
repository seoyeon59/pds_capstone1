import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("12_4_all.csv")

# 상관 관계 분석
num_2_cols=['Price','Mile','Ages','Service_repairs','fuel_capacity','dirver_leg_room','dirver_head_room',
         'curb_weight','dimension_L','dimension_W','dimension_H','mile_per_gallon_city',
         'engine_cly','torque']

corr2 = df[num_2_cols].corr(method = 'pearson') #상관관꼐 분석위해 pearson이라는 메소드 씀
print(corr2)

#상관관계를 수치로 보면 힘들기 때문에 밑에 heatmap 처럼 확인함!
fig = plt.figure(figsize = (16, 12))
ax = fig.gca()

sns.set(font_scale = 1.5)  # heatmap 안의 font-size 설정
heatmap = sns.heatmap(corr2.values, annot = True, fmt='.2f', annot_kws={'size':15},
                      yticklabels = num_2_cols, xticklabels = num_2_cols, ax=ax, cmap = "RdYlBu")

plt.tight_layout()
plt.show()

x_2_cols=['Mile','Ages','Service_repairs','fuel_capacity','dirver_leg_room','dirver_head_room',
         'curb_weight','dimension_L','dimension_W','dimension_H','mile_per_gallon_city',
         'engine_cly','torque']

### 사이킷런은 파이썬에서 머신러닝 분석을 할 때 유용하게 사용할 수 있는 라이브러리 입니다. 여러가지 머신러닝 모듈로 구성되어있습니다.
from sklearn.preprocessing import StandardScaler
#sklearn: 머신러닝 알고리즘 zip
#sklearn의 preprocessing(전처리)의 StandardScaler

### feature standardization  (numerical_columns except dummy var.-"CHAS")

scaler = StandardScaler()#scaler object 만들기  # 평균 0, 분산 1
scale_2_columns = x_2_cols #범주형 변수 빼고 함
df[scale_2_columns] = scaler.fit_transform(df[scale_2_columns])#_를 총해 한번에 fit=run=compile

df[x_2_cols].head()

X_2 = df[x_2_cols]
y = df['Price']

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

y_target_log = np.log1p(y)

from scipy.stats import skew

# object가 아닌 숫자형 피처의 칼럼 index 객체 추출.
features_index = df.dtypes[df.dtypes != 'object'].index
# house_df에 칼럼 index를 [ ]로 입력하면 해당하는 칼럼 데이터 세트 반환. apply lambda로 skew( ) 호출
skew_features = df[features_index].apply(lambda x : skew(x))
# skew(왜곡) 정도가 1 이상인 칼럼만 추출.
skew_features_top = skew_features[skew_features > 1]
print(skew_features_top.sort_values(ascending=False))

df[skew_features_top.index] = df[skew_features_top.index].apply(
    lambda x: np.log1p(x.clip(lower=1e-5))
)

X_train, X_test, y_train, y_test = train_test_split(X_2, y_target_log, test_size=0.2, random_state=156)


from statsmodels.stats.outliers_influence import variance_inflation_factor

#위는 다중공산성을 구하기 위한 문장임
vif = pd.DataFrame()
vif['features'] = X_train.columns
vif["VIF Factor"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif.round(1)

import statsmodels.api as sm

X_train2 = sm.add_constant(X_train)
### 회귀분석모형 수식을 간단하게 만들기 위해 다음과 같이 상수항을 독립변수 데이터에 추가하는 것을 상수항 결합(bias augmentation)작업이라고 합니다.

### ordinary least square 의 약자로, 거리의 최소값을 기준으로 구하는 함수입니다.
model2 = sm.OLS(y_train, X_train2).fit()
model2.summary()

#하이퍼파라미터 튜닝
from sklearn.model_selection import GridSearchCV

def print_best_params(model, params):
    grid_model = GridSearchCV(model, param_grid=params,
                              scoring='neg_mean_squared_error', cv=5)
    grid_model.fit(X_2, y_target_log)
    rmse = np.sqrt(-1* grid_model.best_score_)
    print('{0} 5 CV 시 최적 평균 RMSE 값: {1}, 최적 alpha:{2}'.format(model.__class__.__name__,
                                        np.round(rmse, 4), grid_model.best_params_))
    return grid_model.best_estimator_

#XGBRegressor
from xgboost import XGBRegressor

xgb_params = {'n_estimators':[1000]}
xgb_reg = XGBRegressor(n_estimators=1000, learning_rate=0.05,
                       colsample_bytree=0.5, subsample=0.8)
best_xgb = print_best_params(xgb_reg, xgb_params)

def get_rmse_pred(preds):
    for key in preds.keys():
        pred_value = preds[key]
        mse = mean_squared_error(y_test , pred_value)
        rmse = np.sqrt(mse)
        print('{0} 모델의 RMSE: {1}'.format(key, rmse))

xgb_reg.fit(X_train, y_train)
xgb_pred = xgb_reg.predict(X_test)

print(xgb_reg.score(X_train, y_train))  # training set
print(xgb_reg.score(X_test, y_test))