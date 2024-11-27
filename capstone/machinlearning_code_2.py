import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("12_4_all.csv")

# ## 2차 히트맵
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
print(x_2_cols)

### 사이킷런은 파이썬에서 머신러닝 분석을 할 때 유용하게 사용할 수 있는 라이브러리 입니다. 여러가지 머신러닝 모듈로 구성되어있습니다.

from sklearn.preprocessing import StandardScaler
#sklearn: 머신러닝 알고리즘 zip
#sklearn의 preprocessing(전처리)의 StandardScaler

### feature standardization  (numerical_columns except dummy var.-"CHAS")
scaler = StandardScaler()#scaler object 만들기  # 평균 0, 분산 1
scale_2_columns = x_2_cols #범주형 변수 빼고 함
df[scale_2_columns] = scaler.fit_transform(df[scale_2_columns])#_를 총해 한번에 fit=run=compile
df[scale_2_columns].head()
print(df[x_2_cols])

from sklearn.model_selection import train_test_split
# split dataset into training & test
X_2 = df[x_2_cols]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X_2, y, test_size=0.2, random_state=1)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

print(X_train)
print(y_test)


from statsmodels.stats.outliers_influence import variance_inflation_factor
#위는 다중공산성을 구하기 위한 문장임
vif = pd.DataFrame()
vif['features'] = X_train.columns
vif["VIF Factor"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif.round(1)

from sklearn import linear_model

# fit regression model in training set
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train) #추세선 만듬

# predict in test set
pred_test = lr.predict(X_test)

print(lr.coef_)

import statsmodels.api as sm

X_train2 = sm.add_constant(X_train)
### 회귀분석모형 수식을 간단하게 만들기 위해 다음과 같이 상수항을 독립변수 데이터에 추가하는 것을 상수항 결합(bias augmentation)작업이라고 합니다.

### ordinary least square 의 약자로, 거리의 최소값을 기준으로 구하는 함수입니다.
model2 = sm.OLS(y_train, X_train2).fit()
model2.summary()

df = pd.DataFrame({'actual': y_test, 'prediction': pred_test})
df = df.sort_values(by='actual').reset_index(drop=True)
df.head()

plt.figure(figsize=(12, 9))
plt.scatter(df.index, df['prediction'], marker='x', color='r')
plt.scatter(df.index, df['actual'], alpha=0.3, marker='o', color='black')
plt.title("Prediction Result in Test Set", fontsize=20)
plt.legend(['prediction', 'actual'], fontsize=12)
plt.show()

print(model.score(X_train, y_train))  # training set
print(model.score(X_test, y_test))

# RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt

# training set
pred_train = lr.predict(X_train)
print(sqrt(mean_squared_error(y_train, pred_train)))

# test set
print(sqrt(mean_squared_error(y_test, pred_test)))


# ## 릿지회귀
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

#alpha=10
ridge=Ridge(alpha=10)
neg_mse_scores=cross_val_score(ridge,X_2,y,scoring='neg_mean_squared_error',cv=5)
rmse_scores=np.sqrt(-1*neg_mse_scores)
avg_rmse=np.mean(rmse_scores)

print('5 folds 의 평균 RMSE : {0:.3f}'.format(avg_rmse))