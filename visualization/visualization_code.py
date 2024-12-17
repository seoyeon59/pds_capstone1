# 시각화
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
df=pd.read_csv("all_cars.csv")

df.info() # info 확인

print(df["Price"].describe()) # Price의 기초 통계량 확인
df["Price"].hist(bins=50) # Price를 히스토그램으로 확인
#### 가격이 20000달러~30000달러에 가장 많이 형성되어 있는 것을 알 수 있음

print(df.columns) # df의 컬럼을 리스트로 확인 

num_cols=["Price",'Mile','Ages','Service_repairs','fuel_capacity','wheelbase','dirver_leg_room',
         'dirver_head_room','curb_weight','dimension_L','dimension_W','dimension_H','mile_per_gallon_city',
         'mile_per_gallon_hwy','mile_per_gallon_hwy','engine_cly','engine_L','torque']

# 상관 관계 분석을 위해 pearson 메소드 사용
corr = df[num_cols].corr(method = 'pearson')
print(corr)

# 상관 관계를 수치 heatmap으로 확인
fig = plt.figure(figsize = (16, 12))
ax = fig.gca()

sns.set(font_scale = 2)  # heatmap 안의 font-size 설정
heatmap = sns.heatmap(corr.values, annot = True, fmt='.2f', annot_kws={'size':15},
                      yticklabels = num_cols, xticklabels = num_cols, ax=ax, cmap = "RdYlBu")

plt.tight_layout()
plt.show()

### 가격과 상관 관계가 큰 X변수들 : engine_cly(0.59), engine_L(0.58),Mile(-0.57)
# 가격과 가장 상관 계수가 큰 컷들 따로 시각화해서 보기

### scatter plot (engine_cly(엔진 실린더)와 가격의 관계)
sns.scatterplot(data=df, x='engine_cly', y='Price', markers='o', color='blue', alpha=0.6)
plt.title('Scatter Plot')
plt.show()

### scatter plot (engine_L(엔진의 리터)와 가격의 관계)
sns.scatterplot(data=df, x='engine_L', y='Price', markers='o', color='blue', alpha=0.6)
plt.title('Scatter Plot')
plt.show()

### scatter plot (Mile(마일)과 가격의 관계)
sns.scatterplot(data=df, x='Mile', y='Price', markers='o', color='blue', alpha=0.6)
plt.title('Scatter Plot')
plt.show()

### scatter plot (Ages(연식)과 가격의 관계)
sns.scatterplot(data=df, x='Ages', y='Price', markers='o', color='blue', alpha=0.6)
plt.title('Ages & Price')
plt.show()

## 독립변수들 중에서 상관 계수 큰 것들끼리 분포 확인

### scatter plot (wheelbase(차축)과 dimension_L(차의 하측 치수)의 관계)
sns.scatterplot(data=df, x='wheelbase', y='dimension_L', markers='o', color='blue', alpha=0.6)
#plt.title('scatter plot')
plt.xlabel("wheelbase(inch)", fontsize=20)
plt.ylabel("Price($)", fontsize=20)
plt.show()

### scatter plot (engine_cly(엔진 실린더)와 engine_L(엔진 리터)의 관계)
sns.scatterplot(data=df, x='engine_cly', y='engine_L', markers='o', color='blue', alpha=0.6)
#plt.title('Scatter Plot')
plt.xlabel("engine_cly", fontsize=20)
plt.ylabel("Price($)", fontsize=20)
plt.show()

## 범주형 변수들과 가격간의 상관 관계 확인
# 차종별 가격이 어떻게 형성되는지 학인
sns.barplot(x="type",y="Price",data=df)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel("type", fontsize=13)
plt.ylabel("Price", fontsize=13)
plt.show()

# 사고별 가격이 어떻게 형성되는지 학인
sns.barplot(x="Accident_damages",y="Price",data=df)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel("Accident_damages", fontsize=13)
plt.ylabel("Price", fontsize=13)
plt.show()

# 엔진 연료 가격이 어떻게 형성되는지 학인
sns.barplot(x="engine_fuel",y="Price",data=df)
plt.xticks(fontsize=13, rotation=45)
plt.yticks(fontsize=13)
plt.xlabel("Accident_damages", fontsize=13)
plt.ylabel("Price", fontsize=13)
plt.show()

# 차량 외부 색상별 가격이 어떻게 형성되는지 학인
sns.barplot(x="color_ext",y="Price",data=df)
plt.xticks(fontsize=13, rotation=45)
plt.yticks(fontsize=13)
plt.xlabel("color_ext", fontsize=13)
plt.ylabel("Price", fontsize=13)
plt.show()

# 차량 내부 색상별 가격이 어떻게 형성되는지 학인
sns.barplot(x="color_int",y="Price",data=df)
plt.xticks(fontsize=13, rotation=45)
plt.yticks(fontsize=13)
plt.xlabel("color_int", fontsize=13)
plt.ylabel("Price", fontsize=13)
plt.show()

# transmiddion별 가격이 어떻게 형성되는지 학인
sns.barplot(x="transmission",y="Price",data=df)
plt.xticks(fontsize=13, rotation=45)
plt.yticks(fontsize=13)
plt.xlabel("transmission", fontsize=13)
plt.ylabel("Price", fontsize=13)
plt.show()
