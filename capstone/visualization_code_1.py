import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("12_4_all.csv")

df.info()

df["Price"].describe()
df["Price"].hist(bins=50)
# #### 가격이 20000달러~30000달러에 가장 많이 형성되어 있는 것을 알 수 있음

print(df.columns)

df.info()

num_cols=["Price",'Mile','Ages','Service_repairs','fuel_capacity','wheelbase','dirver_leg_room',
         'dirver_head_room','curb_weight','dimension_L','dimension_W','dimension_H','mile_per_gallon_city',
         'mile_per_gallon_hwy','mile_per_gallon_hwy','engine_cly','engine_L','torque']

corr = df[num_cols].corr(method = 'pearson') #상관관꼐 분석위해 pearson이라는 메소드 씀
print(corr)

#상관관계를 수치로 보면 힘들기 때문에 밑에 heatmap 처럼 확인함!
fig = plt.figure(figsize = (16, 12))
ax = fig.gca()

sns.set(font_scale = 2)  # heatmap 안의 font-size 설정
heatmap = sns.heatmap(corr.values, annot = True, fmt='.2f', annot_kws={'size':15},
                      yticklabels = num_cols, xticklabels = num_cols, ax=ax, cmap = "RdYlBu")

plt.tight_layout()
plt.show()

# ### 가격과 상관관계가 큰 X변수들 : engine_cly(0.59), engine_L(0.58),Mile(-0.57)

# # 가격과 가장 상관계수가 큰 컷들 따로 시각화해서 보기

### scatter plot
sns.scatterplot(data=df, x='engine_cly', y='Price', markers='o', color='blue', alpha=0.6)
plt.title('Scatter Plot')
plt.show()

### scatter plot
sns.scatterplot(data=df, x='engine_L', y='Price', markers='o', color='blue', alpha=0.6)
plt.title('Scatter Plot')
plt.show()

### scatter plot
sns.scatterplot(data=df, x='Mile', y='Price', markers='o', color='blue', alpha=0.6)
plt.title('Scatter Plot')
plt.show()

### scatter plot
sns.scatterplot(data=df, x='Ages', y='Price', markers='o', color='blue', alpha=0.6)
plt.title('Ages & Price')
plt.show()

## 독립변수들 중에서 상관계수 큰 것들끼리 분포확인해보기

### scatter plot
sns.scatterplot(data=df, x='wheelbase', y='dimension_L', markers='o', color='blue', alpha=0.6)
#plt.title('scatter plot')
plt.xlabel("wheelbase(inch)", fontsize=20)
plt.ylabel("Price($)", fontsize=20)
plt.show()

### scatter plot
sns.scatterplot(data=df, x='engine_cly', y='engine_L', markers='o', color='blue', alpha=0.6)
#plt.title('Scatter Plot')
plt.xlabel("engine_cly", fontsize=20)
plt.ylabel("Price($)", fontsize=20)
plt.show()
plt.show()


## 범주형 변수들과 가격간의 상관관계 확인

#차종별 가격이 어떻케 형성되는지 학인
sns.barplot(x="type",y="Price",data=df)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel("type", fontsize=13)
plt.ylabel("Price", fontsize=13)

sns.barplot(x="Accident_damages",y="Price",data=df)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

plt.xlabel("Accident_damages", fontsize=13)
plt.ylabel("Price", fontsize=13)

sns.barplot(x="engine_fuel",y="Price",data=df)
plt.xticks(fontsize=13, rotation=45)
plt.yticks(fontsize=13)
plt.xlabel("Accident_damages", fontsize=13)
plt.ylabel("Price", fontsize=13)

sns.barplot(x="color_ext",y="Price",data=df)
plt.xticks(fontsize=13, rotation=45)
plt.yticks(fontsize=13)
plt.xlabel("color_ext", fontsize=13)
plt.ylabel("Price", fontsize=13)

sns.barplot(x="color_int",y="Price",data=df)
plt.xticks(fontsize=13, rotation=45)
plt.yticks(fontsize=13)
plt.xlabel("color_int", fontsize=13)
plt.ylabel("Price", fontsize=13)

sns.barplot(x="transmission",y="Price",data=df)
plt.xticks(fontsize=13, rotation=45)
plt.yticks(fontsize=13)
plt.xlabel("transmission", fontsize=13)
plt.ylabel("Price", fontsize=13)

## 원핫인코딩 한 csv로 분석
'''
df1=pd.read_csv("output_all.csv")

print(df1)

num_all_cols=['Price', 'Mileage', 'ages', 'service_repairs',
       'fuel_capacity', 'wheelbase', 'dirver_leg_room', 'dirver_head_room',
       'curb_weight', 'dimension_L', 'dimension_W', 'dimension_H',
       'mile_per_gallon_city', 'mile_per_gallon_hwy', 'engine_cly', 'engine_L',
       'torque', 'accident_damages_Multiple Damage Events',
       'accident_damages_No Accidents or Damage Reported',
       'transmission_text_Automatic', 'transmission_text_Manual 5 Speed',
       'transmission_text_Manual 6 Speed', 'transmission_text_Manual 7 Speed',
       'engine_fuel_ Gas', 'engine_fuel_ Hybrid',
       'engine_fuel_ Supercharged Gas', 'engine_fuel_ Turbo Diesel',
       'engine_fuel_ Turbo Gas', 'engine_fuel_ Turbo Hybrid',
       'engine_fuel_ Turbo Supercharged', 'color_ext_Black', 'color_ext_Blue',
       'color_ext_Brown', 'color_ext_Burgundy', 'color_ext_Gold',
       'color_ext_Gray', 'color_ext_Green', 'color_ext_Maroon',
       'color_ext_Orange', 'color_ext_Pearl', 'color_ext_Purple',
       'color_ext_Red', 'color_ext_Silver', 'color_ext_Tan', 'color_ext_White',
       'color_ext_Yellow', 'color_int_Black', 'color_int_Brown',
       'color_int_Burgundy', 'color_int_Champagne', 'color_int_Cream',
       'color_int_Gray', 'color_int_Maroon', 'color_int_Orange',
       'color_int_Red', 'color_int_Tan', 'type_coupes', 'type_crossover',
       'type_sedan', 'type_sportscar', 'type_suv', 'type_truck']

corr = df1[num_all_cols].corr(method = 'pearson') #상관관꼐 분석위해 pearson이라는 메소드 씀
print(corr)

#상관관계를 수치로 보면 힘들기 때문에 밑에 heatmap 처럼 확인함!
fig = plt.figure(figsize = (40, 40))
ax = fig.gca()

sns.set(font_scale = 1.5)  # heatmap 안의 font-size 설정
heatmap = sns.heatmap(corr.values, annot = True, fmt='.2f', annot_kws={'size':15},
                      yticklabels = num_all_cols, xticklabels = num_all_cols, ax=ax, cmap = "RdYlBu")

plt.tight_layout()

plt.show()
'''
