import pandas as pd

df1=pd.read_csv("12_2_re_crossover.csv")
# 'Unnamed: 0.1'과 'Unnamed: 0' 컬럼을 제거
df1=df1.drop(df1.columns[0],axis=1)

## price
df1['Price']=df1['Price'].replace('[\$,*]', '', regex=True)
df1=df1.drop(df1[df1['Price']=="Price unavailable*"].index,axis=0)
df1['Price']=df1['Price'].astype(int)
print(df1['Price'])

## mileage
df1['Mileage']=df1['Mileage'].replace(r'[K,miles]','',regex=True)
df1['Mileage']=df1['Mileage'].astype(int)

df1['type'] = 'crossover'

print(df1["curb_weight"])
print(df1.columns)

df2=pd.read_csv("12_2_coupes.csv")
# 'Unnamed: 0.1'과 'Unnamed: 0' 컬럼을 제거
df2 = df2.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1)

print(df2.columns)

df2['type'] = 'coupes'
print(df2["curb_weight"])

df3=pd.read_csv("12_2_suv.csv")
# 'Unnamed: 0.1'과 'Unnamed: 0' 컬럼을 제거
df3 = df3.drop(['Unnamed: 0'], axis=1)

#Name,Price,Mile 컬럼 맨앞으로 빼기
df3=df3[['Name', 'Price', 'Mileage'] + [col for col in df3.columns if col not in ['Name', 'Price', 'Mileage']]]
df3['type'] = 'suv'
print(df3["curb_weight"])

df4=pd.read_csv("12_2_sportscar.csv")
# 'Unnamed: 0.1'과 'Unnamed: 0' 컬럼을 제거
df4 = df4.drop(['Unnamed: 0'], axis=1)

df4.info()

df4['type']='sportscar'
print(df4["curb_weight"])

df5=pd.read_csv("12_2_truck.csv")
# 'Unnamed: 0.1'과 'Unnamed: 0' 컬럼을 제거
df5 = df5.drop(['Unnamed: 0'], axis=1)

#Name,Price,Mile 컬럼 맨앞으로 빼기
df5=df5[['Name', 'Price', 'Mileage'] + [col for col in df5.columns if col not in ['Name', 'Price', 'Mileage']]]

df5.info()

print(df5['curb_weight'])
df5['type']='truck'

df6=pd.read_csv("12_2_re_sedan.csv")
# 'Unnamed: 0.1'과 'Unnamed: 0' 컬럼을 제거
df6 = df6.drop(['Unnamed: 0'], axis=1)

#Name,Price,Mile 컬럼 맨앞으로 빼기
df6=df6[['Name', 'Price', 'Mileage'] + [col for col in df6.columns if col not in ['Name', 'Price', 'Mileage']]]
print(df6['curb_weight'])

df6.info()

df6["type"]="sedan"

## 여러 개의 DataFrame을 합치기
result_df = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)

# 결과 출력
print(result_df)
# result_df info 확인하기
result_df.info()

# csv 파일 만들기
result_df.to_csv("12_2_all.csv")