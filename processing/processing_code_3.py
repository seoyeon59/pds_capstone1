# step 3
import pandas as pd

## 전처리한 crossover csv 파일 들고오기
df1=pd.read_csv("crossover_prepro_2.csv")
# 'Unnamed: 0.1'과 'Unnamed: 0' 컬럼을 제거
df1=df1.drop(df1.columns[0],axis=1, errors='ignore')
# Name,Price,Mile 컬럼 맨앞으로 빼기
df1=df1[['Name', 'Price', 'Mileage'] + [col for col in df1.columns if col not in ['Name', 'Price', 'Mileage']]]
# 'type' 열에 'crossover' 값 추가
df1['type'] = 'crossover'

print(df1["curb_weight"])
print(df1.columns)

## 전처리한 coupes csv 파일 들고오기
df2=pd.read_csv("coupes_prepro_2.csv")
# 'Unnamed: 0.1'과 'Unnamed: 0' 컬럼을 제거
df2 = df2.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1, errors='ignore')
# Name,Price,Mile 컬럼 맨앞으로 빼기
df2=df2[['Name', 'Price', 'Mileage'] + [col for col in df2.columns if col not in ['Name', 'Price', 'Mileage']]]
# 'type' 열에 'coupes' 값 추가
df2['type'] = 'coupes'

print(df2["curb_weight"])
print(df2.columns)

## 전처리한 suv csv 파일 들고오기
df3=pd.read_csv("suv_prepro_2.csv")
# 'Unnamed: 0.1'과 'Unnamed: 0' 컬럼을 제거
df3 = df3.drop(['Unnamed: 0'], axis=1, errors='ignore')
# Name,Price,Mile 컬럼 맨앞으로 빼기
df3=df3[['Name', 'Price', 'Mileage'] + [col for col in df3.columns if col not in ['Name', 'Price', 'Mileage']]]
# 'type' 열에 'suv' 값 추가
df3['type'] = 'suv'

## 전처리한 sportscar csv 파일 들고오기
df4=pd.read_csv("sportscar_prepro_2.csv")
df4 = df4.drop(['Unnamed: 0'], axis=1, errors='ignore')
# 'type' 열에 'sportscar' 값 추가
df4['type']='sportscar'

print(df4["curb_weight"])
print(df4.columns)

## 전처리한 truck csv 파일 들고오기
df5=pd.read_csv("truck_prepro_2.csv")
# 'Unnamed: 0.1'과 'Unnamed: 0' 컬럼을 제거
df5 = df5.drop(['Unnamed: 0'], axis=1, errors='ignore')
# Name,Price,Mile 컬럼 맨앞으로 빼기
df5=df5[['Name', 'Price', 'Mileage'] + [col for col in df5.columns if col not in ['Name', 'Price', 'Mileage']]]
# 'type' 열에 'truck' 값 추가
df5['type']='truck'

print(df5['curb_weight'])
print(df5.columns)

## 전처리한 sedan csv 파일 들고오기
df6=pd.read_csv("sedan_prepro_2.csv")
# 'Unnamed: 0.1'과 'Unnamed: 0' 컬럼을 제거
df6 = df6.drop(['Unnamed: 0'], axis=1, errors='ignore')
# Name,Price,Mile 컬럼 맨앞으로 빼기
df6=df6[['Name', 'Price', 'Mileage'] + [col for col in df6.columns if col not in ['Name', 'Price', 'Mileage']]]

# 'type' 열에 'sedan' 값 추가
df6["type"]="sedan"

print(df6['curb_weight'])
print(df6.columns)

## 여러 개의 DataFrame을 합치기
result_df = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)

# 열의 이름 교페

old_cols = ['Name', 'Price', 'Mileage', 'ages', 'accident_damages', 'Insurance',
       'service_repairs', 'fuel_capacity', 'wheelbase', 'dirver_leg_room',
       'dirver_head_room', 'cargo_capacity', 'curb_weight', 'towing_capacity',
       'drive_type_text', 'transmission_text', 'torque_num', 'torque_rpm',
       'dimension_L', 'dimension_W', 'dimension_H', 'mile_per_gallon_city',
       'mile_per_gallon_hwy', 'engine_cly', 'engine_fuel', 'engine_L',
       'color_ext', 'color_int', 'torque', 'type']

new_cols = ['Name', "Price",'Mile','Ages','Accident_damages', 'Insurance',
          'Service_repairs','fuel_capacity','wheelbase','dirver_leg_room',
          'dirver_head_room', 'cargo_capacity','curb_weight','towing_capacity',
          'drive_type_text', 'transmission', 'torque_num', 'torque_rpm',
          'dimension_L','dimension_W','dimension_H','mile_per_gallon_city',
         'mile_per_gallon_hwy','engine_cly','engine_fuel', 'engine_L',
          'color_ext', 'color_int','torque', 'type']

result_df.columns = new_cols

# 결과 출력
print(result_df)
# result_df info 확인하기
result_df.info()

# csv 파일 만들기
result_df.to_csv("all_cars.csv", index=False)