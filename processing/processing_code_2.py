# step 2
import pandas as pd

df=pd.read_csv('suv_prepro_1.csv')   #suv, coupes 다시 돌리기
# 인덱스가 있는 csv 파일들만 이 코드를 실행한다
# df = df.drop(df.columns[0], axis=1)

## ages
# ages 열에 Current Year 값 0으로 변경
df['ages'] = df['ages'].replace('Current Year', '0')
age_column = df['ages']
# year(s) 문자 삭제
cleaned_age=age_column.str.replace(r'[year(s)]','',regex=True)
# 타입 int로 변경
cleaned_age=cleaned_age.astype(int)
df['ages']=cleaned_age

## accident_damages
# 결측치 값 No Accidents or Damage Reported으로 대체
df['accident_damages']=df['accident_damages'].fillna("No Accidents or Damage Reported")
# print(df['accident_damages'])

## insurancs
# 'Insurance' 열의 데이터에서 "Null"을 0으로 교체
df['Insurance'] = df['Insurance'].replace("Null", 0)
# print(df['Insurance'])

## service_repairs
# 'service_repairs' 칼럼의 NaN 값을 0으로 대체
df['service_repairs'] = df['service_repairs'].fillna(0)
# [Service Record(s) Reported] 값 삭제
df['service_repairs']=df['service_repairs'].replace(r'[Service Record(s) Reported]','',regex=True)
# 타입 int로 변경
df['service_repairs']=df['service_repairs'].astype(int)
# print(df['service_repairs'])

## price
# $,* 문자 삭제
df['Price']=df['Price'].replace(r'[\$,*]', '', regex=True)
# Price unavailable 값 있는 행 삭제
df=df.drop(df[df['Price']=="Price unavailable"].index,axis=0)
# 타입 int로 변경
df['Price']=df['Price'].astype(int)
# print(df['Price'])

## mileage
# K,miles 문자 삭제
df['Mileage']=df['Mileage'].replace(r'[K,miles]','',regex=True)
# print(df['Mileage'])

## fuel capacity
# gal 문자 삭제
df['fuel_capacity']=df['fuel_capacity'].replace(r'[gal]','',regex=True)
# Null 값이 존재하는 행 삭제
df = df[~df['fuel_capacity'].str.contains("Null", na=False)]
# Nu 값이 존재하는 행 삭제
df = df[~df['fuel_capacity'].str.contains('Nu', na=False)] # suv, coupes 사용
# 타입 float로 변경
df['fuel_capacity']=df['fuel_capacity'].astype(float)
# print(df['fuel_capacity'])

## wheelbase
# Null 값이 존재하면 행 삭제
df = df[~df['wheelbase'].str.contains("Null", na=False)]
# " 문자 지우기
df["wheelbase"]=df["wheelbase"].replace(r'["]','',regex=True)
# 타입 float로 변경
df["wheelbase"]=df["wheelbase"].astype(float)
# print(df["wheelbase"])

## dirver_leg_room
# Null 값이 존재하는 행 삭제
df = df[~df['dirver_leg_room'].str.contains("Null", na=False)]
# " 문자 삭제
df["dirver_leg_room"]=df["dirver_leg_room"].replace(r'["]','',regex=True)
# 'dirver_leg_room' 열에서 빈 문자열을 NaN으로 바꾸기
df['dirver_leg_room'] = df['dirver_leg_room'].replace("", float("nan"))
# NaN 값이 있으면 삭제
df = df.dropna(subset=['dirver_leg_room'])  # NaN이 있는 행을 삭제
# 타입 float로 변경
df["dirver_leg_room"]=df["dirver_leg_room"].astype(float)
# print(df["dirver_leg_room"])

## driver_head_room
## Null 값이 있는 행 삭제
df = df[~df['dirver_head_room'].str.contains("Null", na=False)]
# " 문자 삭제
df["dirver_head_room"]=df["dirver_head_room"].replace(r'["]','',regex=True)
# 값이 없는 해만 선택
df = df[df['dirver_head_room'] != '']
# 타입 float로 변경
df["dirver_head_room"]=df["dirver_head_room"].astype(float)
# print(df["dirver_head_room"])

## cargo capacity
# Null 값 0으로 변경
df["cargo_capacity"]=df["cargo_capacity"].replace("Null",0)
# print(df["cargo_capacity"])

## curb weight
# lds 문자 삭제
df["curb_weight"]=df["curb_weight"].replace("lbs",'',regex=True)
# Null 값이 있으면 행 삭제
df = df[~df['curb_weight'].str.contains("Null", na=False)]
# , 문자 삭제
df["curb_weight"]=df["curb_weight"].replace(',','',regex=True)
# 타입 int로 변경
df["curb_weight"]=df["curb_weight"].astype(int)
# print(df["curb_weight"])

## towing capacity ##
## dimension_L
# Null 값인 행 삭제
df = df[~df['dimension_L'].str.contains("Null", na=False)]
# " L 문자 삭제
df['dimension_L']=df['dimension_L'].replace('" L ', '', regex=True).astype(float)
# print(df['dimension_L'])

## dimension_W
# Null 값인 행 삭제
df = df[~df['dimension_W'].str.contains("Null", na=False)]
# " W 문자 삭제
df['dimension_W']=df['dimension_W'].replace('" W', '', regex=True).astype(float)
# print(df['dimension_W'])

## dimension_H
# Null 값인 행 삭제
df = df[~df['dimension_H'].str.contains("Null", na=False)]
# " H 문자 삭제
df['dimension_H']=df['dimension_H'].replace('" H', '', regex=True).astype(float)
# print(df['dimension_H'])

## mile_per-_gallon_city
# 결측치 있는 행 삭제
df["mile_per_gallon_city"]=df["mile_per_gallon_city"].dropna()
# Null 값인 행 삭제
df = df[~df['mile_per_gallon_city'].str.contains("Null", na=False)]
# hwy 문자가 있는 행 삭제
df = df[~df['mile_per_gallon_city'].str.contains('hwy')]
# city 문자 삭제
df["mile_per_gallon_city"]=df["mile_per_gallon_city"].replace('city', '', regex=True).astype(int)
# print(df["mile_per_gallon_city"])

## mile_per_gallon_hwy
# Null 값인 행 삭제
df = df[~df['mile_per_gallon_hwy'].str.contains("Null", na=False)]
# 결측치 있는 행 삭제
df=df.dropna(subset=['mile_per_gallon_hwy'])
# hwy 문자 삭제
df["mile_per_gallon_hwy"]=df["mile_per_gallon_hwy"].replace('hwy', '', regex=True).astype(int)
# print(df["mile_per_gallon_hwy"])

## engine_cyl
# Null 값인 행 삭제
df = df[~df['engine_cly'].str.contains("Null", na=False)]
# -cyl 문자 삭제
df["engine_cly"]=df["engine_cly"].replace('-cyl', '', regex=True).astype(int)
# print(df["engine_cly"])

## engine_L
# Null 값인 행 삭제
df = df[~df['engine_L'].str.contains("Null", na=False)]
# 결측치 있는 행 삭제
df["engine_L"]=df["engine_L"].dropna()
# L 문자 삭제
df["engine_L"]=df["engine_L"].replace("L",'',regex=True).astype(float)
# print(df["engine_L"])

## torque 계산식 소수점 첫째 자리까지 반환
df['torque'] = df['torque'].round(1)
# print(df['torque'])

# 새로운 csv 파일로 만들기
df.to_csv("suv_prepro_2.csv", index=False)