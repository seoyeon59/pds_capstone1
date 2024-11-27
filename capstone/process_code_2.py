import pandas as pd

df=pd.read_csv("sportscar_prepro.csv")
df = df.drop(df.columns[0], axis=1)

## ages
print(df["ages"])
df['ages'] = df['ages'].replace('Current Year', '0')

# 연식 전처리
age_column = df['ages']
cleaned_age=age_column.str.replace(r'[year(s)]','',regex=True)
print(cleaned_age)
cleaned_age=cleaned_age.astype(int)
df["ages"]=cleaned_age
print(df["ages"])

## accident_damages
print(df['accident_damages'])

df['accident_damages']=df['accident_damages'].fillna("No Accidents or Damage Reported")
print(df['accident_damages'])

## insurancs
print(df['Insurance'])

# 'Insurance' 열의 데이터에서 "Null"을 0으로 바꾸기
df['Insurance'] = df['Insurance'].replace("Null", 0)
print(df['Insurance'])

## service_repairs
print(df["service_repairs"])

# 'service_repairs' 칼럼의 NaN 값을 0으로 대체
df['service_repairs'] = df['service_repairs'].fillna(0)
df['service_repairs']=df['service_repairs'].replace(r'[Service Record(s) Reported]','',regex=True)
df['service_repairs']=df['service_repairs'].astype(int)
print(df['service_repairs'])

## price
df['Price']=df['Price'].replace('[\$,*]', '', regex=True)
df=df.drop(df[df['Price']=="Price unavailable*"].index,axis=0)
df['Price']=df['Price'].astype(int)
print(df['Price'])

## mileage
df['Mileage']=df['Mileage'].replace(r'[K,miles]','',regex=True)
print(df['Mileage'])

## fuel capacity
df['fuel_capacity']=df['fuel_capacity'].replace(r'[gal]','',regex=True)
print(df['fuel_capacity'])

df = df[~df['fuel_capacity'].str.contains("Null", na=False)]
df['fuel_capacity']=df['fuel_capacity'].astype(float)
print(df['fuel_capacity'])

## wheelbase
df = df[~df['wheelbase'].str.contains("Null", na=False)]
print(df["wheelbase"])

df["wheelbase"]=df["wheelbase"].replace(r'["]','',regex=True)
df["wheelbase"]=df["wheelbase"].astype(float)
print(df["wheelbase"])

## dirver_leg_room
df = df[~df['dirver_leg_room'].str.contains("Null", na=False)]
print(df["dirver_leg_room"])

df["dirver_leg_room"]=df["dirver_leg_room"].dropna()
print(df["dirver_leg_room"])

df["dirver_leg_room"]=df["dirver_leg_room"].replace(r'["]','',regex=True)
print(df["dirver_leg_room"])

df = df[~df['dirver_leg_room'].str.contains("Null", na=False)]
df["dirver_leg_room"]=df["dirver_leg_room"].replace(r'["]','',regex=True)

df = df[df['dirver_leg_room'] != '']
df["dirver_leg_room"]=df["dirver_leg_room"].astype(float)
print(df["dirver_leg_room"])

## driver_head_room
df = df[~df['dirver_head_room'].str.contains("Null", na=False)]
print(df["dirver_head_room"])

df["dirver_head_room"]=df["dirver_head_room"].replace(r'["]','',regex=True)

# 조건에 맞는 행 제거
df = df[df['dirver_head_room'] != '']
df["dirver_head_room"]=df["dirver_head_room"].astype(float)
print(df["dirver_head_room"])

## cargo capacity
df["cargo_capacity"]=df["cargo_capacity"].replace("Null",0)
print(df["cargo_capacity"])

## curb weight
print(df["curb_weight"])

df["curb_weight"]=df["curb_weight"].replace("lbs",'',regex=True)
print(df["curb_weight"])

df = df[~df['curb_weight'].str.contains("Null", na=False)]
print(df["curb_weight"])

df["curb_weight"]=df["curb_weight"].replace(',','',regex=True)
df["curb_weight"]=df["curb_weight"].astype(int)
print(df["curb_weight"])

## towing capacity ##
## dimension_L
df = df[~df['dimension_L'].str.contains("Null", na=False)]
df['dimension_L']=df['dimension_L'].replace('" L ', '', regex=True).astype(float)
print(df['dimension_L'])

## dimension_W
df = df[~df['dimension_W'].str.contains("Null", na=False)]
df['dimension_W']=df['dimension_W'].replace('" W', '', regex=True).astype(float)
print(df['dimension_W'])

## dimension_H
df = df[~df['dimension_H'].str.contains("Null", na=False)]
df['dimension_H']=df['dimension_H'].replace('" H', '', regex=True).astype(float)
print(df['dimension_H'])

## mile_per-_gallon_city
print(df["mile_per_gallon_city"])

df["mile_per_gallon_city"]=df["mile_per_gallon_city"].dropna()
print(df["mile_per_gallon_city"])

df = df[~df['mile_per_gallon_city'].str.contains("Null", na=False)]
print(df["mile_per_gallon_city"])

df = df[~df['mile_per_gallon_city'].str.contains('hwy')]
print(df["mile_per_gallon_city"])

df["mile_per_gallon_city"]=df["mile_per_gallon_city"].replace('city', '', regex=True).astype(int)
print(df["mile_per_gallon_city"])

## mile_per_gallon_hwy
print(df["mile_per_gallon_hwy"])

df = df[~df['mile_per_gallon_hwy'].str.contains("Null", na=False)]
print(df["mile_per_gallon_hwy"])

df=df.dropna(subset=['mile_per_gallon_hwy'])
df["mile_per_gallon_hwy"]=df["mile_per_gallon_hwy"].replace('hwy', '', regex=True).astype(int)
print(df["mile_per_gallon_hwy"])

## engine_Cyl
print(df["engine_cly"])

df = df[~df['engine_cly'].str.contains("Null", na=False)]
print(df["engine_cly"])

df["engine_cly"]=df["engine_cly"].replace('-cyl', '', regex=True).astype(int)
print(df["engine_cly"])

df = df[~df['engine_L'].str.contains("Null", na=False)]
df["engine_L"]=df["engine_L"].dropna()
print(df["engine_L"])

df["engine_L"]=df["engine_L"].replace("L",'',regex=True).astype(float)
print(df["engine_L"])

## torque 계산식 소수점 첫째 자리까지 반환
df['torque'] = df['torque'].round(1)
print(df['torque'])

df.to_csv("12_2_sportscar.csv")