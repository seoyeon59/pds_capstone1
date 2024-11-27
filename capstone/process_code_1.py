import pandas as pd

# CSV 파일을 데이터프레임으로 읽기
df = pd.read_csv('result_df_coupes.csv')

# 특정 열을 제거할 열의 이름 리스트
columns_to_remove = ['horsepower', 'front_tire_size', 'prior_use', 'keys']

# 특정 열 제거
df = df.drop(columns=columns_to_remove, axis=1)

## mile_per_gallon
# 1. mile_per_gallon에서 city/hwy 나누기
# city만 있으면 city/Null, hwy만 있으면 Null/hwy로 만들기
target_column_name = 'mile_per_gallon'
target_word1 = ' city'
target_word2 = ' hwy'

# 특정 단어들이 모두 있는 행은 그대로 두고, 각각 하나만 있는 경우 '/Null' 추가
mask1 = df[target_column_name].str.contains(target_word1, case=False, na=False)
mask2 = df[target_column_name].str.contains(target_word2, case=False, na=False)

# 특정 단어1만 있는 경우 '/Null'을 뒤에 추가
df.loc[mask1 & ~mask2, target_column_name] = df.loc[mask1 & ~mask2, target_column_name]
# 특정 단어2만 있는 경우 'Null/'을 앞에 추가
df.loc[~mask1 & mask2, target_column_name] = df.loc[~mask1 & mask2, target_column_name]
# 둘 다 없는 경우 'Null/Null' 추가
df.loc[~mask1 & ~mask2, target_column_name] = 'Null/Null'

## torque
# 특정 열에 특정 단어가 있는 것만 보기
df = df[df['torque'].str.contains('torque@', na=False)]
df = df[~df['torque'].str.contains('-')]

# torque, dimention, engine, color_ext_int, mile_per_gallon
# 특정 열을 n개의 열로 나누기
df[['torque_num', 'torque_rpm']] = df['torque'].str.split('@', expand=True)
df[['dimension_L', 'dimension_W', 'dimension_H']] = df['dimension'].str.split('x', expand=True)
df[['mile_per_gallon_city', 'mile_per_gallon_hwy']] = df['mile_per_gallon'].str.split('/', expand=True)
df[['engine_cly', 'engine_fuel', 'engine_L']] = df['engine_text'].str.split(',', expand=True)
df[['color_ext', 'color_int']] = df['color_ext_int'].str.split('/', expand=True)

# 나눈 열을 제외한 기존 열 삭제
df = df.drop(columns=['torque'])
df = df.drop(columns=['dimension'])
df = df.drop(columns=['mile_per_gallon'])
df = df.drop(columns=['engine_text'])
df = df.drop(columns=['color_ext_int'])

## 토그 숫자만 추출
df['torque_num'] = df['torque_num'].replace('torque', '', regex=True)
df['torque_rpm'] = df['torque_rpm'].replace('rpm', '', regex=True)
df['torque_num'] = df['torque_num'].astype(int)
df['torque_rpm'] = df['torque_rpm'].astype(int)


## 나눌 열 선택
# torque_num 열을 torque_rpm 열로 나누는 함수 정의
def divide_columns(row):
    try:
        # torque_num 열을 torque_rpm 열로 나누기
        result = row['torque_num'] / row['torque_rpm'] * 10000
        return result
    except ZeroDivisionError:
        # 만약 torque_rpm 열이 0이면 예외 처리
        return None


# 새로운 열 추가
df['torque'] = df.apply(divide_columns, axis=1)

# 결과를 새로운 CSV 파일로 저장
df.to_csv('coupe_전처리.csv', index=False)
print(f"특정 열이 제거된 CSV 파일로 저장되었습니다.")

