* preprocessing_code_1 와 preprocessing_code_2 는 크롤링한 csv 파일들 하나씩 처리하는 방식이며, preprocessing_code_3은크롤링 한 모든 파일들을 합쳐서 하나의 파일로 묶는 코드이다. 숫자의 순서대로 코드를 실행하면 된다.
----
### preprocessing_code_1 : 첫 번째 단계 (자동차 기본 정보에 대한 전처리)
1. 우선 필요없는 열([horsepower', 'front_tire_size', 'prior_use', 'keys')을 식제 시킨다. 삭제시킨 열의 기준은 행에 결측치가 많거나 값이 모두 같은 등 중고차 가격에 영향을 많이 미치지 않을 열들이다.
2. mile_per_gallon의 열에 city(국도 연비)와 hwy(고속도로 연비)외의 결과값이 같이 있으므로 city와 hwy의 열을 추가하여 mile_per_gallon에 들어있는 값을 분리시켜 주었다.
3. torque의 열을 살펴보면 단위가 통일되어 있지 않아 토크의 단위를 통일시키기 위하여 '@'의 기준으로 나누어 숫자만 추출한 후, divide_columns 함수를 만들어 단위를 통일시켜 주었다.
4. dimention, engine, color_ext_int, mile_per_gallond의 열들도 한 열에 2개 이상의 정보를 가지고 있어 나누어 새로운 열을 만들었으며, 나눈 기존 열들(torque, dimention, engine, color_ext_int, mile_per_gallon)은 삭제시켜 주었다.
5. 1차 전처리를 했음을 나타내기 위해 전처리 한 파일을 '차종_prepro_1.csv' 파일로 저장해 주었다.

### preprocessing_code_2 : 두 번째 단계 (자동차 이력 정보에 대한 전처리)
1. 1차 전처리를 완료한 csv 파일들을 들고오는데 인덱스가 있는 파일들만 line 6 을 실행한다.(suv, coupes 차종만 실행을 하지 않음)
2. ages 열의 값이 Current Year로 되어있을 시 ages 열을 수치형으로 통일시켜주기 위해 0으로 바꿔주고, year(S) 문자를 삭제하고, 타입을 int로 변경한다.
3. 



### preprocessing_code_3 : 세 번째 단
