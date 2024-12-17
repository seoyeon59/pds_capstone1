# 라이브러리 불러오기
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import pandas as pd

# 브라우저 한 번만 열기
browser = webdriver.Chrome()
# 크라우저 설정 : 차 종별 url 바꿔서 넣기
# url(현재 사이트가 해외 접속 불가와 다운되어 6개의 url 중 남은 url이 3개입니다.) :
# https://www.carmax.com/cars/suvs
# https://www.carmax.com/cars/pickup-trucks
# https://www.carmax.com/cars/minivans-and-vans

browser.get("https://www.carmax.com/cars/pickup-trucks")
time.sleep(1)

## 스펙 크롤링

# 스크롤 내려서 버튼 클릭
for i in range(10):
    # 현재 창의 높이를 가져오기
    current_height = browser.execute_script("return window.innerHeight;")

    # 스크롤을 현재 창의 높이만큼 내리기
    browser.execute_script(f"window.scrollTo(0, {current_height});")
    time.sleep(2)

    # "더 보기" 버튼 클릭
    see_more_button = WebDriverWait(browser, 30).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, '#see-more > div > hzn-button'))
    )
    see_more_button.click()
    time.sleep(2)

html = browser.page_source
soup = BeautifulSoup(html, "html.parser")

car_list_area = soup.select(".sc--make-model-info.make-model-info.kmx-typography--body-2 > a")

website_list = []  # 웹 사이트 리스트
car_herf = []   # 차량 herf 리스트
specs_result = []   # 스팩 리스트
car_number = 0  # 차량 수

# 링크 추출 및 추가
for car_num in car_list_area:
    # 각 요소에서 'herf' 속성 값을 가져옴
    website = car_num.get('href')
    # 가져온 값을 website_list에 추가
    website_list.append(website)

# 웹 사이트 브라우저로 열기
for car_sell in website_list:
    car_number += 1
    browser = webdriver.Chrome()
    car_url = "https://www.carmax.com" + car_sell
    browser.get(car_url)
    car_herf.append(car_url)
    time.sleep(1)

    # 페이지 스크롤 내리기
    scroll_script = "window.scrollTo(0, 1500);"
    browser.execute_script(scroll_script)
    time.sleep(2)

    # view all freatures & specs 버튼 클릭하기
    view_all_feature_specs_button_xpath = '//*[@id="car-page-features-and-specs-section"]/hzn-button'
    view_all_feature_specs_button = WebDriverWait(browser, 100).until(
        EC.element_to_be_clickable((By.XPATH, view_all_feature_specs_button_xpath)))
    # Click the element
    view_all_feature_specs_button.click()

    # specs 버튼 클릭
    specs_button_xpath = '//*[@id="features-and-specs-tab-1"]'
    specs_button = WebDriverWait(browser, 100).until(EC.element_to_be_clickable((By.XPATH, specs_button_xpath)))
    specs_button.click()

    # specs
    specs_text_xpath = '//*[@id="features-and-specs-tabpanel-1"]/hzn-stack'
    specs_text = browser.find_elements(By.XPATH, specs_text_xpath)
    # specs_text 에서 줄 바꿈을 '='으로 대체
    specs_table_data = [td.text.replace('\n', '=') for td in specs_text]

    # specs 데이터를 결과에 추가하기
    specs_result.append([specs_table_data])
    # print(specs_table_data)
    # 차종 200종이 되면 멈추기
    if car_number == 200:
        break
    # print(car_herf) # 결과 출력력

## history부분
ages = []  # 연식
accident_damages = []  # 사고 이력
insurance_list = []  # 보험 이력
service_repairs = []  # 서비스 수리 이력

# 중고차 히스토리 이력 부분 브라우저 실행
for cars in car_herf:
    browser = webdriver.Chrome()
    car_url_h = cars + "/vehicle-history"
    browser.get(car_url_h)
    time.sleep(1)

    # 연식
    age_xpath = "//*[@id='three-box-summary']/div[1]/div/div[2]/div[4]/div[2]"
    # find_element_by_xpath를 사용하여 요소를 찾습니다.
    age_element = WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.XPATH, age_xpath)))

    # 사고이력
    ad_history_xpath = "//*[@id='at-glance']/div[2]/div/div[3]/div/div[1]/div/div/p[2]/span"
    # ad_now_xpath="//*[@id='at-glance']/div[2]/div/div[3]/div/div[2]/div[2]/span/span"
    ad_history_element = WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.XPATH, ad_history_xpath)))

    # 보험이력
    # at-glance > div.section-data > div > div:nth-child(5)
    # 'div.at-glance > div.section-data > div > p.subtitle')
    elements = soup.select('div.at-glance > div.section-data > div > div:nth-child(5) p.subtitle')
    if elements:
    # 결과를 리스트에 추가
        p_element = elements.get_text(strip=True)
        insurance_list.append(p_element)
    # 결과가 없으면 "Null" 추가
    else:
        insurance_list.append("Null")

    # 서비스/수리 이력
    sr_history_path = "//*[@id='at-glance']/div[2]/div/div[8]/div/div[1]/div/div/p[2]/span"
    sr_element = WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.XPATH, sr_history_path)))

    # 요소에서 텍스트를 추출
    age = age_element.text
    ad = ad_history_element.text
    sr = sr_element.text

    # 각 텍스트를 리스트에 추가
    ages.append(age)
    accident_damages.append(ad)
    service_repairs.append(sr)

## 이름 가격 마일
npm_result = []

# 중고차 브라우저 실행
for cars in car_herf:
    browser = webdriver.Chrome()
    car_url_npm = cars
    browser.get(car_url_npm)
    time.sleep(1)

    # 이름, 가격, 마일
    name_xpath = "/html/body/main/section[1]/div[3]/div[1]/h1"
    price_xpath = "/html/body/main/section[1]/div[3]/div[1]/h2/span[1]"
    mile_xpath = "/html/body/main/section[1]/div[3]/div[1]/h2/span[3]"

    # find_element_by_xpath를 사용하여 요소를 찾기
    name_element = WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.XPATH, name_xpath)))
    price_element = WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.XPATH, price_xpath)))
    mile_element = WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.XPATH, mile_xpath)))

    # 요소에서 텍스트를 추출
    name = name_element.text
    price = price_element.text
    mile = mile_element.text

    # 각 요소의 텍스트를 딕셔너리로 추가
    npm_result.append({"Name": name, "Price": price, "Mileage": mile, })
    print(npm_result)

## df 만들기
# specs 데이터프레임 만들기
specs_vans_df=pd.DataFrame(specs_result)

# 결과 출력
print(specs_vans_df)

# history 데이터프레임 만들기
history_data = {'ages': ages, 'accident_damages': accident_damages, 'service_repairs': service_repairs}

# 데이터프레임 생성
history_df = pd.DataFrame(history_data)

# 결과 출력
print(history_df)

# 이름, 가격, 마일 데이터프레임
name_price_mile_df=pd.DataFrame(npm_result)

# 결과 출력
print(name_price_mile_df)

## 3개 병합해서 하나의 데이터프레임 만들기
# 데이터프레임 합치기
result_df = pd.concat([specs_vans_df, history_df, name_price_mile_df], axis=1)

# 결과 출력
print(result_df)

# csv 파일로 만들기
result_df.to_csv('crawling_truck.csv') # 파일명 : crawling_차종.csv
