# linkareer.py

from selenium import webdriver
from selenium.webdriver.common.by import By

def url_crawl(driver:webdriver.Chrome):
    url_list = []
    f=open("C://data/linkcareer_link.txt",'w')
    for page in range(1,573):
        url = "https://linkareer.com/cover-letter/search?page="+str(page)+"&tab=all"
        driver.get(url)
        driver.find_element(By.XPATH,"/html/body/div[1]/div[1]/div/div[4]/div[2]/div/div[3]/div[1]")
        driver.implicitly_wait(3)
        url_tag = driver.find_elements(By.TAG_NAME,'a')
        for tag in url_tag:
            url_name = tag.get_attribute('href')
            if "cover-letter" in url_name and "search" not in url_name:
                print(url_name)
                url_list.append(url_name)
    driver.close()
    for content in list(set(url_list)):
        f.write(content+"\n")
    f.close()

def self_introduction(driver:webdriver.Chrome,url):
    person = {}
    driver.get(url)
    info = driver.find_element(By.XPATH,'//*[@id="__next"]/div[1]/div[4]/div/div[2]/div[1]/div[1]/div/div/div[2]/h1')
    specification=driver.find_element(By.XPATH,'//*[@id="__next"]/div[1]/div[4]/div/div[2]/div[1]/div[1]/div/div/div[3]/p')
    content=driver.find_element(By.ID,"coverLetterContent")
    person['info'] = info.text # 지원자 정보
    person['specification'] = specification.text # 지원자 스펙
    person['self_intro'] = content.text # 지원자 자소서
    print(person)
    return person

from selenium import webdriver
from selenium.webdriver.common.by import By
import linkareer

url="C://data/linkareer_link.txt"
driver = webdriver.Chrome("C:/Program Files/chromedriver/chromedriver")
# linkareer.url_crawl(driver=driver)
f=open(url,'r')
while True: # 11437
    txt_link = f.readline()
    if txt_link=="":
        break
    person = linkareer.self_introduction(driver=driver,url=txt_link)
driver.close()
f.close()