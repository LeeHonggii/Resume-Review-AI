# linkareer_total.py

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