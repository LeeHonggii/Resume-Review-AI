# jobkorea_protocol.py

import jobkorea
from selenium import webdriver
from selenium.webdriver.common.by import By

file = open("C:/data/jobkorea_link.txt", "w")
driver = webdriver.Chrome("C:/Program Files/chromedriver-win64/chromedriver.exe")
jobkorea.login_protocol(driver=driver)
while True:  # 7354개
    file_url = file.readline()
    if file_url == "":
        break
    jobkorea.self_introduction_crawl(driver=driver, file_url=file_url)
