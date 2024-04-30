from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import json


def url_crawl(driver: webdriver.Chrome):
    url_list = []
    f = open("C:/Users/hancomtst/Desktop/Link.txt", "w")
    for page in range(1, 573):
        url = (
            "https://linkareer.com/cover-letter/search?page="
            + str(page)
            + "&sort=SCRAP_COUNT&tab=all"
        )
        driver.get(url)
        driver.implicitly_wait(3)
        url_tags = driver.find_elements(By.TAG_NAME, "a")
        for tag in url_tags:
            url_name = tag.get_attribute("href")
            if "cover-letter" in url_name and "search" not in url_name:
                url_list.append(url_name)
    driver.close()
    # Writing unique URLs to file
    unique_urls = set(url_list)
    for content in unique_urls:
        f.write(content + "\n")
    f.close()


def self_introduction(driver, url):
    person = {}
    driver.get(url)
    info = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(
            (
                By.XPATH,
                '//*[@id="__next"]/div[1]/div[4]/div/div[2]/div[1]/div[1]/div/div/div[2]/h1',
            )
        )
    )
    try:
        specification = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    '//*[@id="__next"]/div[1]/div[4]/div/div[2]/div[1]/div[1]/div/div/div[3]/p',
                )
            )
        )
    except TimeoutException:
        specification = None

    content = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "coverLetterContent"))
    )

    person["info"] = info.text if info else "Not Available"
    person["specification"] = specification.text if specification else "Not Available"
    person["self_intro"] = content.text if content else "Not Available"

    return person


# Main script execution
service = Service("C:/Program Files/chromedriver-win64/chromedriver.exe")
driver = webdriver.Chrome(service=service)

# # Uncomment Crawl URLs first
# url_crawl(driver)

# Then read URLs from Link.txt and write output to Link__1.txt in dictionary form
with open("C:/Users/hancomtst/Desktop/Link.txt", "r") as f, open(
    "C:/Users/hancomtst/Desktop/Link__1.txt", "w"
) as out_file:
    for txt_link in f:
        txt_link = txt_link.strip()
        if txt_link == "":
            continue
        person = self_introduction(driver, txt_link)
        json.dump({"URL": txt_link, "Details": person}, out_file)
        out_file.write("\n")  # Add a newline to separate each dictionary entry

driver.close()
