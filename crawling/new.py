from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import json
import os


def fetch_article_content(driver, url):
    driver.get(url)
    data = {}
    try:
        content_element = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "coverLetterContent"))
        )
        data["content"] = content_element.text
    except TimeoutException:
        data["content"] = "Content could not be loaded."
    return data


# Setup WebDriver
service = Service("C:/Program Files/chromedriver-win64/chromedriver.exe")
driver = webdriver.Chrome(service=service)

# Ensure the file path and permissions are correct
file_path = "crawling/LINKA_2.txt"
output_data = []

# Read URLs from the text file
with open("crawling/link_sorted_400.txt", "r") as file:
    for line in file:
        url = line.strip()
        # Fetch the article content for each URL
        article_data = fetch_article_content(driver, url)
        output_data.append({"URL": url, "Content": article_data})

# Write the fetched contents to a file
try:
    with open(file_path, "w") as out_file:
        json.dump(output_data, out_file, ensure_ascii=False, indent=4)
    print("File written successfully.")
except Exception as e:
    print(f"Error writing to the file: {e}")

driver.quit()
