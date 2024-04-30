# jobkorea.py
from selenium import webdriver
from selenium.webdriver.common.by import By


def link_crawl(driver: webdriver.Chrome):
    array = []
    f = open("C:/data/jobkorea_link.txt", "w")
    for i in range(1, 369):
        driver.get(
            "https://www.jobkorea.co.kr/starter/passassay?schTxt=&Page=" + str(i)
        )
        paper_list = driver.find_element(
            By.XPATH, "/html/body/div[4]/div[2]/div[2]/div[5]/ul"
        )
        driver.implicitly_wait(3)
        urls = paper_list.find_elements(By.TAG_NAME, "a")
        for url in urls:
            if "selfintroduction" in url.get_attribute("href"):
                pass
            else:
                array.append(url.get_attribute("href"))
    array = list(set(array))
    for content in array:
        f.write(content + "\n")
    f.close()


def login_protocol(
    driver: webdriver.Chrome,
):  # 로그인해야지 로그인창때문에 크롤링 멈추는거 막을 수 있음
    driver.get("https://www.jobkorea.co.kr/")
    driver.find_element(
        By.XPATH, "/html/body/div[5]/div/div[1]/div[1]/ul/li[1]/button"
    ).click()
    driver.find_element(By.ID, "lb_id").send_keys("id")
    driver.find_element(By.ID, "lb_pw").send_keys("pw")
    driver.find_element(
        By.XPATH,
        "/html/body/div[5]/div/div[1]/div[1]/ul/li[1]/div/form/fieldset/div[1]/button",
    ).click()
    driver.implicitly_wait(3)
    driver.find_element(By.ID, "closeIncompleteResume")
    driver.implicitly_wait(3)
    print("login success")


def self_introduction_crawl(driver: webdriver.Chrome, file_url):
    print("current URL : " + file_url)
    driver.get(file_url)
    user_info = driver.find_element(
        By.XPATH, '//*[@id="container"]/div[2]/div[1]/div[1]/h2'
    )
    company = user_info.find_element(By.TAG_NAME, "a")
    print(company.text)  # 지원회사
    season = user_info.find_element(By.TAG_NAME, "em")
    print(season.text)  # 지원시기
    specification = driver.find_element(By.CLASS_NAME, "specLists")
    spec_array = specification.text.split("\n")
    print(spec_array[:-2])  # 스펙
    paper = driver.find_element(By.CLASS_NAME, "qnaLists")
    questions = paper.find_elements(By.TAG_NAME, "dt")
    print("question")
    for index in questions:
        question = index.find_element(By.CLASS_NAME, "tx")
        if question.text == "":
            index.find_element(By.TAG_NAME, "button").click()
            question = index.find_element(By.CLASS_NAME, "tx")
            print(question.text)
        else:
            print(question.text)  # 자소서 질문 모아놓은 리스트
    driver.implicitly_wait(3)
    answers = paper.find_elements(By.TAG_NAME, "dd")
    driver.implicitly_wait(3)
    print("answer")
    for index in range(len(answers)):
        answer = answers[index].find_element(By.CLASS_NAME, "tx")
        if answer.text == "":
            questions[index].find_element(By.TAG_NAME, "button").click()
            answer = answers[index].find_element(By.CLASS_NAME, "tx")
        print(answer.text)  # 자소서 답변 모아놓은 리스트
