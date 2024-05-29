# Resume-Review-AI
## NVIDIA-1차-프로젝트-자소서-컨펌
## 자기소개서 필수 항목을 빠뜨리지 않도록 확인하고, 스스로 자기소개서를 점검할 수 있도록 가이드

![LA](/image/LA.png "LA")

## Flowchart
![Flowchart](/image/flowchart.png "flowchart")


## DATACARD
### 회사 인재상 데이터 구조(xls)
| company   | keyword  | sentence |
|-----------|----------|----------|
| Sample Company | Sample Keyword | Sample Sentence |

### 분류 데이터 구조(csv)
| Sentence | Label |
|----------|-------|
| "고등학교 시절 모의고사 수학시험에서 4등급을 받자 누구보다 열정적으로 원인을 분석하고 공부하였습니다." | 성공경험 |

### llm 학습 데이터 구조 (json)
```
{
    "input": "성공경험 값이 들은 자기소개서 작성해줘",
    "output": "대학 시절 연구실에서의 협력을 통해 빠른 시간 안에 충분한 데이터를 확보하여 학술대회에서 성과를 거둔 경험이 있습니다.",
    "instruction": "너는 자기소개서 생성ai 이고 전체을 읽고 요구사항에 맞는 자기소개서을 생성해줘"
}
```



## 프로젝트 구조
- `App/`: 주요 애플리케이션 코드
- `App/initdata` : 가진 데이터 
- `crawling/`: 크롤링 관련 코드
- `image/`: 이미지 파일
- `회의록/`: 회의록 파일
  
```
initdata
├── company.xls # 사전 회사 데이터
├── data_test.csv
├── data_v3.csv # 크롤링 데이터 분류
├── kc_bert_emotion_classifier.pth #모델 pth
├── sample-demo.txt
├── sample.txt
├── vdb 
└── vector.xls 
```

- 
### 설치

1. 리포지토리를 클론합니다:
   ```sh
   git clone https://github.com/LeeHonggii/Resume-Review-AI.git
   ```
2. 가상환경을 설정하고 활성화합니다:
   ```sh
   conda create -n resume python=3.10 
   conda activate resume
   ```
3. 필요한 패키지를 설치합니다:
   ```sh
   pip install -r requirements.txt
   ```
### 사용법

1. OpenAI 키 환경변수 등록:
   ```sh
   export OPENAI_API_KEY='your_openai_api_key'
   ```

2. Cohere 키 환경변수 등록:
   ```sh
   export COHERE_API_KEY='your_cohere_api_key'
   ```

3. 로컬 모델 환경변수 등록:
   ```sh
   export API_URL='your_url'
   ```

4. 프로젝트를 실행하는 방법:
   ```sh
   python App/main.py
   ```

## Demo

![Demo](image/example.gif,"Demo")
