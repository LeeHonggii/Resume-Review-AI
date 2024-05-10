from fastapi import (
    FastAPI,
    Request,
    Form,
    HTTPException,
    Depends,
    UploadFile,
    File,
    Response,
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import hashlib
import uvicorn
import logging
from openai import OpenAI
import base64

app = FastAPI()

# 클라이언트 초기화
def create_openai_client():
    api_key = "sk-proj-1T8b3RZsY7loNQXU3kI0T3BlbkFJsnWje1cI7On6EOUqFL85"

    return OpenAI(api_key=api_key)

client = create_openai_client()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory="App/templates")
styles_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(styles_dir)), name="static")

users = {
    "john_doe": {
        "username": "john_doe",
        "password": hashlib.sha256("password123".encode()).hexdigest(),
    }
}


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    logger.debug(
        f"Received login request with username: {username}, password: {password}"
    )

    user = users.get(username)
    if user and user["password"] == hashlib.sha256(password.encode()).hexdigest():
        logger.debug("Login successful: User authenticated")
        response = RedirectResponse(url=f"/{username}", status_code=302)
        response.set_cookie(key="username", value=username, httponly=True)
        return response
    else:
        logger.error("Login failed: Invalid username or password")
        raise HTTPException(status_code=401, detail="Invalid username or password")


@app.post("/logout")
async def logout(response: Response):
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie("username")
    response.delete_cookie("analysis_result")  # 추가된 쿠키 삭제
    return response


@app.get("/{username}", response_class=HTMLResponse)
async def user_page(request: Request, username: str):
    user_cookie = request.cookies.get("username")
    if not user_cookie or user_cookie != username:
        raise HTTPException(status_code=401, detail="Unauthorized")

    user = users.get(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    analysis_result = request.cookies.get("analysis_result", "")
    return templates.TemplateResponse(
        "user_page.html", {"request": request, "user": {"username": username}, "analysis_result": analysis_result}
    )


@app.route("/upload", methods=["POST"])
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    return {"filename": file.filename, "content_type": file.content_type}


@app.route("/result", methods=["POST"])
async def result():
    pass


@app.post("/classification")
async def chat_analysis(request: Request, job_title: str = Form(...), text: str = Form(...)):
    analysis_result = await get_gpt_response(job_title, text)
    return templates.TemplateResponse("user_page.html", {
        "request": request,
        "job_title": job_title,
        "text": text,
        "analysis_result": analysis_result
    })

async def get_gpt_response(job_title: str, text: str) -> str:
    prompt = f"자소서 내용 분석 (직무: {job_title}): {text}\n" \
             "1. 해당 직무에 필요한 경험, 지원 동기 및 포부, 강점, 단점이 얼마나 잘 매치되는지 분석해줘.\n" \
             "2. 누락된 요소나 추가할 내용이 있는지 제안해줘."
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4-turbo"
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error processing GPT response: {str(e)}")
        return f"서버 오류 발생: {str(e)}"


@app.route("/generation", methods=["POST"])
async def generation():
    pass




if __name__ == "__main__":

    uvicorn.run("main:app", host="0.0.0.0", port=8888, log_level="debug", reload=True)
