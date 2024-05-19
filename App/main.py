from sqlalchemy import Column, String, ForeignKey, Integer, Text
from sqlalchemy.orm import relationship
from fastapi import FastAPI, Request, Form, HTTPException, Response, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base
from sqlalchemy.future import select
from sqlalchemy.orm import sessionmaker
from starlette.middleware.sessions import SessionMiddleware
from pathlib import Path
import hashlib
import uvicorn
import logging
import cohere
import datetime
from openai import OpenAI

app = FastAPI()

# Database setup
DATABASE_URL = "sqlite+aiosqlite:///./test.db"
Base = declarative_base()
engine = create_async_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=engine, class_=AsyncSession
)

# OpenAI client setup


def create_openai_client():
    api_key = ""
    return OpenAI(api_key=api_key)


app.add_middleware(
    SessionMiddleware,
    secret_key="your_secret_key_here",
    session_cookie="session_id",
    max_age=3600,  # Session timeout in seconds (1 hour)
)

client = create_openai_client()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Template and static files configuration
templates = Jinja2Templates(directory="App/templates")
app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).parent / "static")),
    name="static",
)

# User model
class User(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True)
    password = Column(String)
    saved_contents = relationship("SavedPageContent", back_populates="user")


class SavedPageContent(Base):
    __tablename__ = "saved_page_content"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, ForeignKey("users.username"))
    job_title = Column(String)
    text = Column(Text)
    result = Column(Text)
    timestamp = Column(String)
    user = relationship("User", back_populates="saved_contents")


# Database initialization
@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# Dependency to get database session
async def get_session() -> AsyncSession:
    async with SessionLocal() as session:
        yield session

# Utility functions for user handling
async def get_user_by_username(username: str, session: AsyncSession) -> User:
    stmt = select(User).where(User.username == username)
    result = await session.execute(stmt)
    user = result.scalars().first()
    if user:
        logger.debug(f"User found: {user.username}")
    else:
        logger.debug(f"User not found: {username}")
    return user


async def user_exists(username: str, session: AsyncSession) -> bool:
    return await get_user_by_username(username, session) is not None


async def add_user(username: str, password: str, session: AsyncSession):
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    user_instance = User(username=username, password=hashed_password)
    session.add(user_instance)
    try:
        await session.commit()
        logger.debug(
            f"User {username} added successfully with password hash {hashed_password}")
    except Exception as e:
        logger.error(f"Error adding user: {str(e)}")
        await session.rollback()

# Route handlers
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/signup", response_class=HTMLResponse)
async def signup_form(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})


@app.post("/signup")
async def handle_signup(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    session: AsyncSession = Depends(get_session),
):
    try:
        # Check if passwords match
        if password != confirm_password:
            logger.debug("Passwords do not match")
            return templates.TemplateResponse(
                "signup.html",
                {"request": request, "error_message": "Passwords do not match"},
            )

        # Check if the username already exists
        if await user_exists(username, session):
            logger.debug(f"Username {username} already exists")
            return templates.TemplateResponse(
                "signup.html",
                {"request": request, "error_message": "Username already exists"},
            )

        # Add the user to the database
        await add_user(username, password, session)
        logger.debug(f"User {username} added successfully")
        return RedirectResponse(url="/login", status_code=302)

    except Exception as e:
        logger.error(f"Error during signup: {str(e)}")
        return templates.TemplateResponse(
            "signup.html",
            {"request": request,
                "error_message": f"An error occurred: {str(e)}"},
        )


@app.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    session: AsyncSession = Depends(get_session),
):
    user = await get_user_by_username(username, session)
    hashed_password = hashlib.sha256(
        password.encode()).hexdigest()  # Hash the input password

    if user:
        logger.debug(
            f"User found: {user.username}, Stored hash: {user.password}, Provided hash: {hashed_password}")
    else:
        logger.debug(f"User not found: {username}")

    if user and user.password == hashed_password:
        request.session["username"] = username  # Store user name in session
        logger.debug("Login successful")
        response = RedirectResponse(url="/user_page", status_code=302)
        response.set_cookie(
            key="username",
            value=username,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="Lax",
        )
        return response
    else:
        logger.error(
            f"Login failed: Invalid username or password for user {username}")
        return HTMLResponse("Invalid username or password", status_code=401)


@app.post("/logout")
async def logout(request: Request):
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie("username")
    request.session.clear()  # Clear session data
    return response


@app.get("/user_page", response_class=HTMLResponse)
async def user_page(request: Request, session: AsyncSession = Depends(get_session)):
    username = request.session.get("username")
    if not username:
        return RedirectResponse(url="/signup", status_code=303)

    user = await get_user_by_username(username, session) if username else None
    if user:
        return templates.TemplateResponse(
            "user_page.html", {"request": request, "user": user}
        )
    else:
        logger.error("Unauthorized access attempt")
        raise HTTPException(status_code=404, detail="User not found")


@app.get("/history", response_class=HTMLResponse)
async def history(request: Request, session: AsyncSession = Depends(get_session)):
    username = request.session.get("username")
    if not username:
        return RedirectResponse(url="/login", status_code=303)

    saved_contents = await session.execute(select(SavedPageContent).where(SavedPageContent.username == username))
    saved_pages = saved_contents.scalars().all()

    return templates.TemplateResponse(
        "history.html", {"request": request, "saved_pages": saved_pages}
    )


@app.get("/saved_page/{page_id}", response_class=HTMLResponse)
async def saved_page_detail(request: Request, page_id: int, session: AsyncSession = Depends(get_session)):
    page_data = await session.execute(select(SavedPageContent).where(SavedPageContent.id == page_id))
    saved_page = page_data.scalars().first()
    
    if not saved_page:
        raise HTTPException(status_code=404, detail="Saved page not found")

    return templates.TemplateResponse(
        "saved_page_detail.html", {"request": request, "saved_page": saved_page}
    )


@app.post("/save_page_content")
async def save_page_content(
    request: Request,
    page_content: dict,
    session: AsyncSession = Depends(get_session)
):
    try:
        new_page = SavedPage(
            username=request.session.get("username"),
            job_title=page_content.get("job_title"),
            text=page_content.get("text"),
            result=page_content.get("result"),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        session.add(new_page)
        await session.commit()
        return {"message": "Page content saved successfully."}
    except Exception as e:
        logger.error(f"Error saving page content: {str(e)}")
        return JSONResponse(status_code=500, content={"message": "Failed to save page content."})



@app.post("/generation")
async def chat_analysis(request: Request):
    data = await request.json()
    job_title = data.get("job_title")
    text = data.get("text")

    gpt_response = await get_gpt_responses(job_title, text) or "실패"
    cohere_response = await get_cohere_responses(job_title, text) or "실패"
    model_response = await get_model_responses(job_title, text) or "실패"

    analysis_results = [
        {"gpt_response": gpt_response},
        {"cohere_response": cohere_response},
        {"model_response": model_response},
    ]

    return JSONResponse(content=analysis_results)


async def get_gpt_responses(job_title: str, text: str) -> str:
    prompt = (
        f"자소서 내용 분석 (직무: {job_title}): {text}\n"
        "1. 해당 직무에 필요한 경험, 지원 동기 및 포부, 강점, 단점이 얼마나 잘 매치되는지 분석해줘.\n"
        "2. 누락된 요소나 추가할 내용이 있는지 제안해줘.\n"
        "3. 빈줄없이 출력해줘\n"
        "4. 너의 답변을 그대로 자기소개서에 복사 붙여넣기 할 수 있도록, 자기 소개서 본문만 출력해줘.\n"
        "5. 상단의 모든 내용을 반복해서 출력하지마"
    )

    try:
        # Correctly formatted messages list
        messages = [{"role": "user", "content": prompt}]

        # Assuming you have set up the client for the OpenAI API
        chat_completion = client.chat.completions.create(
            messages=messages, model="gpt-4o"
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error processing GPT response: {str(e)}")
        return f"서버 오류 발생: {str(e)}"


# co = cohere.Client(api_key="")
def chat2(prompt1, text):
    response = co.chat(
        chat_history=[
            {"role": "USER", "message": prompt1},
        ],
        message=text,
        connectors=[{"id": "web-search"}],
    )
    return response


async def get_cohere_responses(job_title: str, text: str) -> str:
    prompt = (
        f"자소서 내용 분석 (직무: {job_title}): {text}\n"
        "1. 해당 직무에 필요한 경험, 지원 동기 및 포부, 강점, 단점이 얼마나 잘 매치되는지 분석해줘.\n"
        "2. 누락된 요소나 추가할 내용이 있는지 제안해줘.\n"
        "3. 빈줄없이 출력해줘\n"
        "4. 너의 답변을 그대로 자기소개서에 복사 붙여넣기 할 수 있도록, 자기 소개서 본문만 출력해줘.\n"
        "5. 상단의 모든 내용을 반복해서 출력하지마"
    )

    try:
        response = chat2(prompt, text)
        return response.text  # response 객체의 text 속성을 반환
    except Exception as e:
        logger.error(f"Cohere 응답 처리 중 오류 발생: {str(e)}")
        return f"서버 오류 발생: {str(e)}"


# Colab에서 실행된 서버의 ngrok 공개 URL
public_url = (
    "https://e668-34-66-84-102.ngrok-free.app"  # Colab에서 출력된 public_url로 대체
)
url = f"{public_url}/process"


async def get_model_responses(job_title: str, text: str) -> str:
    prompt = "너는 자기소개서를 감별해주는 AI이고 성공 경험, 지원 동기 및 포부, 강점, 단점이 모두 포함되었는지 확인해서 누락된 부분이 있으면 말해주고 단점이 포함된 문장은 수정을 요청할게"

    try:
        responses = []

        # 서버에 보낼 데이터
        data = {"prompt": prompt, "text": text}

        # 서버에 POST 요청 보내기
        response = requests.post(url, json=data)

        if response.status_code == 200:
            responses.append(response.json().get("reply", "응답 없음"))
        else:
            logger.error(f"서버 오류 발생: {response.status_code}")
            responses.append(f"서버 오류 발생: {response.status_code}")

        return responses[0] if responses else "응답 없음"
    except Exception as e:
        logger.error(f"GPT 응답 처리 중 오류 발생: {str(e)}")
        return f"서버 오류 발생: {str(e)}"


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8888,
                log_level="debug", reload=True)
