from fastapi import FastAPI, Request, Form, HTTPException, Response, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import Column, String
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base
from sqlalchemy.future import select
from sqlalchemy.orm import sessionmaker
from pathlib import Path
import hashlib
import uvicorn
import logging
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
    return result.scalars().first()


async def user_exists(username: str, session: AsyncSession) -> bool:
    return await get_user_by_username(username, session) is not None


async def add_user(username: str, password: str, session: AsyncSession):
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    user_instance = User(username=username, password=hashed_password)
    session.add(user_instance)
    await session.commit()


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
    if password != confirm_password:
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "error_message": "Passwords do not match"},
        )
    if await user_exists(username, session):
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "error_message": "Username already exists"},
        )
    await add_user(username, password, session)
    return RedirectResponse(url="/login", status_code=302)


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
    if user and user.password == hashlib.sha256(password.encode()).hexdigest():
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
        logger.error("Login failed: Invalid username or password")
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


@app.post("/generation")
async def chat_analysis(
    request: Request, job_title: str = Form(...), text: str = Form(...)
):
    analysis_result = await get_gpt_response(job_title, text)
    return Response(content=analysis_result, media_type="text/plain")


async def get_gpt_response(job_title: str, text: str) -> str:
    prompt = (
        f"자소서 내용 분석 (직무: {job_title}): {text}\n"
        "1. 해당 직무에 필요한 경험, 지원 동기 및 포부, 강점, 단점이 얼마나 잘 매치되는지 분석해줘.\n"
        "2. 누락된 요소나 추가할 내용이 있는지 제안해줘.\n"
        "3. 빈줄없이 출력해줘\n"
        "4. 너의 답변을 그대로 자기소개서에 복사 붙여넣기 할 수 있도록, 자기 소개서 본문만 출력해줘."
    )
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model="gpt-4-turbo"
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error processing GPT response: {str(e)}")
        return f"서버 오류 발생: {str(e)}"


if __name__ == "__main__":

    uvicorn.run("main:app", host="0.0.0.0", port=8888, log_level="debug", reload=True)
