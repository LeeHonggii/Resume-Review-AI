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
from fastapi.templating import Jinja2Templates
import hashlib
import uvicorn
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="templates")


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
    return response


@app.get("/{username}", response_class=HTMLResponse)
async def user_page(request: Request, username: str):
    user_cookie = request.cookies.get("username")
    if not user_cookie or user_cookie != username:
        raise HTTPException(status_code=401, detail="Unauthorized")

    user = users.get(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return templates.TemplateResponse(
        "user_page.html", {"request": request, "user": {"username": username}}
    )


@app.route("/upload", methods=["POST"])
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    return {"filename": file.filename, "content_type": file.content_type}


@app.route("/result", methods=["POST"])
async def result():
    pass


@app.route("/classification", methods=["POST"])
async def classification():
    pass


@app.route("/generation", methods=["POST"])
async def generation():
    pass


if __name__ == "__main__":

    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="debug", reload=True)
