from fastapi import FastAPI, Request, HTTPException, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn


app = FastAPI()


templates = Jinja2Templates(directory="App/templates")


users = {}  # In-memory storage for users


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/signup")
async def get_signup(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})


@app.post("/signup")
async def post_signup(username: str = Form(...), password: str = Form(...)):
    if username in users:
        raise HTTPException(
            status_code=400, detail="Username already registered")
    users[username] = {"password": password}
    return templates.TemplateResponse("signup.html", {
        "request": Request(scope={'type': 'http'}),  # Creating a dummy request
        "message": f"User {username} registered successfully"
    })


@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    user = users.get(username)
    if not user or user['password'] != password:
        raise HTTPException(
            status_code=401, detail="Invalid username or password")
    return {"username": username, "message": "Login successful"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    # You could save this file to a server or process it
    return {"filename": file.filename, "content_type": file.content_type}


@app.post("/result")
async def result():
    pass


@app.post("/classification")
async def classification():
    pass


@app.post("/generation")
async def generation():
    pass


if __name__ == "__main__":

    uvicorn.run(
        "main:app", host="127.0.0.1", port=8000, log_level="debug", reload=True
    )
