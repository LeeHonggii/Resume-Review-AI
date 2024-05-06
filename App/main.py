from fastapi import FastAPI, Request, HTTPException, Form, File, UploadFile, status, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import hashlib

app = FastAPI()


templates = Jinja2Templates(directory="App/templates")


users = {
    "john_doe": {
        "username": "john_doe",
        # Hashed password
        "password": hashlib.sha256("password123".encode()).hexdigest()
    }
}


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.route("/signup", methods=["GET", "POST"])
async def signup(request: Request, username: str = Form(None), password: str = Form(None)):
    if request.method == "GET":
        return templates.TemplateResponse("signup.html", {"request": request})

    if username in users:
        raise HTTPException(
            status_code=400, detail="Username already registered")

    users[username] = {"password": password}
    return templates.TemplateResponse("signup.html", {
        "request": request,
        "message": f"User {username} registered successfully"
    })


@app.route("/login", methods=["GET", "POST"])
async def login(request: Request):
    if request.method == "GET":
        # If the request method is GET, render the login page
        return templates.TemplateResponse("login.html", {"request": request})

    # Handle POST request for login
    form_data = await request.form()
    username = form_data.get('username')
    password = form_data.get('password')
    print(f"Received username: {username}")
    print(f"Received password: {password}")

    # Check if the user exists and the password is correct
    if username not in users or hashlib.sha256(password.encode()).hexdigest() != users[username]['password']:
        print("Invalid login attempt")
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error_message": "Invalid username or password"
        })

    print("Login successful")
    return JSONResponse(content={"username": username, "message": "Login successful"})


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
