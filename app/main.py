from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from . import scheduler, sentiment, fraud, paymongocheckout, updatebookingstatus
from fastapi.responses import FileResponse, HTMLResponse
from fastapi import Request, BackgroundTasks

app = FastAPI()

app.include_router(scheduler.router, prefix="/api/scheduler")
app.include_router(sentiment.router, prefix="/api/sentiment")
app.include_router(fraud.router, prefix="/api/fraud")
app.include_router(paymongocheckout.router, prefix="/api/paymongocheckout")
app.include_router(updatebookingstatus.router, prefix="/api/updatebookingstatus")

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/payment-success")
def payment_success(background_tasks: BackgroundTasks):
    webhook_url = "https://hook.us2.make.com/piply2miomfbdrlpo17huvbi95ln2e72"
    import requests
    background_tasks.add_task(requests.get, webhook_url)
    return FileResponse("static/payment_success.html")

@app.get("/payment-failed")
def payment_failed(request: Request):
    redirect_url = request.query_params.get("redirect", "https://default-url.com")
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Payment Failed</title>
        <style>
            body {{ background: #fff; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; }}
            .btn {{ background: #007bff; color: #fff; padding: 12px 24px; border: none; border-radius: 4px; font-size: 18px; cursor: pointer; text-decoration: none; }}
            .btn:hover {{ background: #0056b3; }}
            h1 {{ color: #dc3545; }}
        </style>
    </head>
    <body>
        <h1>Payment Failed</h1>
        <a href="{redirect_url}" class="btn">Back to App</a>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
