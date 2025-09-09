from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
from app.routers import recommend
from app.core.precompute import precompute_and_cache

app = FastAPI(title="AI Recommendation API")

# Run job once a day
scheduler = BackgroundScheduler()
scheduler.add_job(precompute_and_cache, "interval", days=1)  # every 1 day
scheduler.start()

# Initial precompute when app starts


@app.on_event("startup")
def startup_event():
    precompute_and_cache()


app.include_router(recommend.router)


@app.get("/")
def read_root():
    return {"message": "Welcome to AI Recommendation API 🚀"}
