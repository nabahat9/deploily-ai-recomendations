from fastapi import APIRouter, Query
from app.core.recommender import recommend_hybrid

router = APIRouter()  # ✅ this is what main.py expects


@router.get("/recommendations")
def get_recommendations(
    app_id: int = Query(...,
                        description="ID of the app to base recommendations on"),
    user_id: int | None = Query(None, description="Optional user ID"),
    top_n: int = Query(5, description="Number of recommendations to return"),
):
    recs = recommend_hybrid(user_id=user_id, app_id=app_id, top_n=top_n)
    return recs.to_dict(orient="records")
