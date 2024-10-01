from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from typing import List, Dict
import pandas as pd
from restaurants_recomm import restaurants_recomm
from cafe_recomm import cafe_recomm
from accom_recom import rank_accommodation
from trip_recom_realll import combined_recommendation
from course import generate_recommendation

# 데이터 파일 로드
df_tr = pd.read_csv("./data/trip_df_final_v3.csv")
df_ca = pd.read_csv("./data/cafe_df.csv")
df_re = pd.read_csv("./data/restaurant_df.csv")
df_ac = pd.read_csv("./data/accom_Df.csv")
sim_df = pd.read_csv('./data/similarity_df.csv')
master_visit_all = pd.read_csv('./data/master_visit_all.csv')
model_path = './bayesian_regression.pkl'

app = FastAPI()


# Pydantic 모델 정의
class RestSurvey(BaseModel):
    restaurant: List[str]
    requiredRestText: str
    cafe: List[str]


class AccommodationPreferences(BaseModel):
    accomodation: List[str]
    requiredAccomText: str
    accompriority: str


class UserFeatures(BaseModel):
    GENDER: List[int] = Field(..., alias="gender")
    AGE_GRP: List[int] = Field(..., alias="age_grp")
    TRAVEL_STYL_1: List[int] = Field(..., alias="travel_styl_1")
    TRAVEL_STYL_2: List[int] = Field(..., alias="travel_styl_2")
    TRAVEL_STYL_3: List[int] = Field(..., alias="travel_styl_3")
    TRAVEL_STYL_4: List[int] = Field(..., alias="travel_styl_4")


class RecommendationInput(BaseModel):
    user_prefer: List[str]
    rest_survey: RestSurvey
    acc_prefer: AccommodationPreferences
    user_features: UserFeatures
    input_order: List[int]
    user_trip_days: int
    user_difficulty: List[int]
    user_openness: int
    start_time: str


@app.post("/recommend")
def recommend(input_data: RecommendationInput):
    rest_df = restaurants_recomm(df_re, input_data.rest_survey.dict())
    cafe_df = cafe_recomm(df_ca, input_data.rest_survey.dict())
    acco_df = rank_accommodation(input_data.acc_prefer.dict(), df_ac)

    trip_df = combined_recommendation(
        input_data.input_order, sim_df, df_tr, model_path, master_visit_all,
        input_data.user_prefer, input_data.user_features.dict()
    )

    recommendation_result = generate_recommendation(
        rest_df, cafe_df, acco_df, trip_df,
        input_data.user_trip_days, input_data.user_difficulty, input_data.start_time
    )

    return recommendation_result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
