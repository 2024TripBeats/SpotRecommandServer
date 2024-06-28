import joblib
import requests
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from sklearn.preprocessing import MinMaxScaler
from pycaret.regression import predict_model

app = FastAPI()

# CSV 파일 경로
SIMILARITY_DF_PATH = 'extracted_sim_final.csv'
MASTER_VISIT_ALL_PATH = 'master_visit_all.csv'
PLACE_CSV_PATH = 'place.csv'
MODEL_PATH = 'gbr_model.pkl'

# CSV 데이터 읽기
similarity_df = pd.read_csv(SIMILARITY_DF_PATH)
master_visit_all = pd.read_csv(MASTER_VISIT_ALL_PATH)
df = pd.read_csv(PLACE_CSV_PATH)
#similarity_df.set_index('Unnamed: 0', inplace=True)

# 데이터 출력
print("Similarity DataFrame:")
print(similarity_df.head())

print("Master Visit All DataFrame:")
print(master_visit_all.head())

print("Place DataFrame:")
print(df.head())

# 모델 로드
model = joblib.load(MODEL_PATH)


# 입력 데이터 모델 정의
class TravelSegment(BaseModel):
    distance: float


class Place(BaseModel):
    placeId: str
    placeName: str
    duration: int


class RecommendationCandidate(BaseModel):
    itinerary: List[Place]
    travelSegments: List[TravelSegment]


class DayRecommendation(BaseModel):
    dayNumber: int
    candidates: List[RecommendationCandidate]


class InferenceResponseDto(BaseModel):
    recommendations: List[DayRecommendation]


class InferenceRequestDto(BaseModel):
    accountId: int
    destination: str
    period: int
    intensity: List[int]
    stopwords: str
    requirewords: str
    gender: str
    age: str
    distance: str
    activityLevel: str
    scene: str
    openness: int
    musicGenres: List[str]
    genreOpenness: int
    musicTags: List[str]
    tagOpenness: int
    travelSpots: List[int]


def recommendation(travelSpots, travel_style):
    travel_style['TRAVEL_STYL_3'] = [1 if x in [1, 2] else 2 if x == 3 else 3 for x in travel_style['TRAVEL_STYL_3']]
    visit_areas = master_visit_all['VISIT_AREA_NM'].drop_duplicates().dropna(axis=0).tolist()
    repeated_visits = np.tile(visit_areas, len(travel_style['GENDER']))

    testing_dict = {
        'GENDER': np.repeat(travel_style['GENDER'], len(visit_areas)),
        'AGE_GRP': np.repeat(travel_style['AGE_GRP'], len(visit_areas)),
        'TRAVEL_STYL_1': np.repeat(travel_style['TRAVEL_STYL_1'], len(visit_areas)),
        'TRAVEL_STYL_2': np.repeat(travel_style['TRAVEL_STYL_2'], len(visit_areas)),
        'TRAVEL_STYL_3': np.repeat(travel_style['TRAVEL_STYL_3'], len(visit_areas)),
        'TRAVEL_STYL_4': np.repeat(travel_style['TRAVEL_STYL_4'], len(visit_areas)),
        'VISIT_AREA_NM': repeated_visits
    }

    testing = pd.DataFrame(testing_dict).reset_index(drop=True).drop_duplicates()
    result = predict_model(model, data=testing)
    scaler = MinMaxScaler()
    result['output'] = scaler.fit_transform(result[['prediction_label']])
    result = result[['VISIT_AREA_NM', 'output']].sort_values(by='output', ascending=False)

    place_to_number = {
        '안흥지 애련정': 1,
        'KT&G상상마당 홍대': 2,
        '명동난타극장': 3,
        '백운계곡관광지': 4,
        '소래역사관': 5
    }

    number_to_place = {v: k for k, v in place_to_number.items()}
    user_preferences = [number_to_place[num] for num in travelSpots]

    weights = [1.0, 0.8, 0.5, 0.1, 0.05]
    score_dict = {place: weights[i] for i, place in enumerate(user_preferences)}
    total_scores = similarity_df.apply(lambda row: sum(score_dict.get(col, 0) * row[col] for col in user_preferences),
                                       axis=1)
    total_scores_scaled = scaler.fit_transform(total_scores.values.reshape(-1, 1)).flatten()
    recommendations_dict = dict(zip(total_scores.index, total_scores_scaled))
    recommendations_dict = dict(sorted(recommendations_dict.items(), key=lambda item: item[1], reverse=True))
    recommendations_df = pd.DataFrame(list(recommendations_dict.items()), columns=['Place', 'Score'])
    combined_df = pd.merge(result, recommendations_df, left_on='VISIT_AREA_NM', right_on='Place', how='outer')

    combined_df['VISIT_AREA_NM'] = combined_df.apply(
        lambda row: row['VISIT_AREA_NM'] if pd.notna(row['VISIT_AREA_NM']) else row['Place'], axis=1)
    combined_df['Combined_Score'] = combined_df.apply(
        lambda row: (row['output'] / 2 + row['Score'] / 2) if pd.notna(row['output']) and pd.notna(row['Score']) else (
            row['output'] if pd.notna(row['output']) else row['Score']), axis=1)
    final_recommendations_df = combined_df[['VISIT_AREA_NM', 'Combined_Score']].rename(
        columns={'Combined_Score': 'Score'}).sort_values(by='Score', ascending=False)
    final_recommendations = final_recommendations_df.set_index('VISIT_AREA_NM')['Score'].to_dict()

    return final_recommendations


def generate_itinerary(df, similarity_dict, user_difficulty, user_openness, user_days):
    difficulty_map = {
        1: 2,  # 자연관광지
        2: 2,  # 역사
        3: 1,  # 문화시설
        4: 3,  # 상업지구
        5: 5,  # 레저, 스포츠 관련 시설
        6: 5,  # 놀이공원
        7: 2,  # 산책로
        8: 4  # 지역축제
    }

    recommendations = []
    used_lower_similarity_place = False  # 사용자의 개방도에 따른 유사도가 낮은 장소 추천 여부

    for day in range(1, user_days + 1):
        day_plan = {
            "dayNumber": day,
            "candidates": []
        }

        max_difficulty = user_difficulty[day - 1] * 2 + 2
        total_difficulty = 0

        for _ in range(2):  # 각 날에 대해 두 개의 후보 경로 생성
            selected_places = []
            categories = set()
            total_difficulty = 0

            # 시작 지점 선택
            start_place = df.sample().iloc[0]
            selected_places.append(start_place)
            total_difficulty += difficulty_map[start_place['VISIT_AREA_TYPE_CD']]
            categories.add(start_place['VISIT_AREA_TYPE_CD'])

            travel_segments = []

            # 탐욕 알고리즘을 사용하여 경로 생성
            for _ in range(4):
                best_place = None
                best_score = float('inf')

                for _, place in df.iterrows():
                    if place['VISIT_AREA_NM'] in [p['VISIT_AREA_NM'] for p in selected_places]:
                        continue
                    distance = geodesic((selected_places[-1]['lat'], selected_places[-1]['lng']),
                                        (place['lat'], place['lng'])).km
                    difficulty = difficulty_map[place['VISIT_AREA_TYPE_CD']]
                    if total_difficulty + difficulty > max_difficulty:
                        continue
                    similarity = similarity_dict.get(place['VISIT_AREA_NM'], 0)

                    if not used_lower_similarity_place and similarity < user_openness:
                        used_lower_similarity_place = True
                    else:
                        score = distance + (1 - similarity)
                        if score < best_score:
                            best_score = score
                            best_place = place

                if best_place is None:
                    break

                selected_places.append(best_place)
                total_difficulty += difficulty_map[best_place['VISIT_AREA_TYPE_CD']]
                categories.add(best_place['VISIT_AREA_TYPE_CD'])

                if len(selected_places) > 1:
                    distance = geodesic((selected_places[-2]['lat'], selected_places[-2]['lng']),
                                        (selected_places[-1]['lat'], selected_places[-1]['lng'])).km
                    travel_segments.append({"distance": distance})

            itinerary = [{"placeId": str(df[df['VISIT_AREA_NM'] == place['VISIT_AREA_NM']].index[0]),
                          "placeName": place['VISIT_AREA_NM'], "duration": 120} for place in selected_places]

            day_plan["candidates"].append({
                "itinerary": itinerary,
                "travelSegments": travel_segments
            })

        recommendations.append(day_plan)

    return {"recommendations": recommendations}

def gpt_api_zero(api_key, stopwords, requirewords, extracted_sim_final, master_visit_all):
    # DataFrame을 JSON 형식으로 변환
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    message = [
        {
            "role": "user",
            "content": "category_mapping = {'자연': '1', '숲': '1', '산책로': '7', '쇼핑': '4', '역사': '2', '유적': '2', '종교시설': '2', '문화': '3', '공연장': '3', '영화관': '3', '전시관': '3', '상업지구': '4', '시장': '4', '레저': '5', '스포츠': '5', '테마시설': '6', '놀이공원': '6', '워터파크': '6', '축제': '8', '행사': '8'}"
        },
        {
            "role": "user",
            "content": 'stopwords:' + stopwords
        },
        {
            "role": "user",
            "content": "위 stopwords에 대하여 category_mapping을 참고하여 몇번 카테고리를 제외해야 할까? \n답변에는 카테고리 숫자 한가지만 포함해줘. 절대 카테고리 숫자외에 어떠한 말도 답변에 포함시키지마.\n 만약 stop words의 내용이 어떠한 카테고리와도 관련이 없거나 아예 내용이 없다면 아무 내용도 답하지 말아줘"
        },
        {
            "role": "user",
            "content": 'requirewords:' + requirewords
        },
        {
            "role": "user",
            "content": "위 requirewords에 대하여 category_mapping을 참고하여 몇번 카테고리를 우선적으로 넣어야 할까? \n답변에 카테고리 숫자 한가지만 포함해줘. \n절대 카테고리 숫자외에 어떠한 말도 답변에 포함시키지마. 만약 require words의 내용이 어떠한 카테고리와도 관련이 없거나 아예 내용이 없다면 아무 내용도 답하지 말아줘."
        },
        {
            "role": "user",
            "content": "너는 requirewords를 통해 넣어야 할 카테고리 숫자와 stopwords를 통해 제외해야 할 카테고리 숫자 2개를 답변해야 해. \n각각의 숫자는 ,를 통해 구분해줘"
        }
    ]

    payload = {
        "model": "gpt-4",
        "messages": message,
        "max_tokens": 1000,
        "temperature": 0
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
        response_json = response.json()
        content = response_json['choices'][0]['message']['content']
        split_content = content.split(',')
        stop = split_content[0]
        require = split_content[1]
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        stop, require = None, None

    if stop and require:
        categories_to_remove = stop.split(',')
        categories_to_increase = require.split(',')

        # Stopwords 처리
        original_places = master_visit_all['VISIT_AREA_NM'].tolist()
        master_visit_all = master_visit_all[~master_visit_all['VISIT_AREA_TYPE_CD'].isin(categories_to_remove)]

        # extracted_sim_final에서 행 제거
        removed_places = master_visit_all['VISIT_AREA_NM'].tolist()
        to_remove = [item for item in original_places if item not in removed_places]
        extracted_sim_final = extracted_sim_final[~extracted_sim_final['Unnamed: 0'].isin(to_remove)]

        # requirewords 처리
        master_visit_all.loc[master_visit_all['VISIT_AREA_TYPE_CD'].isin(categories_to_increase), 'total_score'] += 5

        # 전처리
        master_visit_all = master_visit_all.drop(columns='Unnamed: 0')
        extracted_sim_final.set_index('Unnamed: 0', inplace=True)

    return master_visit_all, extracted_sim_final

@app.post("/recommend", response_model=InferenceResponseDto)
async def recommend(request: InferenceRequestDto):
    # Apply preprocessing using GPT API
    global master_visit_all, similarity_df
    api_key = ""
    master_visit_all, similarity_df = gpt_api_zero(api_key, request.stopwords, request.requirewords, similarity_df, master_visit_all)

    if request.gender == '1':
        gender = 0
    elif request.gender == '2':
        gender = 1
    else:
        raise HTTPException(status_code=400, detail="Invalid gender value")

    travel_style = {
        "GENDER": [int(gender)],
        "AGE_GRP": [int(request.age)],
        "TRAVEL_STYL_1": [int(request.scene)],
        "TRAVEL_STYL_2": [int(request.distance)],
        "TRAVEL_STYL_3": [int(request.openness)],
        "TRAVEL_STYL_4": [int(request.activityLevel)],
    }

    travel_spots = request.travelSpots
    final_recommendations = recommendation(travelSpots=travel_spots, travel_style=travel_style)

    user_trip_days = request.period
    user_difficulty = request.intensity
    user_openness = request.openness

    itinerary = generate_itinerary(df, final_recommendations, user_difficulty, user_openness, user_trip_days)
    return itinerary


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)