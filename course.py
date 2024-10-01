from datetime import datetime, timedelta
from geopy.distance import geodesic
import random
import pandas as pd
import re

def calculate_distance(place1, place2):
    coords_1 = get_coordinates(place1)
    coords_2 = get_coordinates(place2)
    if coords_1 is None or coords_2 is None:
        return float('inf')
    return geodesic(coords_1, coords_2).km

def get_coordinates(place):
    if '위도' in place and '경도' in place and pd.notnull(place['위도']) and pd.notnull(place['경도']):
        return (place['위도'], place['경도'])
    elif 'lat' in place and 'lon' in place and pd.notnull(place['lat']) and pd.notnull(place['lon']):
        return (place['lat'], place['lon'])
    elif 'X_COORD' in place and 'Y_COORD' in place and pd.notnull(place['X_COORD']) and pd.notnull(place['Y_COORD']):
        return (place['Y_COORD'], place['X_COORD'])
    else:
        return None

def get_place_name(place):
    if '명칭' in place:
        return place['명칭']
    elif '음식점 이름' in place:
        return place['음식점 이름']
    elif 'tourist_x' in place:
        return place['tourist_x']
    return None

def is_within_distance(last_place, new_place, max_distance):
    distance = calculate_distance(last_place, new_place)
    return distance <= max_distance

# 식당 선택 함수
def select_restaurant(restaurant_df, last_place, visited_places, max_distance):
    sorted_restaurants = restaurant_df.sort_index()
    for _, restaurant in sorted_restaurants.iterrows():
        if get_coordinates(restaurant) is None:
            continue
        if restaurant['id'] not in visited_places and is_within_distance(last_place, restaurant, max_distance):
            return restaurant
    return None

# 카페 선택 함수
def select_cafe(cafe_df, last_place, visited_places, max_distance):
    sorted_cafes = cafe_df.sort_index()
    for _, cafe in sorted_cafes.iterrows():
        if get_coordinates(cafe) is None:
            continue
        if cafe['id'] not in visited_places and is_within_distance(last_place, cafe, max_distance):
            return cafe
    return None

# 관광지 선택 함수
def select_tourist_place(trip_df, last_place, visited_places, max_distance, user_difficulty, total_difficulty, day_categories, candidate_categories):
    difficulty_map = {
      '자연관광지': 2,
      '역사유적지': 2,
      '문화시설': 1,
      '테마파크': 6,
      '레저스포츠': 6
  }
    
    for _, place in trip_df.iterrows():
        if get_coordinates(place) is None:
            continue
        if place['tourist_id'] not in visited_places and is_within_distance(last_place, place, max_distance):
            difficulty = difficulty_map.get(place['category'], 0)
            if total_difficulty + difficulty <= user_difficulty and place['category'] not in day_categories:
                # Prioritize categories not yet visited in the entire itinerary
                if place['category'] not in candidate_categories:
                    return place, difficulty
    
    # If all categories have been visited, select any place that fits the criteria
    for _, place in trip_df.iterrows():
        if get_coordinates(place) is None:
            continue
        if place['tourist_id'] not in visited_places and is_within_distance(last_place, place, max_distance):
            difficulty = difficulty_map.get(place['category'], 0)
            if total_difficulty + difficulty <= user_difficulty and place['category'] not in day_categories:
                return place, difficulty
    
    return None, 0

def determine_start_time(first_day_start):
    if first_day_start == '오전':
        return datetime.strptime("09:00:00", "%H:%M:%S")
    elif first_day_start == '오후':
        return datetime.strptime("14:00:00", "%H:%M:%S")
    elif first_day_start == '저녁':
        return datetime.strptime("22:00:00", "%H:%M:%S")

def generate_day_plan(restaurant_df, cafe_df, trip_df, start_datetime, end_time, user_difficulty, visited_places, cafes_added, current_accommodation, candidate_categories):
    end_time = '22:00:00'
    
    
    difficulty_map = {
        '자연관광지': 2,
        '역사유적지': 2,
        '문화시설': 1,
        '테마파크': 6,
        '레저스포츠': 6
    }
    selected_places = []
    total_difficulty = 0
    current_time = start_datetime
    last_place_type = None
    max_travel_distance = 20
    max_daily_places = 4
    tourist_places_added = 0
    day_categories = set()
    lunch_added = False  # Initialize lunch_added
    dinner_added = False  # Initialize dinner_added

    # 숙소 추가
    selected_places.append({
        'place': current_accommodation,
        'type': '숙소',
        'duration': 0
    })
    last_place = current_accommodation
    visited_places.add(current_accommodation['id'])
    last_place_type = '숙소'
    current_time += timedelta(hours=1)

    end_datetime = datetime.combine(current_time.date(), datetime.strptime(end_time, "%H:%M:%S").time())

    while len(selected_places) < max_daily_places and current_time < end_datetime:
        # 점심 시간 체크 및 식당 추가
        if 11 <= current_time.hour < 13 and not lunch_added:
            restaurant = select_restaurant(restaurant_df, last_place, visited_places, max_travel_distance)
            if restaurant is not None:
                selected_places.append({
                    'place': restaurant,
                    'type': '식당 (점심)',
                    'duration': 1.5
                })
                visited_places.add(restaurant['id'])
                last_place = restaurant
                last_place_type = '식당'
                current_time += timedelta(hours=1.5)
                lunch_added = True
                continue

        # 저녁 시간 체크 및 식당 추가
        if 17 <= current_time.hour < 20 and not dinner_added:
            restaurant = select_restaurant(restaurant_df, last_place, visited_places, max_travel_distance)
            if restaurant is not None:
                selected_places.append({
                    'place': restaurant,
                    'type': '식당 (저녁)',
                    'duration': 1.5
                })
                visited_places.add(restaurant['id'])
                last_place = restaurant
                last_place_type = '식당'
                current_time += timedelta(hours=1.5)
                dinner_added = True
                continue

        # 관광지 추가
        if tourist_places_added < 4:
            tourist_place, difficulty = select_tourist_place(trip_df, last_place, visited_places, max_travel_distance, user_difficulty, total_difficulty, day_categories, candidate_categories)
            if tourist_place is not None:
                selected_places.append({
                    'place': tourist_place,
                    'type': tourist_place['category'],
                    'duration': tourist_place['평균 소요 시간']
                })
                visited_places.add(tourist_place['tourist_id'])
                last_place = tourist_place
                last_place_type = '관광지'
                current_time += timedelta(hours=tourist_place['평균 소요 시간'])
                total_difficulty += difficulty
                tourist_places_added += 1
                day_categories.add(tourist_place['category'])
                candidate_categories.add(tourist_place['category'])
                continue

        # 카페 추가
        if cafes_added < 2 and last_place_type != '카페' and last_place_type != '식당' and random.random() < 0.5:
            cafe = select_cafe(cafe_df, last_place, visited_places, max_travel_distance)
            if cafe is not None:
                selected_places.append({
                    'place': cafe,
                    'type': '카페',
                    'duration': 1.5
                })
                visited_places.add(cafe['id'])
                last_place = cafe
                last_place_type = '카페'
                current_time += timedelta(hours=1.5)
                cafes_added += 1
                continue

        # 시간이 남았지만 더 이상 추가할 장소가 없으면 시간 진행
        current_time += timedelta(hours=0.5)

    # 마지막 숙소 추가
    selected_places.append({
        'place': current_accommodation,
        'type': '숙소',
        'duration': 0
    })

    return selected_places, cafes_added

def force_add_restaurant(selected_places, restaurant_df, last_place, visited_places, max_travel_distance, meal_type):
    restaurant = select_restaurant(restaurant_df, last_place, visited_places, max_travel_distance)
    if restaurant is not None:
        selected_places.append({
            'place': restaurant,
            'type': f'식당 ({meal_type})',
            'duration': 1.5
        })
        visited_places.add(restaurant['id'])

def format_itinerary(itinerary):
    def extract_price_from_text(text):
        if isinstance(text, str):
            match = re.search(r'\d+(\.\d+)?', text.replace(',', ''))
            if match:
                price = float(match.group())  # 첫 번째 숫자를 소수점 포함해 추출
                return round(price)  # 소수점 반올림
        return 0
    
    formatted_itinerary = []
    for day_number, day_plan in enumerate(itinerary, 1):
        if not day_plan:
            continue
        places = []
        travel_segments = []
        last_place = None
        for order, place in enumerate(day_plan, 1):  # 수정된 부분
            # 요금 정보에서 숫자를 추출
            price_info = place['place'].get('요금 정보')
            price = extract_price_from_text(price_info)
            
            # 요금 정보에 숫자가 없을 경우 '비수기주말최소' 값을 가져옴
            if price == 0:
                price = place['place'].get('비수기주말최소', 0) 
                price =  round(price)# '비수기주말최소'가 없으면 0 반환

            place_info = {
                "placeId": str(place['place'].get('id') or place['place'].get('tourist_id')),
                "placeName": get_place_name(place['place']),
                "category": place['type'],
                "duration": int(place['duration'] * 60),
                "order": order,
                "price": price
            }
            places.append(place_info)
            if last_place is not None:
                distance = calculate_distance(last_place, place['place'])
                travel_segments.append({"distance": round(distance, 1)})
            last_place = place['place']
        formatted_itinerary.append({
            "dayNumber": day_number,
            "places": places,
            "travelSegments": travel_segments
        })
    return formatted_itinerary


def generate_recommendation(restaurant_df, cafe_df, accommodation_df, trip_df, user_trip_days, user_difficulty, start_time_option):
    visited_places = set()
    candidates = []
    
    # 시작 시간 설정
    start_time_map = {"오전": "10:00:00", "오후": "14:00:00", "저녁": "22:00:00"}
    first_day_start_time = start_time_map[start_time_option]

    for candidate_num in range(2):
        itinerary = []
        cafes_added = 0
        candidate_categories = set()  # 각 후보에 대한 카테고리 집합 초기화
        
        # 데이터프레임 셔플
        restaurant_df = restaurant_df.sample(frac=1).reset_index(drop=True)
        cafe_df = cafe_df.sample(frac=1).reset_index(drop=True)
        trip_df = trip_df.sample(frac=1).reset_index(drop=True)
        current_accommodation = accommodation_df.iloc[candidate_num % len(accommodation_df)]

        for day in range(user_trip_days):
            if day == 0:
                start_datetime = datetime.strptime(first_day_start_time, "%H:%M:%S")
            else:
                start_datetime = datetime.strptime("09:00:00", "%H:%M:%S")
            
            end_time = "22:00:00"
            
            # 첫째 날이 저녁인 경우 숙소만 추천
            if day == 0 and start_time_option == "저녁":
                day_plan = [{
                    'place': current_accommodation,
                    'type': '숙소',
                    'duration': 0
                }]
            else:
                day_plan, cafes_added = generate_day_plan(restaurant_df, cafe_df, trip_df, start_datetime, end_time, user_difficulty[day], visited_places, cafes_added, current_accommodation, candidate_categories)
            
            itinerary.append(day_plan)

        formatted_itinerary = format_itinerary(itinerary)
        candidates.append({"candidates": candidate_num + 1, "itinerary": formatted_itinerary})

    return {"recommendations": candidates}
'''
start_time_option = "오전"  # "오전", "오후", "밤" 중 선택
recommendation_result = generate_recommendation(df_re, df_ca, df_ac, trip_df, user_trip_days, user_difficulty, start_time_option)

# 결과 출력
print(json.dumps(recommendation_result, indent=4, ensure_ascii=False))
# 실행 예시
'''
