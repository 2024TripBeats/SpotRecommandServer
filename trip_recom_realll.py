import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler


def combined_recommendation(input_order, similarity_df, trip_df, model, master_visit_all, user_prefer, user_features):

    model = joblib.load(model)
    common_index = similarity_df.index.intersection(trip_df.index)
    trip_df = trip_df.loc[common_index]
    similarity_df = similarity_df.loc[common_index]

    # 5개 장소의 tourist_id 정의 (입력받는 순서를 기준으로 사용할 장소들)
    place_ids = ['CNTS_200000000010956', 'CONT_000000000500103', 'CNTS_000000000022353', 'CNTS_000000000022082', 'CNTS_000000000022063']

    # 가중치를 순서대로 정의
    weights = [2.0, 1.5, 0.8, 0.5, 0.3]

    # 입력 순서를 기반으로 place_ids의 순서를 재정렬
    ordered_places = [place_ids[i - 1] for i in input_order]
    ordered_weights = [weights[i - 1] for i in input_order]

    # 선택된 장소들에 가중치를 곱한 유사도 합계를 계산
    weighted_sim = np.zeros(similarity_df.shape[0])
    for place, weight in zip(ordered_places, ordered_weights):
        weighted_sim += similarity_df[place].values * weight

    # 선호 카테고리에 해당하는 관광지 가중치 추가
    trip_df['category_match'] = trip_df['category'].apply(lambda x: 2 if x in user_prefer else 0)

    # weighted_similarity에 선호 카테고리 가중치 추가
    similarity_df['weighted_similarity'] = weighted_sim + trip_df['category_match']

    # tourist_id와 weighted_similarity를 함께 반환
    sorted_df = similarity_df[['tourist_id', 'weighted_similarity']].sort_values(by='weighted_similarity', ascending=False)
    sorted_df = pd.merge(sorted_df, trip_df, on='tourist_id', how='left')
    sorted_df = sorted_df[~sorted_df['tourist_id'].isin(ordered_places)]
    sorted_df.rename(columns={'VISIT_AREA_NM':'Place'}, inplace = True)

    # weighted_similarity 열에 대해 Min-Max Scaling 적용
    scaler = MinMaxScaler()
    sorted_df['weighted_similarity'] = scaler.fit_transform(sorted_df[['weighted_similarity']])


    # 필요한 컬럼만 선택
    master_visit_all = master_visit_all[['VISIT_AREA_NM', 'VISIT_AREA_NM_encoded', 'GENDER', 'AGE_GRP', 'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4', 'total_score']]

    # 유저 입력 데이터 생성
    user_prefer = pd.DataFrame(user_features)

    # 'TRAVEL_STYL_3' 수정
    user_prefer['TRAVEL_STYL_3'] = user_prefer['TRAVEL_STYL_3'].replace({1: '1', 2: '1', 3: '2', 4: '3', 5: '3'}).astype(int)
    user_prefer['TRAVEL_STYL_3'] = user_prefer['TRAVEL_STYL_3'].astype(int)

    # 방문 지역 데이터 준비
    visit_areas = master_visit_all['VISIT_AREA_NM_encoded'].drop_duplicates().dropna(axis=0).tolist()
    repeated_visits = np.tile(visit_areas, len(user_prefer))

    # 오버샘플링된 테스트 데이터프레임 생성
    user_prefer_dict = {
        'GENDER': np.repeat(user_prefer['GENDER'], len(visit_areas)),
        'AGE_GRP': np.repeat(user_prefer['AGE_GRP'], len(visit_areas)),
        'TRAVEL_STYL_1': np.repeat(user_prefer['TRAVEL_STYL_1'], len(visit_areas)),
        'TRAVEL_STYL_2': np.repeat(user_prefer['TRAVEL_STYL_2'], len(visit_areas)),
        'TRAVEL_STYL_3': np.repeat(user_prefer['TRAVEL_STYL_3'], len(visit_areas)),
        'TRAVEL_STYL_4': np.repeat(user_prefer['TRAVEL_STYL_4'], len(visit_areas)),
        'VISIT_AREA_NM_encoded': repeated_visits
    }

    user_prefer = pd.DataFrame(user_prefer_dict).reset_index(drop=True).drop_duplicates()
    user_prefer = user_prefer[['VISIT_AREA_NM_encoded', 'GENDER', 'AGE_GRP', 'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4']]

    # 예측
    predictions = model.predict(user_prefer)
    user_prefer['output'] = predictions

    # tourist_id와 VISIT_AREA_NM의 매핑
    label_map = dict(zip(master_visit_all['VISIT_AREA_NM_encoded'], master_visit_all['VISIT_AREA_NM']))
    user_prefer['VISIT_AREA_NM'] = user_prefer['VISIT_AREA_NM_encoded'].map(label_map)

    result = user_prefer[['VISIT_AREA_NM', 'output']].sort_values(by='output', ascending=False)

    #### 병합하는 부분은 따로 보자
    # 결과 병합
    combined_df = pd.merge(result, sorted_df, left_on='VISIT_AREA_NM', right_on='tourist_x', how='outer')

    # output 열과 weighted_similarity 열을 결합
    combined_df['Combined_weighted_similarity'] = combined_df.apply(
        lambda row: (row['output'] / 2 + row['weighted_similarity'] / 2) if pd.notna(row['output']) and pd.notna(row['weighted_similarity'])
        else (row['output'] if pd.notna(row['output']) else row['weighted_similarity']),
        axis=1)

    # 필요한 열만 선택하고 정렬
    final_recommendations_df = combined_df.rename(columns={'Combined_weighted_similarity': 'score'}).sort_values(by='score', ascending=False)

    return final_recommendations_df
