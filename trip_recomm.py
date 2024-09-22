def trip_recomm(input_order, similarity_df, trip_df, user_prefer):
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
        # tourist_id 열을 제외하고 유사도 데이터프레임에서 해당 장소의 열을 찾아 가중치를 적용
        weighted_sim += similarity_df[place].values * weight
    
    # 선호 카테고리에 해당하는 관광지 가중치 추가
    trip_df['category_match'] = trip_df['category'].apply(lambda x: 2 if x in user_prefer else 0)
    
    # weighted_similarity에 선호 카테고리 가중치 추가
    similarity_df['weighted_similarity'] = weighted_sim + trip_df['category_match']
    
    # tourist_id와 weighted_similarity, 추가 정보 열을 함께 반환
    sorted_df = similarity_df[['tourist_id', 'weighted_similarity']].sort_values(by='weighted_similarity', ascending=False)
    sorted_df = pd.merge(sorted_df, trip_df, on='tourist_id', how='left')
    
    # ordered_places에 포함된 장소들을 제외하고 출력
    sorted_df = sorted_df[~sorted_df['tourist_id'].isin(ordered_places)]
    
    return sorted_df

# 예시 입력 (2, 4, 3, 1, 5 순서로 입력 받았을 때)
input_order = [2, 4, 3, 1, 5]
user_prefer = ['역사유적지']

# 가중치 계산 후 정렬된 관광지 출력
sorted_places = weighted_similarity_with_preference(input_order, similarity_df, trip_df, user_prefer)

# 결과 출력
display(sorted_places)
