def restaurants_recomm(df, data):

    # 딕셔너리에서 'restaurant'에 해당하는 키워드 추출
    selected_keywords = data.get("restaurant", [])
    required_foods = data.get("requiredRestText", "").split(",")

    if not selected_keywords and not required_foods:
        print("No keywords or food names selected.")
        return df  # 키워드나 음식 이름이 없으면 원본 데이터프레임 반환



    # 'theme' 열에서 음식 이름이 포함된 행 필터링
    df['has_required_food'] = df['theme'].apply(lambda x: any(food in x for food in required_foods))
    
    # 선택된 키워드 열이 True인 행을 필터링
    filtered_df = df[(df[selected_keywords].sum(axis=1) > 0) | (df['has_required_food'] == True)]
    
    # 키워드에 해당하는 열이 True인 개수를 계산 (우선순위 정렬을 위해)
    filtered_df['true_count'] = filtered_df[selected_keywords].sum(axis=1)

    # 음식 이름이 포함된지 여부를 우선순위에 추가
    filtered_df['food_match'] = filtered_df['has_required_food'].astype(int)

    # '리뷰' 점수를 기준으로 정렬을 위한 추가 계산
    filtered_df['total_score'] = filtered_df['리뷰']

    # 먼저 음식 이름이 포함된지 여부, 'true_count'로 정렬한 다음 'total_score'로 정렬
    df_sorted = filtered_df.sort_values(by=['food_match', 'true_count', 'total_score'], ascending=[False, False, False])

    # 필요 없는 임시 컬럼을 제거
    df_sorted.drop(columns=['true_count', 'total_score', 'food_match', 'has_required_food'], inplace=True)

    return df_sorted


