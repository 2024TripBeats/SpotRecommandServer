def cafe_recomm(df, data):
    # 딕셔너리에서 'cafe'에 해당하는 키워드를 추출
    selected_keywords = data.get("cafe", [])

    if not selected_keywords:
        print("No keywords selected for cafe.")
        return df  # 키워드가 없으면 원본 데이터프레임 반환


    # 선택된 키워드 열이 True인 행을 필터링
    filtered_df = df[df[selected_keywords].sum(axis=1) > 0]
    
    # 키워드에 해당하는 열이 True인 개수를 계산 (우선순위 정렬을 위해)
    filtered_df['true_count'] = filtered_df[selected_keywords].sum(axis=1)

    # '리뷰'를 추가하여 정렬에 사용
    filtered_df['total_score'] = filtered_df['리뷰']

    # 먼저 'true_count'로 정렬한 다음 'total_score'로 정렬
    df_sorted = filtered_df.sort_values(by=['true_count', 'total_score'], ascending=[False, False])

    # 필요 없는 임시 컬럼을 제거
    df_sorted.drop(columns=['true_count', 'total_score'], inplace=True)

    return df_sorted




