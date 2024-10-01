def rank_accommodation(user_prefer, acco_df):

    # 조건 충족 점수 초기화
    acco_df['조건 충족 점수'] = 0

    # 1. 기본 조건 - 주차시설 등
    if '주차시설' in user_prefer["accomodation"]:
        acco_df['조건 충족 점수'] += acco_df['주차 가능'].apply(lambda x: 1 if x == '가능' else 0)

    # 부대 시설에서 기본 조건 추가
    basic_facilities = ['사우나', '수영장', '조식', '바베큐장']
    selected_basic_facilities = [facility for facility in basic_facilities if facility in user_prefer['accomodation']]
    
    # NaN 값을 빈 문자열로 대체
    acco_df['부대 시설'] = acco_df['부대 시설'].fillna('')

    for facility in selected_basic_facilities:
        acco_df['조건 충족 점수'] += acco_df['부대 시설'].apply(lambda x: 1 if facility in x else 0)

    # 2. 객실 크기 조건
    if '20평 이상' in user_prefer["accomodation"]:
        acco_df['조건 충족 점수'] += acco_df['객실크기'].apply(lambda x: 1 if x >= 20 else 0)
    
    if '20평 ~ 10평' in user_prefer["accomodation"]:
        acco_df['조건 충족 점수'] += acco_df['객실크기'].apply(lambda x: 1 if 10 < x < 20 else 0)
    
    if '10평 이하' in user_prefer["accomodation"]:
        acco_df['조건 충족 점수'] += acco_df['객실크기'].apply(lambda x: 1 if x <= 10 else 0)

    # 3. 필요 설비 추가 조건
    # 필요 설비가 빈 문자열이 아닌 경우에만 설비 조건 확인
    if user_prefer["requiredAccomText"].strip():
        necessary_facilities = [facility.strip().lower() for facility in user_prefer["requiredAccomText"].split(',')]
    
        for facility in necessary_facilities:
            acco_df['조건 충족 점수'] += acco_df['부대 시설'].apply(lambda x: 1 if facility.lower() in x.lower() else 0)

    # 4. 기준에 따른 가중치 적용 (좋은 품질 vs 가성비)
    if user_prefer["accompriority"] == '가성비':
        # 가성비: 조건 충족 점수와 가격의 비율을 중요시하여 가격이 낮은 것을 우선함
        acco_df['최종 점수'] = acco_df['조건 충족 점수'] / acco_df['비수기주말최소']
    elif user_prefer["accompriority"] == "좋은 품질":
        # 좋은 품질: 충족된 조건 개수를 2배로 중요하게 고려함
        acco_df['최종 점수'] = acco_df['조건 충족 점수'] * 2 - acco_df['비수기주말최소'] / 1000  # 가격은 작아질수록 좋음
    
    # 최종 점수에 따라 내림차순 정렬
    acco_df = acco_df.sort_values(by='최종 점수', ascending=False)

    return acco_df
