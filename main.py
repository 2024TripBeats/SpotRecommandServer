from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

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

@app.post("/recommend", response_model=InferenceResponseDto)
async def recommend(request: InferenceRequestDto):
    print(request)
    sample_response = InferenceResponseDto(
        recommendations=[
            DayRecommendation(
                dayNumber=1,
                candidates=[
                    RecommendationCandidate(
                        itinerary=[
                            Place(
                                placeId="1",
                                placeName="Central Park",
                                duration=120
                            ),
                            Place(
                                placeId="2",
                                placeName="Statue of Liberty",
                                duration=180
                            )
                        ],
                        travelSegments=[
                            TravelSegment(
                                distance=5.0
                            )
                        ]
                    ),
                    RecommendationCandidate(
                        itinerary=[
                            Place(
                                placeId="3",
                                placeName="Empire State Building",
                                duration=90
                            ),
                            Place(
                                placeId="4",
                                placeName="Times Square",
                                duration=60
                            )
                        ],
                        travelSegments=[
                            TravelSegment(
                                distance=2.0
                            )
                        ]
                    )
                ]
            ),
            DayRecommendation(
                dayNumber=2,
                candidates=[
                    RecommendationCandidate(
                        itinerary=[
                            Place(
                                placeId="5",
                                placeName="Brooklyn Bridge",
                                duration=110
                            ),
                            Place(
                                placeId="6",
                                placeName="Fifth Avenue",
                                duration=75
                            )
                        ],
                        travelSegments=[
                            TravelSegment(
                                distance=3.5
                            )
                        ]
                    ),
                    RecommendationCandidate(
                        itinerary=[
                            Place(
                                placeId="7",
                                placeName="Broadway",
                                duration=100
                            ),
                            Place(
                                placeId="8",
                                placeName="Wall Street",
                                duration=130
                            )
                        ],
                        travelSegments=[
                            TravelSegment(
                                distance=4.0
                            )
                        ]
                    )
                ]
            )
        ]
    )
    return sample_response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
