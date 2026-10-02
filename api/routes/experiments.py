"""Experiments API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.schemas import (
    ExperimentCreateRequest, ExperimentResponse,
    RecommendationResponse, RecommendationStatusUpdate,
)
from database.db import get_db
from database.repositories import ExperimentRepository, RecommendationRepository

router = APIRouter()


@router.get("/", response_model=list[ExperimentResponse])
def list_experiments(db: Session = Depends(get_db)):
    return ExperimentRepository(db).list_all()


@router.post("/", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
def create_experiment(request: ExperimentCreateRequest, db: Session = Depends(get_db)):
    exp = ExperimentRepository(db).create(
        name=request.name,
        description=request.description,
        framework=request.framework,
        model_type=request.model_type,
        hyperparams=request.hyperparams,
        dataset_ref=request.dataset_ref,
        tags=request.tags,
    )
    return exp


@router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(experiment_id: int, db: Session = Depends(get_db)):
    exp = ExperimentRepository(db).get(experiment_id)
    if not exp:
        raise HTTPException(404, f"Experiment {experiment_id} not found.")
    return exp


@router.get("/{experiment_id}/recommendations", response_model=list[RecommendationResponse])
def get_recommendations(experiment_id: int, db: Session = Depends(get_db)):
    return RecommendationRepository(db).list_all()


@router.post("/{experiment_id}/analyse")
def analyse_experiment(experiment_id: int):
    """Trigger AI advisor analysis for the experiment."""
    from advisor.advisor import Advisor
    advisor = Advisor()
    recs = advisor.analyse_experiment(experiment_id)
    return {"experiment_id": experiment_id, "recommendations_created": len(recs)}


@router.patch("/recommendations/{rec_id}", response_model=RecommendationResponse)
def update_recommendation_status(
    rec_id: int,
    body: RecommendationStatusUpdate,
    db: Session = Depends(get_db),
):
    rec = RecommendationRepository(db).update_status(rec_id, body.status)
    if not rec:
        raise HTTPException(404, f"Recommendation {rec_id} not found.")
    return rec
