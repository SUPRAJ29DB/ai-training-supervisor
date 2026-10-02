"""Reports and evaluation API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.schemas import EvaluationResponse
from database.db import get_db
from database.repositories import EvaluationRepository

router = APIRouter()


@router.get("/experiment/{experiment_id}", response_model=EvaluationResponse)
def get_evaluation(experiment_id: int, db: Session = Depends(get_db)):
    result = EvaluationRepository(db).get_for_experiment(experiment_id)
    if not result:
        raise HTTPException(404, f"No evaluation found for experiment {experiment_id}.")
    return result
