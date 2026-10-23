from typing import Literal

from pydantic import BaseModel, Field, model_validator

TestPolicy = Literal[1, 3, 5]


class InstitutionCycle(BaseModel):
    institution_id: str | None = Field(default=None, description="Partner or IPEDS identifier, echoed back.")
    applicants: int = Field(gt=0, description="Applicants this cycle.")
    offers: int = Field(ge=0, description="Offers made or planned this cycle.")
    last_cycle_applicants: int = Field(gt=0)
    last_cycle_offers: int = Field(gt=0)
    last_cycle_yield: float = Field(ge=0, le=1, description="Enrolled divided by offers last cycle.")
    test_policy: TestPolicy = Field(description="IPEDS ADMCON7: 1 required, 3 neither required nor recommended, 5 considered but not required.")
    last_cycle_test_policy: TestPolicy | None = None
    control: Literal[1, 2, 3] = Field(description="1 public, 2 private nonprofit, 3 private for-profit.")
    size_band: Literal[1, 2, 3, 4, 5] = Field(description="IPEDS INSTSIZE enrolment band.")
    locale: int | None = Field(default=None, ge=11, le=43)
    region: int | None = Field(default=None, ge=0, le=9)
    carnegie_class: int | None = Field(default=None, ge=1, le=33)
    hbcu: bool = Field(default=False, description="Used for fairness monitoring only, never as a model input.")

    @model_validator(mode="after")
    def offers_cannot_exceed_applicants(self):
        if self.offers > self.applicants:
            raise ValueError("offers cannot exceed applicants")
        if self.last_cycle_offers > self.last_cycle_applicants:
            raise ValueError("last_cycle_offers cannot exceed last_cycle_applicants")
        return self


class Interval(BaseModel):
    lower: float
    upper: float


class Driver(BaseModel):
    feature: str
    value: float | None
    contribution_points: float


class Forecast(BaseModel):
    institution_id: str | None
    forecast_yield: float
    yield_interval_80: Interval
    expected_enrolled: int
    expected_enrolled_interval_80: Interval
    yield_drop_alert: bool
    fairness_group: str
    top_drivers: list[Driver]
    model_version: str
    approved_for_release: bool
    notice: str
    request_id: str | None = Field(default=None, description="Key for the stored explanation at GET /explanations/{request_id}.")


class Health(BaseModel):
    status: str
    model_version: str
    approved_for_release: bool


class StoredExplanation(BaseModel):
    request_id: str
    created_at: str
    model_version: str
    institution_id: str | None
    fairness_group: str
    inputs: dict
    forecast: dict
    record_hash: str
