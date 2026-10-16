import numpy as np
import pandas as pd
from faker import Faker

MOCK_TENANTS = {
    "lagos-campus-a": "Lagos",
    "abuja-campus-b": "FCT",
    "enugu-campus-c": "Enugu",
}
PROGRAMMES = ["CSC", "ACC", "LAW", "NUR", "MCM"]
CYCLE = "2026/2027"
FUNNEL_PROBABILITIES = {"offered": 0.55, "accepted": 0.6, "enrolled": 0.85}


def generate_partner_extract(applicants_per_tenant: int = 400, seed: int = 2026) -> pd.DataFrame:
    """Synthetic applicant export used only to exercise PII removal. It never enters model training."""
    fake = Faker("en_NG")
    Faker.seed(seed)
    rng = np.random.default_rng(seed)
    records = []
    for tenant_id, state in MOCK_TENANTS.items():
        for _ in range(applicants_per_tenant):
            offered = rng.random() < FUNNEL_PROBABILITIES["offered"]
            accepted = offered and rng.random() < FUNNEL_PROBABILITIES["accepted"]
            enrolled = accepted and rng.random() < FUNNEL_PROBABILITIES["enrolled"]
            records.append({
                "tenant_id": tenant_id,
                "cycle": CYCLE,
                "programme_code": rng.choice(PROGRAMMES),
                "full_name": fake.name(),
                "email": fake.email(),
                "phone_number": fake.phone_number(),
                "nin": "".join(str(d) for d in rng.integers(0, 10, size=11)),
                "date_of_birth": fake.date_of_birth(minimum_age=16, maximum_age=35).isoformat(),
                "home_address": fake.address().replace("\n", ", "),
                "state_of_origin": state,
                "offered": offered,
                "accepted": accepted,
                "enrolled": enrolled,
            })
    return pd.DataFrame.from_records(records)
