import pandas as pd
import pytest

from src.privacy.anonymize import (
    DIRECT_IDENTIFIERS,
    PersonalDataLeak,
    aggregate_partner_extract,
    assert_no_personal_data,
    pseudonymise,
)
from src.privacy.partner_mock import generate_partner_extract
from src.privacy.suppression import suppress_small_cells, suppression_mask

KEY = b"test-key"


def test_single_small_cell_triggers_complementary_suppression():
    parts = pd.DataFrame({"M": [40.0], "W": [55.0], "AN": [3.0], "UN": [0.0]})
    mask = suppression_mask(parts, threshold=10)
    assert mask.loc[0].tolist() == [True, False, True, False]


def test_small_total_hides_every_split():
    frame = pd.DataFrame({"T": [6.0], "M": [4.0], "W": [2.0]})
    published, suppressed = suppress_small_cells(frame, {"T": ["M", "W"]}, threshold=10)
    assert published[["T", "M", "W"]].isna().all(axis=None)
    assert suppressed == 2


def test_pseudonym_is_stable_per_key_and_changes_with_key():
    assert pseudonymise("lagos-campus-a", KEY) == pseudonymise("lagos-campus-a", KEY)
    assert pseudonymise("lagos-campus-a", KEY) != pseudonymise("lagos-campus-a", b"other-key")


def test_aggregates_contain_no_personal_data():
    extract = generate_partner_extract(applicants_per_tenant=200)
    assert set(DIRECT_IDENTIFIERS) <= set(extract.columns)
    aggregates = aggregate_partner_extract(extract, KEY)
    assert_no_personal_data(aggregates)
    assert aggregates["applications"].sum() == len(extract)


def test_leaked_identifier_is_rejected():
    with pytest.raises(PersonalDataLeak):
        assert_no_personal_data(pd.DataFrame({"email": ["a@example.com"]}))
