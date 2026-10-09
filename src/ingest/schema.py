import pandera.pandas as pa

admissions_schema = pa.DataFrameSchema(
    {
        "UNITID": pa.Column(int, pa.Check.ge(100000)),
        "APPLCN": pa.Column(float, pa.Check.ge(0), nullable=True),
        "ADMSSN": pa.Column(float, pa.Check.ge(0), nullable=True),
        "ENRLT": pa.Column(float, pa.Check.ge(0), nullable=True),
        "ADMCON7": pa.Column(float, pa.Check.isin([1, 3, 5]), nullable=True),
        "YEAR": pa.Column(int, pa.Check.in_range(2000, 2100)),
    },
    checks=[
        pa.Check(lambda df: (df["ADMSSN"] <= df["APPLCN"]) | df[["ADMSSN", "APPLCN"]].isna().any(axis=1),
                 error="admits exceed applicants"),
        pa.Check(lambda df: (df["ENRLT"] <= df["ADMSSN"]) | df[["ENRLT", "ADMSSN"]].isna().any(axis=1),
                 error="enrolled exceed admits"),
    ],
    unique=["UNITID", "YEAR"],
)
