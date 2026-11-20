# Data Dictionary — Real Lending Club Sample

| Column              | Type     | Description |
|---------------------|----------|-------------|
| loan_amnt           | numeric  | Requested / funded loan amount (USD) |
| term                | categorical | Loan term in months (36 or 60) |
| annual_inc          | numeric  | Self-reported annual income |
| dti                 | numeric  | Debt-to-income ratio |
| payment_inc_ratio   | numeric  | Monthly payment / monthly income |
| revol_bal           | numeric  | Revolving balance |
| revol_util          | numeric  | Revolving utilisation (%) |
| purpose             | categorical | Loan purpose (e.g. debt_consolidation, credit_card, small_business) |
| home_ownership      | categorical | RENT / MORTGAGE / OWN / OTHER |
| delinq_2yrs_zero    | binary   | 1 if no delinquencies in past 2 years |
| pub_rec_zero        | binary   | 1 if no public records |
| open_acc            | numeric  | Number of open credit lines |
| grade               | numeric  | Numeric encoding of Lending Club grade (higher = better) |
| emp_length          | numeric  | Employment length in years |
| borrower_score      | numeric  | Composite borrower score derived in the cleaned sample |
| default             | binary   | Target: 1 = default / charged-off, 0 = fully paid |

Source: cleaned extract from public Lending Club data via Practical Statistics for Data Scientists materials.

**Download:** `python scripts/download_data.py` → `data/raw/loan_data_lc.csv`
