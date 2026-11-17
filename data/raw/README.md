# Raw Data

## Preferred: Real Lending Club sample

**File:** `loan_data_lc.csv`

**How to get it (recommended):**

```bash
# From the project root
python scripts/download_data.py
```

This downloads the cleaned sample from:

https://github.com/gedeck/practical-statistics-for-data-scientists/raw/master/data/loan_data.csv.gz

(originally derived from public Lending Club loan data releases) and saves it as `loan_data_lc.csv`.

**Manual download:**
1. Open the URL above.
2. Save the `.gz` file into this folder as `loan_data_lc.csv.gz`.
3. Decompress it, or re-run `python scripts/download_data.py`.

## Fallback: Synthetic data

If you cannot download the real file (offline, firewall, etc.):

```bash
python -m src.data.generate_data
```

This creates `loan_data.csv` (synthetic but realistic). The training code auto-detects which file is present.

## Attribution

- Real data ultimately comes from Lending Club public loan data.
- The cleaned sample used here is distributed via the Practical Statistics for Data Scientists materials.
- Respect the original terms of use of any upstream source.
