# Data Card: OFR Financial Stress Index Snapshot

## Source and intended use

The project downloads `fsi.csv` from the U.S. Office of Financial Research
(OFR): <https://www.financialresearch.gov/financial-stress-index/data/fsi.csv>.
OFR describes the FSI as a daily, market-based snapshot of stress in global
financial markets. It aggregates five indicator categories and three regions.

The repository does not redistribute the CSV. The MIT license applies to this
project's source code and documentation, not automatically to the upstream data
or the proprietary inputs from which OFR constructs the index. Users must review
the OFR site and any applicable upstream terms for their own use.

## Frozen reference

- retrieval recorded for the reference run: 2026-08-10
- analysis cutoff: 2026-08-05
- rows: 6,730
- raw file SHA-256:
  `2d4a955fb0d72993fae454a731628d1deb4aca980a19121b989e80de09bf8478`
- canonical cleaned snapshot SHA-256:
  `38535be9eadd819493c3b77e11885deb14e344d97007551f87c76700cc829c9c`

The canonical hash normalizes the selected columns, dates, line endings and
numeric serialization. A live OFR download can contain later rows; preprocessing
truncates at the cutoff and verifies the canonical snapshot before evaluation.
If OFR revises a historical value, verification fails visibly instead of silently
claiming reproduction.

## Fields

`Date` plus nine numeric columns are used: `OFR FSI`, `Credit`,
`Equity valuation`, `Safe assets`, `Funding`, `Volatility`, `United States`,
`Other advanced economies`, and `Emerging markets`.

Cleaning parses and sorts dates, rejects duplicates, checks required columns and
rejects missing required values. No imputation is performed. The target is the
mean of the next five OFR FSI observations minus the current OFR FSI value.

## Known limitations

- The file is a current publication, not a database of historical data vintages.
- OFR methodology and component indicators can change over time.
- Trading days are observations, not equal calendar intervals.
- The nine published series do not cover all drivers of systemic stress.
- Aggregate market data avoids direct personal attributes but does not make an
  eventual downstream decision system automatically fair or safe.
