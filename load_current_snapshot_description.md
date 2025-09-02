# Description of `load_current_snapshot` DataFrame

## Function Purpose
The `load_current_snapshot` function loads **current prospects** for lead generation scoring. It retrieves data for companies that are **non-members TODAY** and could potentially become members in the future.

## Key Characteristics

### Data Scope
- **Temporal**: Uses current date (`GETDATE()`) as the snapshot date
- **Target Population**: Non-members as of today who are eligible prospects
- **Purpose**: Scoring/ranking for lead generation (no historical labels needed)

### Filtering Logic
The function applies three main filters to identify viable prospects:

1. **Non-Members Filter**: 
   ```sql
   (b.Eintritt IS NULL OR b.Eintritt > s.snapshot_date)
   ```
   - Companies that never joined (`Eintritt IS NULL`)
   - OR companies that joined after today (future joiners)

2. **Company Existence Filter**:
   ```sql
   (b.Gruendung_Jahr IS NULL OR b.Gruendung_Jahr <= YEAR(s.snapshot_date))
   ```
   - Only companies founded on or before current year
   - Excludes companies with unrealistic future founding dates

3. **Active Company Filter**:
   ```sql
   (b.DT_LoeschungAusfall IS NULL 
    OR b.DT_LoeschungAusfall = '1888-12-31' 
    OR b.DT_LoeschungAusfall > s.snapshot_date)
   ```
   - Excludes companies that were deleted/bankrupt before today
   - `1888-12-31` is used as a NULL sentinel value

## Returned DataFrame Structure

### Company Identification
- **CrefoID**: Unique company identifier
- **Name_Firma**: Company name

### Temporal Information  
- **snapshot_date**: Current date (same for all rows)
- **Gruendung_Jahr**: Company founding year

### Geographic Information
- **PLZ**: Postal code
- **Kanton**: Canton/region
- **Ort**: City/location

### Company Classification
- **Rechtsform**: Legal form of company
- **RechtsCode**: Legal form code
- **BrancheText_06/04/02**: Industry descriptions (different classification levels)
- **BrancheCode_06/04/02**: Industry codes (different classification levels)

### Size & Financial Metrics
- **MitarbeiterBestand**: Number of employees
- **MitarbeiterBestandKategorie**: Employee count category
- **MitarbeiterBestandKategorieOrder**: Employee category ordering
- **Umsatz**: Revenue/turnover
- **UmsatzKategorie**: Revenue category  
- **UmsatzKategorieOrder**: Revenue category ordering
- **GroessenKategorie**: Size category
- **V_Bestand_Kategorie**: Inventory/stock category

### Risk Assessment
- **Risikoklasse**: Risk classification

### Membership Information (for reference)
- **Eintritt**: Entry/join date (NULL or future date for prospects)
- **Austritt**: Exit date (if applicable)
- **DT_LoeschungAusfall**: Deletion/bankruptcy date

## Usage Context
This dataframe is used in the lead generation pipeline to:
1. **Score prospects** using the trained model
2. **Rank companies** by conversion probability
3. **Generate lead lists** for sales teams
4. **Export rankings** to CSV and database tables

## Data Quality Notes
- All rows have the same `snapshot_date` (current date)
- No `Target` column (unlike training data) since these are current prospects
- Companies are pre-filtered to exclude inactive/invalid entities
- Date columns are automatically parsed during SQL query execution

## Expected Output Size
The function typically returns thousands to tens of thousands of prospect companies, depending on the size of the company database and filtering criteria.
