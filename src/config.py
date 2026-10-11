"""Central settings for the whole project. Change things here, not in the other files."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW_CSV = DATA / "raw" / "complaints.csv"      # the CFPB file you downloaded
CFPB_META_CSV = RAW_CSV                        # the same file also gives dates + products for the trends tab
TRENDS_PARQUET = DATA / "trends.parquet"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"

SEED = 42

# What text does the classifier read?  (This download has no narrative column, so we use "issue".)
#   "auto"      -> use the complaint narrative if the CSV really has one, else Issue + Sub-issue
#   "narrative" -> force the narrative column (stops with an error if it is missing)
#   "issue"     -> force Issue + Sub-issue
TEXT_MODE = "issue"

# How product names become category labels:
#   "auto"  -> use LABEL_MAP if the file looks like the CFPB file, else use the file's own product names
#   "map"   -> always use LABEL_MAP        "as_is" -> always use the file's own names (credit_card -> Credit card)
LABEL_MODE = "map"

CAP_PER_LABEL = 5000      # max complaints kept per category (keeps training fast, balances classes)
MIN_CLASS_ROWS = 300      # categories with fewer rows than this are dropped
MIN_WORDS = {"narrative": 5, "issue": 2}   # drop texts shorter than this many words

# The CFPB renamed its products over the years. Old and new names map to one clean label.
LABEL_MAP = {
    "Credit reporting or other personal consumer reports": "Credit reporting",
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
    "Credit reporting": "Credit reporting",
    "Debt collection": "Debt collection",
    "Checking or savings account": "Bank account",
    "Bank account or service": "Bank account",
    "Credit card": "Credit/prepaid card",
    "Credit card or prepaid card": "Credit/prepaid card",
    "Prepaid card": "Credit/prepaid card",
    "Student loan": "Student loan",
    "Vehicle loan or lease": "Vehicle loan",
    "Consumer Loan": "Vehicle loan",
    "Mortgage": "Mortgage",
    "Payday loan, title loan, personal loan, or advance loan": "Payday/personal loan",
    "Payday loan, title loan, or personal loan": "Payday/personal loan",
    "Payday loan": "Payday/personal loan",
    "Money transfer, virtual currency, or money service": "Money transfer",
    "Money transfers": "Money transfer",
    "Virtual currency": "Money transfer",
}
