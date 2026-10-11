"""Light text cleaning. Keeps real words, turns CFPB redactions into single tokens."""
import re


def clean(text) -> str:
    t = "" if text is None else str(text).lower()
    t = re.sub(r"x{2,}(?:[/-]x{2,})*", " <redacted> ", t)   # XXXX, XX/XX/XXXX
    t = re.sub(r"\{\$[\d.,]+\}", " <amount> ", t)           # {$1,234.00}
    t = re.sub(r"\s+", " ", t).strip()
    return t
