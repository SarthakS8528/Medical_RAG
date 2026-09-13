import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.hyde import generate_hypothetical_document


query = """
What role does ferroptosis play in pancreatic beta-cell dysfunction
in type 2 diabetes?
"""

hypothetical_document = generate_hypothetical_document(query)

print("\nOriginal Question:")
print(query)

print("\nHypothetical Document:")
print(hypothetical_document)