import pandas as pd
from fpdf import FPDF
import os

os.makedirs("data/raw", exist_ok=True)

# 1. Generate CSV
csv_data = {
    "Employee ID": [101, 102, 103, 104],
    "Name": ["Alice Smith", "Bob Johnson", "Charlie Brown", "Diana Prince"],
    "Department": ["Engineering", "Marketing", "Sales", "Engineering"],
    "Role": ["Senior Developer", "Marketing Manager", "Sales Lead", "DevOps Engineer"],
    "Email": ["alice@company.com", "bob@company.com", "charlie@company.com", "diana@company.com"]
}
df_csv = pd.DataFrame(csv_data)
df_csv.to_csv("data/raw/employee_directory.csv", index=False)
print("Created data/raw/employee_directory.csv")

# 2. Generate XLSX
xlsx_data = {
    "Quarter": ["Q1 2026", "Q2 2026", "Q3 2026"],
    "Revenue (M)": [12.5, 14.2, 18.9],
    "Expenses (M)": [8.1, 9.0, 10.5],
    "Net Profit (M)": [4.4, 5.2, 8.4],
    "Key Highlights": ["Launched Product X", "Expanded to Europe", "Record breaking sales in Asia"]
}
df_xlsx = pd.DataFrame(xlsx_data)
df_xlsx.to_excel("data/raw/q3_financials.xlsx", index=False)
print("Created data/raw/q3_financials.xlsx")

# 3. Generate PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

handbook_text = """
Company Handbook - 2026 Edition

1. Introduction
Welcome to our company! We are dedicated to innovation and excellence. Our core values are Integrity, Teamwork, and Customer Success.

2. Work Hours
Standard working hours are from 9:00 AM to 5:00 PM in your local timezone. We support a hybrid work model, allowing up to 3 days of remote work per week.

3. Vacation Policy
All full-time employees are entitled to 20 days of paid time off (PTO) per year. Please submit your vacation requests at least two weeks in advance.

4. IT Security
Do not share your passwords. Always lock your computer when stepping away from your desk. Phishing is our #1 security threat, so report any suspicious emails to the IT helpdesk immediately.
"""

for line in handbook_text.split('\n'):
    pdf.cell(200, 10, txt=line, ln=True, align='L')

pdf.output("data/raw/company_handbook.pdf")
print("Created data/raw/company_handbook.pdf")
