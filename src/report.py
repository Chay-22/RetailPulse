from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import os

def generate_pdf(title, kpis, insights, recommendations, charts=None, filename="report.pdf"):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    content = []

    # ---------------- LOGO ----------------
    logo_path = "assets/logo.png"  # put logo here

    if os.path.exists(logo_path):
        content.append(Image(logo_path, width=2*inch, height=1*inch))
        content.append(Spacer(1, 10))

    # ---------------- TITLE ----------------
    content.append(Paragraph(title, styles['Heading1']))
    content.append(Spacer(1, 12))

    # ---------------- KPIs ----------------
    content.append(Paragraph("Key Metrics", styles['Heading2']))
    for k, v in kpis.items():
        content.append(Paragraph(f"{k}: {v}", styles['BodyText']))
    content.append(Spacer(1, 10))

    # ---------------- INSIGHTS ----------------
    content.append(Paragraph("Insights", styles['Heading2']))
    for i in insights:
        content.append(Paragraph(f"• {i}", styles['BodyText']))
    content.append(Spacer(1, 10))

    # ---------------- RECOMMENDATIONS ----------------
    content.append(Paragraph("Recommendations", styles['Heading2']))
    for r in recommendations:
        content.append(Paragraph(f"• {r}", styles['BodyText']))
    content.append(Spacer(1, 12))

    # ---------------- CHARTS ----------------
    if charts:
        content.append(Paragraph("Visual Insights", styles['Heading2']))
        content.append(Spacer(1, 10))

        for chart_path in charts:
            if os.path.exists(chart_path):
                content.append(Image(chart_path, width=5*inch, height=3*inch))
                content.append(Spacer(1, 10))

    doc.build(content)
    return filename