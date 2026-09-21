from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import os

def generate_pdf(*args, charts=None, filename="report.pdf"):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    content = []

    # -------- LOGO (optional) --------
    logo_path = "assets/logo.png"
    if os.path.exists(logo_path):
        content.append(Image(logo_path, width=2*inch, height=1*inch))
        content.append(Spacer(1, 10))

    # -------- OLD FORMAT --------
    if len(args) == 1:
        text = args[0]
        content.append(Paragraph("RetailPulse Report", styles['Heading1']))
        content.append(Spacer(1, 12))

        for line in str(text).split("\n"):
            content.append(Paragraph(line, styles['BodyText']))
            content.append(Spacer(1, 6))

    # -------- NEW FORMAT --------
    elif len(args) == 4:
        title, kpis, insights, recommendations = args

        content.append(Paragraph(title, styles['Heading1']))
        content.append(Spacer(1, 12))

        content.append(Paragraph("Key Metrics", styles['Heading2']))
        for k, v in kpis.items():
            content.append(Paragraph(f"{k}: {v}", styles['BodyText']))
        content.append(Spacer(1, 10))

        content.append(Paragraph("Insights", styles['Heading2']))
        for i in insights:
            content.append(Paragraph(f"• {i}", styles['BodyText']))
        content.append(Spacer(1, 10))

        content.append(Paragraph("Recommendations", styles['Heading2']))
        for r in recommendations:
            content.append(Paragraph(f"• {r}", styles['BodyText']))
        content.append(Spacer(1, 12))

    # -------- CHARTS --------
    if charts:
        content.append(Paragraph("Visual Insights", styles['Heading2']))
        content.append(Spacer(1, 10))

        for chart in charts:
            if os.path.exists(chart):
                content.append(Image(chart, width=5*inch, height=3*inch))
                content.append(Spacer(1, 10))

    doc.build(content)
    return filename