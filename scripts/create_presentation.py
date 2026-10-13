from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pathlib import Path

# Create presentation
prs = Presentation()
prs.slide_width = Inches(10)
prs.slide_height = Inches(7.5)

# Define colors
DARK_BLUE = RGBColor(46, 134, 171)
DARK_MAGENTA = RGBColor(162, 59, 114)
LIGHT_GRAY = RGBColor(240, 240, 240)

def add_title_slide(prs, title, subtitle):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = DARK_BLUE

    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(2.5), Inches(9), Inches(1.5))
    title_frame = title_box.text_frame
    title_frame.word_wrap = True
    p = title_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(54)
    p.font.bold = True
    p.font.color.rgb = RGBColor(255, 255, 255)
    p.alignment = PP_ALIGN.CENTER

    subtitle_box = slide.shapes.add_textbox(Inches(0.5), Inches(4.2), Inches(9), Inches(2))
    subtitle_frame = subtitle_box.text_frame
    subtitle_frame.word_wrap = True
    p = subtitle_frame.paragraphs[0]
    p.text = subtitle
    p.font.size = Pt(28)
    p.font.color.rgb = RGBColor(255, 255, 255)
    p.alignment = PP_ALIGN.CENTER

def add_content_slide(prs, title):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor(255, 255, 255)

    title_shape = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(10), Inches(0.8))
    title_shape.fill.solid()
    title_shape.fill.fore_color.rgb = DARK_BLUE
    title_shape.line.color.rgb = DARK_BLUE

    title_box = title_shape.text_frame
    title_box.text = title
    title_box.paragraphs[0].font.size = Pt(32)
    title_box.paragraphs[0].font.bold = True
    title_box.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
    title_box.vertical_anchor = 1

    return slide

# SLIDE 1: Title
add_title_slide(prs, "RRWWTF Flow Model", "Phase 6 R4S2 Configuration\nActual vs Predicted Performance")

# SLIDE 2: Model Overview
slide = add_content_slide(prs, "Model Configuration")
y_pos = 1.2

content = [
    ("Gauge Network Rainfall", "Average of 10 nearby NOAA/CoCoRaHS stations (0.5-6 miles)"),
    ("4-Day Antecedent Rain", "Rolling sum captures recent soil saturation (short-term memory)"),
    ("2-Day Antecedent Soil", "Multi-source average (ERA5, ERA5-Land, ECMWF over 3x3 grid)"),
    ("Rain x Antecedent", "Interactions: captures how storms scale with catchment wetness"),
    ("Features", "23 total (base + interactions + soil)"),
    ("Hyperparameters", "XGBoost: depth=2, lr=0.05, n_est=71 (original Phase 6 tuning)"),
]

for title, desc in content:
    title_box = slide.shapes.add_textbox(Inches(0.7), Inches(y_pos), Inches(2.5), Inches(0.4))
    tf = title_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = DARK_BLUE

    desc_box = slide.shapes.add_textbox(Inches(3.3), Inches(y_pos), Inches(6), Inches(0.4))
    tf = desc_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = desc
    p.font.size = Pt(11)

    y_pos += 0.85

# SLIDE 3: Key Metrics
slide = add_content_slide(prs, "Performance Metrics (Reference Test Set)")

metrics = [
    ["Metric", "R4S2 Model", "vs Baseline", "Status"],
    ["R2 (Correlation)", "0.605", "+0.012", "IMPROVED"],
    ["MAE (MGD)", "0.196", "-0.003", "IMPROVED"],
    ["Wet R2 (Rainy Days)", "0.642", "+0.011", "IMPROVED"],
    ["High-Flow Bias", "-0.477 MGD", "+0.034", "BETTER PEAKS"],
    ["Wet-Day Accuracy", "106.2%", "+6.2%", "EXCELLENT"],
]

left = Inches(0.7)
top = Inches(1.3)
width = Inches(8.6)
height = Inches(4.5)

table_shape = slide.shapes.add_table(len(metrics), len(metrics[0]), left, top, width, height).table

for i, row in enumerate(metrics):
    for j, cell_text in enumerate(row):
        cell = table_shape.cell(i, j)
        cell.text = cell_text

        if i == 0:
            cell.fill.solid()
            cell.fill.fore_color.rgb = DARK_BLUE
            tf = cell.text_frame
            tf.paragraphs[0].font.bold = True
            tf.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
            tf.paragraphs[0].font.size = Pt(11)
        else:
            tf = cell.text_frame
            tf.paragraphs[0].font.size = Pt(11)
            if i % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT_GRAY
            if j == 3:
                tf.paragraphs[0].font.bold = True
                if "IMPROVED" in cell_text or "BETTER" in cell_text or "EXCELLENT" in cell_text:
                    tf.paragraphs[0].font.color.rgb = RGBColor(0, 128, 0)

# SLIDE 4: Scatter Plot
slide = add_content_slide(prs, "Actual vs Predicted (Scatter)")
slide.shapes.add_picture("output/22_phase6_gauge_network/r4s2_final/01_scatter_actual_vs_predicted.png",
                         Inches(0.5), Inches(1), width=Inches(9), height=Inches(6))

# SLIDE 5: Time Series
slide = add_content_slide(prs, "Time Series Overlay (2020-2024)")
slide.shapes.add_picture("output/22_phase6_gauge_network/r4s2_final/02_timeseries_actual_vs_predicted.png",
                         Inches(0.3), Inches(1), width=Inches(9.4), height=Inches(6))

# SLIDE 6: Residuals
slide = add_content_slide(prs, "Prediction Errors Analysis")
slide.shapes.add_picture("output/22_phase6_gauge_network/r4s2_final/03_residuals_analysis.png",
                         Inches(0.3), Inches(1), width=Inches(9.4), height=Inches(6))

# SLIDE 7: Performance by Period
slide = add_content_slide(prs, "Performance by Test Period")
slide.shapes.add_picture("output/22_phase6_gauge_network/r4s2_final/04_performance_by_period.png",
                         Inches(0.3), Inches(1), width=Inches(9.4), height=Inches(6))

# SLIDE 8: Key Improvements
slide = add_content_slide(prs, "Why R4S2 Outperforms")

improvements = [
    "Shorter Antecedent Windows",
    "Recent saturation (4d rain, 2d soil) predicts flow better than month-long memory",
    "",
    "Soil-Rain Physics Captured",
    "Same storm = different flow depending on catchment wetness; soil interacts with rain",
    "",
    "Better Peak Prediction",
    "High-flow bias improved by 0.034 MGD (critical for flood forecasting)",
    "",
    "Multi-Source Data",
    "Gauge network + ERA5/ECMWF soil provides robustness across different data sources",
]

y = 1.3
for text in improvements:
    if text == "":
        y += 0.3
    elif any(keyword in text for keyword in ["Shorter", "Soil", "Better", "Multi"]):
        box = slide.shapes.add_textbox(Inches(0.7), Inches(y), Inches(8.6), Inches(0.35))
        tf = box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = text
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = DARK_MAGENTA
        y += 0.4
    else:
        box = slide.shapes.add_textbox(Inches(1.2), Inches(y), Inches(8.1), Inches(0.35))
        tf = box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = text
        p.font.size = Pt(11)
        y += 0.4

# SLIDE 9: Conclusion
slide = add_content_slide(prs, "Deployment Ready")

conclusion = [
    ("Model Location", "output/22_phase6_gauge_network/r4s2_final/production_model.json"),
    ("Test Performance", "R2 = 0.605, MAE = 0.196 MGD on 1,214 validation days (2020-2024)"),
    ("Production Data", "Trained on full 2,017 complete days (all available data)"),
    ("Peak Prediction", "High-flow bias -0.477 MGD (improved from -0.511 baseline)"),
    ("Wet-Day R2", "0.642 (excellent accuracy on rainy days)"),
    ("Status", "READY FOR OPERATIONAL FORECASTING"),
]

y = 1.3
for label, value in conclusion:
    label_box = slide.shapes.add_textbox(Inches(0.7), Inches(y), Inches(2.8), Inches(0.35))
    tf = label_box.text_frame
    p = tf.paragraphs[0]
    p.text = label + ":"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = DARK_BLUE

    value_box = slide.shapes.add_textbox(Inches(3.6), Inches(y), Inches(5.7), Inches(0.35))
    tf = value_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = value
    p.font.size = Pt(11)
    if "READY" in value:
        p.font.bold = True
        p.font.color.rgb = RGBColor(0, 128, 0)

    y += 0.75

# Save presentation
output_path = Path("output/22_phase6_gauge_network/r4s2_final/Phase6_R4S2_Model_Performance.pptx")
prs.save(str(output_path))
print("PowerPoint created: Phase6_R4S2_Model_Performance.pptx")
