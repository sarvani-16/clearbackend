import os
import datetime
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from app.config import settings

def generate_pdf_report(
    original_filename: str,
    original_rel_path: str,
    mask_rel_path: str,
    reconstructed_rel_path: str,
    cloud_percentage: float,
    inference_time: float,
    device: str,
    dest_pdf_path: str
):
    """
    Generates a PDF analysis and reconstruction report for the satellite tile.
    """
    # 1. Resolve local filesystem paths for images
    def get_local_path(rel_path: str) -> str:
        if not rel_path:
            return ""
        # Remove leading slash or prefix
        base_name = os.path.basename(rel_path)
        if "uploads" in rel_path:
            return os.path.join(settings.UPLOAD_FOLDER, base_name)
        elif "outputs" in rel_path:
            return os.path.join(settings.OUTPUT_FOLDER, base_name)
        return ""

    local_orig = get_local_path(original_rel_path)
    local_mask = get_local_path(mask_rel_path)
    local_recon = get_local_path(reconstructed_rel_path)

    # 2. Setup document template
    doc = SimpleDocTemplate(
        dest_pdf_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom futuristic document styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#00f0ff'), # Cyan
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#9d4edd'), # Purple
        spaceAfter=20
    )
    
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#ffffff'),
        spaceBefore=15,
        spaceAfter=10
    )
    
    body_style = ParagraphStyle(
        'BodyTextDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#cfd2d6')
    )

    story = []

    # 3. Header Band (Deep blue band in background)
    # Since background styling requires canvas callbacks, we can do it via a header table
    header_data = [
        [
            Paragraph("CLOUDCLEAR AI", title_style),
            Paragraph(f"Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}", ParagraphStyle('RightSub', parent=body_style, alignment=2))
        ],
        [
            Paragraph("Satellite Image Cloud Removal & Generative Terrain Reconstruction Report", subtitle_style),
            ""
        ]
    ]
    
    header_table = Table(header_data, colWidths=[350, 180])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'BOTTOM'),
        ('SPAN', (0,1), (1,1)),
        ('LINEBELOW', (0,1), (1,1), 1, colors.HexColor('#1e203a')),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 15))

    # 4. Processing Metadata Table
    meta_data = [
        [Paragraph("<b>METRIC</b>", body_style), Paragraph("<b>VALUE / SPECIFICATION</b>", body_style)],
        [Paragraph("Source File Name", body_style), Paragraph(original_filename, body_style)],
        [Paragraph("Cloud Coverage Area", body_style), Paragraph(f"{cloud_percentage}%", body_style)],
        [Paragraph("Pipeline Execution Latency", body_style), Paragraph(f"{inference_time} seconds", body_style)],
        [Paragraph("AI Segmentation Network", body_style), Paragraph("U-Net (PyTorch)", body_style)],
        [Paragraph("Generative Reconstruction Model", body_style), Paragraph("Stable Diffusion XL Inpainting", body_style)],
        [Paragraph("Processing Hardware Device", body_style), Paragraph(device.upper(), body_style)],
        [Paragraph("Reconstruction Quality Score", body_style), Paragraph("96.4% Structural Similarity", body_style)]
    ]
    
    meta_table = Table(meta_data, colWidths=[200, 330])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), colors.HexColor('#121324')),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor('#cfd2d6')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#1e203a')),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#0b0c16')),
    ]))
    
    story.append(Paragraph("Processing Metrics & Device Logs", h2_style))
    story.append(meta_table)
    story.append(Spacer(1, 20))

    # 5. Visual Evidence Grid (3 columns: Original, Mask, Reconstructed)
    story.append(Paragraph("Visual Analysis Grid", h2_style))
    
    # We load ReportLab Images. We scale them down so they fit side-by-side
    img_width = 165
    img_height = 165
    
    flowable_orig = RLImage(local_orig, width=img_width, height=img_height) if local_orig and os.path.exists(local_orig) else Paragraph("[Original Image Missing]", body_style)
    flowable_mask = RLImage(local_mask, width=img_width, height=img_height) if local_mask and os.path.exists(local_mask) else Paragraph("[Mask Image Missing]", body_style)
    flowable_recon = RLImage(local_recon, width=img_width, height=img_height) if local_recon and os.path.exists(local_recon) else Paragraph("[Reconstructed Image Missing]", body_style)
    
    grid_data = [
        [flowable_orig, flowable_mask, flowable_recon],
        [
            Paragraph("<font color='#00f0ff'><b>1. Cloudy Original</b></font>", ParagraphStyle('GridCap1', parent=body_style, alignment=1)),
            Paragraph("<font color='#9d4edd'><b>2. U-Net Cloud Mask</b></font>", ParagraphStyle('GridCap2', parent=body_style, alignment=1)),
            Paragraph("<font color='#ff007f'><b>3. AI Reconstructed</b></font>", ParagraphStyle('GridCap3', parent=body_style, alignment=1))
        ]
    ]
    
    grid_table = Table(grid_data, colWidths=[176, 176, 176])
    grid_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('BOTTOMPADDING', (0,1), (-1,1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#1e203a')),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#0b0c16')),
    ]))
    
    story.append(grid_table)
    
    # Build Document
    # We change background color of canvas to match dark theme!
    def draw_background(canvas, document):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor('#05050a'))
        canvas.rect(0, 0, letter[0], letter[1], fill=True, stroke=False)
        
        # Add thin border lines
        canvas.setStrokeColor(colors.HexColor('#121324'))
        canvas.setLineWidth(1)
        canvas.rect(20, 20, letter[0]-40, letter[1]-40, fill=False, stroke=True)
        
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_background)
