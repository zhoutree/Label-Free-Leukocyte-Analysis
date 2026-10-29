"""用 ReportLab 排版患者汇总及真实图块图册；不计算模型决策。"""

import hashlib
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont, TTFError
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph

from .patient_summary import CellResult, report_groups, summarize_sample


GRAY = colors.HexColor("#555555")
PALE = colors.HexColor("#f5f5f5")
WIDTH, HEIGHT = A4
MARGIN = 28.35
TABLE_WIDTH = 402.52
ROW_HEIGHT = 19.84
LABEL_X, VALUE_X, COUNT_X, PERCENT_X = 70.03, 234.44, 282.44, 380.71
GALLERY_COLUMNS = 10
IMAGE_SIZE, IMAGE_STEP, IMAGE_PADDING = 48, 53.67, 2.83


def _font_name(path: Path) -> str:
    """同一路径使用同一注册名称，避免多患者导出重复解析字体。"""
    return "CellReport-" + hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]


def resolve_font(path: Path | None = None) -> Path:
    """查找并嵌入支持中文的 TrueType 字体，不退回缺字的西文字体。"""
    candidates = [path] if path is not None else [
        Path("C:/Windows/Fonts/simhei.ttf"), Path("C:/Windows/Fonts/msyh.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/usr/share/fonts/truetype/arphic/uming.ttc"),
        Path("/System/Library/Fonts/STHeiti Light.ttc"),
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            name = _font_name(candidate)
            if name not in pdfmetrics.getRegisteredFontNames():
                font = TTFont(name, str(candidate))
                if any(ord(char) not in font.face.charToGlyph for char in "白细胞患者样本质量复核"):
                    raise ValueError("所选字体缺少报告所需中文")
                pdfmetrics.registerFont(font)
            return candidate.resolve()
        except (TTFError, ValueError):
            if path is not None:
                raise ValueError("字体必须支持中文并可由 ReportLab 嵌入") from None
    raise ValueError("未找到可嵌入的中文字体，请用 --font-ttf 指定中文 TrueType 字体")


def _paragraph(canvas: Canvas, text: str, x: float, top: float, width: float,
               font: str, *, size: float = 9, color=GRAY) -> float:
    """使用 CJK 自动换行并转义输入，避免长编号溢出或被当成排版标记。"""
    style = ParagraphStyle("cell-report", fontName=font, fontSize=size,
                           leading=size * 1.5, wordWrap="CJK", textColor=color)
    paragraph = Paragraph(escape(text).replace("\n", "<br/>"), style)
    _, height = paragraph.wrap(width, HEIGHT)
    paragraph.drawOn(canvas, x, top - height)
    return top - height


def _display_image(tile: np.ndarray) -> Image.Image:
    """仅为图册做逐图 1-99 百分位灰度拉伸，不修改模型输入或预测。"""
    if not np.isfinite(tile).all():
        raise ValueError("图块包含非有限像素，不能写入患者报告")
    low, high = np.percentile(tile, [1, 99])
    pixels = (np.clip((tile - low) / (high - low), 0, 1) * 255
              if high > low else np.full(tile.shape, 128))
    return Image.fromarray(np.rint(pixels).astype(np.uint8)).resize(
        (220, 220), Image.Resampling.NEAREST)


def _table_row(canvas: Canvas, font: str, top: float, title: str, count: int,
               proportion: float, *, shaded: bool = False, outlined: bool = False) -> float:
    """按参考 PDF 的三列位置排名称、数量与占全部输入图块的百分比。"""
    canvas.setFillColor(PALE if shaded else colors.white)
    canvas.setStrokeColor(PALE)
    canvas.setLineWidth(1.2)
    canvas.rect(MARGIN, top - ROW_HEIGHT, TABLE_WIDTH, ROW_HEIGHT,
                fill=1, stroke=int(outlined))
    canvas.setFillColor(colors.black)
    canvas.setFont(font, 12)
    baseline = top - 14
    canvas.drawString(LABEL_X, baseline, title)
    canvas.drawCentredString(COUNT_X, baseline, str(count))
    canvas.drawCentredString(PERCENT_X, baseline, f"{proportion:.1%}")
    return top - ROW_HEIGHT


def _header(canvas: Canvas, font: str, patient_id: str, sample_id: str,
            generated: str, continuation: bool = False) -> float:
    """仿参考样本信息表；续页直接接图册，页脚仅保留身份和页码。"""
    canvas.setFillColor(GRAY)
    canvas.setFont(font, 6)
    short = lambda value: value[:22] + ("..." if len(value) > 22 else "")
    canvas.drawString(MARGIN, 23, f"患者：{short(patient_id)}  样本：{short(sample_id)}")
    canvas.drawRightString(WIDTH - MARGIN, 23, f"第 {canvas.getPageNumber()} 页")
    top = HEIGHT - MARGIN
    if continuation:
        return top
    canvas.setFillColor(PALE)
    canvas.rect(MARGIN, top - ROW_HEIGHT, TABLE_WIDTH, ROW_HEIGHT, fill=1, stroke=0)
    canvas.setFillColor(colors.black)
    canvas.setFont(font, 12)
    canvas.drawString(VALUE_X, top - 14, "样本信息")
    top -= ROW_HEIGHT
    for label, value in (("患者编号", patient_id), ("样本编号", sample_id),
                         ("报告生成时间", generated)):
        canvas.setFillColor(colors.black)
        canvas.setFont(font, 12)
        canvas.drawString(LABEL_X, top - 14, label)
        bottom = _paragraph(canvas, value, VALUE_X, top - 2,
                            MARGIN + TABLE_WIDTH - VALUE_X - 4, font, size=10.5,
                            color=colors.black)
        top = min(top - ROW_HEIGHT, bottom - 3)
    return top


def _summary(canvas: Canvas, cells: list[CellResult], font: str, top: float) -> float:
    """先列含六个互斥分组的总表，全部输入为统一分母，不仿造临床标记。"""
    summary = summarize_sample(cells)
    canvas.setFillColor(PALE)
    canvas.rect(MARGIN, top - ROW_HEIGHT, TABLE_WIDTH, ROW_HEIGHT, fill=1, stroke=0)
    canvas.setFillColor(colors.black)
    canvas.setFont(font, 12)
    canvas.drawString(VALUE_X, top - 14, "白细胞（模型候选）")
    top = _table_row(canvas, font, top - ROW_HEIGHT, "输入图块总数", summary["total"], 1,
                     shaded=True)
    for row in summary["report_rows"]:
        top = _table_row(canvas, font, top, row["title"], row["count"], row["proportion"])
    return top - 14


def _gallery_row(canvas: Canvas, cells: list[CellResult], tiles: np.ndarray,
                 top: float) -> None:
    """按参考 PDF 每行十张、48 点方图排列；逐图明细移到 CSV，保持图册紧凑。"""
    for column, cell in enumerate(cells):
        x = MARGIN + IMAGE_PADDING + column * IMAGE_STEP
        canvas.drawImage(ImageReader(_display_image(tiles[cell.packed_index])),
                         x, top - IMAGE_PADDING - IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE)


def render_patient_pdf(path: Path, patient_id: str, samples: dict[str, list[CellResult]],
                       tiles: np.ndarray, metadata: dict, generated: str,
                       font_path: Path) -> None:
    """每位患者一个 PDF，各样本独立汇总，图册自动续页且不丢弃拦截项。"""
    font = _font_name(font_path)
    canvas = Canvas(str(path), pagesize=A4, pageCompression=1)
    canvas.setTitle("明场白细胞分析结果")
    canvas.setAuthor("Label-Free Leukocyte Analysis")
    first_page = True
    for sample_id, cells in samples.items():
        if not first_page:
            canvas.showPage()
        first_page = False
        top = _header(canvas, font, patient_id, sample_id, generated)
        top = _summary(canvas, cells, font, top)
        for row, selected in report_groups(cells):
            if not selected:
                continue
            # 类别栏与第一行图片保持同页；后续图片无多余标题地连续排版。
            if top - ROW_HEIGHT - IMAGE_STEP < 36:
                canvas.showPage()
                top = _header(canvas, font, patient_id, sample_id, generated, True)
            top = _table_row(canvas, font, top, row["title"], row["count"],
                             row["proportion"], outlined=True)
            for start in range(0, len(selected), GALLERY_COLUMNS):
                if top - IMAGE_STEP < 36:
                    canvas.showPage()
                    top = _header(canvas, font, patient_id, sample_id, generated, True)
                _gallery_row(canvas, selected[start:start + GALLERY_COLUMNS], tiles, top)
                top -= IMAGE_STEP
            top -= 14
    canvas.save()
