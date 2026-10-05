# -*- coding: utf-8 -*-
"""
Module Định Dạng Bảng Hiển Thị Console & Báo Cáo Markdown
Cung cấp các công cụ tạo bảng lưới (Grid Table) có viền, khoảng chắn ngăn cách
rõ ràng giữa các hàng và cột, tự động căn lề và tính toán độ rộng chuẩn xác
cho môi trường dòng lệnh (Console/Terminal) và tệp Markdown.

TopTop Project - Causal Debiasing & Click Recommendation
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional


def strip_formatting(text: str) -> str:
    """Loại bỏ ký tự markdown (bold, italic) và ANSI escape codes để tính độ dài thực."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    text = ansi_escape.sub('', str(text))
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    return text


def get_display_width(text: Any) -> int:
    """
    Tính độ rộng hiển thị thực tế trên terminal (Monospace).
    Chuẩn hóa NFC và tính toán ký tự độ rộng kép (East Asian Wide/Fullwidth).
    """
    s = strip_formatting(str(text))
    s_norm = unicodedata.normalize('NFC', s)
    width = 0
    for ch in s_norm:
        ea = unicodedata.east_asian_width(ch)
        if ea in ('W', 'F'):
            width += 2
        else:
            width += 1
    return width


def pad_text(text: Any, target_width: int, align: str = "left") -> str:
    """
    Căn lề chuỗi (left, center, right) dựa trên độ rộng hiển thị thực tế.
    """
    s = str(text)
    cur_width = get_display_width(s)
    pad_len = max(0, target_width - cur_width)

    if align == "right":
        return " " * pad_len + s
    elif align == "center":
        left_pad = pad_len // 2
        right_pad = pad_len - left_pad
        return " " * left_pad + s + " " * right_pad
    else:  # left
        return s + " " * pad_len


def format_grid_table(
    headers: List[str],
    rows: List[List[Any]],
    col_aligns: Optional[List[str]] = None,
    title: Optional[str] = None,
    row_divider: bool = True
) -> str:
    """
    Tạo bảng lưới chuyên nghiệp cho màn hình Console với đường kẻ ngang phân tách
    từng hàng và cột (khoảng chắn), ngăn chặn dính chữ và lệch dòng.

    Args:
        headers: Danh sách tiêu đề cột.
        rows: Danh sách các hàng dữ liệu.
        col_aligns: Danh sách căn lề cho từng cột ('left', 'center', 'right').
        title: Tiêu đề trên cùng của bảng (tùy chọn).
        row_divider: Nếu True, vẽ đường kẻ ngang giữa TẤT CẢ các hàng.
    """
    n_cols = len(headers)
    if col_aligns is None:
        col_aligns = ["left"] + ["center"] * (n_cols - 1)

    # 1. Tính toán độ rộng tối đa cho từng cột (dựa trên chuỗi đã lọc định dạng)
    col_widths = []
    for col_idx in range(n_cols):
        max_w = get_display_width(headers[col_idx])
        for row in rows:
            if col_idx < len(row):
                w = get_display_width(row[col_idx])
                if w > max_w:
                    max_w = w
        col_widths.append(max_w)

    border_sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_sep = "+" + "+".join("=" * (w + 2) for w in col_widths) + "+"
    total_inner_width = sum(w + 2 for w in col_widths) + (n_cols - 1)

    output_lines = []

    # 2. Tiêu đề bảng (nếu có)
    if title:
        top_bar = "+" + "-" * total_inner_width + "+"
        output_lines.append(top_bar)
        title_clean = strip_formatting(title.strip())
        title_padded = pad_text(f"  {title_clean}  ", total_inner_width, align="center")
        output_lines.append(f"|{title_padded}|")
        output_lines.append(header_sep)
    else:
        output_lines.append(border_sep)

    # 3. Hàng tiêu đề các cột
    header_cells = [
        pad_text(strip_formatting(headers[i]), col_widths[i], align="center")
        for i in range(n_cols)
    ]
    output_lines.append("| " + " | ".join(header_cells) + " |")
    output_lines.append(header_sep)

    # 4. Các hàng dữ liệu có khoảng chắn phân cách
    for r_idx, row in enumerate(rows):
        cells = []
        for c_idx in range(n_cols):
            raw_val = row[c_idx] if c_idx < len(row) else ""
            clean_val = strip_formatting(raw_val)
            align = col_aligns[c_idx] if c_idx < len(col_aligns) else "left"
            cells.append(pad_text(clean_val, col_widths[c_idx], align=align))

        output_lines.append("| " + " | ".join(cells) + " |")

        # Khoảng chắn giữa các hàng (Divider)
        if row_divider or r_idx == len(rows) - 1:
            output_lines.append(border_sep)

    return "\n".join(output_lines)


def format_markdown_table(
    headers: List[str],
    rows: List[List[Any]],
    col_aligns: Optional[List[str]] = None
) -> str:
    """
    Tạo bảng Markdown chuẩn hóa với các cột được đệm khoảng trắng thẳng đều tắp.
    Sử dụng độ dài chuỗi ký tự thô để đảm bảo các dấu gạch đứng (|) thẳng hàng
    tuyệt đối khi xem tệp dạng Raw Markdown trong Text Editor.
    """
    n_cols = len(headers)
    if col_aligns is None:
        col_aligns = ["left"] + ["center"] * (n_cols - 1)

    def raw_len(val: Any) -> int:
        return len(str(val))

    def pad_raw(val: Any, target_w: int, align: str) -> str:
        s = str(val)
        pad = max(0, target_w - len(s))
        if align == "right":
            return " " * pad + s
        elif align == "center":
            left_pad = pad // 2
            right_pad = pad - left_pad
            return " " * left_pad + s + " " * right_pad
        else:
            return s + " " * pad

    col_widths = []
    for col_idx in range(n_cols):
        max_w = max(raw_len(headers[col_idx]), 4)
        for row in rows:
            if col_idx < len(row):
                w = raw_len(row[col_idx])
                if w > max_w:
                    max_w = w
        col_widths.append(max_w)

    output_lines = []

    # Hàng Header
    h_cells = [
        pad_raw(headers[i], col_widths[i], align="left" if col_aligns[i] == "left" else "center")
        for i in range(n_cols)
    ]
    output_lines.append("| " + " | ".join(h_cells) + " |")

    # Hàng Separator Markdown (:---, :---:, ---:)
    sep_cells = []
    for i in range(n_cols):
        w = col_widths[i]
        align = col_aligns[i]
        if align == "center":
            sep_cells.append(":" + "-" * max(w - 2, 1) + ":")
        elif align == "right":
            sep_cells.append("-" * max(w - 1, 1) + ":")
        else:
            sep_cells.append(":" + "-" * max(w - 1, 1))
    output_lines.append("| " + " | ".join(sep_cells) + " |")

    # Các hàng dữ liệu
    for row in rows:
        cells = []
        for c_idx in range(n_cols):
            val = row[c_idx] if c_idx < len(row) else ""
            align = col_aligns[c_idx]
            cells.append(pad_raw(val, col_widths[c_idx], align=align))
        output_lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(output_lines)


def format_metrics_table(
    metrics: Dict[str, Any],
    title: str = "KẾT QUẢ ĐÁNH GIÁ (EVALUATION METRICS)",
    col1_name: str = "Chỉ Số Đánh Giá (Metric)",
    col2_name: str = "Giá Trị Đạt Được"
) -> str:
    """
    Tạo bảng lưới 2 cột đóng khung đẹp mắt cho một từ điển chỉ số.
    """
    rows = []
    for k, v in metrics.items():
        if isinstance(v, float):
            val_str = f"{v:.4f}"
        else:
            val_str = str(v)
        rows.append([str(k), val_str])

    return format_grid_table(
        headers=[col1_name, col2_name],
        rows=rows,
        col_aligns=["left", "center"],
        title=title,
        row_divider=True
    )


class EpochTableTracker:
    """
    Theo dõi tiến trình huấn luyện qua các epoch và in bảng lưới Console
    chuẩn 80 cột (VT100), vừa vặn mọi màn hình terminal mà không bị ngắt dòng.
    """
    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        # Chuẩn hóa 8 cột vừa khít 80 ký tự chiều ngang
        self.headers = ["Epoch", "Loss", "NDCG@5", "Recall@5", "Prec@5", "HR@5", "Train", "Eval"]
        self.col_widths = [7, 7, 8, 8, 7, 7, 6, 5]
        self.aligns = ["center", "center", "center", "center", "center", "center", "center", "center"]
        self.border_sep = "+" + "+".join("-" * (w + 2) for w in self.col_widths) + "+"
        self.header_sep = "+" + "+".join("=" * (w + 2) for w in self.col_widths) + "+"

    def print_header(self, title: Optional[str] = None):
        total_inner_width = sum(w + 2 for w in self.col_widths) + (len(self.col_widths) - 1)
        if title:
            top_bar = "+" + "-" * total_inner_width + "+"
            print(top_bar)
            clean_title = strip_formatting(title.strip())
            print(f"|{pad_text(f'  {clean_title}  ', total_inner_width, align='center')}|")
            print(self.header_sep)
        else:
            print(self.border_sep)

        h_cells = [pad_text(self.headers[i], self.col_widths[i], align="center") for i in range(len(self.headers))]
        print("| " + " | ".join(h_cells) + " |")
        print(self.header_sep)

    def print_epoch(
        self,
        epoch: int,
        loss: float,
        ndcg: float,
        recall: float,
        precision: float,
        hitrate: float,
        train_time: float,
        eval_time: float
    ):
        row = [
            f"{epoch:02d}/{self.total_epochs:02d}",
            f"{loss:.4f}",
            f"{ndcg:.4f}",
            f"{recall:.4f}",
            f"{precision:.4f}",
            f"{hitrate:.4f}",
            f"{train_time:.1f}s",
            f"{eval_time:.1f}s"
        ]
        r_cells = [pad_text(row[i], self.col_widths[i], align=self.aligns[i]) for i in range(len(row))]
        print("| " + " | ".join(r_cells) + " |")
        print(self.border_sep)
