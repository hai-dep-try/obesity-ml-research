# ai_log — Nhật ký sử dụng AI

Bắt buộc theo `docs/AI_Midterm_Project.pdf` §II.

Với mỗi tương tác LLM substantive, ghi:

1. Thời gian + công cụ AI
2. Mục đích
3. Prompt đầy đủ
4. Phản hồi đầy đủ
5. Đánh giá của nhóm
6. Nội dung chấp nhận / chỉnh / loại bỏ + lý do
7. Cách kiểm chứng + kết quả áp dụng

Có thể dùng bảng (ví dụ `AI-001`, …) hoặc file theo phiên. Hội thoại nhiều lượt phải giữ thứ tự.

## Phiên gần đây (tóm tắt vận hành)

| ID | Mục đích | Chấp nhận? | Kiểm chứng |
|----|----------|------------|------------|
| AI-001 | Scaffold repo + AGENTS.md | Có | tree + git push `41d9754` |
| AI-002 | Tải dataset Kaggle (`kagglehub`) | Có (`src/download_data.py`) | CSV 2111×17 trong `data/` |
| AI-003 | Docs tiếng Việt + tái lập CatBoost paper trên CPU + push GitHub | Có (`src/train.py`, reports) | Acc CatBoost local **95.51%** vs paper **95.98%**; `python -m src.train` |

Chi tiết prompt/response đầy đủ nên được nhóm bổ sung vào bảng/log riêng nếu giảng viên yêu cầu bản đầy đủ theo §II.
