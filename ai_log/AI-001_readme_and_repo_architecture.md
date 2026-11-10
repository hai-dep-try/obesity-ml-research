# Nhật Ký Sử Dụng AI — Phiên AI-001

* **Mã nhật ký:** `AI-001`
* **Thời gian:** 10/11/2026
* **Công cụ AI:** OpenCode Assistant
* **Người thực hiện:** Nhóm 10 — INT3401 Research Track

---

### 1. Mục đích
Khảo sát 4 repository mẫu chuẩn quốc tế trên GitHub (`emiddell/minimal-project`, `bavadeve/repo-template-secure`, `StefKirsch/amazing_github_research_repo`, `matiassingers/awesome-readme`) để học hỏi bố cục, chuẩn mực trình bày đề tài nghiên cứu khoa học trên GitHub và nâng cấp toàn diện tài liệu `README.md`, bổ sung tệp siêu dữ liệu trích dẫn `CITATION.cff`, giấy phép mã nguồn `LICENSE` và tệp `.gitignore` chống rò rỉ dữ liệu/file rác.

---

### 2. Prompt chính từ nhóm
> "Nếu bạn muốn tìm cảm hứng trình bày README nói chung (không chỉ cho NCKH), repo matiassingers/awesome-readme là một bộ sưu tập khổng lồ các README đẹp và chuyên nghiệp... bắt đầu với emiddell/minimal-project... bavadeve/repo-template-secure... StefKirsch/amazing_github_research_repo... nghiên cứu các repo này và trình bày cho tôi xem chúng ta có thể học hỏi được gì."
> 
> "oke làm theo kế hoạch đi"

---

### 3. Tóm tắt nội dung phản hồi từ AI
1. **Phân tích từng repo:**
   * `minimal-project`: Triết lý tối giản theo *The Good Research Code Handbook*, phân chia sạch giữa code module (`src/`), code chạy thử nghiệm và cấu trúc thư mục rõ ràng.
   * `repo-template-secure`: Khung 3 câu hỏi cốt lõi (Summary), quản trị bảo mật dữ liệu, phân công tác giả (Authors & Affiliations), kiểm tra rò rỉ dữ liệu lớn qua `.pre-commit-config.yaml`.
   * `amazing_github_research_repo`: Chuẩn mực nghiên cứu có thể tái lập (FAIR Software), bổ sung `CITATION.cff` để GitHub tự tạo nút trích dẫn, khối lệnh cài đặt tách biệt Windows vs Linux/macOS, và mục tuyên bố sử dụng AI (Statement of AI Use).
   * `awesome-readme`: Phương châm "Show, don't just tell" — nhúng trực tiếp biểu đồ kết quả, ma trận nhầm lẫn vào README, vẽ sơ đồ Mermaid trực quan hóa quy trình, và dùng `<details>` cho nội dung mở rộng.
2. **Kế hoạch triển khai:**
   * Viết lại `README.md` tích hợp đầy đủ các tinh hoa trên mà vẫn bám sát số liệu thực nghiệm CatBoost (95.51%), phát hiện phản biện khoa học (typo Naive Bayes, hội tụ Logistic Regression).
   * Tạo tệp `LICENSE` (MIT) và `CITATION.cff` ở thư mục gốc.
   * Cập nhật `.gitignore` chặn các file tạm (`.idea/`, `*.bak`, `*.sav`, `results/_test/`, `~$*`).

---

### 4. Đánh giá của nhóm sinh viên
* **Đánh giá phản hồi:** Ý tưởng phân tích sâu sắc, đối chiếu thực tế với repo hiện tại của nhóm, không làm loãng cấu trúc gốc theo quy ước `AGENTS.md`.
* **Nội dung chấp nhận:** Chấp nhận toàn bộ khung nâng cấp `README.md`, việc tạo `CITATION.cff`, `LICENSE` và siết chặt `.gitignore`.
* **Nội dung điều chỉnh/giữ vững:** Giữ ngôn ngữ tiếng Việt làm chủ đạo theo quy ước đề bài, bảo toàn các đường dẫn file và số liệu thực nghiệm gốc đã chạy trên CPU.

---

### 5. Kết quả kiểm chứng và áp dụng
* Đã tạo thành công `LICENSE` và `CITATION.cff` chuẩn CFF 1.2.0.
* Đã cập nhật `README.md` mới hoàn chỉnh, hiển thị đẹp mắt cả trên GitHub lẫn trình xem Markdown cục bộ.
* `.gitignore` hoạt động tốt, ngăn chặn các file cấu hình IDE và file tạm thời phát sinh.
