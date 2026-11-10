# Nhật Ký Sử Dụng AI — Phiên AI-003

* **Mã nhật ký:** `AI-003`
* **Thời gian:** 10/11/2026
* **Công cụ AI:** OpenCode Assistant
* **Người thực hiện:** Nhóm 10 — Môn học Trí tuệ Nhân tạo (Research Track)

---

### 1. Mục đích
Tạo quy trình lọc 5 đặc trưng theo đúng bài báo Yagin et al. (2023) [PDF 1] để sinh tệp cấu hình `data/splits/top5_features.json` và 3 bộ dữ liệu CSV 5 đặc trưng (`chi2`, `f_classify`, `mutual_info`) phục vụ các kịch bản thực nghiệm ANN & Logistic Regression.

---

### 2. Prompt chính từ nhóm
> "teammate của tôi bảo tôi xử lý dữ liệu lọc 5 features là làm gì thế"
>
> "ủa thế chử minh đã lọc chưa hay giờ chúng ta phải lọc"
>
> "giờ bạn cứ tạo 1 bộ data lọc 5 cái y như bài đã nhé"

---

### 3. Tóm tắt công việc đã triển khai & kiểm chứng
1. **Xây dựng [`src/select_features.py`](../src/select_features.py):**
   * Tạo sẵn 2 chế độ: `--mode paper` (mặc định, xuất đúng 3 bộ 5 đặc trưng của Yagin et al. 2023 trang 9) và `--mode selectkbest` (tính lại Top 5 theo điểm số cao nhất trên `train_for_val`).
   * Xuất tệp `data/splits/top5_features.json` và 3 bộ dữ liệu tách sẵn 5 cột đặc trưng + `NObeyesdad` tại `data/splits/top5_{chi2,f_classify,mutual_info}/`.
   * Lưu bảng đối chiếu điểm `SelectKBest` trên `train_for_val` tại `results/feature_selection_scores.csv`.
2. **Sửa lỗi tương thích Pandas 3.0 & Windows Console trong [`src/lr_baseline.py`](../src/lr_baseline.py):**
   * Thay điều kiện `df[col].dtype == object` thành `not pd.api.types.is_numeric_dtype(df[col])` để encode chính xác cột kiểu `StringDtype` trên Pandas 3.0+, đồng thời cấu hình `utf-8` cho stdout.
3. **Chạy kiểm chứng thực tế:**
   * `python -m src.select_features --mode paper`
   * `python src/lr_baseline.py` $\rightarrow$ Đã tự động nhận đủ 4 bộ (`full`: 96.02%, `chi2`: 67.05%, `f_classify`: 34.66%, `mutual_info`: 58.14%) và ghi ra `results/lr_results.csv` & `results/lr_per_class_f1.csv`.
