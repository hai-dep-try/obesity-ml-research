# Nhật Ký Sử Dụng AI — Phiên AI-002

* **Mã nhật ký:** `AI-002`
* **Thời gian:** 10/11/2026
* **Công cụ AI:** OpenCode Assistant
* **Người thực hiện:** Nhóm 10 — Môn học Trí tuệ Nhân tạo (Research Track)

---

### 1. Mục đích
1. Tích hợp và chuẩn hóa mô hình XGBoost từ nhánh `xgboost` của thành viên HoTuAn (`src/train_xgboost.py`) vào pipeline chung (`src/preprocess.py`, `src/evaluate.py`).
2. Xây dựng mô-đun phân tích lỗi mô hình, ma trận đồng thuận & tương quan lỗi giữa các thuật toán, và thí nghiệm phản biện luật BMI (`src/analyze_models.py`).
3. Cập nhật kết quả, biểu đồ và phân tích vào `README.md`, `results/` và báo cáo markdown `reports/ket_luan_sau_khi_thuc_nghiem_tu_bai_bao_nay.md`.

---

### 2. Prompt chính từ nhóm
> "ơ mà bạn chạy thử kết quả của hotuan vs xgboost rồi thêm vào bảng đi nha với cả in ra xem các model thường hay bị lỗi như nào nữa xem các kết quả mô hình trả ra có mức độ tương quan như nào nha"
>
> "ơ mà bộ dữ liệu này cho weight với height thì tính ra bmi là ra chuẩn xác rồi thì cần gì train mô hình nhỉ"
>
> "ồ oke thể thêm các phần đấy vào đi"

---

### 3. Tóm tắt nội dung thực nghiệm & phản hồi
1. **Chuẩn hóa XGBoost (`src/train_xgboost.py`):**
   * Chạy cấu hình của HoTuAn (`n_estimators=100, lr=0.1, max_depth=6`) trên pipeline chuẩn đạt **94.56%** (23/423 lỗi).
   * Tối ưu GridSearchCV 5-Fold (`lr=0.1, max_depth=4, n_estimators=200`) đạt **95.51%** (19/423 lỗi), cân bằng với CatBoost (95.51%).
2. **Phân tích lỗi & Tương quan (`src/analyze_models.py`):**
   * Ranh giới `Overweight_Level_I <-> Normal_Weight` (BMI = 25.0) chiếm 36.8% lỗi CatBoost, 50.0% lỗi Random Forest, 51.7% lỗi Logistic Regression.
   * `CatBoost` và `XGBoost_Tuned` đồng thuận **96.69%**, nhưng chỉ có $r_{\text{err}} = 0.30$–$0.39$ khi ghép với `LogisticRegression`. Nhờ sự bổ trợ giữa mô hình cây và tuyến tính, tổ hợp Soft Voting (`CatBoost + XGBoost Tuned + LR`) đạt **96.45%** (15/423 lỗi).
3. **Thí nghiệm phản biện BMI (Ablation Study):**
   * Áp luật cứng BMI WHO lên toàn bộ 2,111 mẫu chỉ đạt **91.95%** (sai 170 mẫu do nội suy tuyến tính SMOTE làm sai lệch biên phi tuyến $W/H^2$).
   * Bỏ hoàn toàn `Weight` và `Height`, CatBoost vẫn đạt **82.98%** chỉ từ 14 biến thói quen sinh hoạt/nhân khẩu học, khẳng định giá trị thực tế của ML ngoài phép chia BMI.

---

### 4. Đánh giá & Kiểm chứng của nhóm
* Toàn bộ mã nguồn được kiểm tra và thực thi trực tiếp trên CPU (`python -m src.train_xgboost`, `python -m src.analyze_models`), kết quả lưu vào `results/` và trực quan hóa tại `results/figures/`.
* Đã kiểm tra trực tiếp từng biểu đồ PNG sinh ra (`model_comparison_extended.png`, `model_correlation_heatmaps.png`, `bmi_ablation_study.png`, `confusion_matrix_xgboost.png`) để đảm bảo không bị chồng chữ hay sai lệch nhãn.
