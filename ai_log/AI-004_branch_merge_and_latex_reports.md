# Nhật Ký Sử Dụng AI — Phiên AI-004

* **Mã nhật ký:** `AI-004`
* **Thời gian:** 27/11/2026
* **Công cụ AI:** Antigravity
* **Người thực hiện:** Nhóm 1 — Môn học Trí tuệ Nhân tạo (Research Track)

---

### 1. Mục đích
1. Kiểm tra, đồng bộ và hợp nhất các nhánh làm việc của 4 thành viên (`main`, `model-optimization-lead`, `KNN`, `xgboost`) về nhánh `main` duy nhất mà vẫn giữ nguyên lịch sử đóng góp của từng người.
2. Dọn dẹp tệp rác/lỗi thời vi phạm `.gitignore` (`.idea/.name`, `*.bak`, script nháp cũ `bayes_optimize.py`, tệp `*_preds.npz` cũ) và tích hợp mô hình `KNN_Tuned` của Đặng Đình Đức vào `src/analyze_models.py`.
3. Tạo thêm các biểu đồ trực quan từ kết quả đã xác minh (`src/plot_paper_figures.py`) và biên soạn lại **Sản phẩm 1** & **Sản phẩm 2** dưới dạng bài báo khoa học 2 cột bằng LuaLaTeX, đồng bộ trên bộ dữ liệu 2.111 mẫu cho cả hai công trình Maulana et al. (2024) và Yagin et al. (2023).

---

### 2. Prompt chính từ nhóm
> "bạn kiểm tra các branch và pull về rồi phân tích cho tôi xem mọi người đang làm gì đi nha"
>
> "Bạn lên cho tôi 1 kế hoạch để merge các nhánh của mọi người lại nha /plan"
>
> "oke thế đầu tiên bạn hãy lên kế hoạch merge xong rồi viết lại sản phẩm 1 và 2 dưới dạng latex cho giống bài báo khoa học có oke không /plan"
>
> "vẫn giữ lịch sử commit của an nhé. oke gỡ các tệp đấy đi. oke bổ sung knn tuned vào đi, push luôn lên main nhé. nó là nhóm 1 á bạn sửa lại readme cho tôi luôn. cập nhật bản pdf nộp cùng repo nha. có bạn hãy vẽ thêm vài biểu đồ nữa vào cho dễ minh hoạ với ít chữ hơn"

---

### 3. Tóm tắt nội dung thực hiện & kiểm chứng
1. **Hợp nhất Git sạch & bảo toàn lịch sử (`obesity-ml-research`):**
   * Merge nhánh `model-optimization-lead` (`c014161`, 0 conflict) vào `main`.
   * Dùng `git merge --no-ff -s ours xgboost` để nối lịch sử commit của HoTuAn (`fea5efa`) vào `main` mà không ghi đè `src/train_xgboost.py` bản chuẩn hóa.
   * Gỡ khỏi Git tracking các tệp rác/cũ: `.idea/.name`, `results/bayes/*.bak`, `src/bayes_optimize.py`, `results/{chi2,f_classify,mutual_info}_preds.npz`.
2. **Tích hợp `KNN_Tuned` & tạo biểu đồ mới:**
   * Bổ sung `KNN_Tuned` (`manhattan, n_neighbors=3, weights=distance`) vào `src/analyze_models.py` và chạy lại `python -m src.analyze_models` xác nhận `KNN_Tuned` đạt **87.71%** (52/423 lỗi) trong `results/extended_comparison.csv`, không làm sai lệch kết quả CatBoost (95.51%), XGBoost Tuned (95.51%) và Soft Voting (96.45%).
   * Viết `src/plot_paper_figures.py` sinh 5 biểu đồ minh họa mới (`dataset_composition_bmi.png`, `sp1_literature_benchmark.png`, `ann_vs_lr_comparison.png`, `f_classify_critique.png`, `per_class_f1_heatmap.png`) từ dữ liệu và CSV thực tế, kiểm tra trực quan từng ảnh tránh chồng chữ.
3. **Biên soạn & biên dịch LaTeX (LuaLaTeX 2026):**
   * Cập nhật `README.md`, `CITATION.cff` đúng tên **Nhóm 1** và đầy đủ 4 thành viên.
   * Viết mới `sp1_khao_sat.tex` (Sản phẩm 1) và nâng cấp `main.tex` (Sản phẩm 2), biên dịch sạch không lỗi qua `lualatex` + `bibtex`, rà soát từng trang PDF và cập nhật 2 tệp PDF vào thư mục gốc repository.
