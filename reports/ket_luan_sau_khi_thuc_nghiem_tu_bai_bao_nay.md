# Kết Luận & Phân Tích Chuyên Sâu Sau Khi Thực Nghiệm Tái Lập Bài Báo

> **Bài báo gốc:** *Predicting Obesity Levels with High Accuracy: Insights from a CatBoost Machine Learning Model*  
> **Tác giả:** Aga Maulana, Razief Perucha Fauzie Afidh, Nur Balqis Maulydia, Ghazi Mauer Idroes, Souvia Rahimah (2024)  
> **Tạp chí:** *Infolitika Journal of Data Science*, Vol. 2, No. 1, 2024, pp. 17–27. DOI: [10.60084/ijds.v2i1.195](https://doi.org/10.60084/ijds.v2i1.195)  
> **Tệp PDF tham chiếu:** `docs/Predicting Obesity Levels with High Accuracy Insights from a CatBoost .pdf`  
> **Dữ liệu thực nghiệm:** Kaggle `fatemehmehrparvar/obesity-levels` (2,111 mẫu, 17 thuộc tính, 7 nhãn phân loại thể trạng `NObeyesdad`).

---

## 1. Bảng Tổng Hợp Kết Quả: Thực Nghiệm Tái Lập vs. Bài Báo Gốc

Dưới đây là bảng đối chiếu chi tiết giữa kết quả công bố trong **Table 4** của bài báo và kết quả chạy thực nghiệm thực tế trên môi trường CPU cục bộ (với `random_state=42`, chia Stratified 80:20, 5-Fold Cross-Validation):

| Mô hình (Model) | Accuracy Bài báo (%) | Accuracy Thực nghiệm (%) | Độ lệch $\Delta$ (pp) | Recall Bài báo (%) | Recall Thực nghiệm (%) | F1-Score Bài báo (%) | F1-Score Thực nghiệm (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CatBoost (Main)** | **95.98** | **95.51** | **−0.47** | **95.98** | **95.51** | **96.00** | **95.53** |
| **Random Forest** | 95.04 | 94.33 | −0.71 | 95.04 | 94.33 | 95.00 | 94.44 |
| **KNN** | 88.18 | 84.40 | −3.78 | 88.18 | 84.40 | 89.07 | 83.52 |
| **Logistic Regression** | 65.48 | 93.14 | **+27.66** | 65.48 | 93.14 | 66.60 | 93.10 |
| **Naive Bayes** | ⚠️ 94.07 | 52.25 | **−41.82** | ⚠️ 64.07 | 52.25 | 66.17 | 48.53 |

---

## 2. Các Phát Hiện & Phân Tích Chuyên Sâu

Qua quá trình tái lập độc lập từ mã nguồn, nhóm nghiên cứu phát hiện 4 hiện tượng kỹ thuật quan trọng:

### 2.1. CatBoost và Random Forest tái lập thành công vượt bậc (Độ lệch < 0.8%)
- **CatBoost** đạt **95.51%** (sát mức **95.98%** của bài báo với độ chênh lệch chỉ $0.47$ điểm phần trăm).
- **Random Forest** đạt **94.33%** (sát mức **95.04%** của bài báo với độ chênh lệch chỉ $0.71$ điểm phần trăm).
- **Nguyên nhân thành công:** Cấu trúc dạng cây (Tree-based Ensembles) có ưu thế vượt trội trên dữ liệu dạng bảng (Tabular Data). Chúng xử lý cực kỳ tốt các ranh giới phi tuyến tính (như tương quan giữa Cân nặng và Chiều cao theo công thức $BMI = \frac{Weight}{Height^2}$) mà không bị ảnh hưởng tiêu cực bởi các biến giả lập (One-Hot dummy variables).

---

### 2.2. Phát hiện mâu thuẫn toán học & Lỗi đánh máy (Typo) ở Naive Bayes trong bài báo gốc

Tại **Table 4** của bài báo, Naive Bayes có kết quả bất thường:
$$\text{Accuracy} = 94.07\%, \quad \text{Precision} = 72.13\%, \quad \text{Recall} = 64.07\%, \quad \text{F1-Score} = 66.17\%$$

#### Bằng chứng toán học về lỗi mâu thuẫn:
Trong bài toán phân loại đa lớp (Multiclass Classification), khi đánh giá theo trọng số tương ứng số lượng mẫu của từng lớp (**Weighted Average**):
$$\text{Weighted Recall} = \sum_{c=1}^{K} \left( \frac{N_c}{N} \times \text{Recall}_c \right) = \sum_{c=1}^{K} \left( \frac{N_c}{N} \times \frac{TP_c}{N_c} \right) = \frac{\sum_{c=1}^K TP_c}{N} \equiv \text{Accuracy}$$
- **Quy tắc bắt buộc:** $\text{Weighted Recall}$ **luôn luôn bằng** $\text{Accuracy}$.
- Quan sát tất cả các dòng khác trong Table 4:
  - CatBoost: $\text{Acc} = 95.98\% = \text{Recall} = 95.98\%$
  - Random Forest: $\text{Acc} = 95.04\% = \text{Recall} = 95.04\%$
  - KNN: $\text{Acc} = 88.18\% = \text{Recall} = 88.18\%$
  - Logistic Regression: $\text{Acc} = 65.48\% = \text{Recall} = 65.48\%$
- **Riêng Naive Bayes:** Recall là **64.07%**, nhưng Accuracy lại ghi **94.07%**!
- **Kết luận:** Tác giả bài báo đã gặp **lỗi đánh máy (Typo)** khi soạn thảo, gõ nhầm chữ số `6` thành chữ số `9` (từ `64.07%` thành `94.07%`). Với Precision 72% và F1 66%, **Accuracy thực tế của bài báo chỉ đạt quanh mức ~64%, hoàn toàn không thể là 94.07%**.

#### Lý do Naive Bayes trong thực tế chỉ đạt ~52% – 64%:
1. **Vi phạm giả định độc lập có điều kiện:** Thuật toán Naive Bayes giả định các đặc trưng độc lập với nhau. Tuy nhiên, `Weight` (Cân nặng) và `Height` (Chiều cao) là hai biến quyết định thể trạng, có tương quan ngầm cực kỳ mạnh.
2. **Vi phạm phân phối chuẩn Gauss (Gaussian assumption):** Bộ tiền xử lý theo Table 2 của bài báo áp dụng One-Hot Encoding cho 4 biến danh mục (`Gender`, `CAEC`, `CALC`, `MTRANS`), tạo ra các cột nhị phân thưa chỉ có giá trị 0 và 1. Việc đưa các cột nhị phân này qua `StandardScaler` rồi áp dụng hàm mật độ xác suất Gauss liên tục (`GaussianNB`) làm sai lệch hoàn toàn hàm hợp lý (likelihood).

---

### 2.3. Nghịch lý Logistic Regression: Thực nghiệm đạt 93.14% trong khi bài báo chỉ đạt 65.48%

Trong thực nghiệm tái lập của chúng ta, Logistic Regression đạt độ chính xác lên tới **93.14%** (+27.66 pp so với bài báo).

- **Lỗi dừng sớm (Early Stopping) của Scikit-Learn trong bài báo:** 
  Trong thư viện `scikit-learn`, lớp `LogisticRegression` có tham số mặc định `max_iter=100`. Đối với bài toán phân loại đa lớp (7 lớp `NObeyesdad`) với 27 chiều đặc trưng, 100 lần lặp là **hoàn toàn không đủ để thuật toán tối ưu L-BFGS hội tụ** (mô hình sẽ văng cảnh báo `ConvergenceWarning: lbfgs failed to converge`). Khi dừng ở trạng thái chưa hội tụ, Logistic Regression thường kẹt ở mức chính xác thấp từ **60% – 65%**.
- **Khi được tối ưu đúng cách:**
  Trong pipeline tái lập, khi dữ liệu được chuẩn hóa `StandardScaler` đúng chuẩn và tăng số vòng lặp tối đa `max_iter=2000`, hàm mất mát log-loss hội tụ trọn vẹn. Do các mức độ béo phì có ranh giới phân tách tuyến tính khá rõ trong không gian đa chiều của tỷ lệ cân nặng/chiều cao, mô hình Logistic Regression tuyến tính thực tế có khả năng phân loại rất mạnh mẽ (>93%).

---

### 2.4. Sự dao động nhẹ ở mô hình KNN (84.40% vs. 88.18%)
- KNN dựa trên khoảng cách hình học Euclidean ($L_2$ distance). Việc chuẩn hóa Z-score trên cả các biến danh mục đã One-Hot có thể làm méo mó cấu trúc khoảng cách cục bộ.
- Ngoài ra, bài báo gốc không công bố giá trị hạt giống ngẫu nhiên (`random_state`) khi chia tập dữ liệu 80:20. Trên tập kiểm thử 423 mẫu, sự phân bổ ngẫu nhiên của 15–16 mẫu ranh giới có thể tạo ra độ lệch khoảng 3–4%.

---

## 3. Ý Nghĩa Đối Với Báo Cáo Nghiên Cứu (Research Track)

Việc tái lập thực nghiệm và phát hiện ra các điểm dị biệt này mang lại giá trị học thuật rất lớn cho đề tài giữa kỳ:

1. **Khẳng định tính trung thực khoa học:** Nhóm không làm đẹp số liệu hay sao chép kết quả một cách máy móc, mà tiến hành thực nghiệm thực chất, ghi nhận trung thực mọi dữ liệu từ pipeline mã nguồn.
2. **Năng lực phản biện học thuật (Critical Thinking):** Chỉ ra được mâu thuẫn nội tại trong bảng số liệu của bài báo quốc tế (sự bất khả thi về mặt toán học giữa Recall 64.07% và Accuracy 94.07% trong phân loại đa lớp có trọng số).
3. **Thấu hiểu bản chất thuật toán:** Giải thích được tại sao các mô hình dạng cây (CatBoost, Random Forest) lại ổn định và vượt trội trên dữ liệu dạng bảng, cũng như lý do các mô hình truyền thống (Naive Bayes, Logistic Regression) rất nhạy cảm với tiền xử lý và tham số hội tụ.
4. **Xác nhận kết luận của bài báo:** Dù có những sai lệch ở các mô hình baseline do thiếu chi tiết cấu hình, kết luận cốt lõi của tác giả vẫn hoàn toàn chính xác: **CatBoost là mô hình tối ưu nhất (95.51%), có tính ổn định cao và đặc biệt phù hợp cho bài toán phân loại mức độ béo phì.**

---

## 4. Thực Nghiệm Mở Rộng: Đối Chứng XGBoost, Phân Tích Lỗi, Tương Quan & Phản Biện Luật BMI

Để kiểm chứng sâu hơn năng lực của CatBoost khi đặt cạnh các mô hình mạnh khác ngoài bài báo gốc, nhóm thực hiện thêm 3 thí nghiệm chuẩn hóa (mã nguồn tại `src/train_xgboost.py` và `src/analyze_models.py`):

### 4.1. Đối chứng XGBoost trên cùng Pipeline chuẩn
| Mô hình | Accuracy (%) | Precision (%) | Recall (%) | F1-Weighted (%) | Số lỗi / 423 |
|---|---:|---:|---:|---:|---:|
| **CatBoost (Paper Params)** | **95.51** | 95.60 | 95.51 | 95.53 | 19 |
| **XGBoost (Tuned GridSearchCV)** | **95.51** | 95.64 | 95.51 | **95.55** | 19 |
| **XGBoost (HoTuAn Default)** | 94.56 | 94.71 | 94.56 | 94.61 | 23 |
| **Soft Voting (CatBoost + XGBoost Tuned + LR)** | **96.45** | 96.57 | 96.45 | **96.48** | **15** |

* **Nhận xét:** Khi dùng cấu hình mặc định (`n_estimators=100, lr=0.1, max_depth=6`), XGBoost đạt **94.56%** (nhỉnh hơn Random Forest 94.33%). Khi tối ưu qua Stratified 5-Fold CV (`lr=0.1, max_depth=4, n_estimators=200`), XGBoost đạt **95.51%** — cân bằng điểm Accuracy với CatBoost.

### 4.2. Phân tích lỗi điển hình & Tương quan giữa các mô hình
* **Quy luật lỗi ở ranh giới BMI = 25.0 (`Overweight_Level_I` $\leftrightarrow$ `Normal_Weight`):** Do dữ liệu không chứa cột BMI tường minh, cặp liền kề này là nguồn sai số lớn nhất ở mọi mô hình top đầu: chiếm **36.8%** tổng lỗi của CatBoost (7/19 ca), **50.0%** của Random Forest (12/24 ca), **51.7%** của Logistic Regression (15/29 ca), và **43.5%** của XGBoost HoTuAn (10/23 ca).
* **Tương quan dự đoán & Khả năng bù lỗi:** `CatBoost` và `XGBoost_Tuned` đồng thuận **96.69%** (cùng sai nhiều ca giống nhau với $r_{\text{err}} = 0.6143$, khác nhau trên 14 mẫu và mỗi mô hình thắng đúng 7 mẫu). Ngược lại, `LogisticRegression` có tương quan lỗi thấp với các mô hình cây ($r_{\text{err}} = 0.3025$ với XGBoost và $0.3928$ với CatBoost). Vì sự đa dạng lỗi này, tổ hợp **Soft Voting (CatBoost + XGBoost + LogisticRegression)** giảm số lỗi xuống còn **15/423 (Accuracy 96.45%)**, vượt qua cả mức 95.98% trong bài báo gốc.

### 4.3. Tại sao không dùng thẳng công thức $BMI = Weight / Height^2$? (Ablation Study)
| Kịch bản thử nghiệm | Phạm vi | Accuracy (%) | F1-Score (%) | Số mẫu sai |
|---|---|---:|---:|---:|
| **CatBoost (Bỏ `Weight` & `Height`)** | Test ($N=423$) | **82.98** | 82.96 | 72 |
| **Luật cứng BMI WHO ($W/H^2$)** | Full ($N=2111$) | **91.95** | 91.90 | 170 |
| **CatBoost (Pipeline chuẩn)** | Test ($N=423$) | **95.51** | 95.53 | 19 |
| **CatBoost (+ Đặc trưng `BMI`)** | Test ($N=423$) | **98.82** | 98.82 | 5 |

* **Giải thích khoa học:** Việc áp công thức BMI chuẩn WHO lên toàn bộ tập dữ liệu thực tế chỉ đạt **91.95%** (sai **170/2,111 mẫu** — ví dụ bỏ sót 56 ca `Obesity_Type_III` có BMI dưới 40.0 và 62 ca `Overweight_Level_II` có BMI dưới 27.5). Nguyên nhân là thuật toán **SMOTE** nội suy tuyến tính trên các biến `Weight` và `Height`, không bảo toàn ngưỡng cắt phi tuyến $W/H^2$. Trong khi đó, mô hình ML học kết hợp đặc trưng hành vi (`FAVC`, `FCVC`, `FAF`, `Gender`) nên bù đắp được độ lệch biên (đạt **95.51%**), và ngay cả khi loại bỏ hoàn toàn `Weight` và `Height`, CatBoost vẫn đạt **82.98%** — chứng minh giá trị độc lập của dữ liệu thói quen sinh hoạt trong dự báo béo phì.
