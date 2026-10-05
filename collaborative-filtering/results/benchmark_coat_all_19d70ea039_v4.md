# Benchmark v4: validation và test độc lập

**Benchmark đầy đủ dữ liệu; một seed huấn luyện, chọn checkpoint bằng validation.**

**Thời gian cập nhật:** 2026-10-01 08:55:03

Chỉ xếp hạng các exposure đã ghi log. Chọn epoch trên validation; test giữ riêng.
Chỉ số chính: NDCG@5 và Recall@5. AUC là chỉ số phụ lấy trung bình theo user có hai lớp; không mặc định tương đương AUC trong bài CDR.
Coat và KuaiRand chia random log theo user, 30% validation / 70% test; macro metric chỉ chấm user có nhãn dương.
Kết quả địa phương chưa thể so trực tiếp với bảng bài báo: mô hình, siêu tham số, seed và giao thức giữa các bài báo còn khác. Bảng tham chiếu có chú giải riêng tại `docs/paper_reference_comparison.md`.

### Kết quả trên held-out test:

| Tập Dữ Liệu | Mô Hình Thuật Toán | NDCG@5 | Recall@5 | Precision@5 | HitRate@5 | AUC (macro-user) |
| :---------- | :----------------- | :----: | :------: | :---------: | :-------: | :--------------: |
| **Coat**    | MF                 | 0.4629 |  0.4898  |   0.3293    |  0.8503   |      0.6460      |
| **Coat**    | DR-BIAS + CDR      | 0.4841 |  0.4932  |   0.3485    |  0.8263   |      0.6826      |
| **Coat**    | IViDR              | 0.5454 |  0.5714  |   0.3653    |  0.8982   |      0.7064      |

### Candidate pool coat

- Users: 203; zero-positive users: 36; candidates: 3,248.
- Positive prevalence: 0.1930; candidate size p10/p50/p90: [16.0, 16.0, 16.0].
- Cold users: 0.00%; cold candidates: 0.00%.
- Validation positive prevalence: 0.1674; validation candidate p50: 16.

- MF: ΔNDCG@5 so với Popularity = +0.1118; CI95 NDCG của mô hình, theo user ≈ ±0.0448.
- IViDR: ΔNDCG@5 so với Popularity = +0.1943; CI95 NDCG của mô hình, theo user ≈ ±0.0467.
- DR-BIAS + CDR: ΔNDCG@5 so với Popularity = +0.1330; CI95 NDCG của mô hình, theo user ≈ ±0.0476.

Coat train/random overlap: 366 cặp; test overlap: 247 cặp, gồm 63 cặp dương.

### Giới hạn sai số

- Khoảng tin cậy trong JSON là xấp xỉ theo người dùng của một lần chạy; chưa bao gồm biến thiên do seed hoặc chọn cấu hình.
- Random (expected) là kỳ vọng xếp hạng ngẫu nhiên có điều kiện trên candidate pool hiện có; Popularity dùng tần suất exposure trong train.
- KuaiRand dùng duy nhất is_click, ghép hai standard log, lọc user/item tối thiểu 10 exposure một lượt và chia random log theo user. CDR dùng tần suất exposure theo item làm proxy propensity.
- Coat dùng rating ≥ 4 là dương; 6.960 biased ratings để train và 4.640 random ratings chia theo user seed 1234. Các cặp trùng train/test giữ nguyên để đối chiếu giao thức bài báo.
- IViDR huấn luyện decoder tái tạo exposure từ biased train: toàn bộ item ở Coat và 128 item lấy mẫu đều mỗi batch ở KuaiRand.
