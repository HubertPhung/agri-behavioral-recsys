> Tài liệu lịch sử: artifact chạy cũ đã được xóa ngày 2026-09-27. Các số liệu cũ dưới đây không phải kết quả hiện tại.

# Giao thức benchmark v4 và phân tích sai số

Tên file v2 được giữ để tương thích liên kết; nội dung hiện mô tả v4.

## Mục tiêu

Đo NDCG@5 và Recall@5 của MF, IViDR và DR-BIAS+CDR khi xếp hạng **những video đã được phơi bày trong random log KuaiRand-Pure**. Nhãn duy nhất là `is_click ∈ {0,1}`. Đây là re-ranking trên logged candidates; những user–video không được ghi log không có nhãn để chấm.

## Giao thức đã triển khai

1. **Train:** ghép hai file standard log KuaiRand-Pure. Lọc một lượt user có ít nhất 10 exposure, rồi video có ít nhất 10 exposure trong phần còn lại. Giữ cả click và non-click quan sát được. Dữ liệu raw trong repo tạo đúng **23.533 user, 6.712 video, 1.413.574 train exposure** theo [notebook iDCF](https://github.com/BgmLover/iDCF/blob/main/data_process/kuairand/build_dataset.ipynb).
2. **Validation/test:** lọc random log theo user/video có trong train, còn **954.814 exposure**. Chia theo user với `RandomState(1234)`: khoảng 30% user để validation và 70% để test, khớp [hàm split của iDCF](https://github.com/BgmLover/iDCF/blob/main/utils.py). User validation và test không giao nhau.
3. **Nhãn và huấn luyện:** `is_click` là nguồn nhãn duy nhất. BCE mặc định có trọng số 1 cho cả hai lớp (`alpha=0`). Chọn epoch và siêu tham số chỉ bằng validation; đánh giá test một lần cho checkpoint đã chọn. Cache khóa theo file input, alpha và định nghĩa nhãn/split.
4. **Metric:** macro NDCG@5 và Recall@5 trên user có ít nhất một click dương, theo evaluator iDCF. Báo thêm số user không có dương, candidate size, positive rate, Random và Popularity baseline. AUC chỉ tính trên user có cả hai lớp và ghi `auc_users`.
5. **Báo cáo:** mỗi lần chạy xuất JSON và Markdown với hash cấu hình, không nhập điểm từ lần chạy khác. `--quick` chỉ là smoke test.

## Kết quả kiểm tra dữ liệu và smoke test

### Coat

Coat dùng toàn bộ 6.960 rating tự chọn để train, rating ≥ 4 là dương. Random log có 4.640 rating từ 290 user, mỗi user 16 rating. Theo [giao thức iDCF](https://arxiv.org/pdf/2302.05052) và [mã split](https://github.com/BgmLover/iDCF/blob/main/utils.py), chia user bằng `RandomState(1234)`: 87 user / 1.392 rating cho validation; 203 user / 3.248 rating cho test. Macro NDCG@5 và Recall@5 chỉ chấm user có ít nhất một rating dương trong phần đánh giá.

Bộ dữ liệu gốc có **366 cặp user–item xuất hiện ở cả biased train và random log**; 247 cặp nằm trong test, tức 7,6% test candidates, gồm 63 rating dương. Rating trùng khớp hoàn toàn giữa hai ma trận. Đây là đặc điểm của dữ liệu gốc; khi so với bài báo cần ghi rõ vì mô hình đã được học nhãn của một số cặp được chấm. Muốn đánh giá các cặp hoàn toàn chưa thấy thì cần một thí nghiệm riêng và không nên so điểm của nó với bảng tác giả.

Lượt v4 tối đa 25 epoch, seed 42, chọn bằng validation: [Artifact cũ đã xóa; xem trạng thái hiện tại](../results/README.md), [Artifact cũ đã xóa; xem trạng thái hiện tại](../results/README.md).

| Mô hình | NDCG@5 | Recall@5 |
| :-- | --: | --: |
| Random, kỳ vọng | 0,2986 | 0,3125 |
| Popularity | 0,3543 | 0,3727 |
| MF | 0,5086 | 0,4865 |
| IViDR địa phương | 0,4407 | 0,4572 |
| DR-BIAS + CDR địa phương | 0,4325 | 0,4499 |

Test có 203 user, 3.248 candidates; 36 user không positive được loại, còn 167 user được chấm. Bảng trên là lượt cấu hình mặc định trước tuning; kết quả tuning và nhiều seed được bổ sung ở dưới. Sau khi khôi phục gradient, IViDR/CDR chưa vượt MF; lỗi được sửa không bảo đảm điểm số sẽ tăng. Các giá trị v2 từng ghi trong tài liệu là lịch sử chưa kiểm chứng lại do artifact gốc không còn.

CDR có loss ước lượng âm ở một số epoch vì mục tiêu chứa số hạng hiệu chỉnh bị trừ; điều đó không đồng nghĩa BCE âm hoặc lỗi số học. Trong lượt này imputation loss lớn và validation tốt nhất ở epoch 2 rồi giảm: cần tuning chỉ trên validation và phân tích độ nhạy propensity/regularization trước khi kết luận về hiệu quả mô hình.

## Sau tuning và kiểm tra 5 seed

Đã quét 17 cấu hình (MF 5, IViDR 6, CDR 6), tối đa 50 epoch, chọn bằng validation ở seed 42. Sau đó cố định cấu hình và lặp seeds 42–46; mỗi seed chọn epoch bằng validation.

| Mô hình | NDCG@5 mean ± std | Recall@5 mean ± std |
| :-- | --: | --: |
| MF | 0.5007 ± 0.0062 | 0.5083 ± 0.0171 |
| IVIDR | 0.5120 ± 0.0220 | 0.5204 ± 0.0209 |
| CDR | 0.4338 ± 0.0195 | 0.4521 ± 0.0180 |

IViDR có mean NDCG cao hơn MF khoảng 0,0114 nhưng dao động lớn hơn và chỉ cao hơn MF ở 3/5 seed. CDR còn thấp hơn MF ở cả 5 seed. Không chọn lại cấu hình dựa trên các điểm test này; khác biệt kiến trúc/giao thức với tác giả vẫn còn.

Std đo biến thiên do seed trên cùng split. Seed 42 đã tham gia tuning; kết quả chưa bao gồm bất định do thay đổi split hay dữ liệu mới.

[Artifact cũ đã xóa; xem trạng thái hiện tại](../results/README.md) · [Artifact cũ đã xóa; xem trạng thái hiện tại](../results/README.md).

### KuaiRand

| Đặc điểm | Giá trị |
| :-- | --: |
| Train exposure | 1.413.574 |
| Random exposure sau lọc | 954.814 |
| Test exposure theo split seed 1234 | 671.377 |
| Tỷ lệ click trong test | 17,64% |
| User có candidate trong test | 16.473 |
| User được chấm, có click dương | 13.502 |
| User không có click dương | 2.971 |

Các số MF một epoch v2 trước đây không còn artifact gốc để kiểm chứng. V4 mới chạy smoke cả ba mô hình trên mẫu 64 user, chưa chạy benchmark KuaiRand đầy đủ. Mẫu nhỏ giảm candidate pool nên metric cao hơn không chứng minh cải thiện mô hình. Xem [danh mục lượt chạy thực tế](../results/README.md).

## Lỗi đã sửa trong v4

- CDR: BCE imputation từng bị `no_grad()` làm recommendation gradient giống IPS. Giữ gradient về recommendation, tách mẫu imputation và mặt nạ lọc; dùng phương sai tổng thể.
- IViDR: `iv_proj` từng không nhận gradient, context phụ thuộc batch và treatment trung bình catalog dùng chung cho mọi user. V4 dùng ridge khả vi, context biased-train cố định và trung bình lịch sử exposure từng user bằng CSR.
- iVAE decoder được huấn luyện tái tạo exposure: toàn catalog Coat, 128 item mẫu đều mỗi dòng batch KuaiRand. Nhãn exposure khác nhãn click.
- MF chạy thường và grid dùng chung trainer, optimizer và regularization, reset seed từng trial. Tuning nạp trực tiếp checkpoint thắng.
- CLI từ chối tổ hợp grid không hỗ trợ và tham số số học không hợp lệ. Metric xử lý pool rỗng, catalog nhỏ, tên theo k; full-sort loại truth bị mask.
- Checkpoint/report v4 chứa metadata và hash dữ liệu/cấu hình/mã nguồn; nạp sai phiên bản bị từ chối. Đường dẫn log và feature KuaiRand dùng chung resolver.

## Giới hạn kiến trúc địa phương

IViDR dùng ridge `P = solve(G + 1e-4 I, G)` trong không gian feature. Đây không phải projector trực giao, không bảo đảm fitted/residual trực giao và chưa xác nhận tương đương estimator 2SLS của bài báo. Context và lịch sử dùng biased train; proxy KuaiRand lấy từ cùng user features nên điều kiện IV/proxy cần được kiểm định riêng.

CDR dùng tỷ số độ lệch chuẩn/kỳ vọng loss để lọc, dropout trên embedding, hệ số hiệu chỉnh `(inv_prop - 1)` và lấy mẫu riêng cho từng dataset. Thứ tự BCE trong bước imputation giữ theo mã tác giả; các lựa chọn còn lại được ghi là biến thể địa phương. KuaiRand dùng tần suất exposure item thay propensity thật. Vì vậy tên mô hình không đủ để suy ra tính tương đương bài báo.

## Các yếu tố sai số cần kiểm soát

| Yếu tố | Ảnh hưởng | Cách kiểm soát |
| :-- | :-- | :-- |
| Chọn epoch/hyperparameter trên test | Làm điểm test lạc quan | Chỉ chọn trên validation; khóa cấu hình trước khi xem test |
| Click rate thấp và candidate pool khác nhau | NDCG/Recall phụ thuộc số dương và độ dài danh sách | Báo positive rate, candidate p10/p50/p90, Random và Popularity |
| Loại user không có click dương | Nâng macro metric so với đánh giá mọi user | Ghi rõ quy ước theo iDCF và số user bị loại |
| Train standard log và test random log khác logging policy | Phân phối click, item và user thay đổi | Báo thống kê từng split và so với Popularity |
| Exposure user–video lặp | Một cặp được học nhiều lần; random log có thể có nhiều outcome | Giữ exposure train; trong candidate pool gộp cặp, dương nếu từng có click; ghi quy ước |
| Cold user/video | Embedding thiếu dữ liệu huấn luyện | Lọc random log theo ID trong train và báo tỷ lệ cold |
| Một seed mô hình, nhiều user chung video | CI theo user chưa bao trùm bất định do seed/item | Chạy nhiều seed; tính chênh lệch ghép cặp theo cùng user |
| MF/IViDR/CDR dùng kiến trúc và tuning địa phương | Điểm khác tác giả dù dữ liệu giống | Báo cấu hình, ngân sách và độ lệch chuẩn; không gọi là tái lập nguyên bản |
| CDR dùng tần suất exposure làm propensity proxy | Không chứng minh được xác suất phơi bày thật | Ghi rõ là proxy; không suy diễn hiệu quả nhân quả chỉ từ NDCG |
| IViDR lấy IV/proxy từ user features | Điều kiện nhận dạng có thể không thỏa | Kiểm định giả định IV/proxy trước khi kết luận causal |
| Coat có cặp user–item trùng giữa train và random log | Điểm có thể lạc quan do một số rating test đã được học | Báo số cặp trùng; cân nhắc thêm thí nghiệm loại cặp trùng |

`NDCG@5_CI95_halfwidth` là xấp xỉ `1,96 × sd(user_NDCG) / √n` của **một lần chạy**. Nó không bao gồm độ bất định do seed, chọn mô hình hay thay đổi dữ liệu.

## Cách chạy và tiêu chí hoàn tất benchmark

```powershell
.\venv\Scripts\python.exe experiments\run.py test
.\venv\Scripts\python.exe experiments\run.py benchmark --dataset kuairand --model mf --quick --epochs-kr 1
.\venv\Scripts\python.exe experiments\run.py benchmark --dataset kuairand
```

Trước khi đặt số địa phương cạnh bài báo, chạy mỗi mô hình đến khi validation hội tụ, lặp nhiều seed và công bố mean ± std. Dùng bảng [nguồn bài báo](paper_reference_comparison.md) để ghi rõ các paper dùng split khác nhau.
