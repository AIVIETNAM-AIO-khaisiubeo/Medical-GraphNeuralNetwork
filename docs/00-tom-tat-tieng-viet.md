# 00 — Tóm tắt lý thuyết (tiếng Việt)

Bản tóm tắt toàn bộ lý thuyết của repo, viết cho người đã biết neural network,
backprop và Transformer. Các file `01`–`05` là bản chi tiết bằng tiếng Anh.

---

## 1. Vấn đề cốt lõi: tại sao không thể đưa graph vào MLP

Một graph `G = (V, E)` gồm **node** (thực thể) và **edge** (quan hệ). Thông tin
có thể nằm ở node features `X ∈ R^{N×F}`, edge features, hoặc global.

Trở ngại lớn nhất là **permutation invariance**. Cách đánh số node là tùy ý:
đổi thứ tự node ta được `A' = P A Pᵀ` nhưng vẫn là *cùng một graph*. Nếu làm
phẳng `A` rồi đưa vào MLP, model sẽ coi `N!` ma trận tương đương là `N!` input
khác nhau. Không học nổi.

Ta cần:
- **invariance** cho bài toán graph-level: đổi thứ tự node → kết quả không đổi;
- **equivariance** cho node-level: đổi thứ tự node → kết quả hoán vị theo.

Thêm hai trở ngại: graph có **kích thước thay đổi** (MLP cần input cố định) và
là không gian **non-Euclidean** (không có "bên trái", chỉ có "cách mấy hop").

**Ba loại bài toán:** node-level (phân loại từng node — Cora), edge-level (dự
đoán liên kết), graph-level (một nhãn cho cả graph — ESOL).

**Biểu diễn thưa:** không dùng ma trận `A` dày `O(N²)` mà dùng danh sách cạnh
`edge_index` dạng `[2, E]`, tốn `O(E)`. Cora có 2708 node nhưng chỉ ~10.500
cạnh = 0.14% số ô của `A`.

## 2. Message passing — ý tưởng duy nhất

```
h_u^(k+1) = UPDATE( h_u^(k),  AGGREGATE({ h_v^(k) : v ∈ N(u) }) )
```

Ba bước: lấy hàng xóm → gộp lại (hàm không phụ thuộc thứ tự: sum/mean/max) →
cập nhật node. Vì định nghĩa trên *tập* hàng xóm nên tự động có tính
permutation-equivariant.

**Độ sâu = số hop.** Sau `k` layer, node thấy được vùng bán kính `k` hop. Đây
là khác biệt lớn nhất so với CNN: mỗi layer nhân vùng tiếp nhận với bậc trung
bình của graph, nên GNN thường chỉ 2–4 layer.

**Nguyên thủy duy nhất cần có là scatter-add:**
```python
out.index_add_(0, index, src)   # out[i] = Σ messages gửi tới node i
```
Đây là lý do repo này không cần thư viện graph nào.

## 3. GCN: suy ra công thức

Bắt đầu từ `A X` (cộng hàng xóm). Hai lỗi:

1. **Node quên chính nó** → thêm self-loop: `Ã = A + I`.
2. **Độ lớn tỉ lệ với bậc node** → chuẩn hoá. GCN dùng chuẩn hoá đối xứng:

```
Â = D̃^(-1/2) Ã D̃^(-1/2)          Â_ij = 1 / √(d_i · d_j)
```

Vì sao đối xứng mà không phải trung bình theo hàng `D̃⁻¹Ã`? Vì (a) tin nhắn từ
một node "quen biết nhiều" bị chia thêm `√d_j` nên ít đặc trưng hơn, và (b) `Â`
đối xứng nên trị riêng thực, nằm trong `[-1,1]`, không nổ cũng không triệt tiêu
khi nhân nhiều lần.

Layer GCN:  `H^(k+1) = σ( Â H^(k) W^(k) )`, trong đó chỉ `W` là tham số học.

Trong code ta **không** dựng ma trận `N×N`: `Â_ij` là một số vô hướng *trên mỗi
cạnh*, nên phép nhân ma trận thành weighted scatter-add (3 dòng trong
`gnn/layers.py`). Test `test_gcn_matches_dense` khẳng định kết quả trùng khớp
với `Â H W` dạng dày.

**So với CNN:** kernel 3×3 có trọng số *khác nhau cho từng vị trí tương đối*.
GCN không làm được vậy vì graph không có "góc trên trái" — mọi hàng xóm dùng
chung `W`, chỉ khác nhau ở số vô hướng `Â_ij`. Phần còn lại của khoá học chính
là để lấy lại khả năng đó (GAT, R-GCN, edge features).

## 4. Over-smoothing — sâu hơn là tệ hơn

Đo trên Cora:

| số layer | 1 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| accuracy | 0.652 | **0.810** | 0.743 | 0.510 | 0.138 | 0.151 | 0.082 |
| MAD | 0.682 | 0.319 | 0.335 | 0.288 | 0.046 | 0.001 | 0.010 |

**Nguyên nhân 1:** sau `k` layer, computational graph của các node trùng nhau
gần hết → đầu vào giống nhau → đầu ra giống nhau.

**Nguyên nhân 2 (bản chất):** bỏ `W` và `σ` đi thì `k` layer chính là `Â^k`.
`Â` có trị riêng lớn nhất đúng bằng 1 ứng với vector hằng (theo trọng số bậc),
mọi thành phần khác tắt dần theo `λ_i^k`. Vậy `Â^k X` hội tụ về **một vector
duy nhất cho mọi node** — mọi thông tin phân biệt bị trung bình hoá mất.

**Hai thước đo:**
- **MAD** = khoảng cách cosine trung bình giữa các embedding. MAD → 0 nghĩa là
  các node đã sụp về cùng một hướng.
- **MADGap** = MAD(cặp ở xa) − MAD(cặp hàng xóm). Model khoẻ thì kéo hàng xóm
  lại gần nhưng giữ node xa tách biệt → MADGap dương lớn. Trên Cora nó đi từ
  `+0.611` (1 layer) xuống `−0.004` (8 layer).

**Cách khắc phục.** Slide đề xuất NGNN (chèn MLP phi tuyến *bên trong* mỗi
layer). Đo thực tế thì nó **không cứu được**: giúp chút ở 6–8 layer (lúc cả hai
đều vô dụng) và *làm tệ đi* ở 2–3 layer. Thứ thực sự hiệu quả, theo thứ tự:
(1) **đừng làm sâu** — 2 layer là đáp án đúng; (2) **residual connection**;
(3) **normalisation** giữa các layer (PairNorm); (4) giữ lại feature gốc
(APPNP, GCNII); (5) jumping knowledge.

## 5. Edge features và quan hệ

GCN thuần chỉ dùng được một số vô hướng `Â_uj` trên mỗi cạnh — *loại* quan hệ
hoàn toàn vô hình. Ba mức độ bổ sung:

**(a) Edge weight** — một số thực trên mỗi cạnh (cường độ tương tác). Không cần
đổi kiến trúc; mọi layer trong repo đã nhận `edge_weight`.

**(b) R-GCN** — mỗi *loại* quan hệ một ma trận riêng:
```
h_u' = W_0 h_u + Σ_r (1/|N_u^r|) Σ_{v ∈ N_u^r} W_r h_v
```

**(c) Edge features nhiều chiều (MPNN)** — message là MLP trên cả hai đầu mút
và vector đặc trưng của cạnh:
```
m_uv = MLP([h_u ‖ h_v ‖ e_uv])
```
Dạng này bao trùm cả (a) và (b), nhưng MLP chạy *trên mỗi cạnh* nên đắt nhất.

**Kết quả đo (experiment 05).** Graph tổng hợp: nhãn của node = tín hiệu của
*một* hàng xóm loại `family`, còn `friend`/`colleague` là nhiễu.

| model | accuracy |
|---|---|
| GCN | 0.322 |
| GAT | 0.439 |
| **R-GCN** | **1.000** |
| ngẫu nhiên | 0.333 |

Điểm tinh tế: **attention cũng không cứu được**, vì GAT tính `α_ij` từ
*feature* của node, mà ở đây hàng xóm nhiễu trông y hệt hàng xóm hữu ích. Thứ
cần thiết là *loại cạnh* — thứ GAT không nhìn thấy.

## 6. Attention (GAT)

```
e_ij = LeakyReLU( aᵀ [W h_i ‖ W h_j] )
α_ij = softmax trên j ∈ N(i)
h_i' = σ( Σ_j α_ij W h_j )
```

**Nếu bạn biết Transformer:** đây chính là self-attention với hai khác biệt —
(1) chỉ attend tới **hàng xóm trong graph** thay vì mọi token (ma trận kề đóng
vai trò **attention mask**), và (2) điểm số là MLP cộng thay vì tích vô hướng
có scale. Self-attention là trường hợp đặc biệt khi graph đầy đủ. Chi phí cũng
khác: Transformer `O(N²)`, GAT `O(E)`.

Chi tiết cài đặt quan trọng: softmax phải tính **theo từng node đích**
(segment softmax), và phải trừ max từng segment trước khi `exp` để không tràn
số. Nhớ thêm self-loop trước khi attend.

**Kết quả và một cảnh báo.** Trên Cora: GCN 0.810, GAT 0.818 — thắng rất ít. Soi
vào hệ số, `α_ij` chia cho trọng số đều `1/degree` nằm trong khoảng 0.98–1.02
(layer 1): attention học ra **gần như đồng đều**. Không phải bug — tham số
attention thay đổi 60–130% khi train. Lý do nằm ở dữ liệu: Cora **homophilous**
rất mạnh, gần như mọi trích dẫn đều nối hai bài cùng chủ đề, nên không có gì để
attention phải giảm trọng số. Attention chỉ đáng giá khi các hàng xóm thực sự
khác nhau về mức hữu ích.

## 7. Dự đoán mức graph

Cần một **readout** bất biến hoán vị để gộp số node thay đổi thành một vector:
```
h_graph = POOL({h_v})        ŷ = MLP(h_graph)
```

Lựa chọn readout không hề nhỏ. Trên ESOL, cùng model, chỉ đổi pooling:

| readout | test RMSE |
|---|---|
| **sum** | **0.621** |
| mean | 0.864 |
| max | 0.980 |

Sum thắng rõ rệt vì độ tan tỉ lệ với **kích thước phân tử**, mà mean pooling
chia đúng thông tin đó đi. Quy tắc: đại lượng *extensive* (tăng theo kích
thước) → dùng sum; đại lượng *intensive* (tỉ lệ, trung bình) → dùng mean.

**Gộp batch nhiều graph nhỏ:** xếp chúng thành *một graph lớn rời rạc* — nối
các ma trận feature, dịch `edge_index` theo offset, và giữ vector `batch` ghi
mỗi node thuộc graph nào. Không cạnh nào nối hai graph nên message không rò rỉ.

---

## 8. Ba chỗ số liệu không khớp với slide

Đây là phần đáng chú ý nhất khi chạy thật:

1. **"GCN chưa train đã thấy community" (Deck 2 p.36).** Đo trên 12 seed,
   untrained GCN chỉ đạt **0.529** (ngẫu nhiên là 0.5). Claim này **không
   reproduce**. Sự thật: cấu trúc nằm trong *toán tử truyền tin* `Â`, không
   nằm trong random weights. Propagation thuần `h ← Â h` đạt **0.880**, và dấu
   của **eigenvector thứ 2** của `Â` đạt **0.941** — đó chính là spectral
   clustering.

2. **Kiến trúc GCN trong slide mất ~4.5 điểm.** Deck 1 p.71 mô tả "2 MP layer +
   1 linear output" tức `1433 → 16 → 16 → 7`, ép feature qua *hai* nút thắt 16
   chiều, đạt **0.765**. GCN chuẩn (`1433 → 16 → 7`, layer thứ 2 xuất thẳng
   logit) đạt **0.810**. Repo cài cả hai (`head="linear"` / `head="direct"`).

3. **Attention của GAT gần như không làm gì trên Cora** — xem mục 6.

---

## Kết quả chính

| thí nghiệm | kết quả |
|---|---|
| Cora: MLP (không graph) | 0.537 |
| Cora: GCN | **0.810** — graph đáng giá **+51% tương đối** |
| Cora: GAT | 0.818 |
| Over-smoothing | 0.810 (2 layer) → 0.082 (8 layer), MAD 0.682 → 0.001 |
| Relational: GCN / GAT / R-GCN | 0.322 / 0.439 / **1.000** |
| ESOL graph regression | RMSE **0.621** (baseline đoán trung bình: 2.087) |

Con số 0.810 của GCN khớp với kết quả công bố (~0.815) — đây là bằng chứng
chính cho thấy bản cài đặt from-scratch này đúng.
