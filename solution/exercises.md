# K4 — Ngày 1: Bài Tập & Phản Ánh

## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature

Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)

> Khi temperature tăng từ 0.0 lên 1.5, câu trả lời có xu hướng sáng tạo hơn. Ở temperature thấp, model thường trả lời ổn định và tập trung vào một thông tin; ở temperature cao hơn, phản hồi có tính ngẫu nhiên đa dạng hơn

### Câu 1.2 — Chọn temperature cho sản phẩm

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**

> Em sẽ đặt temperature khoảng 0.3 – 0.5 cho chatbot hỗ trợ khách hàng. Với chatbot hỗ trợ khách hàng, tính chính xác và nhất quán quan trọng hơn sự sáng tạo. Mức temperature thấp giúp câu trả lời ổn định, nhất quán và ít tạo ra thông tin không cần thiết, trong khi đó vẫn đủ linh hoạt để diễn đạt tự nhiên.

### Câu 1.3 — Đánh đổi chi phí

Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**

> GPT-4o đắt hơn GPT-4o-mini khoảng 18.75 lần đối với workload này. GPT-4o dùng cho các tác vụ phức tạp, cần khả năng suy luận và chất lượng phản hồi cao, ví dụ phân tích vấn đề chuyên sâu. Trong khi đó, GPT-4o-mini dùng cho các tác vụ đơn giản, có số lượng request lớn như chatbot FAQ, phân loại nội dung.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona

Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:

- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)

> Khi sử dụng cùng một câu hỏi nhưng thay đổi system prompt, mô hình cho ra cách trả lời khác nhau rõ rệt. Với persona giáo viên tiểu học, câu trả lời sử dụng cách diễn đạt đơn giản, thân thiện và hướng đến trẻ em; trong khi với chuyên gia tài chính, mô hình có xu hướng sử dụng thuật ngữ chuyên môn và cách trình bày chuyên sâu hơn. Điều này cho thấy system prompt có ảnh hưởng trực tiếp đến phong cách, mức độ phức tạp của từ vựng và cách mô hình trình bày nội dung. Vì vậy, có thể sử dụng system prompt để định hướng mô hình phù hợp với từng đối tượng và mục đích sử dụng.

### Câu 2.2 — tiktoken vs đếm từ

Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**

> Với đoạn văn khoảng 21 từ, phương pháp tiktoken cho kết quả 44 token, trong khi công thức ước lượng số từ/0.75 chỉ cho khoảng 28 token. Hai kết quả chênh lệch khoảng 57,14%. Nguyên nhân là token không tương đương với từ. Tokenizer có thể chia một từ thành nhiều token dựa trên các chuỗi ký tự hoặc subword. Đối với tiếng Việt, các từ có dấu và các chuỗi ký tự ít phổ biến trong dữ liệu huấn luyện thường bị tách thành nhiều token hơn, do đó số token có thể cao hơn đáng kể so với cách ước lượng dựa trên số từ.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming

**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)

> Streaming quan trọng nhất khi câu trả lời của AI dài và người dùng phải chờ một khoảng thời gian đáng kể. Nếu đợi toàn bộ kết quả rồi mới hiện thì sẽ gây cho người dùng cảm giác chờ đợi, còn khi hiện thị từng phần ngay thì người dùng sẽ có cảm giác hệ thống phản hồi nhanh hơn và có thể bắt đầu đọc ngay. Ngược lại, non-streaming phù hợp hơn với các tác vụ cần nhận toàn bộ kết quả cùng lúc, chẳng hạn như chương trình cần chờ toàn bộ response để thực hiện bước xử lý tiếp theo

### Câu 3.2 — Vì sao backoff theo cấp số nhân?

**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**

> Exponential backoff giúp hệ thống giảm áp lực lên API khi xảy ra quá tải hoặc lỗi tạm thời.Thời gian chờ sẽ tăng dần sau mỗi lần retry, điều này giúp client không liên tục gửi request vào một hệ thống đang quá tải, tạo cơ hội để hệ thống phục hồi. Nếu hàng nghìn client cùng retry với delay cố định 1 giây, chúng có thể gửi lại request gần như cùng một thời điểm, tạo ra các đợt request dồn dập, khiến API càng quá tải và có thể tiếp tục thất bại.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona

**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**

> Em chọn từ "trợ giảng thân thiện" để định hướng cách trả lời của model theo hướng dễ hiểu, gần gũi và phù hợp với người đang học. Từ "ngắn gọn" giúp câu trả lời tập trung vào nội dung chính, tránh giải thích quá dài.

### Câu 4.2 — Hạn chế & cải thiện

**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**

> Hạn chế lớn nhất của trợ lý hiện tại là history chỉ lưu tối đa 3 lượt hội thoại gần nhất, tương đương 6 message. Vì vậy, nếu cuộc trò chuyện dài hơn, trợ lý có thể không còn nhớ những thông tin được trao đổi ở các lượt đầu. Một cải thiện cụ thể là xây dựng bộ nhớ dài hạn bằng cách lưu lịch sử hội thoại vào cơ sở dữ liệu. Khi người dùng bắt đầu một phiên chat mới, hệ thống có thể lấy lại những thông tin quan trọng từ các cuộc trò chuyện trước và đưa chúng vào context của model. Để tránh context quá dài, có thể tóm tắt các cuộc hội thoại cũ và chỉ đưa những thông tin liên quan vào prompt khi cần.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
