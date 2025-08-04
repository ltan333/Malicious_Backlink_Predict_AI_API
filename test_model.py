import re
import torch
from contextlib import nullcontext
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- LABEL MAPPING ---
label2id = {
    "gambling": 0, "movies": 1, "ecommerce": 2, "government": 3, "education": 4, "technology": 5,
    "tourism": 6, "health": 7, "finance": 8, "media": 9, "nonprofit": 10, "realestate": 11,
    "services": 12, "industries": 13, "agriculture": 14
}
id2label = {v: k for k, v in label2id.items()}

# --- DEVICE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autocast_ctx = torch.amp.autocast(device_type="cuda") if torch.cuda.is_available() else nullcontext()

# --- LOAD MODEL ---
def load_model():
    """
    Load the pre-trained classification model and tokenizer from local directory.
    This is called once during app startup.
    """
    global model, tokenizer
    model_path = r"Models\phobert_base_v7"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to(device).eval()
    except Exception as e:
        print("[ERROR] Failed to load model. Check logs for details.")

# --- CLEAN TEXT ---
def clean_text(text):   
    text = text.lower()

    # Preserve domain dots, decimal dots, and URL hyphens
    text = re.sub(r'(\w)\.(?=\w)', r'\1<DOMAIN>', text)
    text = re.sub(r'(\d)\.(?=\d)', r'\1<DECIMAL>', text)
    text = re.sub(r'(\w)-(?=\w)', r'\1<HYPHEN>', text)

    # Remove remaining dots and hyphens
    text = text.replace('.', ' ')
    text = text.replace('-', ' ')

    # Replace one or more underscores with a single space
    text = re.sub(r'_+', ' ', text)

    # Restore preserved characters
    text = text.replace('<DOMAIN>', '.')
    text = text.replace('<DECIMAL>', '.')
    text = text.replace('<HYPHEN>', '-')

    # Handle commas
    text = re.sub(r'(?<=[a-z0-9]),(?=[a-z])', ' ', text)
    text = re.sub(r'(?<=[a-z]),(?=[0-9])', ' ', text)
    text = re.sub(r',(?=\D)|(?<=\D),', '', text)

    # Remove unwanted punctuation (keep quotes, %, /)
    text = re.sub(r'[^\w\s\.,/%"]', '', text)

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Testing manually
def predict_text_class(text_input: str):
    text_input_cleaned = clean_text(text_input)

    # Ensure truncation and padding match the training setup
    inputs = tokenizer(
        text_input_cleaned,
        return_tensors="pt",
        truncation=True,
        padding='max_length',
        max_length=64
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    predicted_label = id2label[predicted_class_id]

    return predicted_label

# Text samples
example_texts = [
    # "Hệ thống nha khoa Tâm Đức Smile dẫn đầu về các dịch vụ cấy ghép Implant, răng sứ thẩm mỹ, niềng răng uy tín hàng đầu, ưu đãi lên đến 60%.",
    # "Nhà Xe Mỹ Duyên - Vận chuyển hành khách và hàng hóa chuyên tuyến Hồ Chí Minh đi Sóc Trăng và ngược lại Nhận vé Nhận mã vé, xác nhận và lên xe",
    # "Công ty tư vấn du học Mỹ cập nhật mới nhất về điều kiện, chi phí, hồ sơ xin visa du học mỹ, cơ hội định cư. Công ty du học Á - Âu 24 năm kinh nghiệm, đỉnh cao uy tín và chất lượng.",
    # "máy rung cầm tay tặng bạn gái",
    # "thd cybersecurity",
    # "Vay Cầm Cố, Giải Ngân 15p Tổng hợp các bài viết từ cơ quan báo chí truyền thông uy tín, phản ánh hoạt động, thành tựu nổi bật và chiến lược phát triển của F88.",
    # "Đường dây đánh bạc, nỗ hủ, cá độ bóng đá rửa tiền lên đến 1.000 tỷ đồng bị triệt phá",
    # "Samsung Galaxy Z Fold7 | Foldable meets Ultra Sleek | Samsung Australia",
    # "đường vào tim em ôi băng giá",
    #  "Nghị định số 46/2017/NĐ-CP ngày 21/4/2017 của Chính phủ quy định về hoạt động đầu tư giáo dục trong các chương trình đào tạo mầm non, phổ thông, đại học; ...",
    # "5 ngày trước — tachi,Nền tảng xổ số trực tuyến với tỷ lệ hoa hồng cho đại lý cao nhất. Hợp tác và kiếm thu nhập thụ động cùng chúng tôi.",
    # "Thông báo tuyển sinh đi học tại Căm-pu-chia diện Hiệp định năm 2025 · THÔNG BÁO TUYỂN SINH ĐI HỌC TẠI MA-RỐC NĂM 2025 · THÔNG BÁO TUYỂN SINH ĐI HỌC TẠI MÔNG CỔ ...",
    # "7 ngày trước — qq188bet! Nền tảng xổ số trực tuyến với tỷ lệ hoa hồng cho đại lý cao nhất. Hợp tác và kiếm thu nhập thụ động cùng chúng tôi. Trở thành đối tác ...",
    # "bốn đôi thông chặt được gì? cách dùng bốn đôi thông hiệu quả. 19 thg 10, 2024 - bốn đôi thông là một tổ hợp bài đặc biệt trong các trò chơi bài miền nam, đặc biệt là tiến lên. khi một người chơi sở hữu bốn đôi bài giống nhau ...",
    # "Nghị định số 46/2017/NĐ-CP ngày 21/4/2017 của Chính phủ quy định về hoạt động đầu tư giáo dục trong các chương trình đào tạo mầm non cá cược phổ thông, đại học; ...",
    "+85 sản phẩm sàn gỗ Malaysia siêu chịu nước giá tốt nhất",
    "Sàn gỗ Malaysia chính hãng, chất lượng, giá cả cạnh tranh được phân phối bởi JANHOME là hệ thống bán lẻ sàn gỗ, sàn nhựa, giấy dán tường, vật liệu nội thất ...",
    "+85 sản phẩm sàn gỗ Malaysia siêu chịu nước giá tốt nhất Sàn gỗ Malaysia chính hãng, chất lượng, giá cả cạnh tranh được phân phối bởi JANHOME là hệ thống bán lẻ sàn gỗ, sàn nhựa, giấy dán tường, vật liệu nội thất ...",
    "giọng_nữ_trầm",
    "giọng_nữ_trầm-Chủ tịch HĐQT Bệnh viện thẩm mỹ Sao Hàn chia sẻ rằng, thẩm mỹ hay phẫu thuật thẩm mỹ thì yêu cầu, mong muốn đầu tiên chắc chắn phải đẹp.",
    "giọng_nữ_trầm giọng_nữ_trầm-Chủ tịch HĐQT Bệnh viện thẩm mỹ Sao Hàn chia sẻ rằng, thẩm mỹ hay phẫu thuật thẩm mỹ thì yêu cầu, mong muốn đầu tiên chắc chắn phải đẹp.",
    "Hệ thống QLVBDH: Trang chủ",
    "Đăng nhập Đăng nhập. Chuyển tới trang đầy đủ. Đăng nhập hệ thống. Phím chuyển chữ hoa đang bật. Đăng nhập. Đăng nhập qua hệ thống xác thực TP. Cần Thơ.",
    "Hệ thống QLVBDH: Trang chủ Đăng nhập Đăng nhập. Chuyển tới trang đầy đủ. Đăng nhập hệ thống. Phím chuyển chữ hoa đang bật. Đăng nhập. Đăng nhập qua hệ thống xác thực TP. Cần Thơ.",
    "Đăng nhập VIC",
    "Tên đăng nhập : Mật khẩu : Đăng nhập. Thoát. VIC 6.5 Được phát triển bởi công ty CINOTEC 282 Lê Quang Định, Phường 11, Quận Bình Thạnh, TP HCM."
    "Đăng nhập VIC Tên đăng nhập : Mật khẩu : Đăng nhập. Thoát. VIC 6.5 Được phát triển bởi công ty CINOTEC 282 Lê Quang Định, Phường 11, Quận Bình Thạnh, TP HCM."
    "Đối tượng bảo trợ",
    "Trung tâm Bảo trợ Xã hội là nơi quản lý chăm sóc, nuôi dưỡng điều trị đối tượng Bảo trợ theo quy định của nhà nước - Địa chỉ: Khu vực Bình Hòa A, ...",
    "Đối tượng bảo trợ Trung tâm Bảo trợ Xã hội là nơi quản lý chăm sóc, nuôi dưỡng điều trị đối tượng Bảo trợ theo quy định của nhà nước - Địa chỉ: Khu vực Bình Hòa A, ...",
    "Đăng ký doanh nghiệp qua mạng điện tử",
    "Hình I.1.2. Biểu mẫu nhập thông tin đăng ký tài khoản. Quý doanh nghiệp nhập đầy đủ các trường thông tin của biểu mẫu theo Hình.",
    "Đăng ký doanh nghiệp qua mạng điện tử Hình I.1.2. Biểu mẫu nhập thông tin đăng ký tài khoản. Quý doanh nghiệp nhập đầy đủ các trường thông tin của biểu mẫu theo Hình.",
    "Đăng nhập hệ thống",
    "Đăng nhập hệ thống. Lưu tài khoản đăng nhập. Đăng nhập. Quên mật khẩu.",
    "Đăng nhập hệ thống Đăng nhập hệ thống. Lưu tài khoản đăng nhập. Đăng nhập. Quên mật khẩu.",
    "Liên đoàn Taekwondo TP Cần Thơ tăng cường chuyển đổi ...",
    "Mục tiêu của Liên đoàn Taekwondo TP Cần Thơ trong năm 2024 là đẩy mạnh chuyển đổi số, xây dựng mô hình quản lý phù hợp điều kiện, xu thế và quy định của Liên ...",
    "Liên đoàn Taekwondo TP Cần Thơ tăng cường chuyển đổi ... Mục tiêu của Liên đoàn Taekwondo TP Cần Thơ trong năm 2024 là đẩy mạnh chuyển đổi số, xây dựng mô hình quản lý phù hợp điều kiện, xu thế và quy định của Liên ...",
    "MẪU CHUYỆN “ĐÔI DÉP BÁC HỒ”",
    "3 thg 10, 2023 — Một anh nhanh tay giành lấy chiếc dép, giơ lên nhưng ngớ ra, lúng túng. Anh bên cạnh liếc thấy, “vượt vây” chạy biến… Bác phải giục:“Ơ kìa, ngắm ...",
    "MẪU CHUYỆN “ĐÔI DÉP BÁC HỒ” 3 thg 10, 2023 — Một anh nhanh tay giành lấy chiếc dép, giơ lên nhưng ngớ ra, lúng túng. Anh bên cạnh liếc thấy, “vượt vây” chạy biến… Bác phải giục:“Ơ kìa, ngắm ...",
    "Ảnh hưởng của các loại thức ăn đến sinh trưởng và tỉ lệ ...",
    "6 thg 12, 2022 — là một trong những loài chân bụng nước ngọt được tìm thấy trong ao nước ngọt, vũng, bể, hồ, đầm lầy, ruộng lúa và đôi khi ở sông suối. Hiện nay, ...",
    "Ảnh hưởng của các loại thức ăn đến sinh trưởng và tỉ lệ ... 6 thg 12, 2022 — là một trong những loài chân bụng nước ngọt được tìm thấy trong ao nước ngọt, vũng, bể, hồ, đầm lầy, ruộng lúa và đôi khi ở sông suối. Hiện nay, ...",
    "Lịch sử hình thành",
    "Ngày 01 tháng 01 năm 2004, tỉnh Cần Thơ được chia tách thành 02 đơn vị hành chính là TP. Cần Thơ và tỉnh Hậu Giang. Bảo tàng tỉnh Cần đổi tên cho phù hợp với ...",
    "Lịch sử hình thành Ngày 01 tháng 01 năm 2004, tỉnh Cần Thơ được chia tách thành 02 đơn vị hành chính là TP. Cần Thơ và tỉnh Hậu Giang. Bảo tàng tỉnh Cần đổi tên cho phù hợp với ...",
    "Hội Cựu chiến binh thành phố Cần Thơ",
    "Thông tin liên hệ. Hội Cựu chiến binh - Thành phố Cần Thơ Địa chỉ : 22 Trần Văn Hoài, P.Xuân Khánh, Q.Ninh Kiều, TP Cần Thơ Điện thoại: (0710) 3832735",
    "Hội Cựu chiến binh thành phố Cần Thơ Thông tin liên hệ. Hội Cựu chiến binh - Thành phố Cần Thơ Địa chỉ : 22 Trần Văn Hoài, P.Xuân Khánh, Q.Ninh Kiều, TP Cần Thơ Điện thoại: (0710) 3832735",
    "văn hóa Địa điểm Chiến thắng Ông Đưa năm 1960",
    "DI TÍCH LỊCH SỬ - VĂN HÓA ĐỊA ĐIỂM CHIẾN THẮNG ÔNG ĐƯA NĂM 1960 ... Di tích lịch sử - văn hóa Địa điểm Chiến thắng Ông Đưa năm 1960 tọa lạc tại ấp Định Khánh A, ...",
    "văn hóa Địa điểm Chiến thắng Ông Đưa năm 1960 DI TÍCH LỊCH SỬ - VĂN HÓA ĐỊA ĐIỂM CHIẾN THẮNG ÔNG ĐƯA NĂM 1960 ... Di tích lịch sử - văn hóa Địa điểm Chiến thắng Ông Đưa năm 1960 tọa lạc tại ấp Định Khánh A, ...",
    "Ước ao của thiếu nhi qua bài hát",
    "ƯỚC AO CỦA THIẾU NHI QUA BÀI HÁT “EM MƠ GẶP BÁC HỒ” CỦA NHẠC SĨ XUÂN GIAO. Nhạc sĩ Xuân Giao quê gốc ở Như Quỳnh, Văn Lâm, Hưng Yên, sinh năm 1932 tại Hải ...",
    "Ước ao của thiếu nhi qua bài hát ƯỚC AO CỦA THIẾU NHI QUA BÀI HÁT “EM MƠ GẶP BÁC HỒ” CỦA NHẠC SĨ XUÂN GIAO. Nhạc sĩ Xuân Giao quê gốc ở Như Quỳnh, Văn Lâm, Hưng Yên, sinh năm 1932 tại Hải ...",
    "BẢNG THANH TOÁN PHỤ CẤP CÁN BỘ CÔNG ĐOÀN",
    "BẢNG THANH TOÁN PHỤ CẤP CÁN BỘ CÔNG ĐOÀN · Các biểu mẫu tài chính Công đoàn cơ sở · Mẫu hướng dẫn Công đoàn cơ sở · Mẫu biểu dự toán, quyết toán tài chính CĐ ...",
    "BẢNG THANH TOÁN PHỤ CẤP CÁN BỘ CÔNG ĐOÀN BẢNG THANH TOÁN PHỤ CẤP CÁN BỘ CÔNG ĐOÀN · Các biểu mẫu tài chính Công đoàn cơ sở · Mẫu hướng dẫn Công đoàn cơ sở · Mẫu biểu dự toán, quyết toán tài chính CĐ ...",
    "dung tin co ay tap 2 Sòng bạc thông thường của Việt Nam",
    "dung tin co ay tap 2 -Xuáº¥t hiá»‡n cÃ¹ng kiá»ƒu tÃ³c layer vuá»'t ngÆ°á»£c vá»›i pháº§n tÃ³c tá»« hai mang tai Ä'á» u Ä'Æ°á»£c háº¥t ngÆ°á»£c ra sau vÃ ...",
    "dung tin co ay tap 2 Sòng bạc thông thường của Việt Nam dung tin co ay tap 2 -Xuáº¥t hiá»‡n cÃ¹ng kiá»ƒu tÃ³c layer vuá»'t ngÆ°á»£c vá»›i pháº§n tÃ³c tá»« hai mang tai Ä'á» u Ä'Æ°á»£c háº¥t ngÆ°á»£c ra sau vÃ ...",
    "Thông tin truy nã, đình nã",
    "Thông tin truy nã, đình nã · Thông báo · Thông tin truy nã · Liên kết webiste · Thăm dò ý kiến · Số lượt truy cập. Trong ngày: Tất cả:.","Cờ bạc"
    "Thông tin truy nã, đình nã Thông tin truy nã, đình nã · Thông báo · Thông tin truy nã · Liên kết webiste · Thăm dò ý kiến · Số lượt truy cập. Trong ngày: Tất cả:.","Cờ bạc"
    "+79 sản phẩm sàn gỗ Florton chính hãng, chất lượng, giá rẻ",
    "SÀN GỖ FLORTON chất lượng, giá rẻ, đạt tiêu chuẩn Châu Âu được cung cấp bởi JANHOME là hệ thống bán hàng tại kho cung cấp vật liệu sàn gỗ giấy dán tường ...",
    "+79 sản phẩm sàn gỗ Florton chính hãng, chất lượng, giá rẻ SÀN GỖ FLORTON chất lượng, giá rẻ, đạt tiêu chuẩn Châu Âu được cung cấp bởi JANHOME là hệ thống bán hàng tại kho cung cấp vật liệu sàn gỗ giấy dán tường ...",
    "Trang chủ - Cần Thơ",
    "Bộ Công Thương vừa ban hành Thông tư quy định việc nhập khẩu mặt hàng gạo và lá thuốc lá khô có xuất xứ từ Campuchia theo hạn ngạch thuế quan năm 2023 và 2024.",
    "Trang chủ - Cần Thơ Bộ Công Thương vừa ban hành Thông tư quy định việc nhập khẩu mặt hàng gạo và lá thuốc lá khô có xuất xứ từ Campuchia theo hạn ngạch thuế quan năm 2023 và 2024.",
    "Login",
    "???login.label.loginheading.left??? ???login.label.userid??? ???login.label.password??? Help. Product documentation · Product wiki · Media gallery ...",
    "Login ???login.label.loginheading.left??? ???login.label.userid??? ???login.label.password??? Help. Product documentation · Product wiki · Media gallery ...",
    "Đăng ký tư vấn từ QR code",
    "Đăng ký tư vấn từ QR code. Xoá Dán (Paste)",
    "Đăng ký tư vấn từ QR code Đăng ký tư vấn từ QR code. Xoá Dán (Paste)",
]

# Load model and tokenizer first
load_model()

# Then run predictions
for i, text in enumerate(example_texts):
    predicted_label = predict_text_class(text)
    text_input_cleaned = clean_text(text)

    print(f"  Input: '{text}'")
    print(f"  After Cleaned: {text_input_cleaned}")
    print(f"  Predicted: {predicted_label}")
    print()