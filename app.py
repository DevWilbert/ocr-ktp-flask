from flask import Flask, request, jsonify, make_response
import easyocr
import numpy as np
import cv2, os
from ultralytics import YOLO
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy.ndimage import rotate
from io import BytesIO
import re, datetime, textdistance, time

app = Flask(__name__)
app.json.sort_keys = False  # Ensure JSON response keeps order

MODEL_DIR = 'models/best.pt'  # Update with your model path
model = YOLO(MODEL_DIR)  # Load YOLO model

API_KEY = os.getenv('API_KEY', 'TEST') 

def validate_api_key(api_key):
    return api_key == API_KEY

def api_key_required(f):
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-KEY')
        if not api_key or not validate_api_key(api_key):
            error_response = make_response(jsonify({
                'error': True, 
                'message': 'API key required',
                'data':{}
                }), 401)
            return error_response
        return f(*args, **kwargs)
    return decorated_function

def correct_skew(img_blur, delta=1, limit=90):
    def determine_score(arr, angle):
        data = rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = img_blur.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(img_blur, M, (w, h), flags=cv2.INTER_CUBIC, \
                               borderMode=cv2.BORDER_REPLICATE)

    return best_angle, corrected

def fix_nik_characters(extracted_text):
    # Convert common misrecognized characters to their number equivalents
    replacements = {
        "!": "1", "l": "1", ")": "1", "L": "1", "|": "1", "]": "1", "I": "1", "i": "1",
        "o": "0", "O": "0", "D": "0", "Q": "0",
        "s": "5", "S": "5",
        "b": "6", "G": "6",
        "?": "7", "T": "7",
        "B": "8", "&": "8",
        "g": "9", "q": "9"
    }

    for char, replacement in replacements.items():
        extracted_text = extracted_text.replace(char, replacement)

    # Remove any non-numeric characters that might remain
    extracted_text = re.sub(r'[^0-9]', '', extracted_text)

    # NIK should be exactly 16 digits in Indonesia
    if len(extracted_text) > 16:
        extracted_text = extracted_text[:16]

    return extracted_text

def validate_nik(nik):
    # Check if NIK has exactly 16 digits
    is_digits_16 = bool(re.match(r'^\d{16}$', nik))
    
    # Validate NIK with the detailed regex (matching Node.js implementation)
    is_valid_nik = bool(re.match(r'^(1[1-9]|21|[37][1-6]|5[1-3]|6[1-5]|[89][12])\d{2}\d{2}([04][1-9]|[1256][0-9]|[37][01])(0[1-9]|1[0-2])\d{2}\d{4}$', nik))
    
    return {
        "digits": len(nik),
        "validDigits": is_digits_16,
        "validNIK": is_valid_nik,
        "success": is_digits_16 and is_valid_nik
    }

def extract_date(date_text):
    try:
        match = re.search(r"(\d{1,2})([-/\.])(\d{2})\2(\d{4})", date_text)
        if match:
            day, month, year = int(match.group(1)), int(match.group(3)), int(match.group(4))
            return datetime.datetime(year, month, day)
        parsed_date = datetime.datetime.strptime(date_text, "%Y %m-%d")
        return parsed_date
    except ValueError:
        pass

    date_pattern = r"(\d{1,4})(?:[-/\.])(\d{1,2})(?:[-/\.])(\d{2,4})"
    match = re.search(date_pattern, date_text)
    if match:
        day, month, year = map(lambda x: int(x) if 1 <= int(x) <= 31 else None, match.groups())
        if day is not None and month is not None and year is not None:
            try:
                return datetime.datetime(year, month, day)
            except ValueError:
                return None
    return None

@app.route('/')
def hello():
    return jsonify({"error": False, "message": "System Ready!","data":{}}), 200
            
@app.route('/healthz')
def healthz():
    return jsonify({"error": False, "message": "System Healthy!","data":{}}), 200

@app.route('/ocr', methods=['POST'])
@api_key_required
def ocr_ktp():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No file part", "message": "No image file uploaded"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file", "message": "No image file uploaded"}), 400
    
    # Check if the request is for KTP only validation
    is_only_ktp = request.args.get('ktp') == 'true'
    
    try:
        img = Image.open(BytesIO(file.read()))
        print('Preprocess:', file)
        img = img.convert('RGB')
        img = ImageOps.exif_transpose(img)
        img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_cv2 = cv2.resize(img_cv2, (640, 480))
        img_blur = cv2.GaussianBlur(img_cv2, (3, 3), 0)

        angle, corrected = correct_skew(img_blur)
        print('Rotate angle:', angle)
        img_pil = Image.fromarray(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
        img_pil = img_pil.filter(ImageFilter.SHARPEN)
        enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = enhancer.enhance(2)
        img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        results = model.predict(np.array(img_cv2), imgsz=(480, 640), iou=0.7, conf=0.5)
        pil_img = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

        extracted_data = {}
        start_time = time.time()
        has_detections = False

        # Initialize EasyOCR reader
        reader = easyocr.Reader(['id'])

        # Process YOLO detections if any
        for result in results:
            if len(result.boxes) > 0:
                has_detections = True
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    class_id = box.cls[0].item()
                    confidence = box.conf[0].item()
                    class_name = model.names[class_id]

                    cropped_img_pil = pil_img.crop((x1, y1, x2, y2))
                    cropped_img_cv2 = cv2.cvtColor(np.array(cropped_img_pil), cv2.COLOR_RGB2BGR)

                    # Use EasyOCR to extract text
                    ocr_result = reader.readtext(cropped_img_cv2, workers=0)
                    extracted_text = " ".join([detection[1] for detection in ocr_result])

                    extracted_data[class_name] = extracted_text

                    if class_name == 'jk':
                        if textdistance.levenshtein(extracted_text.upper(), "LAKI-LAKI") < textdistance.levenshtein(extracted_text.upper(), "PEREMPUAN"):
                            extracted_data[class_name] = "LAKI-LAKI"
                        else:
                            extracted_data[class_name] = "PEREMPUAN"
                    if class_name == 'nik':
                        extracted_data[class_name] = fix_nik_characters(extracted_text)
                    if class_name == 'ttl':
                        match = re.search(r'\d', extracted_text)
                        if match:
                            index = match.start()
                            extracted_data['tempat_lahir'] = extracted_text[:index].strip()
                            extracted_data['tgl_lahir'] = extract_date(extracted_text[index:].strip())

        # If no detections found or NIK is missing, use fallback method
        if not has_detections or 'nik' not in extracted_data or not extracted_data['nik']:
            print("Using fallback method for entire image or missing NIK...")
            
            # Define regions of interest for different fields in an Indonesian ID card
            h, w = img_cv2.shape[:2]
            roi_regions = {
                'nik': (0, int(h * 0.1), int(w * 0.6), int(h * 0.15)),
            }
            
            # Only process NIK region if NIK is missing or no detections
            if 'nik' not in extracted_data or not extracted_data['nik']:
                for field, (x1, y1, x2, y2) in roi_regions.items():
                    roi = img_cv2[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    
                    # Use EasyOCR for ROI
                    ocr_result = reader.readtext(roi, workers=0)
                    extracted_text = " ".join([detection[1] for detection in ocr_result])
                    
                    extracted_data[field] = fix_nik_characters(extracted_text)
            
            # If still no NIK or NIK is empty, process the entire image
            if 'nik' not in extracted_data or not extracted_data['nik']:
                # Use EasyOCR for full image
                full_ocr_result = reader.readtext(img_cv2, workers=0)
                full_text = " ".join([detection[1] for detection in full_ocr_result])
                
                # Extract NIK using regex pattern
                nik_pattern = r'NIK\s*:?\s*([0-9lLI|!][-0-9lLI|!\s]*)'
                nik_match = re.search(nik_pattern, full_text)
                if nik_match:
                    nik_text = nik_match.group(1)
                    extracted_data['nik'] = fix_nik_characters(nik_text)

        # Process province and kabupaten data
        prov_kab = extracted_data.get('prov_kab', '')
        provinsi = ""
        kabupaten = ""

        if "KOTA" in prov_kab:
            provinsi, kabupaten = prov_kab.split("KOTA", 1)
            kabupaten = "KOTA " + kabupaten.strip()
        elif "KABUPATEN" in prov_kab:
            provinsi, kabupaten = prov_kab.split("KABUPATEN", 1)
            kabupaten = "KABUPATEN " + kabupaten.strip()
        elif "JAKARTA" in prov_kab:
            provinsi, kabupaten = prov_kab.split("JAKARTA", 1)
            kabupaten = kabupaten.strip()
            provinsi = "PROVINSI DKI JAKARTA"
        else:
            provinsi = prov_kab
            kabupaten = ""
        provinsi = provinsi.strip()

        finish_time = time.time() - start_time
        
        # Get the NIK and perform validation
        nik = extracted_data.get('nik', '').strip()
        nik_validation = validate_nik(nik)
        
        # If the request is for KTP only validation, return simplified response
        if is_only_ktp:
            response = {
                "success": nik_validation["success"],
                "nik": nik,
                "digits": nik_validation["digits"],
                "validDigits": nik_validation["validDigits"],
                "validNIK": nik_validation["validNIK"]
            }
            
            status_code = 200 if nik_validation["success"] else 400
            return jsonify(response), status_code
        
        # Otherwise, return the full response with KTP data
        ktp_data = {
            "nik": nik,
            "nama": extracted_data.get('nama', '').upper().strip().replace(":", ""),
            "tempat_lahir": extracted_data.get('tempat_lahir', '').upper().strip(),
            "tgl_lahir": extracted_data.get('tgl_lahir', '').strftime('%d-%m-%Y') if extracted_data.get('tgl_lahir') else '',
            "jenis_kelamin": extracted_data.get('jk', '').upper().strip(),
            "agama": extracted_data.get('agama', '').upper().strip(),
            "status_perkawinan": extracted_data.get('perkawinan', '').upper().strip(),
            "pekerjaan": extracted_data.get('pekerjaan', '').upper().strip(),
            "alamat": {
                "name": extracted_data.get('alamat', '').upper().strip(),
                "rt_rw": extracted_data.get('rt_rw', '').strip(),
                "kel_desa": extracted_data.get('kel_desa', '').upper().strip(),
                "kecamatan": extracted_data.get('kecamatan', '').upper().strip(),
                "kabupaten": kabupaten.upper(),
                "provinsi": provinsi.upper()
            },
            "detection_method": "yolo" if has_detections else "fallback",
            "time_elapsed": round(finish_time, 3)
        }
        
        response = {
            "success": nik_validation["success"],
            "nik": nik,
            "digits": nik_validation["digits"],
            "validDigits": nik_validation["validDigits"],
            "validNIK": nik_validation["validNIK"],
            "data": ktp_data
        }
        
        status_code = 200 if nik_validation["success"] else 400
        return jsonify(response), status_code

    except Exception as e:
        print('KTP extraction failed:', str(e))
        return jsonify({
            "success": False,
            "error": "Failed to extract KTP data",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)