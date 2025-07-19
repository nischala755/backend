from flask import Flask, render_template, request, redirect, url_for, send_file
from PIL import Image
import io
import os
import cv2
import numpy as np
import google.generativeai as genai
import base64
import json

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure Gemini API
API_KEY = "AIzaSyDLlDmcpbq_lip5UNLW_J1LMnHp4JLRt3k"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# ───────────────────────────────────────────────────────────────
# Image Enhancement Function
# ───────────────────────────────────────────────────────────────
def enhance_image(image):
    """Enhance uploaded image for better analysis."""
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    denoised = cv2.bilateralFilter(opencv_image, 9, 75, 75)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    return enhanced_pil

# ───────────────────────────────────────────────────────────────
# Prompt Dictionary
# ───────────────────────────────────────────────────────────────
prompts = {
    "comprehensive": """🎯 MILITARY IMAGE ANALYSIS - COMPREHENSIVE REPORT ... [TRUNCATED FOR BREVITY]""",
    "quick": """Provide a QUICK military analysis of this image: ...""",
    "detection": """Focus on OBJECT DETECTION in this military image: ...""",
    "threat": """Focus on THREAT ASSESSMENT for this military image: ..."""
}

# ───────────────────────────────────────────────────────────────
# Analysis Function
# ───────────────────────────────────────────────────────────────
def analyze_image_with_gemini(pil_image, mode):
    try:
        response = model.generate_content([prompts[mode], pil_image])
        return response.text
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ───────────────────────────────────────────────────────────────
# Routes
# ───────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(url_for('home'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('home'))

    analysis_type = request.form.get('analysis_type', 'comprehensive')
    image = Image.open(file.stream).convert("RGB")
    enhanced = enhance_image(image)

    # Run analysis
    result = analyze_image_with_gemini(enhanced, analysis_type)

    # Save enhanced image temporarily
    enhanced_io = io.BytesIO()
    enhanced.save(enhanced_io, format='PNG')
    enhanced_io.seek(0)
    encoded_img = base64.b64encode(enhanced_io.read()).decode('utf-8')

    return render_template('result.html',
                           analysis=result,
                           image_data=encoded_img,
                           original_filename=file.filename,
                           analysis_type=analysis_type)

@app.route('/api_status')
def api_status():
    try:
        test_response = model.generate_content("Hello, are you working?")
        return f"<h3>✅ API is active</h3><p>{test_response.text}</p>"
    except Exception as e:
        return f"<h3>❌ API Error</h3><p>{str(e)}</p>"

# ───────────────────────────────────────────────────────────────
# Run App
# ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)
