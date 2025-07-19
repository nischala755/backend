# ========================================================================================
# FLASK MILITARY VIDEO ANALYZER - WEB APPLICATION
# Enhanced with First Detection Tracking and Real-time Progress Updates
# ========================================================================================

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
import google.generativeai as genai
from PIL import Image
import cv2
import numpy as np
import os
import json
import time
from datetime import timedelta, datetime
import re
from collections import defaultdict
import threading
import uuid
from functools import wraps
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# ========================================================================================
# FLASK APP CONFIGURATION
# ========================================================================================

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Google AI Configuration
API_KEY = "AIzaSyBquxNYulLUPXDjkpnbJEADl_7NPIgpeuA"  # Move to environment variable in production
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Global analysis status tracking
analysis_status = {}
analysis_results = {}

# ========================================================================================
# UTILITY FUNCTIONS
# ========================================================================================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_video_info(video_path):
    """Get basic video information"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return {
        'duration': duration,
        'fps': fps,
        'frame_count': frame_count,
        'resolution': f"{width}x{height}",
        'estimated_tokens': int(duration * 295)
    }

def extract_frames_with_timestamps(video_path, interval_seconds=1.0):
    """Extract frames at specified intervals"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames_data = []
    frame_interval = max(1, int(fps * interval_seconds))
    
    frame_idx = 0
    while frame_idx < frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            timestamp = frame_idx / fps
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_data.append({
                'frame_number': frame_idx,
                'timestamp': timestamp,
                'timestamp_str': str(timedelta(seconds=timestamp)),
                'time_formatted': f"{int(timestamp//60):02d}:{timestamp%60:06.3f}",
                'image': Image.fromarray(frame_rgb)
            })
        
        frame_idx += frame_interval
    
    cap.release()
    return frames_data

def analyze_frame_for_detection(frame_image, frame_number, timestamp):
    """Analyze frame for military object detection"""
    detection_prompt = f"""
    🔍 MILITARY OBJECT DETECTION - Frame #{frame_number} at {timestamp:.3f}s
    
    Analyze this frame for military objects and threats:
    
    **DETECTED OBJECTS:**
    List each object with:
    - OBJECT_TYPE: [tank/vehicle/aircraft/weapon/personnel/building/threat]
    - CONFIDENCE: [0-100%]
    - POSITION: [location description]
    - DETAILS: [specific characteristics]
    
    **THREAT ASSESSMENT:**
    - OVERALL_THREAT: [NONE/LOW/MEDIUM/HIGH/CRITICAL]
    - SPECIFIC_THREATS: [weapons, hostile activities, etc.]
    
    **OBJECT_COUNTS:**
    - Vehicles: [number]
    - Personnel: [number]
    - Aircraft: [number]
    - Weapons: [number]
    
    Be precise and detailed for accurate tracking.
    """
    
    try:
        response = model.generate_content([detection_prompt, frame_image])
        return response.text
    except Exception as e:
        return f"❌ Analysis error: {str(e)}"

def parse_detection_results(analysis_text):
    """Parse analysis to extract structured object data"""
    objects_detected = []
    lines = analysis_text.lower().split('\n')
    
    current_object = {}
    object_types = ['tank', 'vehicle', 'aircraft', 'weapon', 'personnel', 'building', 'threat']
    
    for line in lines:
        line = line.strip()
        
        # Look for object types
        for obj_type in object_types:
            if obj_type in line and ('object_type' in line or obj_type in line.split()[:3]):
                if current_object:
                    objects_detected.append(current_object.copy())
                
                current_object = {
                    'type': obj_type,
                    'confidence': 70,  # default confidence
                    'position': 'unknown',
                    'details': line
                }
                break
        
        # Extract confidence percentage
        confidence_match = re.search(r'(\d+)%', line)
        if confidence_match and current_object:
            current_object['confidence'] = int(confidence_match.group(1))
        
        # Extract position
        position_keywords = ['left', 'right', 'center', 'top', 'bottom', 'front', 'back']
        if any(pos in line for pos in position_keywords) and current_object:
            current_object['position'] = line
    
    if current_object:
        objects_detected.append(current_object)
    
    return objects_detected

def create_detection_timeline_chart(first_detections, output_path):
    """Create timeline visualization of first detections"""
    if not first_detections:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sorted_detections = sorted(first_detections.items(), key=lambda x: x[1]['timestamp'])
    timestamps = [d[1]['timestamp'] for d in sorted_detections]
    objects = [d[0] for d in sorted_detections]
    confidences = [d[1]['confidence'] for d in sorted_detections]
    
    # Color mapping
    color_map = {
        'tank': '#FF4444', 'vehicle': '#FF8844', 'aircraft': '#4488FF',
        'weapon': '#AA44FF', 'personnel': '#44AA44', 'building': '#8B4513',
        'threat': '#CC0000'
    }
    colors = [color_map.get(obj, '#666666') for obj in objects]
    
    # Create scatter plot
    scatter = ax.scatter(timestamps, range(len(objects)), 
                        c=colors, s=[conf*3 for conf in confidences], 
                        alpha=0.7, edgecolors='black', linewidth=1)
    
    # Add labels
    for i, (timestamp, obj, conf) in enumerate(zip(timestamps, objects, confidences)):
        ax.annotate(f'{obj.upper()}\n{conf}%', 
                   (timestamp, i), xytext=(10, 0), 
                   textcoords='offset points', fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_yticks(range(len(objects)))
    ax.set_yticklabels([f'{i+1}. {obj.upper()}' for i, obj in enumerate(objects)])
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_title('🎯 First Detection Timeline - Military Objects', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

# ========================================================================================
# ANALYSIS FUNCTIONS
# ========================================================================================

def analyze_video_comprehensive(video_path, analysis_id, confidence_threshold=60):
    """Comprehensive video analysis with progress tracking"""
    try:
        analysis_status[analysis_id] = {
            'status': 'starting',
            'progress': 0,
            'message': 'Initializing analysis...',
            'start_time': datetime.now()
        }
        
        # Get video info
        video_info = get_video_info(video_path)
        analysis_status[analysis_id].update({
            'status': 'extracting_frames',
            'progress': 10,
            'message': f'Extracting frames from {video_info["duration"]:.1f}s video...',
            'video_info': video_info
        })
        
        # Extract frames
        frames_data = extract_frames_with_timestamps(video_path, interval_seconds=1.0)
        total_frames = len(frames_data)
        
        analysis_status[analysis_id].update({
            'progress': 20,
            'message': f'Analyzing {total_frames} frames...'
        })
        
        # Track detections
        first_detections = {}
        detection_timeline = []
        all_detected_objects = defaultdict(list)
        
        for i, frame_data in enumerate(frames_data):
            # Update progress
            progress = 20 + int((i / total_frames) * 60)  # 20-80% for frame analysis
            analysis_status[analysis_id].update({
                'status': 'analyzing_frames',
                'progress': progress,
                'message': f'Analyzing frame {i+1}/{total_frames} at {frame_data["time_formatted"]}'
            })
            
            # Analyze frame
            analysis = analyze_frame_for_detection(
                frame_data['image'], 
                frame_data['frame_number'], 
                frame_data['timestamp']
            )
            
            detected_objects = parse_detection_results(analysis)
            frame_first_detections = []
            
            # Track first detections
            for obj in detected_objects:
                if obj['confidence'] >= confidence_threshold:
                    obj_type = obj['type']
                    
                    if obj_type not in first_detections:
                        first_detections[obj_type] = {
                            'frame_number': frame_data['frame_number'],
                            'timestamp': frame_data['timestamp'],
                            'time_formatted': frame_data['time_formatted'],
                            'confidence': obj['confidence'],
                            'position': obj['position'],
                            'details': obj['details']
                        }
                        frame_first_detections.append(obj_type)
                    
                    all_detected_objects[obj_type].append({
                        'frame_number': frame_data['frame_number'],
                        'timestamp': frame_data['timestamp'],
                        'confidence': obj['confidence']
                    })
            
            detection_timeline.append({
                'frame_number': frame_data['frame_number'],
                'timestamp': frame_data['timestamp'],
                'time_formatted': frame_data['time_formatted'],
                'analysis': analysis,
                'detected_objects': detected_objects,
                'first_detections_in_frame': frame_first_detections
            })
            
            time.sleep(0.2)  # Rate limiting
        
        # Generate visualizations
        analysis_status[analysis_id].update({
            'status': 'generating_report',
            'progress': 85,
            'message': 'Generating timeline visualization...'
        })
        
        # Create timeline chart
        chart_path = None
        if first_detections:
            chart_filename = f'timeline_{analysis_id}.png'
            chart_path = os.path.join(app.config['RESULTS_FOLDER'], chart_filename)
            create_detection_timeline_chart(first_detections, chart_path)
        
        # Compile final results
        results = {
            'analysis_id': analysis_id,
            'video_info': video_info,
            'first_detections': first_detections,
            'detection_timeline': detection_timeline,
            'all_detected_objects': dict(all_detected_objects),
            'chart_path': chart_path,
            'summary': {
                'total_object_types': len(first_detections),
                'total_frames_analyzed': total_frames,
                'detection_rate': len(first_detections) / video_info['duration'] if video_info['duration'] > 0 else 0,
                'analysis_duration': (datetime.now() - analysis_status[analysis_id]['start_time']).total_seconds()
            }
        }
        
        # Save results
        results_filename = f'results_{analysis_id}.json'
        results_path = os.path.join(app.config['RESULTS_FOLDER'], results_filename)
        with open(results_path, 'w') as f:
            # Convert datetime and other non-serializable objects
            serializable_results = json.loads(json.dumps(results, default=str))
            json.dump(serializable_results, f, indent=2)
        
        # Update final status
        analysis_status[analysis_id] = {
            'status': 'completed',
            'progress': 100,
            'message': f'Analysis complete! Detected {len(first_detections)} object types.',
            'completion_time': datetime.now(),
            'results_path': results_path
        }
        
        analysis_results[analysis_id] = results
        
    except Exception as e:
        analysis_status[analysis_id] = {
            'status': 'error',
            'progress': 0,
            'message': f'Analysis failed: {str(e)}',
            'error_time': datetime.now()
        }

def analyze_image_military(image_path):
    """Analyze military image content"""
    try:
        img = Image.open(image_path)
        
        prompt = """
        Analyze this military image and provide detailed assessment:
        
        **DETECTED OBJECTS:**
        - Military vehicles (tanks, APCs, trucks)
        - Aircraft (helicopters, jets, drones)
        - Personnel and their activities
        - Weapons and equipment
        - Structures and installations
        
        **THREAT ASSESSMENT:**
        - Overall threat level: [NONE/LOW/MEDIUM/HIGH/CRITICAL]
        - Specific threats identified
        - Combat readiness indicators
        
        **STRATEGIC ANALYSIS:**
        - Geographic context
        - Tactical positioning
        - Force composition
        - Operational significance
        
        Be thorough and professional in your analysis.
        """
        
        response = model.generate_content([prompt, img])
        
        return {
            'analysis': response.text,
            'image_info': {
                'size': img.size,
                'mode': img.mode,
                'format': img.format
            }
        }
        
    except Exception as e:
        return {'error': f'Image analysis failed: {str(e)}'}

# ========================================================================================
# FLASK ROUTES
# ========================================================================================

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start analysis"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_filename = f"{timestamp}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(file_path)
    
    # Generate analysis ID
    analysis_id = str(uuid.uuid4())
    
    # Determine file type and start appropriate analysis
    file_ext = filename.lower().split('.')[-1]
    
    if file_ext in ['mp4', 'avi', 'mov', 'mkv', 'wmv']:
        # Start video analysis in background thread
        confidence_threshold = int(request.form.get('confidence', 60))
        thread = threading.Thread(
            target=analyze_video_comprehensive,
            args=(file_path, analysis_id, confidence_threshold)
        )
        thread.start()
        
        return jsonify({
            'analysis_id': analysis_id,
            'file_type': 'video',
            'message': 'Video analysis started'
        })
    
    else:
        # Analyze image immediately
        result = analyze_image_military(file_path)
        
        return jsonify({
            'analysis_id': analysis_id,
            'file_type': 'image',
            'result': result
        })

@app.route('/status/<analysis_id>')
def get_analysis_status(analysis_id):
    """Get analysis progress status"""
    if analysis_id not in analysis_status:
        return jsonify({'error': 'Analysis ID not found'}), 404
    
    return jsonify(analysis_status[analysis_id])

@app.route('/results/<analysis_id>')
def get_analysis_results(analysis_id):
    """Get complete analysis results"""
    if analysis_id not in analysis_results:
        return jsonify({'error': 'Results not found or analysis not complete'}), 404
    
    return jsonify(analysis_results[analysis_id])

@app.route('/download/<analysis_id>')
def download_results(analysis_id):
    """Download analysis results as JSON"""
    if analysis_id not in analysis_status:
        return jsonify({'error': 'Analysis not found'}), 404
    
    if analysis_status[analysis_id]['status'] != 'completed':
        return jsonify({'error': 'Analysis not completed'}), 400
    
    results_path = analysis_status[analysis_id]['results_path']
    return send_file(results_path, as_attachment=True, 
                    download_name=f'military_analysis_{analysis_id}.json')

@app.route('/chart/<analysis_id>')
def get_timeline_chart(analysis_id):
    """Serve timeline chart image"""
    if analysis_id not in analysis_results:
        return jsonify({'error': 'Chart not found'}), 404
    
    chart_path = analysis_results[analysis_id].get('chart_path')
    if not chart_path or not os.path.exists(chart_path):
        return jsonify({'error': 'Chart not available'}), 404
    
    return send_file(chart_path)

@app.route('/api/health')
def health_check():
    """API health check"""
    try:
        # Test Google AI API
        test_response = model.generate_content("Hello")
        api_status = "active"
    except:
        api_status = "error"
    
    return jsonify({
        'status': 'healthy',
        'api_status': api_status,
        'timestamp': datetime.now().isoformat()
    })

# ========================================================================================
# ERROR HANDLERS
# ========================================================================================

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 500MB'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# ========================================================================================
# MAIN APPLICATION
# ========================================================================================

if __name__ == '__main__':
    print("🛡️ FLASK MILITARY VIDEO ANALYZER")
    print("=" * 50)
    print("🚀 Starting web server...")
    print("📹 Video analysis with first detection tracking")
    print("🖼️ Image analysis capabilities")
    print("📊 Real-time progress tracking")
    print("💾 Results download and visualization")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)