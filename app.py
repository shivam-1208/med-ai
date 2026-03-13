"""
PNEUMONIA DETECTION & CONSULTATION SYSTEM v2.0
===============================================
Complete Flask application with:
- User Authentication (Patient & Doctor)
- DenseNet121 CNN Model for Pneumonia Detection
- LLM Report Generation
- AI Chatbot
- Doctor Consultation with Video Call
- Report Management
- Dashboard for Both Users
"""

import os
import io
import base64
import json
import uuid
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, render_template, request, jsonify, session, send_file, redirect, url_for, flash
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
import numpy as np
from groq import Groq
from dotenv import load_dotenv
import logging

# PyTorch imports for DenseNet121 model
import torch
from torchvision import models, transforms

load_dotenv()  # loads variables from .env file

# ============================================================================
# LOGGING CONFIGURATION (MUST BE FIRST!)
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# DENSENET121 PNEUMONIA DETECTION MODEL
# ============================================================================
class PneumoniaDetectionModel:
    """DenseNet121 PyTorch model for pneumonia detection"""
    
    def __init__(self, model_path='densenet121_pneumonia.pth'):
        """Initialize the DenseNet121 model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        
        logger.info(f"🔧 Using device: {self.device}")
        
        # Load DenseNet121 model
        self.model = models.densenet121(pretrained=False)
        num_features = self.model.classifier.in_features
        self.model.classifier = torch.nn.Linear(num_features, 1)
        
        # Load trained weights
        if not os.path.exists(model_path):
            logger.warning(f"⚠️ Model file '{model_path}' not found!")
            logger.warning(f"   Current directory: {os.getcwd()}")
            logger.warning(f"   Using UNTRAINED model - predictions will be RANDOM")
            self.model_loaded = False
        else:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                self.model.to(self.device)
                self.model_loaded = True
                logger.info(f"✅ Model loaded successfully from {model_path}")
            except Exception as e:
                logger.error(f"❌ Error loading model: {e}")
                self.model_loaded = False
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path, threshold=0.6):
        """
        Predict pneumonia from chest X-ray image
        
        Args:
            image_path: Path to X-ray image or PIL Image object
            threshold: Probability threshold for pneumonia detection (default: 0.6)
        
        Returns:
            dict: Prediction results with confidence scores
        """
        
        # If model not loaded, return error immediately
        if not self.model_loaded:
            logger.warning("⚠️ Model not loaded - returning error response")
            return {
                'prediction': 'Error - Model Not Loaded',
                'confidence': 0.0,
                'pneumonia_probability': 0.0,
                'normal_probability': 0.0,
                'severity': None,
                'risk_level': 'Unknown',
                'error': 'Model file not found. Please place densenet121_pneumonia.pth in the project directory.'
            }
        
        try:
            # Load and convert image to RGB
            if isinstance(image_path, str):
                image = Image.open(image_path).convert("RGB")
            elif isinstance(image_path, Image.Image):
                image = image_path.convert("RGB")
            else:
                # Try to open as file-like object
                image = Image.open(image_path).convert("RGB")
            
            # Transform image
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(image_tensor)
                pneumonia_prob = torch.sigmoid(output).item()
            
            # Determine prediction based on threshold
            is_pneumonia = pneumonia_prob > threshold
            confidence = pneumonia_prob if is_pneumonia else (1 - pneumonia_prob)
            
            # Determine severity and risk level
            severity = None
            risk_level = 'Low'
            
            if is_pneumonia:
                if pneumonia_prob >= 0.85:
                    severity = 'Severe'
                    risk_level = 'High'
                elif pneumonia_prob >= 0.70:
                    severity = 'Moderate'
                    risk_level = 'Medium'
                else:
                    severity = 'Mild'
                    risk_level = 'Medium'
            
            result = {
                'prediction': 'Pneumonia Detected' if is_pneumonia else 'Normal',
                'confidence': confidence,
                'pneumonia_probability': pneumonia_prob,
                'normal_probability': 1 - pneumonia_prob,
                'severity': severity,
                'risk_level': risk_level,
                'raw_score': pneumonia_prob,
                'threshold_used': threshold
            }
            
            logger.info(f"✅ Prediction: {result['prediction']} (Prob: {pneumonia_prob:.2%}, Threshold: {threshold})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'prediction': 'Error - Prediction Failed',
                'confidence': 0.0,
                'pneumonia_probability': 0.0,
                'normal_probability': 0.0,
                'severity': None,
                'risk_level': 'Unknown',
                'error': str(e)
            }

# Initialize model
logger.info("="*60)
logger.info("🚀 INITIALIZING PNEUMONIA DETECTION SYSTEM")
logger.info("="*60)
model = PneumoniaDetectionModel()
logger.info("="*60)

# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================
# Initialize Flask
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'pneumonia-detection-secret-key-2024')
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

# Initialize SocketIO for video calls
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize Groq for LLM
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================================================
# DATABASE (In-memory - Replace with PostgreSQL/MongoDB)
# ============================================================================
class Database:
    def __init__(self):
        self.users = {}
        self.patients = {}
        self.doctors = {}
        self.reports = {}
        self.chat_sessions = {}
        self.consultations = {}
        self.notifications = {}
        self._create_default_users()
    
    def _create_default_users(self):
        """Create default doctor and patient accounts"""
        # Default Doctor
        self.users['doc123'] = {
            'user_id': 'doc123',
            'email': 'doctor@hospital.com',
            'password': generate_password_hash('doctor123'),
            'user_type': 'doctor',
            'name': 'Dr. Sarah Johnson',
            'specialization': 'Pulmonologist',
            'license_number': 'MED-2024-12345',
            'experience': '15 years',
            'verified': True,
            'created_at': datetime.now().isoformat()
        }
        self.doctors['doc123'] = {
            'total_consultations': 0,
            'pending_reviews': 0,
            'rating': 4.8
        }
        
        # Default Patient
        self.users['pat123'] = {
            'user_id': 'pat123',
            'email': 'patient@email.com',
            'password': generate_password_hash('patient123'),
            'user_type': 'patient',
            'name': 'John Doe',
            'age': 35,
            'phone': '+1234567890',
            'created_at': datetime.now().isoformat()
        }
        self.patients['pat123'] = {
            'reports': [],
            'consultations': []
        }
    
    def create_user(self, email, password, user_type, **kwargs):
        user_id = str(uuid.uuid4())
        self.users[user_id] = {
            'user_id': user_id,
            'email': email.lower(),
            'password': generate_password_hash(password),
            'user_type': user_type,
            'created_at': datetime.now().isoformat(),
            'verified': user_type == 'patient',
            **kwargs
        }
        if user_type == 'doctor':
            self.doctors[user_id] = {
                'total_consultations': 0,
                'pending_reviews': 0,
                'rating': 0.0
            }
        else:
            self.patients[user_id] = {
                'reports': [],
                'consultations': []
            }
        return user_id
    
    def get_user_by_email(self, email):
        for user in self.users.values():
            if user['email'].lower() == email.lower():
                return user
        return None
    
    def get_user(self, user_id):
        return self.users.get(user_id)
    
    def save_report(self, report_id, data):
        self.reports[report_id] = data
        patient_id = data.get('patient_id')
        if patient_id in self.patients:
            if 'reports' not in self.patients[patient_id]:
                self.patients[patient_id]['reports'] = []
            self.patients[patient_id]['reports'].append(report_id)
    
    def get_report(self, report_id):
        return self.reports.get(report_id)
    
    def get_patient_reports(self, patient_id):
        if patient_id not in self.patients:
            return []
        report_ids = self.patients[patient_id].get('reports', [])
        return [self.reports.get(rid) for rid in report_ids if rid in self.reports]
    
    def get_pending_reports(self):
        return [r for r in self.reports.values() if r.get('status') == 'pending_review']
    
    def save_chat_message(self, report_id, message):
        if report_id not in self.chat_sessions:
            self.chat_sessions[report_id] = []
        self.chat_sessions[report_id].append(message)
    
    def get_chat_history(self, report_id):
        return self.chat_sessions.get(report_id, [])
    
    def create_consultation(self, patient_id, doctor_id, report_id):
        consult_id = str(uuid.uuid4())
        self.consultations[consult_id] = {
            'consultation_id': consult_id,
            'patient_id': patient_id,
            'doctor_id': doctor_id,
            'report_id': report_id,
            'status': 'scheduled',
            'created_at': datetime.now().isoformat(),
            'notes': None
        }
        return consult_id
    
    def add_notification(self, user_id, message, notification_type='info'):
        if user_id not in self.notifications:
            self.notifications[user_id] = []
        self.notifications[user_id].append({
            'message': message,
            'type': notification_type,
            'timestamp': datetime.now().isoformat(),
            'read': False
        })
    
    def get_notifications(self, user_id):
        return self.notifications.get(user_id, [])

db = Database()

# ============================================================================
# DECORATORS
# ============================================================================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def doctor_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        user = db.get_user(session['user_id'])
        if not user or user['user_type'] != 'doctor':
            flash('Access denied. Doctors only.', 'error')
            return redirect(url_for('patient_dashboard'))
        return f(*args, **kwargs)
    return decorated_function

def patient_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        user = db.get_user(session['user_id'])
        if not user or user['user_type'] != 'patient':
            flash('Access denied. Patients only.', 'error')
            return redirect(url_for('doctor_dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def generate_llm_report(prediction_data, patient_name, patient_age):
    """Generate medical report using LLM"""
    prompt = f"""You are an expert radiologist AI. Generate a professional medical report for:

**Patient Information:**
- Name: {patient_name}
- Age: {patient_age}

**AI Analysis Results:**
- Classification: {prediction_data['prediction']}
- Confidence: {prediction_data['confidence']*100:.1f}%
- Risk Level: {prediction_data['risk_level']}
{f"- Severity: {prediction_data['severity']}" if prediction_data.get('severity') else ""}

Generate a comprehensive medical report with these sections:

## FINDINGS
Detailed radiological observations from the chest X-ray.

## CLINICAL IMPRESSION
Professional interpretation of the findings.

## RECOMMENDATIONS
1. Immediate actions needed
2. Medication suggestions (if applicable)
3. Follow-up timeline
4. Lifestyle modifications

## RISK ASSESSMENT
Patient's current risk level and prognosis.

## ADDITIONAL NOTES
Any other relevant medical information.

Format professionally with clear section headers. Be empathetic but accurate."""

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an expert radiologist generating medical reports."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"**Error generating report:** {str(e)}\n\nPlease consult with a doctor for detailed analysis."

def save_file(file, user_id):
    """Save uploaded file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = secure_filename(f"{user_id}_{timestamp}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return filepath

# ============================================================================
# AUTHENTICATION ROUTES
# ============================================================================

@app.route('/')
def index():
    if 'user_id' in session:
        user = db.get_user(session['user_id'])
        if user['user_type'] == 'doctor':
            return redirect(url_for('doctor_dashboard'))
        return redirect(url_for('patient_dashboard'))
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = db.get_user_by_email(email)
        
        if user and check_password_hash(user['password'], password):
            if user['user_type'] == 'doctor' and not user.get('verified'):
                flash('Your account is pending verification. Please contact admin.', 'warning')
                return redirect(url_for('login'))
            
            session.permanent = True
            session['user_id'] = user['user_id']
            session['user_type'] = user['user_type']
            session['user_name'] = user['name']
            flash(f'Welcome back, {user["name"]}!', 'success')
            
            if user['user_type'] == 'doctor':
                return redirect(url_for('doctor_dashboard'))
            return redirect(url_for('patient_dashboard'))
        
        flash('Invalid email or password', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        user_type = request.form.get('user_type')
        name = request.form.get('name')
        
        if db.get_user_by_email(email):
            flash('Email already registered', 'error')
            return redirect(url_for('signup'))
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('signup'))
        
        if len(password) < 6:
            flash('Password must be at least 6 characters', 'error')
            return redirect(url_for('signup'))
        
        if user_type == 'doctor':
            user_id = db.create_user(
                email=email,
                password=password,
                user_type='doctor',
                name=name,
                specialization=request.form.get('specialization', 'General'),
                license_number=request.form.get('license_number', ''),
                experience=request.form.get('experience', ''),
                verified=False
            )
            flash('Account created! Pending admin verification.', 'info')
        else:
            user_id = db.create_user(
                email=email,
                password=password,
                user_type='patient',
                name=name,
                age=request.form.get('age', 0),
                phone=request.form.get('phone', '')
            )
            flash('Account created successfully! Please login.', 'success')
        
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('index'))

# ============================================================================
# PATIENT ROUTES
# ============================================================================

@app.route('/doctor/pending-calls')
@login_required
@doctor_required
def doctor_pending_calls():
    """View pending call requests"""
    user = db.get_user(session['user_id'])
    
    # Get reports where doctor is assigned and patient is waiting
    pending_calls = []
    for report_id, room_data in active_consultations.items():
        report = db.get_report(report_id)
        if report:
            # Show all pending reports to doctor or specifically assigned ones
            if report.get('doctor_id') == session['user_id'] or report.get('status') == 'pending_review':
                if room_data.get('patient'):  # Patient is waiting
                    pending_calls.append({
                        'report_id': report_id,
                        'patient_name': room_data['patient_name'],
                        'patient_age': report.get('patient_age', 'N/A'),
                        'timestamp': report.get('timestamp', ''),
                        'prediction': report['prediction']['prediction'],
                        'patient_waiting': True
                    })
    
    return render_template('doctor_calls.html', user=user, pending_calls=pending_calls)
@app.route('/patient/dashboard')
@login_required
@patient_required
def patient_dashboard():
    user = db.get_user(session['user_id'])
    reports = db.get_patient_reports(session['user_id'])
    notifications = db.get_notifications(session['user_id'])
    return render_template('patient_dashboard.html', user=user, reports=reports, notifications=notifications)

@app.route('/patient/upload', methods=['GET'])
@login_required
@patient_required
def upload_page():
    """Render upload page"""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
@login_required
@patient_required
def upload_xray():
    """Upload and analyze X-ray"""
    try:
        if 'xray' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['xray']
        user = db.get_user(session['user_id'])
        
        patient_name = request.form.get('patient_name', '').strip()
        patient_age = request.form.get('patient_age', '').strip()
        
        if not patient_name:
            patient_name = user.get('name', 'Unknown')
        if not patient_age:
            patient_age = user.get('age', 'N/A')
        
        report_id = str(uuid.uuid4())
        
        # Save file
        filepath = save_file(file, session['user_id'])
        
        # Read image as base64
        with open(filepath, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Predict using DenseNet121 - pass filepath directly
        logger.info("🔬 Running DenseNet121 prediction...")
        prediction = model.predict(filepath, threshold=0.6)
        logger.info(f"✅ Prediction complete: {prediction['prediction']}")
        
        # Check for errors
        if 'error' in prediction:
            logger.error(f"Prediction failed: {prediction['error']}")
            return jsonify({
                'error': 'Prediction failed',
                'message': prediction.get('error', 'Unknown error'),
                'success': False
            }), 500
        
        # Generate LLM report
        llm_report = generate_llm_report(prediction, patient_name, patient_age)
        
        # Save report
        report_data = {
            'report_id': report_id,
            'patient_id': session['user_id'],
            'patient_name': patient_name,
            'patient_age': patient_age,
            'prediction': prediction,
            'llm_report': llm_report,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending_review',
            'image_path': filepath,
            'image_base64': image_data,
            'doctor_id': None,
            'doctor_notes': None
        }
        
        db.save_report(report_id, report_data)
        
        # Notify doctors
        for uid, udata in db.users.items():
            if udata['user_type'] == 'doctor' and udata.get('verified'):
                db.add_notification(uid, f"New X-ray from {patient_name} pending review", 'new_report')
        
        flash('X-ray analyzed successfully!', 'success')
        return jsonify({
            'success': True,
            'report_id': report_id,
            'redirect': url_for('view_report', report_id=report_id)
        })
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/report/<report_id>')
@login_required
def view_report(report_id):
    report = db.get_report(report_id)
    if not report:
        flash('Report not found', 'error')
        return redirect(url_for('patient_dashboard'))
    
    # Get current user
    user = db.get_user(session['user_id'])
    
    # Check access permissions
    if user['user_type'] == 'patient' and report['patient_id'] != session['user_id']:
        flash('Access denied', 'error')
        return redirect(url_for('patient_dashboard'))
    
    # Pass both report and user to template
    return render_template('report.html', report=report, user=user)

@app.route('/download_report/<report_id>')
@login_required
def download_report(report_id):
    report = db.get_report(report_id)
    if not report:
        return "Report not found", 404
    
    user = db.get_user(session['user_id'])
    if user['user_type'] == 'patient' and report['patient_id'] != session['user_id']:
        return "Access denied", 403
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Medical Report - {report_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .header {{ background: #667eea; color: white; padding: 20px; border-radius: 10px; }}
            .section {{ margin: 20px 0; padding: 15px; background: #f9f9f9; border-radius: 8px; }}
            .prediction {{ background: #e8f4f8; padding: 15px; border-left: 4px solid #667eea; }}
            h1 {{ margin: 0; }}
            h2 {{ color: #2c3e50; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🫁 Pneumonia Detection Report</h1>
            <p>Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
        <div class="section">
            <h2>Patient Information</h2>
            <p><strong>Name:</strong> {report['patient_name']}</p>
            <p><strong>Age:</strong> {report['patient_age']}</p>
        </div>
        <div class="prediction">
            <h2>AI Results</h2>
            <p><strong>Classification:</strong> {report['prediction']['prediction']}</p>
            <p><strong>Confidence:</strong> {report['prediction']['confidence']*100:.1f}%</p>
        </div>
        <div class="section">
            <h2>Medical Report</h2>
            {report['llm_report'].replace('\n', '<br>')}
        </div>
    </body>
    </html>
    """
    return html_content

# ============================================================================
# CHATBOT ROUTES
# ============================================================================

@app.route('/chatbot/<report_id>')
@login_required
def chatbot(report_id):
    report = db.get_report(report_id)
    if not report:
        flash('Report not found', 'error')
        return redirect(url_for('patient_dashboard'))
    
    chat_history = db.get_chat_history(report_id)
    return render_template('chatbot.html', report_id=report_id, chat_history=chat_history)

@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    data = request.json
    report_id = data.get('report_id')
    message = data.get('message')
    
    report = db.get_report(report_id)
    if not report:
        return jsonify({'error': 'Report not found'}), 404
    
    chat_history = db.get_chat_history(report_id)
    
    context = f"""You are a compassionate medical AI assistant.

**Report Summary:**
- Prediction: {report['prediction']['prediction']}
- Confidence: {report['prediction']['confidence']*100:.1f}%

**Patient's Question:** {message}

Provide a helpful, caring response in simple language:"""
    
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful medical AI assistant."},
                {"role": "user", "content": context}
            ],
            temperature=0.7,
            max_tokens=600
        )
        
        response = completion.choices[0].message.content
        
        db.save_chat_message(report_id, {'role': 'user', 'content': message, 'timestamp': datetime.now().isoformat()})
        db.save_chat_message(report_id, {'role': 'assistant', 'content': response, 'timestamp': datetime.now().isoformat()})
        
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# DOCTOR ROUTES
# ============================================================================

@app.route('/doctor/dashboard')
@login_required
@doctor_required
def doctor_dashboard():
    user = db.get_user(session['user_id'])
    pending_reports = db.get_pending_reports()
    notifications = db.get_notifications(session['user_id'])
    doctor_stats = db.doctors.get(session['user_id'], {})
    return render_template('doctor_dashboard.html', user=user, pending_reports=pending_reports, notifications=notifications, stats=doctor_stats)

@app.route('/doctor/review/<report_id>', methods=['GET', 'POST'])
@login_required
@doctor_required
def review_report(report_id):
    report = db.get_report(report_id)
    if not report:
        flash('Report not found', 'error')
        return redirect(url_for('doctor_dashboard'))
    
    if request.method == 'POST':
        doctor_notes = request.form.get('doctor_notes')
        diagnosis = request.form.get('diagnosis')
        
        report['doctor_id'] = session['user_id']
        report['doctor_notes'] = doctor_notes
        report['doctor_diagnosis'] = diagnosis
        report['status'] = 'reviewed'
        report['reviewed_at'] = datetime.now().isoformat()
        
        db.save_report(report_id, report)
        
        if session['user_id'] in db.doctors:
            db.doctors[session['user_id']]['total_consultations'] += 1
        
        db.add_notification(report['patient_id'], f"Report reviewed by Dr. {db.get_user(session['user_id'])['name']}", 'review_complete')
        
        flash('Report reviewed successfully', 'success')
        return redirect(url_for('doctor_dashboard'))
    
    return render_template('doctor_review.html', report=report)

# ============================================================================
# VIDEO CONSULTATION ROUTES
# ============================================================================

@app.route('/consultation/<report_id>')
@login_required
def consultation(report_id):
    report = db.get_report(report_id)
    if not report:
        flash('Report not found', 'error')
        return redirect(url_for('patient_dashboard' if session.get('user_type') == 'patient' else 'doctor_dashboard'))
    
    user = db.get_user(session['user_id'])
    
    # Auto-assign first available doctor if not assigned
    if not report.get('doctor_id'):
        available_doctors = [
            uid for uid, udata in db.users.items()
            if udata['user_type'] == 'doctor' and udata.get('verified')
        ]
        
        if available_doctors:
            doctor_id = available_doctors[0]
            report['doctor_id'] = doctor_id
            db.save_report(report_id, report)
            
            # Notify doctor
            doctor = db.get_user(doctor_id)
            db.add_notification(
                doctor_id,
                f"🎥 Video consultation requested by {report['patient_name']}",
                'consultation_request'
            )
            
            logger.info(f"✅ Auto-assigned doctor {doctor['name']} to report {report_id}")
    
    # Get doctor name
    doctor_name = None
    if report.get('doctor_id'):
        doctor = db.get_user(report['doctor_id'])
        if doctor:
            doctor_name = doctor['name']
    
    return render_template('consultation.html', report=report, user=user, room_id=report_id, doctor_name=doctor_name)

@app.route('/api/consultation/status/<report_id>')
@login_required
def consultation_status(report_id):
    """Get consultation room status"""
    if report_id in active_consultations:
        return jsonify({
            'active': True,
            'patient_present': active_consultations[report_id]['patient'] is not None,
            'doctor_present': active_consultations[report_id]['doctor'] is not None,
            'patient_name': active_consultations[report_id]['patient_name'],
            'doctor_name': active_consultations[report_id]['doctor_name']
        })
    return jsonify({
        'active': False,
        'patient_present': False,
        'doctor_present': False
    })


# Add route for doctors to see pending call requests
@app.route('/doctor/calls')
@login_required
@doctor_required
def doctor_calls():
    """View pending call requests"""
    user = db.get_user(session['user_id'])
    
    # Get reports where doctor is assigned
    pending_calls = []
    for report_id, report in db.reports.items():
        if report.get('doctor_id') == session['user_id']:
            if report_id in active_consultations:
                if active_consultations[report_id]['patient']:
                    pending_calls.append({
                        'report_id': report_id,
                        'patient_name': report['patient_name'],
                        'patient_waiting': True
                    })
    
    return render_template('doctor_calls.html', user=user, pending_calls=pending_calls)

@app.route('/patient/request_consultation/<report_id>', methods=['POST'])
@login_required
@patient_required
def request_consultation(report_id):
    """Patient requests consultation with a doctor"""
    report = db.get_report(report_id)
    if not report:
        return jsonify({'error': 'Report not found'}), 404
    
    if report['patient_id'] != session['user_id']:
        return jsonify({'error': 'Access denied'}), 403
    
    # Find available verified doctors
    available_doctors = [
        uid for uid, user in db.users.items()
        if user['user_type'] == 'doctor' and user.get('verified')
    ]
    
    if not available_doctors:
        return jsonify({'error': 'No doctors available'}), 404
    
    # Assign first available doctor (or implement your own logic)
    doctor_id = available_doctors[0]
    
    # Update report
    report['doctor_id'] = doctor_id
    report['consultation_requested_at'] = datetime.now().isoformat()
    db.save_report(report_id, report)
    
    # Create consultation
    consult_id = db.create_consultation(session['user_id'], doctor_id, report_id)
    
    # Notify doctor
    db.add_notification(
        doctor_id,
        f"Video consultation requested by {report['patient_name']}",
        'consultation_request'
    )
    
    return jsonify({
        'success': True,
        'consultation_id': consult_id,
        'doctor_id': doctor_id,
        'redirect': url_for('consultation', report_id=report_id)
    })

@app.route('/api/doctors/online')
@login_required
def get_online_doctors():
    """Get list of online doctors"""
    online_doctors = []
    
    for user_id, user in db.users.items():
        if user['user_type'] == 'doctor' and user.get('verified'):
            doctor_info = db.doctors.get(user_id, {})
            online_doctors.append({
                'user_id': user_id,
                'name': user['name'],
                'specialization': user.get('specialization', 'General'),
                'rating': doctor_info.get('rating', 0.0),
                'total_consultations': doctor_info.get('total_consultations', 0)
            })
    
    return jsonify({'doctors': online_doctors})


@app.route('/api/notifications/<user_id>')
@login_required
def get_notifications_api(user_id):
    """Get user notifications via API"""
    if session.get('user_id') != user_id:
        return jsonify({'error': 'Access denied'}), 403
    
    notifications = db.get_notifications(user_id)
    return jsonify({'notifications': notifications})

@app.route('/api/doctor/active-calls')
@login_required
@doctor_required
def get_active_calls():
    """Get active consultation requests for doctor"""
    doctor_id = session['user_id']
    active_calls = []
    
    for report_id, room_data in active_consultations.items():
        report = db.get_report(report_id)
        if report and (report.get('doctor_id') == doctor_id or report.get('status') == 'pending_review'):
            if room_data.get('patient'):  # Patient is waiting
                active_calls.append({
                    'report_id': report_id,
                    'patient_name': room_data['patient_name'],
                    'patient_age': report.get('patient_age', 'N/A'),
                    'timestamp': report.get('timestamp', ''),
                    'prediction': report['prediction']['prediction']
                })
    
    return jsonify({'calls': active_calls})

# ============================================================================
# SOCKETIO EVENTS (Video Call)
# ============================================================================

active_consultations = {}  # {report_id: {'patient': sid, 'doctor': sid, 'patient_name': '', 'doctor_name': ''}}
# Add this global variable at the top with active_consultations
call_notifications = {}  # {doctor_id: [list of pending calls]}

# Update the handle_join function
@socketio.on('join_consultation')
def handle_join(data):
    """Handle user joining consultation room"""
    room = data['report_id']
    user_type = session.get('user_type', 'unknown')
    user_name = session.get('user_name', 'User')
    user_id = session.get('user_id')
    
    join_room(room)
    
    # Initialize room if not exists
    if room not in active_consultations:
        active_consultations[room] = {
            'patient': None,
            'doctor': None,
            'patient_name': '',
            'doctor_name': '',
            'patient_id': None,
            'doctor_id': None
        }
    
    # Store user's SID and ID based on type
    if user_type == 'patient':
        active_consultations[room]['patient'] = request.sid
        active_consultations[room]['patient_name'] = user_name
        active_consultations[room]['patient_id'] = user_id
        
        # Get report and notify assigned doctor
        report = db.get_report(room)
        if report and report.get('doctor_id'):
            doctor_id = report['doctor_id']
            # Send real-time notification to doctor
            socketio.emit('new_call_notification', {
                'report_id': room,
                'patient_name': user_name,
                'patient_id': user_id,
                'message': f'{user_name} is waiting for video consultation'
            }, room=f"doctor_{doctor_id}")
            
            # Also add to database notification
            db.add_notification(
                doctor_id,
                f"🎥 {user_name} is waiting for video consultation",
                'call_request'
            )
        
    elif user_type == 'doctor':
        active_consultations[room]['doctor'] = request.sid
        active_consultations[room]['doctor_name'] = user_name
        active_consultations[room]['doctor_id'] = user_id
    
    logger.info(f"✅ {user_name} ({user_type}) joined room {room}")
    
    # Notify all users in the room
    emit('user_joined', {
        'user_type': user_type,
        'user_name': user_name,
        'user_id': user_id,
        'message': f'{user_name} joined the consultation',
        'room_status': {
            'patient_present': active_consultations[room]['patient'] is not None,
            'doctor_present': active_consultations[room]['doctor'] is not None,
            'patient_name': active_consultations[room]['patient_name'],
            'doctor_name': active_consultations[room]['doctor_name']
        }
    }, room=room)
    
    # If both patient and doctor are present, notify them to start call
    if active_consultations[room]['patient'] and active_consultations[room]['doctor']:
        emit('ready_to_call', {
            'message': 'Both parties are present. You can start the video call.',
            'patient_name': active_consultations[room]['patient_name'],
            'doctor_name': active_consultations[room]['doctor_name']
        }, room=room)
        logger.info(f"🎥 Both parties present in room {room}")


@socketio.on('leave_consultation')
def handle_leave(data):
    """Handle user leaving consultation room"""
    room = data['report_id']
    user_name = session.get('user_name', 'User')
    user_type = session.get('user_type', 'unknown')
    
    # Remove user from active consultations
    if room in active_consultations:
        if user_type == 'patient':
            active_consultations[room]['patient'] = None
            active_consultations[room]['patient_name'] = ''
        elif user_type == 'doctor':
            active_consultations[room]['doctor'] = None
            active_consultations[room]['doctor_name'] = ''
        
        # Clean up empty rooms
        if not active_consultations[room]['patient'] and not active_consultations[room]['doctor']:
            del active_consultations[room]
    
    leave_room(room)
    
    logger.info(f"❌ {user_name} ({user_type}) left room {room}")
    
    emit('user_left', {
        'user_type': user_type,
        'user_name': user_name,
        'message': f'{user_name} left the consultation'
    }, room=room)


@socketio.on('offer')
def handle_offer(data):
    """Handle WebRTC offer"""
    room = data['report_id']
    offer = data['offer']
    sender_type = session.get('user_type')
    
    logger.info(f"📤 Offer from {sender_type} in room {room}")
    
    # Forward offer to the other party
    if room in active_consultations:
        target_sid = None
        if sender_type == 'patient':
            target_sid = active_consultations[room]['doctor']
        elif sender_type == 'doctor':
            target_sid = active_consultations[room]['patient']
        
        if target_sid:
            emit('offer', {
                'offer': offer,
                'from': sender_type
            }, room=target_sid)
            logger.info(f"✅ Offer forwarded to {target_sid}")
        else:
            logger.warning(f"⚠️ No target found for offer in room {room}")
            emit('error', {'message': 'Other party not in room'})


@socketio.on('answer')
def handle_answer(data):
    """Handle WebRTC answer"""
    room = data['report_id']
    answer = data['answer']
    sender_type = session.get('user_type')
    
    logger.info(f"📤 Answer from {sender_type} in room {room}")
    
    # Forward answer to the other party
    if room in active_consultations:
        target_sid = None
        if sender_type == 'patient':
            target_sid = active_consultations[room]['doctor']
        elif sender_type == 'doctor':
            target_sid = active_consultations[room]['patient']
        
        if target_sid:
            emit('answer', {
                'answer': answer,
                'from': sender_type
            }, room=target_sid)
            logger.info(f"✅ Answer forwarded to {target_sid}")
        else:
            logger.warning(f"⚠️ No target found for answer in room {room}")


@socketio.on('ice_candidate')
def handle_ice_candidate(data):
    """Handle ICE candidate"""
    room = data['report_id']
    candidate = data['candidate']
    sender_type = session.get('user_type')
    
    # Forward ICE candidate to the other party
    if room in active_consultations:
        target_sid = None
        if sender_type == 'patient':
            target_sid = active_consultations[room]['doctor']
        elif sender_type == 'doctor':
            target_sid = active_consultations[room]['patient']
        
        if target_sid:
            emit('ice_candidate', {
                'candidate': candidate,
                'from': sender_type
            }, room=target_sid)


@socketio.on('call_request')
def handle_call_request(data):
    """Handle call initiation request"""
    room = data['report_id']
    sender_name = session.get('user_name', 'User')
    sender_type = session.get('user_type')
    
    logger.info(f"📞 Call request from {sender_name} ({sender_type}) in room {room}")
    
    # Notify other party
    emit('incoming_call', {
        'from_name': sender_name,
        'from_type': sender_type,
        'report_id': room
    }, room=room, skip_sid=request.sid)
    
    # Send notification to doctor if they're not in the room yet
    if sender_type == 'patient':
        report = db.get_report(room)
        if report and report.get('doctor_id'):
            db.add_notification(
                report['doctor_id'],
                f"Video call request from {sender_name}",
                'call_request'
            )
            # Emit real-time notification to doctor if they're connected
            emit('new_call_notification', {
                'report_id': room,
                'patient_name': sender_name,
                'message': f'{sender_name} is requesting a video consultation'
            }, room=f"doctor_{report['doctor_id']}")


@socketio.on('call_accepted')
def handle_call_accepted(data):
    """Handle call acceptance"""
    room = data['report_id']
    user_name = session.get('user_name', 'User')
    
    logger.info(f"✅ Call accepted by {user_name} in room {room}")
    
    emit('call_accepted', {
        'message': f'{user_name} accepted the call',
        'accepted_by': user_name
    }, room=room)


@socketio.on('call_rejected')
def handle_call_rejected(data):
    """Handle call rejection"""
    room = data['report_id']
    user_name = session.get('user_name', 'User')
    
    logger.info(f"❌ Call rejected by {user_name} in room {room}")
    
    emit('call_rejected', {
        'message': f'{user_name} rejected the call',
        'rejected_by': user_name
    }, room=room)


@socketio.on('chat_message')
def handle_chat_message(data):
    """Handle text chat during video call"""
    room = data['report_id']
    user_name = session.get('user_name', 'User')
    user_type = session.get('user_type', 'unknown')
    message = data['message']
    
    emit('chat_message', {
        'user_name': user_name,
        'user_type': user_type,
        'message': message,
        'timestamp': datetime.now().isoformat()
    }, room=room)


@socketio.on('disconnect')
def handle_disconnect():
    """Handle user disconnection"""
    user_name = session.get('user_name', 'User')
    user_type = session.get('user_type', 'unknown')
    
    # Remove user from all active consultations
    for room_id, room_data in list(active_consultations.items()):
        if room_data.get('patient') == request.sid or room_data.get('doctor') == request.sid:
            logger.info(f"🔌 {user_name} disconnected from room {room_id}")
            
            emit('user_disconnected', {
                'user_name': user_name,
                'user_type': user_type,
                'message': f'{user_name} disconnected'
            }, room=room_id)
            
            # Clean up
            if room_data.get('patient') == request.sid:
                room_data['patient'] = None
                room_data['patient_name'] = ''
            if room_data.get('doctor') == request.sid:
                room_data['doctor'] = None
                room_data['doctor_name'] = ''
            
            if not room_data['patient'] and not room_data['doctor']:
                del active_consultations[room_id]

@socketio.on('join_doctor_room')
def handle_join_doctor_room(data):
    """Allow doctors to join their personal notification room"""
    doctor_id = data.get('doctor_id')
    if doctor_id:
        room = f"doctor_{doctor_id}"
        join_room(room)
        logger.info(f"✅ Doctor {doctor_id} joined notification room {room}")
        emit('joined_doctor_room', {'room': room})

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    return "<h1>404 - Page Not Found</h1>", 404

@app.errorhandler(500)
def server_error(e):
    return "<h1>500 - Internal Server Error</h1>", 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🏥 PNEUMONIA DETECTION SYSTEM v2.0")
    print("=" * 60)
    print("\n📋 Default Accounts:")
    print("\n👨‍⚕️ DOCTOR:")
    print("   Email: doctor@hospital.com")
    print("   Password: doctor123")
    print("\n👤 PATIENT:")
    print("   Email: patient@email.com")
    print("   Password: patient123")
    print("\n🔧 Model Status:")
    if model.model_loaded:
        print("   ✅ DenseNet121 loaded successfully")
    else:
        print("   ⚠️  Model file not found - using untrained model")
        print("   📁 Place 'densenet121_pneumonia.pth' in project root")
    print("\n🌐 Starting server on http://localhost:5000")
    print("=" * 60)
    print()
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)