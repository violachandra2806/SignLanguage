from flask import Flask, request, jsonify
from src.inference.video_processor import get_sibi_translation
import os

app = Flask(__name__)

@app.route('/SignLanguage/translate', methods=['POST'])
def translate_video():
    """API endpoint for SIBI translation"""
    try:
        data = request.get_json()
        
        if not data or 'video_url' not in data:
            return jsonify({
                "status": "error",
                "code": 400,
                "message": "Missing video_url in request",
                "data": []
            }), 400
        
        video_url = data['video_url']
        
        print(f"🎯 Received translation request for: {video_url}")
        
        # Use your local SIBI model
        predictions = get_sibi_translation(video_url)
        
        # Format response to match your existing structure
        formatted_predictions = [
            {"second": pred["second"], "text": pred["text"]} 
            for pred in predictions
        ]
        
        response = {
            "status": "success",
            "code": 0,
            "message": "SIBI translation completed",
            "data": formatted_predictions
        }
        
        print(f"✅ Translation completed: {len(formatted_predictions)} signs detected")
        return jsonify(response), 200
        
    except Exception as e:
        print(f"❌ Error in translation API: {e}")
        return jsonify({
            "status": "error",
            "code": 500,
            "message": str(e),
            "data": []
        }), 500

@app.route('/SignLanguage/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "success",
        "message": "SIBI Translation API is running",
        "version": "1.0"
    }), 200

if __name__ == '__main__':
    print("🚀 Starting SIBI Translation API Server...")
    print("📍 Endpoints:")
    print("   POST /SignLanguage/translate - Translate video to SIBI signs")
    print("   GET  /SignLanguage/health    - Health check")
    
    app.run(host='0.0.0.0', port=5001, debug=True)