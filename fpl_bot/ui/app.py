"""
Flask web UI for FPL Bot

Displays analysis results in a clean web interface while keeping
debug output in the terminal.
"""

from flask import Flask, render_template, jsonify
import json
import os
from datetime import datetime

app = Flask(__name__)

# Global storage for latest report
latest_report = None
latest_timestamp = None

def set_report(report):
    """Store the latest report data"""
    global latest_report, latest_timestamp
    latest_report = report
    latest_timestamp = datetime.now()

@app.route('/')
def index():
    """Main dashboard page"""
    if not latest_report:
        return render_template('waiting.html')
    return render_template('dashboard.html', report=latest_report, timestamp=latest_timestamp)

@app.route('/api/report')
def get_report():
    """API endpoint to get current report as JSON"""
    if not latest_report:
        return jsonify({'error': 'No report available yet'}), 404
    return jsonify({
        'report': latest_report,
        'timestamp': latest_timestamp.isoformat() if latest_timestamp else None
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'has_report': latest_report is not None})

def run_ui(host='127.0.0.1', port=5000, debug=False):
    """Start the Flask UI server"""
    app.run(host=host, port=port, debug=debug, use_reloader=False)

