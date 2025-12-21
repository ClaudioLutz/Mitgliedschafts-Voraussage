from flask import Flask, render_template, send_from_directory, jsonify
import os
import json

app = Flask(__name__)

DATA_DIR = os.path.join(app.root_path, 'static', 'data')
STATUS_FILE = os.path.join(app.root_path, 'static', 'status.json')

@app.route('/')
def dashboard():
    """
    Serves the dashboard page if data exists, otherwise instructions.
    """
    if os.path.exists(os.path.join(DATA_DIR, 'shab_monthly.json')):
        return render_template('dashboard.html')
    else:
        return "<h1>Missing Data</h1><p>Run <code>python refresh_data.py</code> to generate dashboard data.</p>"

@app.route('/api/status')
def status():
    """
    Returns the status JSON.
    """
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    else:
        return jsonify({"error": "Status file not found"}), 404

# Flask serves static files from 'static/' automatically at /static/
# We don't need explicit routes for static/data/*.json if we access them via /static/data/...

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
