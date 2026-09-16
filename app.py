"""
backend/app.py

Implements the REST API (Section 34) and a minimal Dashboard (Section 66)
on top of the SQLite demo store (backend/db.py).

Only the endpoints that make sense with data this demo actually produces
are wired up:
    GET  /api/events
    GET  /api/events/<event_id>
    GET  /api/alerts
    GET  /api/blockchain/verify/<event_id>

Person/vehicle/camera/auth endpoints from Section 34.1 are NOT built yet
-- they need the Person Enrollment (Section 38) and Authentication
(Section 48) modules first, which we haven't built. Listed as TODO
rather than stubbed with fake behavior.
"""

from flask import Flask, jsonify, render_template_string

from backend import db

app = Flask(__name__)

DASHBOARD_TEMPLATE = """
<!doctype html>
<html>
<head>
    <title>IBVAP Demo Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 2rem; background:#111; color:#eee; }
        h1 { color: #4caf50; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 2rem; }
        th, td { border: 1px solid #444; padding: 6px 10px; text-align: left; font-size: 14px; }
        th { background: #222; }
        tr:nth-child(even) { background: #1a1a1a; }
        .alert-row { background: #3a1a1a !important; }
    </style>
</head>
<body>
    <h1>IBVAP -- Demo Dashboard (Section 66)</h1>
    <p>SQLite-backed demo store. Data comes from run_demo.py's synthetic run.</p>

    <h2>Alerts (Section 68)</h2>
    <table>
        <tr><th>Alert ID</th><th>Event ID</th><th>Type</th><th>Camera</th><th>Confidence</th><th>Time</th><th>Message</th></tr>
        {% for a in alerts %}
        <tr class="alert-row">
            <td>{{a.alert_id}}</td><td>{{a.event_id}}</td><td>{{a.event_type}}</td>
            <td>{{a.camera_id}}</td><td>{{'%.2f' % a.confidence}}</td>
            <td>{{a.timestamp}}</td><td>{{a.message}}</td>
        </tr>
        {% endfor %}
    </table>

    <h2>Event History (Section 69)</h2>
    <table>
        <tr><th>Event ID</th><th>Type</th><th>Camera</th><th>Confidence</th><th>Time</th><th>Metadata</th></tr>
        {% for e in events %}
        <tr>
            <td>{{e.event_id}}</td><td>{{e.event_type}}</td><td>{{e.camera_id}}</td>
            <td>{{'%.2f' % e.confidence}}</td><td>{{e.timestamp}}</td><td>{{e.metadata}}</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
"""


@app.route("/")
def dashboard():
    return render_template_string(
        DASHBOARD_TEMPLATE,
        events=db.get_all_events(),
        alerts=db.get_all_alerts(),
    )


@app.route("/api/events", methods=["GET"])
def api_events():
    return jsonify(db.get_all_events())


@app.route("/api/events/<event_id>", methods=["GET"])
def api_event_detail(event_id):
    event = db.get_event(event_id)
    if event is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(event)


@app.route("/api/alerts", methods=["GET"])
def api_alerts():
    return jsonify(db.get_all_alerts())


@app.route("/api/blockchain/verify/<event_id>", methods=["GET"])
def api_blockchain_verify(event_id):
    record = db.get_blockchain_record(event_id)
    if record is None:
        return jsonify({"found": False}), 404
    return jsonify({"found": True, "record": record})


if __name__ == "__main__":
    db.init_db()
    app.run(host="127.0.0.1", port=5000, debug=True)
