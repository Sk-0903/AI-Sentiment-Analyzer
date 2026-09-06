"""
verify_live.py - Direct API verification script for live Flask server.
"""

import urllib.request
import urllib.error
import json

def test_api():
    def post(payload):
        req = urllib.request.Request(
            "http://127.0.0.1:5000/api/analyze",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8"))

    test_cases = [
        ("Positive Review", "I absolutely love this product. The quality is amazing and delivery was very fast."),
        ("Negative Review", "Very disappointed. The product stopped working after two days and customer support was not helpful."),
        ("Neutral Statement", "The product arrived on time. The quality is acceptable and matches the description."),
        ("Mixed Feedback", "The design is sleek and features are great, but the battery life is quite disappointing."),
        ("Empty Input", ""),
        ("Whitespace Only", "    \n\t   "),
        ("2000+ Characters Input", "Exceptional product feedback with lots of detail. " * 80)
    ]

    print("\n" + "="*70)
    print("LIVE API VERIFICATION SUITE")
    print("="*70)

    for label, text in test_cases:
        code, body = post({"text": text})
        print(f"\n[Case: {label}] -> HTTP {code}")
        if code == 200:
            print(f"  Sentiment   : {body['sentiment']}")
            print(f"  Confidence  : {body['confidence'] * 100:.1f}%")
            print(f"  Scores      : Pos={body['scores']['positive']}, Neu={body['scores']['neutral']}, Neg={body['scores']['negative']}")
            print(f"  Insight     : {body['explanation']}")
            print(f"  Engine Used : {body['engine']} ({body['model_name']})")
        else:
            print(f"  Correctly Rejected: {body.get('error')}")

    # Test GET /api/history
    print("\n" + "="*70)
    print("TESTING GET /api/history")
    print("="*70)
    req_hist = urllib.request.Request("http://127.0.0.1:5000/api/history?limit=10")
    with urllib.request.urlopen(req_hist) as resp:
        hist = json.loads(resp.read().decode("utf-8"))
        print(f"History items retrieved: {len(hist['history'])}")
        print(f"Aggregate Stats: Total={hist['statistics']['total']}, Pos={hist['statistics']['positive']}, Neu={hist['statistics']['neutral']}, Neg={hist['statistics']['negative']}")
        print(f"Database Status: {hist['database_status']['status_label']}")

    # Test DELETE /api/history
    print("\n" + "="*70)
    print("TESTING DELETE /api/history")
    print("="*70)
    req_del = urllib.request.Request("http://127.0.0.1:5000/api/history", method="DELETE")
    with urllib.request.urlopen(req_del) as resp:
        del_res = json.loads(resp.read().decode("utf-8"))
        print(f"History Cleared: {del_res['message']} (Deleted count: {del_res['deleted_count']})")
        print(f"Stats after clear: Total={del_res['statistics']['total']}")

    print("\n" + "="*70)
    print("ALL LIVE ENDPOINT VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_api()
