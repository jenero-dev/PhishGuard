import requests

def test_url(url):
    api_url = "http://127.0.0.1:8000/predict"
    payload = {"url": url}

    print(f"\nTesting: {url}")
    try:
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            data = response.json()
            result = "🚨 PHISHING" if data['is_phishing'] else "✅ LEGITIMATE"
            print(f"Result: {result} ({data['confidence']}% confidence)")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Connection Failed: {e}")

if __name__ == "__main__":
    test_url("https://google.com")
    test_url("http://secure-login-paypal-verify.com/login")
