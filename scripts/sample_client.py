#!/usr/bin/env python3
"""
HTTP client for ed-dit-server.
"""
import sys
import json
import urllib.request
import time

def query_sample(server_url, latent_path, output_path, steps=6, seed=42, budget_mb=200):
    url = f"{server_url.rstrip('/')}/sample"
    payload = {
        "latent_path": latent_path,
        "output_path": output_path,
        "steps": steps,
        "seed": seed,
        "budget_mb": budget_mb
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    print(f"Sending sample request to {url}...")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"Response in {time.time()-t0:.2f}s: {json.dumps(data, indent=2)}")
            return data
    except Exception as e:
        print(f"Request failed: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: sample_client.py <latent_path.edbin> <output_path.pt> [steps] [budget_mb] [server_url]")
        sys.exit(1)
    latent = sys.argv[1]
    out = sys.argv[2]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    budget = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    host = sys.argv[5] if len(sys.argv) > 5 else "http://127.0.0.1:9005"
    query_sample(host, latent, out, steps=steps, budget_mb=budget)
