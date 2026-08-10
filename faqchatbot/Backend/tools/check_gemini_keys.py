"""
Check each Gemini API key in the configured pool by attempting a small embed call
so you can identify which keys are producing permission errors (403).

Usage:
  source .venv/bin/activate
  python tools/check_gemini_keys.py
"""
import json
import time
from app.services.gemini_key_pool import gemini_key_pool


def main():
    entries = getattr(gemini_key_pool, "_entries", None)
    if not entries:
        print("No Gemini keys configured in the pool.")
        return

    sample = "Test embed for key validation."
    for entry in entries:
        label = entry.label
        client = entry.client
        print(f"Checking {label}...")
        try:
            # call the embed API with a trivial text
            resp = client.models.embed_content(model="gemini-embedding-001", contents=[sample])
            # If we got embeddings, success
            if getattr(resp, 'embeddings', None):
                print(f"  {label}: OK")
            else:
                print(f"  {label}: unexpected response")
        except Exception as e:
            msg = str(e)
            if 'PERMISSION_DENIED' in msg or 'denied' in msg or '403' in msg:
                print(f"  {label}: PERMISSION_DENIED - {msg[:200]}")
            else:
                print(f"  {label}: ERROR - {msg[:200]}")
        # Small delay so we don't trigger immediate rate limits
        time.sleep(0.5)


if __name__ == "__main__":
    main()
