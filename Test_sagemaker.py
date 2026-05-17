#!/usr/bin/env python3
"""
WYA — SageMaker autotag diagnostic script.
Run this on EC2 (inside or outside Docker) to check if the endpoint works.

Usage:
    python3 test_sagemaker.py                        # uses a solid-color test image
    python3 test_sagemaker.py path/to/garment.jpg    # test with a real image
"""
import sys, os, base64, io, json, traceback

ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT", "wya-fashionclip-serverless")
REGION   = os.getenv("AWS_REGION", "ap-south-1")
LABELS   = [
    "t-shirt", "dress", "jeans", "trousers", "skirt",
    "shoes", "boots", "jacket", "watch", "necklace", "bag",
]

print(f"\n{'='*55}")
print(f" WYA SageMaker Diagnostic")
print(f"{'='*55}")
print(f" Endpoint : {ENDPOINT}")
print(f" Region   : {REGION}")
print(f" Labels   : {LABELS}\n")

# ── 1. Check boto3 / credentials ─────────────────────────────────────────────
print("Step 1 — boto3 + AWS credentials...")
try:
    import boto3
    sts = boto3.client("sts", region_name=REGION)
    identity = sts.get_caller_identity()
    print(f"    Credentials OK  → Account: {identity['Account']}, ARN: {identity['Arn']}")
except Exception as e:
    print(f"    Credentials FAILED:\n      {e}")
    print("      → EC2 instance profile may not be attached, or Docker can't reach 169.254.169.254")
    sys.exit(1)

# ── 2. Check endpoint status ─────────────────────────────────────────────────
print("\nStep 2 — SageMaker endpoint status...")
try:
    sm = boto3.client("sagemaker", region_name=REGION)
    ep = sm.describe_endpoint(EndpointName=ENDPOINT)
    status = ep["EndpointStatus"]
    itype  = ep.get("ProductionVariants", [{}])[0].get("CurrentInstanceType", "unknown")
    print(f"    Endpoint status: {status}  | Instance: {itype}")
    if status != "InService":
        print(f"     Endpoint is NOT InService — cannot invoke until it is ready.")
        sys.exit(1)
except Exception as e:
    print(f"   Could not describe endpoint:\n      {e}")
    sys.exit(1)

# ── 3. Build test image ───────────────────────────────────────────────────────
print("\nStep 3 — Building test image...")
try:
    from PIL import Image as PILImage
    import numpy as np

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        print(f"  Using provided image: {img_path}")
        pil_img = PILImage.open(img_path).convert("RGB")
    else:
        print("  Using synthetic 224×224 blue rectangle (simulates jeans/trousers)")
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        arr[30:200, 60:164] = [30, 80, 160]   # blue rectangle
        pil_img = PILImage.fromarray(arr)

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    print(f"    Image encoded ({len(img_b64)} chars base64)")
except Exception as e:
    print(f"   Image prep failed:\n      {e}")
    sys.exit(1)

# ── 4. Invoke endpoint ────────────────────────────────────────────────────────
print("\nStep 4 — Invoking endpoint...")
try:
    runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    payload = {"inputs": img_b64, "parameters": {"candidate_labels": LABELS}}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    raw = response["Body"].read()
    results = json.loads(raw)
    print(f"    Raw response:\n      {json.dumps(results, indent=6)}")

    if isinstance(results, list) and results:
        top_label = results[0].get("label", "?")
        top_score = results[0].get("score", 0)
        print(f"\n    Top prediction: '{top_label}'  (score={top_score:.3f})")
    else:
        print(f"\n     Unexpected response shape — may need payload format adjustment")

except Exception as e:
    print(f"    Endpoint invocation FAILED:\n      {e}")
    print(f"\n  Full traceback:")
    traceback.print_exc()
    print("""
  Common causes:
    • Wrong payload format  → endpoint expects different 'inputs' key
    • IAM permissions       → role missing sagemaker:InvokeEndpoint
    • Wrong endpoint name   → double-check SAGEMAKER_ENDPOINT env var
    • Network / VPC issue   → EC2 can't reach SageMaker endpoint
  """)
    sys.exit(1)

print(f"\n{'='*55}")
print(" ✅  All checks passed — SageMaker is reachable from this environment!")
print(f"{'='*55}\n")
