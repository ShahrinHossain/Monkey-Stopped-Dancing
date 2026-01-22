import json

REQUIRED = ["title", "body", "url", "date", "language"]

def count(path):
    total = 0          # total non-empty lines processed
    bad = 0            # valid JSON but missing required fields
    decode_errors = 0  # lines that are not valid JSON

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # skip empty lines

            total += 1

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                decode_errors += 1
                preview = line[:200].replace("\n", "\\n")
                print(f"\n❌ JSONDecodeError in {path} at line {lineno}: {e}")
                print(f"Line preview: {preview}")
                continue  # keep going instead of crashing

            if any(k not in obj or obj[k] is None for k in REQUIRED):
                bad += 1

    return total, bad, decode_errors


print("BN:", count("data/processed/bn.jsonl"))
print("EN:", count("data/processed/en.jsonl"))
