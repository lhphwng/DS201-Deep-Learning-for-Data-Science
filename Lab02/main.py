import subprocess
import sys
import time

ASSIGNMENTS = [
    ("Bài 2 - GoogLeNet", "assignment-2.py"),
    ("Bài 3 - ResNet-18", "assignment-3.py"),
    ("Bài 4 - Fine-tune ResNet50 (HuggingFace)", "assignment-4.py"),
]

if __name__ == "__main__":
    results = []
    for name, script in ASSIGNMENTS:
        print(f"\n{'=' * 70}\n{name}  ({script})\n{'=' * 70}", flush=True)

        start = time.time()
        return_code = subprocess.run([sys.executable, script]).returncode
        elapsed = (time.time() - start) / 6

        results.append((name, return_code, elapsed))

    print(f"\n{'=' * 70}\nTỔNG KẾT\n{'=' * 70}")
    for name, return_code, elapsed in results:
        status = "OK" if return_code == 0 else f"LỖI (exit code {return_code})"
        print(f"- {name}: {status} - {elapsed:.1f} phút")
