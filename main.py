import os
for root, dirs, files in os.walk("."):
    if "sots" in root.lower() or "indoor" in root.lower():
        print(root, "->", dirs, files[:3])