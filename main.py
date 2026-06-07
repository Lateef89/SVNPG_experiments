"""
Convenience script to run all experiments and generate figures.
"""
import os
import sys

print("="*60)
print("SVNPG Experiment Suite")
print("="*60)

# Run synthetic LAD
print("\n[1/3] Running synthetic LAD experiments...")
os.system("python experiments.py")

# Generate figures
print("\n[2/3] Generating figures...")
os.system("python generate_figures.py")

print("\n[3/3] Done. Results saved to ./results/ and ./figures/")
