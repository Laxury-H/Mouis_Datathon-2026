import os
import subprocess
from pathlib import Path

def run_script(script_path):
    print(f"\n🚀 Running {script_path}...")
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error in {script_path}:\n{result.stderr}")
        exit(1)
    else:
        print(f"✅ Success!\n{result.stdout.strip()[-500:]}") # Print last 500 chars

def main():
    print("=====================================================")
    print("🏆 REPRODUCING V32 TIME-DECAY BLEND (665k MAE) 🏆")
    print("=====================================================")
    
    src_dir = Path("src")
    
    # 1. Run V18 Deep Learning Stack
    run_script(src_dir / "model_v18_dl_stack.py")
    
    # 2. Run V25 Components Stack (uses V18 as anchor)
    run_script(src_dir / "model_v25_components.py")
    
    # 2.1 Run V25 Sweep to get the Base a30 blend
    run_script(src_dir / "blend_v25_sweep.py")
    
    # 3. Run V28 One-Shot Components
    run_script(src_dir / "model_v28_oneshot_components.py")
    
    # 4. Run Time-Decay Blending (10% to 50% shift)
    run_script(src_dir / "blend_time_decay.py")
    
    print("\n🎉 All models executed successfully!")
    print("📂 The final submission is located at: Results/submissions/final_sweeps/submission_time_decay_10_50.csv")

if __name__ == "__main__":
    # Ensure working directory is the project root for paths to work correctly
    current_dir = Path.cwd()
    if current_dir.name == "V32_TimeDecay_665k":
        os.chdir("../../..")
    elif current_dir.name == "Champions":
        os.chdir("../..")
        
    main()
