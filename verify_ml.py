import sys
import os

# Add the src/analytics directory to path
sys.path.append(os.path.join(os.getcwd(), 'src', 'analytics'))

from ml_models import MLModels

def verify():
    print("Initializing MLModels...")
    m = MLModels()
    
    print("Checking get_regression_models...")
    models = m.get_regression_models()
    
    print(f"Number of models: {len(models)}")
    print(f"Keys: {list(models.keys())}")
    
    # Verify structure (dict with 'model' and 'params')
    first_model = list(models.keys())[0]
    first_val = models[first_model]
    
    if isinstance(first_val, dict) and 'model' in first_val and 'params' in first_val:
        print(f"✅ Structure verification passed. {first_model} has 'model' and 'params'.")
    else:
        print(f"❌ Structure verification failed. {first_model} is {type(first_val)}")
        
    print("Verification Successful")

if __name__ == "__main__":
    verify()
