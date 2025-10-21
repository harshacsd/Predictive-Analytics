"""
Automatic Fix Script for IndexError in Healthcare Predictive Analytics
Run this script to automatically fix all model files
"""

import os
import sys

def fix_model_file(filepath):
    """Fix the probability indexing issue in a model file"""
    print(f"Fixing {filepath}...")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Old problematic code
        old_code = "risk_score = probability[1] * 100"
        
        # New fixed code
        new_code = """# Handle probability array - it should have shape (2,) with [prob_class_0, prob_class_1]
        if len(probability) > 1:
            risk_score = probability[1] * 100  # Probability of class 1 (high risk)
        else:
            risk_score = probability[0] * 100 if prediction == 1 else (1 - probability[0]) * 100"""
        
        if old_code in content:
            content = content.replace(old_code, new_code)
            
            # Write back
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Fixed {filepath}")
            return True
        else:
            print(f"⚠️  {filepath} - Already fixed or pattern not found")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing {filepath}: {e}")
        return False

def main():
    """Main function to fix all model files"""
    print("=" * 60)
    print("Healthcare Predictive Analytics - Automatic Fix Script")
    print("=" * 60)
    print()
    
    # Check if we're in the right directory
    if not os.path.exists('ml_model'):
        print("❌ Error: ml_model directory not found!")
        print("Please run this script from the Predictive-Analytics-Healthcare folder")
        print()
        print("Current directory:", os.getcwd())
        print()
        print("Navigate to the correct directory and try again:")
        print("  cd Predictive-Analytics-Healthcare")
        print("  python fix_models.py")
        sys.exit(1)
    
    # List of model files to fix
    model_files = [
        'ml_model/heart_model.py',
        'ml_model/diabetes_model.py',
        'ml_model/kidney_model.py',
        'ml_model/stroke_model.py',
        'ml_model/hypertension_model.py'
    ]
    
    print("Fixing model files...")
    print()
    
    fixed_count = 0
    for filepath in model_files:
        if os.path.exists(filepath):
            if fix_model_file(filepath):
                fixed_count += 1
        else:
            print(f"⚠️  {filepath} not found - skipping")
    
    print()
    print("=" * 60)
    print(f"Fix complete! {fixed_count} file(s) fixed.")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Restart your Streamlit app: streamlit run app.py")
    print("2. Test the heart disease prediction")
    print("3. The error should be gone!")
    print()

if __name__ == "__main__":
    main()