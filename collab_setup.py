"""
Colab setup script for Physics repo.
Run this first in any Colab notebook.
"""
import subprocess
import sys
import os

def setup_colab():
    """Clone repo and install dependencies for Colab."""
    
    # Clone repo
    if not os.path.exists('/content/Physics'):
        subprocess.run([
            'git', 'clone', 
            'https://github.com/mpast043/Physics.git',
            '/content/Physics'
        ], check=True)
    
    # Change to repo directory
    os.chdir('/content/Physics')
    
    # Install dependencies
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 
                   'numpy', 'scipy', 'matplotlib', 'quimb', 'opt_einsum'],
                  check=True)
    
    # Add to path
    if '/content/Physics' not in sys.path:
        sys.path.insert(0, '/content/Physics')
    
    print("✓ Colab setup complete")
    print(f"  Repo: /content/Physics")
    print(f"  Working directory: {os.getcwd()}")
    
    return '/content/Physics'

if __name__ == '__main__':
    setup_colab()
