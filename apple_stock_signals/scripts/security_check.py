#!/usr/bin/env python3
"""
Security Check Script
Scans codebase for potential security issues
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Patterns that might indicate hardcoded secrets
SECRET_PATTERNS = [
    # API Keys
    (r'api[_-]?key\s*=\s*[\'"][\w\-]{20,}[\'"]', 'Potential API key'),
    (r'apikey\s*=\s*[\'"][\w\-]{20,}[\'"]', 'Potential API key'),
    
    # Passwords
    (r'password\s*=\s*[\'"](\w|\S){8,}[\'"]', 'Hardcoded password'),
    (r'passwd\s*=\s*[\'"](\w|\S){8,}[\'"]', 'Hardcoded password'),
    (r'pwd\s*=\s*[\'"](\w|\S){8,}[\'"]', 'Hardcoded password'),
    
    # Tokens
    (r'token\s*=\s*[\'"][\w\-\.]{20,}[\'"]', 'Potential token'),
    (r'auth[_-]?token\s*=\s*[\'"][\w\-\.]{20,}[\'"]', 'Potential auth token'),
    
    # Secrets
    (r'secret\s*=\s*[\'"](\w|\S){10,}[\'"]', 'Potential secret'),
    (r'private[_-]?key\s*=\s*[\'"](\w|\S){10,}[\'"]', 'Potential private key'),
    
    # URLs with credentials
    (r'https?://[^:]+:[^@]+@', 'URL with embedded credentials'),
    
    # AWS Keys
    (r'AKIA[0-9A-Z]{16}', 'Potential AWS access key'),
    (r'aws[_-]?secret[_-]?access[_-]?key\s*=\s*[\'"](\w|\S){40}[\'"]', 'Potential AWS secret key'),
]

# Files/directories to skip
SKIP_DIRS = {'.git', '__pycache__', 'venv', 'env', '.env', 'node_modules', '.idea', '.vscode'}
SKIP_FILES = {'.env', '.env.example', 'security_check.py'}
SKIP_EXTENSIONS = {'.pyc', '.pyo', '.pyd', '.so', '.dylib', '.dll', '.exe', '.bin'}

def scan_file(filepath: Path) -> List[Tuple[int, str, str]]:
    """Scan a single file for security issues"""
    issues = []
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                for pattern, description in SECRET_PATTERNS:
                    if re.search(pattern, line, re.IGNORECASE):
                        # Skip if it's a variable reference (e.g., os.environ.get)
                        if 'os.environ' in line or 'getenv' in line:
                            continue
                        # Skip if it's a comment
                        if line.strip().startswith('#') or line.strip().startswith('//'):
                            continue
                        
                        issues.append((line_num, description, line.strip()))
    except Exception as e:
        print(f"Error scanning {filepath}: {e}")
    
    return issues

def scan_directory(root_dir: Path) -> Dict[str, List[Tuple[int, str, str]]]:
    """Scan directory recursively for security issues"""
    all_issues = {}
    
    for root, dirs, files in os.walk(root_dir):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        
        for file in files:
            # Skip certain files and extensions
            if file in SKIP_FILES or Path(file).suffix in SKIP_EXTENSIONS:
                continue
            
            # Only scan Python files and config files
            if not (file.endswith('.py') or file.endswith('.json') or 
                    file.endswith('.yaml') or file.endswith('.yml') or
                    file.endswith('.conf') or file.endswith('.cfg')):
                continue
            
            filepath = Path(root) / file
            issues = scan_file(filepath)
            
            if issues:
                rel_path = filepath.relative_to(root_dir)
                all_issues[str(rel_path)] = issues
    
    return all_issues

def print_report(issues: Dict[str, List[Tuple[int, str, str]]]):
    """Print security scan report"""
    if not issues:
        print("✅ No security issues found!")
        return
    
    print("⚠️  SECURITY ISSUES FOUND:")
    print("=" * 80)
    
    total_issues = sum(len(file_issues) for file_issues in issues.values())
    print(f"Total issues: {total_issues}")
    print("=" * 80)
    
    for filepath, file_issues in issues.items():
        print(f"\n📄 {filepath}:")
        for line_num, description, line_content in file_issues:
            print(f"  Line {line_num}: {description}")
            print(f"    > {line_content[:100]}..." if len(line_content) > 100 else f"    > {line_content}")
    
    print("\n" + "=" * 80)
    print("🔧 RECOMMENDATIONS:")
    print("1. Move all secrets to environment variables")
    print("2. Use the config.env_config module for accessing secrets")
    print("3. Never commit .env files to version control")
    print("4. Consider using a secrets management service for production")

def main():
    """Run security scan"""
    print("🔍 Security Scan for Trading System")
    print("=" * 80)
    
    # Get root directory
    if len(sys.argv) > 1:
        root_dir = Path(sys.argv[1])
    else:
        root_dir = Path(__file__).parent.parent
    
    print(f"Scanning directory: {root_dir}")
    print("This may take a moment...\n")
    
    # Run scan
    issues = scan_directory(root_dir)
    
    # Print report
    print_report(issues)
    
    # Return exit code based on findings
    return 1 if issues else 0

if __name__ == "__main__":
    sys.exit(main())