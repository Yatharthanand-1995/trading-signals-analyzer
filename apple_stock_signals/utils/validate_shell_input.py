#!/usr/bin/env python3
"""
Shell Script Input Validator
Validates inputs from run_analysis.sh before processing
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.validators import Validators, ValidationError

def validate_command(command: str) -> str:
    """Validate shell script command"""
    valid_commands = {
        'top5', 'top10', 'top50', 'top100', 'sp500',
        'add', 'remove', 'list', 'active', 'backtest',
        'swing', 'signals', 'paper', 'health', 'ml',
        'phase1', 'phase2', 'phase3', 'enhanced'
    }
    
    command = command.lower().strip()
    
    if command not in valid_commands:
        raise ValidationError(
            f"Invalid command '{command}'. Valid commands: {', '.join(sorted(valid_commands))}"
        )
    
    return command

def validate_add_remove_args(args: list) -> tuple:
    """Validate arguments for add/remove commands"""
    if len(args) < 2:
        raise ValidationError("Add/remove commands require a stock symbol")
    
    command = args[0].lower()
    symbol = Validators.validate_stock_symbol(args[1], allow_lowercase=True)
    
    return command, symbol

def main():
    """Main validation entry point"""
    if len(sys.argv) < 2:
        print("Usage: validate_shell_input.py <command> [args...]")
        sys.exit(1)
    
    try:
        command = validate_command(sys.argv[1])
        
        # Special handling for add/remove commands
        if command in ['add', 'remove'] and len(sys.argv) > 2:
            _, symbol = validate_add_remove_args(sys.argv[1:3])
            print(f"VALID:{command}:{symbol}")
        else:
            print(f"VALID:{command}")
        
        sys.exit(0)
        
    except ValidationError as e:
        print(f"ERROR:{e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR:Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()