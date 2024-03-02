import os
import sys
from typing import Optional
from datetime import datetime


def get_env(key: str, default: Optional[str]) -> str:
	value = os.getenv(key, default)
	if not validate_input(value):
		value = default
	return value

def log(content: str, error: bool = False):
	"""logs a message to the console"""
	now = datetime.now()
	print(f'[{now.isoformat()}] {content}', flush=True, file=sys.stderr if error else sys.stdout)

def validate_input(value: Optional[str]) -> bool:
	"""checks if the input is valid (not None and not empty)"""
	return value is not None and value.strip() != ''

