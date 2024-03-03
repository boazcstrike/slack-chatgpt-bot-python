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

def sanitize_prompt(prompt):
  allowed = string.ascii_letters + '"!()[];:-.,/_ ' + string.digits
  prompt = prompt.replace('\n'," ").replace('\t', " ")
  prompt = ''.join(filter(lambda x: x in allowed, prompt.encode('ASCII', "ignore").decode('ASCII')))

  forbidden_strings = [" -o", " --out"]
  for p in forbidden_strings:
    if p in prompt:
      return ""

  return prompt.strip()

def remove_args(args, args_to_remove):
  # e.g. args_to_remove = ["n", "U"]
  olist = args.split("-")

  new_args = []
  for o in olist:
    ok = True
    for p in args_to_remove:
      if o.startswith(p):
        ok = False
    if ok:
      n = o
      if len(o) > 0 and not o.endswith(' '):
        n = n + " "

      new_args.append(n)

  return new_args