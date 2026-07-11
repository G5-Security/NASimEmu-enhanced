import os
import sys

# NASimEmu-agents/ needs to be on sys.path for `llm_teacher`, `nasim_problem`,
# `config`, `net`, etc. to import -- same pattern llm_teacher/label_states.py
# and experiments/evaluate_llm_selector.py already use for standalone scripts.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "src"))
