import json,sys
from pathlib import Path
b=json.loads(Path('run_logs/paired_audit/baseline_1234.json').read_text())['summary']
r=json.loads(Path(sys.argv[1]).read_text())['summary']
passed=r['static_placement']['invalid_cases']==0 and r['success_rate']>=b['success_rate']-0.03 and all(r['rates']['pose'][k]>=v-0.05 for k,v in b['rates']['pose'].items())
print('25 iteration screening:',passed,'baseline',b['success_rate'],'trial',r['success_rate'])
if not passed: raise SystemExit(1)
