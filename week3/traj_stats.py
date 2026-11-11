"""Summarise mini-swe-agent v2 trajectories: commands, reading vs acting, and git-history lookups."""
import json, re, glob, sys
for f in sorted(glob.glob(sys.argv[1] + '/*/*.traj.json')):
    d = json.load(open(f)); cmds = []
    for m in d['messages']:
        if m.get('role') == 'assistant':
            for tc in m.get('tool_calls') or []:
                try: cmds.append(json.loads(tc['function']['arguments']).get('command', ''))
                except Exception: pass
    core = [re.sub(r'^cd /testbed\s*&&\s*', '', c).strip() for c in cmds]
    read = [c for c in core if re.match(r'^(cat|grep|ls|find|head|tail|sed -n|rg|wc|pwd|tree)', c)]
    hist = [c for c in core if re.search(r'git (log|show|blame|merge-base)', c)]
    print(f"{f.split('/')[-2]}: {len(cmds)} commands, {len(read)} reading, {len(hist)} git-history lookups, ${d['info']['model_stats']['instance_cost']:.3f}")
    for c in hist: print("   ", c[:110])
