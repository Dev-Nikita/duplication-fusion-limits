"""Add the R11 load condition (from meta.json) to a units CSV. Usage: python3 -I add_cond.py <root> <in.csv> <out.csv>"""
import glob, json, os, sys
import pandas as pd
cond = {}
for p in glob.glob(os.path.join(sys.argv[1], "episodes", "ep_*", "meta.json")):
    m = json.load(open(p))
    if "load_cond" in m:
        cond[m["episode"]] = m["load_cond"]
u = pd.read_csv(sys.argv[2]); u["load_cond"] = u.episode.map(cond)
u = u[u.load_cond.isin(["control", "loaded"])]
u.to_csv(sys.argv[3], index=False)
print(len(u), "rows;", u.drop_duplicates("episode").load_cond.value_counts().to_dict())
