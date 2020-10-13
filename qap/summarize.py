# @license: %MIT License%:~ http://www.opensource.org/licenses/MIT
# @project: qap_lp
# @file: /summarize.py
# @created: Monday, 12th October 2020
# @author: brentian (chuwzhang@gmail.com)
# @modified: brentian (chuwzhang@gmail.com>)
#    Monday, 12th October 2020 10:10:36 pm
# @description:
# summarize the current results and produce a benchmark as of today here.

import pandas as pd
import json
import os
import datetime

d = datetime.datetime.now()
dt_str = d.strftime("%Y%m%d-%H%M")
smr = [f for f in os.listdir('./result/') if f.endswith("json")]
data = [
    v for f in smr for k, v in json.load(open(f"result/{f}")).items()
    if k != 'best'
]
df = pd.DataFrame.from_records(data)

print(df)
df.to_csv(f"summary_{dt_str}.csv")
