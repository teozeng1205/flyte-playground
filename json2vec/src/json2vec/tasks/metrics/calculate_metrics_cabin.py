import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
from metrics_reader import read_metrics_dataframe

"""Generate per-validatingCarrier and cabin metrics table.

Requested columns:
validatingcarrier, cabin, samplecount, median_totalamount, avg_totalamount,
avg_yq, avg_yr, avg_yqyr, rmse, mae,
abs_acc2, abs_acc5, abs_acc10, acc_pct2, acc_pct5, acc_pct10
"""

path = sys.argv[1] if len(sys.argv) > 1 else "data/rank-0.parquet"
out_path = sys.argv[2] if len(sys.argv) > 2 else "metrics_cabin.csv"
df = read_metrics_dataframe(path)

for col in ['yrAmount', 'yqAmount', 'totalAmount', 'predicted_YR_tax', 'predicted_YQ_tax', 'predicted_total_tax', 'total_tax']:
    if col in df.columns:
        df[col] = df[col].fillna(0)

df['actual_tax_sum'] = df['yrAmount'] + df['yqAmount']
if 'predicted_total_tax' in df.columns:
    df['pred_tax_sum'] = df['predicted_total_tax']
elif 'predicted_YR_tax' in df.columns or 'predicted_YQ_tax' in df.columns:
    df['pred_tax_sum'] = df.get('predicted_YR_tax', 0) + df.get('predicted_YQ_tax', 0)
elif 'total_tax' in df.columns:
    df['pred_tax_sum'] = df['total_tax']
else:
    raise ValueError(f"No prediction columns found in input data: {path}")
df['abs_error'] = (df['actual_tax_sum'] - df['pred_tax_sum']).abs()

# Relative error, guard division by zero
df['rel_error'] = np.where(df['actual_tax_sum'] != 0, df['abs_error'] / df['actual_tax_sum'], np.nan)

group_cols = ['validatingCarrier', 'cabin']
for col in group_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in input data: {path}")

def pct(condition_series):
    return condition_series.mean() if len(condition_series) else np.nan

rows = []
for (carrier, cabin), g in df.groupby(group_cols):
    samplecount = len(g)
    median_totalamount = g['totalAmount'].median() if 'totalAmount' in g else np.nan
    avg_totalamount = g['totalAmount'].mean() if 'totalAmount' in g else np.nan
    avg_yq = g['yqAmount'].mean()
    avg_yr = g['yrAmount'].mean()
    avg_yqyr = (g['yqAmount'] + g['yrAmount']).mean()
    mae = mean_absolute_error(g['actual_tax_sum'], g['pred_tax_sum'])
    rmse = np.sqrt(mean_squared_error(g['actual_tax_sum'], g['pred_tax_sum']))

    abs_acc2 = pct(g['abs_error'] <= 2)
    abs_acc5 = pct(g['abs_error'] <= 5)
    abs_acc10 = pct(g['abs_error'] <= 10)

    rel_nonzero = g[g['actual_tax_sum'] != 0]
    acc_pct2 = pct(rel_nonzero['rel_error'] <= 0.02)
    acc_pct5 = pct(rel_nonzero['rel_error'] <= 0.05)
    acc_pct10 = pct(rel_nonzero['rel_error'] <= 0.10)

    rows.append({
        'validatingcarrier': carrier,
        'cabin': cabin,
        'samplecount': samplecount,
        'median_totalamount': median_totalamount,
        'avg_totalamount': avg_totalamount,
        'avg_yq': avg_yq,
        'avg_yr': avg_yr,
        'avg_yqyr': avg_yqyr,
        'rmse': rmse,
        'mae': mae,
        'abs_acc2': abs_acc2,
        'abs_acc5': abs_acc5,
        'abs_acc10': abs_acc10,
        'acc_pct2': acc_pct2,
        'acc_pct5': acc_pct5,
        'acc_pct10': acc_pct10,
    })

out_df = pd.DataFrame(rows)

# Order columns explicitly
cols = [
    'validatingcarrier','cabin','samplecount','median_totalamount','avg_totalamount','avg_yq','avg_yr','avg_yqyr',
    'rmse','mae','abs_acc2','abs_acc5','abs_acc10','acc_pct2','acc_pct5','acc_pct10'
]
out_df = out_df[cols]

csv_str = out_df.to_csv(index=False)
with open(out_path, 'w') as f:
    f.write(csv_str)
print(f"Metrics written to {out_path} ({len(out_df)} rows).")
