import pandas as pd

matches = pd.read_csv("matches.csv", index_col = 0)

matches["date"] = pd.to_datetime(matches["date"])

matches["venue_code"] = matches["venue"].astype("category").cat.codes

matches["opp_code"] = matches["opponent"].astype("category").cat.codes

print(matches)