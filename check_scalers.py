import pickle

scalers = pickle.load(open('data/processed/scalers.pkl', 'rb'))
print('Scalers:', list(scalers.keys()))
for k, v in scalers.items():
    print(f'{k}: {getattr(v, "n_features_in_", "unknown")} features')
