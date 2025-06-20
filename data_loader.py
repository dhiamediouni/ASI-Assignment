def load_mauna_loa_data():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    file_path = "./co2-mm-mlo.csv"
    df = pd.read_csv(file_path)
    df.columns = [col.strip() for col in df.columns]

    df = df[["Decimal Date", "Average"]].dropna()
    df = df.rename(columns={"Decimal Date": "time", "Average": "average"})

    X = df["time"].values.reshape(-1, 1)
    y = df["average"].values.reshape(-1, 1)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    return train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False), scaler_x, scaler_y, df
