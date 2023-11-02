class CustomLabelEncoder:
    def __init__(self):
        self.mapping = {}

    def fit(self, series):
        unique_values = series.unique()
        self.mapping = {value: idx for idx, value in enumerate(unique_values)}

    def transform(self, series):
        return series.map(lambda x: self.mapping.get(x, -1)).values

    def fit_transform(self, series):
        self.fit(series)
        return self.transform(series)
