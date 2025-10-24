import numpy as np
from sklearn.linear_model import LinearRegression

class LinearRegression_Acumulado:
    def __init__(self):
        self.model = None

    def train(self, data):
        x = np.arange(len(data)).reshape(-1, 1)
        y = np.array(data['Acumulado'])
        self.model = LinearRegression()
        self.model.fit(x, y)
        # return f"valores de x: {x}, y: {y}"

    def prediction(self, indices: int | list[int]) -> int | np.ndarray:
        if self.model is None:
            raise ValueError("O modelo não foi treinado.")

        pred = self.model.predict(np.array(indices).reshape(-1, 1))

        if isinstance(indices, int):
            return pred[0]
        
        return pred