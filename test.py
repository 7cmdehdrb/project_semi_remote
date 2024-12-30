import numpy as np


class PositionAverageFilter:
    def __init__(self):
        self.i = 0
        self.average = np.array([0.0, 0.0, 0.0])

    def filter(self, data):
        # 샘플 수 +1 (+1 the number of sample)
        self.i += 1

        # 평균 필터의 alpha 값 (alpha of average filter)
        alpha = (self.i - 1) / (self.i + 0.0)

        # 평균 필터의 재귀식 (recursive expression of average filter)
        average = alpha * self.average + (1 - alpha) * data

        # 평균 필터의 이전 상태값 업데이트 (update previous state value of average filter)
        self.average = average

        return average


filter = PositionAverageFilter()

d1 = np.array([1.0, 2.0, 3.0])
d2 = np.array([2.0, 3.0, 4.0])
d3 = np.array([3.0, 4.0, 5.0])

print(filter.filter(d1))
print(filter.filter(d2))
print(filter.filter(d3))
