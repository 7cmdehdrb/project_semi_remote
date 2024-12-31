import numpy as np


class Plane(object):
    def __init__(self, n: np.array, d: float):
        if not isinstance(n, np.ndarray):
            raise ValueError("n must be a numpy array")

        self.n = n
        self.d = d

    def distance(self, p: np.array):
        """Calculate the distance of a point to the plane"""
        return np.abs(np.dot(self.n, p) + self.d) / np.linalg.norm(self.n)


def fit_plane_with_normal(points):
    """
    주어진 3D 점들로부터 평면의 법선 벡터와 d 값을 계산합니다.
    :param points: (N, 3) 형태의 numpy 배열. 각 행은 (x, y, z) 좌표를 나타냄.
    :return: (n, d) 형태로 반환. n은 법선 벡터, d는 평면 방정식 상수.
    """
    # 점들의 중심(평균) 계산
    centroid = np.mean(points, axis=0)

    # 중심에서 점들로의 벡터 계산
    centered_points = points - centroid

    # 공분산 행렬 계산
    covariance_matrix = np.cov(centered_points, rowvar=False)

    # 공분산 행렬의 고유값과 고유벡터 계산
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # 가장 작은 고유값에 대응하는 고유벡터가 법선 벡터
    normal_vector = eigenvectors[:, 0]

    # 평면의 d 값 계산
    d = -np.dot(normal_vector, centroid)

    return Plane(np.array(normal_vector), d)


p1 = {
    "x": 0.9317888326636833,
    "y": 0.11156090208858072,
    "z": 0.18073836479915795,
}

p2 = {
    "x": 0.944986591132788,
    "y": 0.11561529748327867,
    "z": 0.02864824894943209,
}

p3 = {
    "x": 0.9382411074153857,
    "y": 0.3867322757907743,
    "z": 0.02616219654130511,
}

p4 = {"x": 0.9307690064474197, "y": 0.386145576612526, "z": 0.18353896555058538}

points = np.array(
    [
        [p1["x"], p1["y"], p1["z"]],
        [p2["x"], p2["y"], p2["z"]],
        [p3["x"], p3["y"], p3["z"]],
        [p4["x"], p4["y"], p4["z"]],
    ]
)

plane = fit_plane_with_normal(points)

print(plane.n, plane.d)
