import math


def euclidean_distance(point1, point2):
    """
    :param point1: (int, int) first point
    :param point2: (int, int) second point
    :return: the distance between the points
    """
    x1, y1 = point1
    x2, y2 = point2

    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance
