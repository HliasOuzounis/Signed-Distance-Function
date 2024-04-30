import numpy as np


class Vertice:
    def __init__(self, idx: int, point: np.array) -> None:
        self.idx: int = idx
        self.x: float = point[0]
        self.y: float = point[1]
        self.cw_edge_list: list["Edge"] = []

    def add_edge(self, edge: 'Edge') -> 'Edge':
        if self == edge.vertice2:
            edge = Edge(edge.vertice2, edge.vertice1)
            
        idx = 0
        while idx < len(self.cw_edge_list) and edge.angle < self.cw_edge_list[idx].angle:
            idx += 1
        self.cw_edge_list.insert(idx, edge)
        
        return edge

    def remove_edge(self, v: object) -> 'Edge':
        for edge in self.cw_edge_list:
            if edge.vertice2 == v:
                self.cw_edge_list.remove(edge)
                return edge

    def get_cw_edge(self, idx) -> "Edge":
        return self.cw_edge_list[idx]

    def get_cc_edge(self, idx) -> "Edge":
        return self.cw_edge_list[-idx - 1]

    def get_next_edge(self, edge) -> "Edge":
        idx = (self.cw_edge_list.index(edge) + 1) % len(self.cw_edge_list)
        return self.get_cw_edge(idx)

    def get_prev_edge(self, edge) -> "Edge":
        idx = self.cw_edge_list.index(edge) - 1
        return self.get_cw_edge(idx)

    def __eq__(self, value: object) -> bool:
        return self.x == value.x and self.y == value.y

    @property
    def points(self):
        return np.array([self.x, self.y])


class Edge:
    def __init__(self, vertice1: Vertice, vertice2: Vertice) -> None:
        if vertice1.x > vertice2.x:
            vertice1, vertice2 = vertice2, vertice1
        if vertice1.x == vertice2.x and vertice1.y > vertice2.y:
            vertice1, vertice2 = vertice2, vertice1

        self.vertice1 = vertice1
        self.vertice2 = vertice2
        self.angle = arctan(vertice2.x - vertice1.x, vertice2.y - vertice1.y)
        self.in_triangles: set[str] = set()

    def get_opposite(self, vertice: Vertice) -> Vertice:
        return self.vertice1 if vertice == self.vertice2 else self.vertice2

    def is_left(self, vertice: Vertice) -> bool:
        v1v = (vertice.x - self.vertice1.x, vertice.y - self.vertice1.y)
        v1v2 = (self.vertice2.x - self.vertice1.x, self.vertice2.y - self.vertice1.y)

        cross = v1v[0] * v1v2[1] - v1v2[0] * v1v[1]

        return cross <= 0

    def is_right(self, vertice: Vertice) -> bool:
        v1v = (vertice.x - self.vertice1.x, vertice.y - self.vertice1.y)
        v1v2 = (self.vertice2.x - self.vertice1.x, self.vertice2.y - self.vertice1.y)

        cross = v1v[0] * v1v2[1] - v1v2[0] * v1v[1]

        return cross >= 0

    def is_inside_circumcircle(self, v3: Vertice, test_vert: Vertice) -> bool:
        """
        Computes 
            | a.x  a.y  a.x²+a.y²  1 |\n
            | b.x  b.y  b.x²+b.y²  1 | > 0\n
            | c.x  c.y  c.x²+c.y²  1 |\n
            | d.x  d.y  d.x²+d.y²  1 |\n

         Return true if d = test_vert is in the circumcircle of a = self.vertice1, b = self.vertice2, c = v3
         From Jon Shewchuk's "Fast Robust predicates for Computational geometry"
        """
        a1 = self.vertice1.x - test_vert.x
        a2 = self.vertice1.y - test_vert.y

        b1 = self.vertice2.x - test_vert.x
        b2 = self.vertice2.y - test_vert.y

        c1 = v3.x - test_vert.x
        c2 = v3.y - test_vert.y

        a3 = pow(a1, 2) + pow(a2, 2)
        b3 = pow(b1, 2) + pow(b2, 2)
        c3 = pow(c1, 2) + pow(c2, 2)

        det = (a1 * b2 * c3 + a2 * b3 * c1 + a3 * b1 * c2) - (
            a3 * b2 * c1 + a1 * b3 * c2 + a2 * b1 * c3
        )

        return det > 0  # det > 0 means that the point is inside the circumcircle

    def is_right_turn(self, edge: "Edge") -> bool:
        if self.vertice1 in (edge.vertice1, edge.vertice2):
            common_v = self.vertice1
        elif self.vertice2 in (edge.vertice1, edge.vertice2):
            common_v = self.vertice2
        else:
            raise AssertionError("Must have one vertice in common")

        v1 = self.get_opposite(common_v).points - common_v.points
        v2 = edge.get_opposite(common_v).points - common_v.points

        return np.cross(v1, v2) < 0


def arctan(x: float, y: float) -> float:
    angle = np.arctan2(y, x)
    if angle < 0:
        angle += np.pi * 2

    return angle


if __name__ == "__main__":
    v = Vertice(0, np.array([-1, 0.1]))
    v1 = Vertice(1, np.array([0, 0]))
    v2 = Vertice(2, np.array([1, 0]))

    e1 = Edge(v1, v2)
    e2 = Edge(v2, v)

    print(e1.is_right_turn(e2))
