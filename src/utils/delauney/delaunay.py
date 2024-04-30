import numpy as np

from helpers import Vertice, Edge

class Delauney:
    vertices: list[Vertice] = []
    edges: set[tuple[int, int]] = set()
    triangles: set[tuple[int, int, int]] = set()

    def __new__(cls, points: np.array) -> list[int, int, int]:
        cls.vertices.clear()
        cls.edges.clear()
        cls.triangles.clear()
        return cls._get_triangulation(points)

    @classmethod
    def _get_triangulation(cls, points: np.array) -> list[np.array]:
        vertices = [Vertice(idx, point) for idx, point in enumerate(points)]
        cls.vertices = sorted(vertices, key=lambda v: (v.x, v.y))

        cls._delauney(cls.vertices)
        return list(cls.triangles)

    @classmethod
    def _delauney(cls, vertices: list[Vertice]) -> None:
        """
        Divide and Conquer Delauney Triangulation

        Params:
        - vertices: list[Vertice] The vertices to triangulate.
        """
        if len(vertices) < 3:
            cls._add_edge(vertices[0], vertices[1])
            return

        if len(vertices) == 3:
            edge1 = Edge(vertices[0], vertices[1])
            edge2 = Edge(vertices[1], vertices[2])
            edge3 = Edge(vertices[2], vertices[0])
            cls._add_edge(edge1)
            cls._add_edge(edge2)
            cls._add_edge(edge3)
            cls._add_triangle(edge1, vertices[2])
            return

        # split the vertices in two halves
        mid = len(vertices) // 2
        LL = vertices[:mid]
        RR = vertices[mid:]

        cls._delauney(LL)
        cls._delauney(RR)

        base_LR = cls._base_LR(LL, RR)

        cls._merge(LL, RR, base_LR)

    @classmethod
    def _add_triangle(cls, edge1: Edge, v2: Vertice):
        v0, v1 = edge1.vertice1, edge1.vertice2
        
        print(cls.edges)


        edge2 = cls.edges[(v1.idx, v2.idx)]
        edge3 = cls.edges[(v2.idx, v0.idx)]


        triangle_str = f"{v0.idx}-{v1.idx}-{v2.idx}"
        edge1.in_triangles.add(triangle_str)
        edge2.in_triangles.add(triangle_str)
        edge3.in_triangles.add(triangle_str)

        cls.triangles[triangle_str] = (
            np.array([v0.idx, v1.idx, v2.idx]),
            edge1,
            edge2,
            edge3,
        )
        
    @classmethod
    def _add_edge(cls, v1: Vertice, v2: Vertice):
        e1 = v1.add_edge(Edge(v1, v2))
        e2 = v2.add_edge(Edge(v2, v1))
        
        cls.edges.add((v1.idx, v2.idx))

        if v1.idx > v2.idx:
            v1, v2 = v2, v1
        
        cls.edges.add((v1.idx, v2.idx))

    @classmethod
    def _remove_edge(cls, v1: Vertice, v2: Vertice):
        e1 = v1.remove_edge(v2)
        e2 = v2.remove_edge(v1)

        cls.edges.remove((v1.idx, v2.idx))

    @classmethod
    def _merge(cls, LL: list[Vertice], RR: list[Vertice], base_LR: Edge) -> None:
        lv: Vertice = cls._left_candidate(base_LR)
        rv: Vertice = cls._right_candidate(base_LR)

        if lv is None and rv is None:
            return

        if lv is None:
            new_base_LR: Edge = Edge(base_LR.vertice1, rv)
            cls._add_edge(new_base_LR)
            cls._add_triangle(base_LR, rv)
            
            return cls._merge(LL, RR, new_base_LR)

        if rv is None:
            new_base_LR: Edge = Edge(lv, base_LR.vertice2)
            cls._add_edge(new_base_LR)
            cls._add_triangle(base_LR, lv)

            return cls._merge(LL, RR, new_base_LR)

        # check if lv is inside circumcircle of base_LR + rv
        if not base_LR.is_inside_circumcircle(rv, lv):
            new_base_LR: Edge = Edge(lv, base_LR.vertice2)
            cls._add_edge(new_base_LR)
            cls._add_triangle(base_LR, lv)
            
        # check if rv is inside circumcircle of base_LR + lv
        elif not base_LR.is_inside_circumcircle(lv, rv):
            new_base_LR: Edge = Edge(base_LR.vertice1, rv)
            cls._add_edge(new_base_LR)
            cls._add_triangle(base_LR, rv)
            
        else:
            raise ValueError("Invalid triangulation")

        return cls._merge(LL, RR, new_base_LR)

    @classmethod
    def _base_LR(cls, LL: list[Vertice], RR: list[Vertice]) -> Edge:
        lcand: Vertice = LL[-1]
        rcand: Vertice = RR[0]

        ledge: Edge = lcand.get_cw_edge(0)
        redge: Edge = rcand.get_cw_edge(0)

        while True:
            if ledge.is_left(rcand):
                lcand = ledge.get_opposite(lcand)
                ledge = lcand.get_cw_edge(0)
            elif redge.is_right(lcand):
                rcand = redge.get_opposite(rcand)
                redge = rcand.get_cw_edge(0)
            else:
                break

        base_LR = Edge(lcand, rcand)
        cls._add_edge(base_LR)

        return base_LR

    @classmethod
    def _left_candidate(cls, base_LR: Edge) -> Vertice | None:
        lv: Vertice = base_LR.vertice1
        cand_edge = lv.get_prev_edge(base_LR)
        potential_cand = cand_edge.get_opposite(lv)

        while True:
            # check if angle with base_LR > 180
            if base_LR.is_right_turn(cand_edge):
                return None

            next_edge = lv.get_prev_edge(cand_edge)
            next_cand = next_edge.get_opposite(lv)

            if cand_edge.is_inside_circumcircle(potential_cand, next_cand):
                cls._remove_edge(cand_edge)

                potential_cand = next_cand
                cand_edge = next_edge
            else:
                return potential_cand

    @classmethod
    def _right_candidate(cls, base_LR: Edge) -> Vertice | None:
        rv: Vertice = base_LR.vertice2
        cand_edge = rv.get_next_edge(base_LR)
        potential_cand = cand_edge.get_opposite(rv)
        

        while True:
            # check if angle with base_LR > 180
            if not base_LR.is_right_turn(cand_edge):
                return None

            next_edge = rv.get_next_edge(cand_edge)
            next_cand = next_edge.get_opposite(rv)

            if cand_edge.is_inside_circumcircle(potential_cand, next_cand):
                cls._remove_edge(cand_edge)

                potential_cand = next_cand
                cand_edge = rv.get_next_edge(cand_edge)
            else:
                return potential_cand


if __name__ == "__main__":
    test_points = np.array([
        [-1, 0],
        [0, 0],
        [1, 0],
        [0, 1], 
        [0, -1]
    ])
    print(Delauney(test_points))
