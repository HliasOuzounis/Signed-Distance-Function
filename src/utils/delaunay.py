import numpy as np

from vvrpywork.shapes import Triangle2D, Point2D

    
def get_delaunay(points: np.array) -> list[Triangle2D]:
    minx = np.min(points[:, 0])
    maxx = np.max(points[:, 0])
    miny = np.min(points[:, 1])
    maxy = np.max(points[:, 1])
    
    # bounding rectangle for the points
    triangles = [
        Triangle2D(Point2D((minx, miny)), Point2D((minx, maxy)), Point2D((maxx, miny))),
        Triangle2D(Point2D((maxx, miny)), Point2D((minx, maxy)), Point2D((maxx, maxy)))             
    ]
    
    for new_point in points:
        _add_point(triangles, Point2D(new_point))
    
    return triangles

def _add_point(triangles_list: list[Triangle2D], point: Point2D):
    for i, triangle in enumerate(triangles_list):
        if triangle.contains(point):
            triangles_list.pop(i)
            
            p1, p2, p3 = (
                triangle.getPoint1(),
                triangle.getPoint2(),
                triangle.getPoint3(),
            )
            
            triangles_list.append(Triangle2D(point, p1, p2))
            triangles_list.append(Triangle2D(point, p1, p3))
            triangles_list.append(Triangle2D(point, p2, p3))
            
            return
        
    raise RuntimeError("Point is not inside any triangle")