import shapefile

with shapefile.Writer("test", shapeType=shapefile.POLYGON) as w:
    w.field("name", "C")
    w.poly([[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]])
    w.record("Square")



