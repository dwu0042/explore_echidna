#! /usr/bin/env -S manim -qh

from manim import *

config.background_color = WHITE
config.background_opacity = 0.0

class BiLayer(ThreeDScene):

    def construct(self):

        self.set_camera_orientation(
            phi=45 * DEGREES,
        )

        separation = 2.5
        levels = [separation, -separation]
        plane_height = 3
        plane_width = 4
        plane_shear_angle = 60 * DEGREES
        plane_shear = plane_height / np.tan(plane_shear_angle)
        colours = [YELLOW, GREEN]
        names = ['Direct', 'Indirect']
        for level, colour, name in zip(levels, colours, names):
            plane = Polygon(
                *[
                    (-plane_width - plane_shear, -plane_height, level),
                    (-plane_width + plane_shear, plane_height, level),
                    (plane_width + plane_shear, plane_height, level),
                    (plane_width - plane_shear, -plane_height, level),
                ],
                color = colour, 
                fill_opacity=0.4,
                stroke_width=0.0,
            )
            plane.set_z_index(0)
            self.add(plane)
            label_text = Text(name, color=BLACK, font_size=18)
            label_text.align_to(label_text.get_corner(DOWN + LEFT), DOWN + LEFT)
            label_text.move_to(plane.get_vertices()[0] + UP/2 + RIGHT, aligned_edge=DOWN + LEFT)
            self.add(label_text)

        vertices = ['A', 'B', 'C']
        
        graph_config = dict(
            labels = True,
            layout = 'circular',
            vertex_config = dict(
                fill_opacity = 1.0,
                stroke_color = BLACK,
            ),
            edge_config = dict(
                path_arc = 1.2,
                loop_config = {
                    "angle_between_points": PI, 
                    "path_arc": 3 * PI / 2,
                },
                color = BLACK,
            )
        )

        G0 = DiGraph(
            vertices = vertices,
            edges = [
                ('A', 'B'),
                ('B', 'C'),
                ('A', 'C'),
            ],
            **graph_config
        )

        G0.set_z(levels[0])
        for v in G0.vertices.values():
            v.set_z_index(5)
        for e in G0.edges.values():
            e.set_z_index(4)

        self.add(G0)

        G1 = DiGraph(
            vertices = vertices,
            edges = [
                ('A', 'B'),
                ('B', 'A'),
                ('B', 'C'),
                ('C', 'A'),
                ('A', 'A'),
                ('B', 'B'),
                ('C', 'C'),
            ],
            **graph_config
        )

        G1.set_z(levels[1])
        for v in G1.vertices.values():
            v.set_z_index(5)
        for e in G1.edges.values():
            e.set_z_index(4)

        self.add(G1)
                
        for vertex in vertices:
            start, end = G0[vertex].get_center(), G1[vertex].get_center()
            ln = DashedLine(start, end, dashed_ratio=0.2, color=BLACK)
            ln.set_z_index(3)
            self.add(ln)