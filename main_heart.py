import taichi as ti
import taichi.math as tm
import numpy as np
from Geometry.body import Body
import Electrophysiology.ep_heart as ep
import Dynamics.XPBD.XPBD_SNH as xpbd
# from data.LV1 import meshData
# from data.cube import meshData
from data.heart import meshData


def read_data():
    # 顶点位置
    pos_np = np.array(meshData['verts'], dtype=float)
    pos_np = pos_np.reshape((-1, 3))
    # 四面体顶点索引
    tet_np = np.array(meshData['tetIds'], dtype=int)
    tet_np = tet_np.reshape((-1, 4))
    # edge
    edge_np = np.array(meshData['tetEdgeIds'], dtype=int)
    edge_np = edge_np.reshape((-1, 2))
    # surface tri index
    # surf_tri_np = np.array(meshData['tetSurfaceTriIds'], dtype=int)
    # surf_tri_np = surf_tri_np.reshape((-1, 3))

    # tet_fiber方向
    fiber_tet_np = np.array(meshData['fiberDirection'], dtype=float)
    fiber_tet_np = fiber_tet_np.reshape((-1, 3))

    # tet_sheet方向
    sheet_tet_np = np.array(meshData['sheetDirection'], dtype=float)
    sheet_tet_np = sheet_tet_np.reshape((-1, 3))
    # num_edge_set
    num_edge_set_np = np.array(meshData['num_edge_set'], dtype=int)[0]
    # edge_set
    edge_set_np = np.array(meshData['edge_set'], dtype=int)
    # num_tet_set
    num_tet_set_np = np.array(meshData['num_tet_set'], dtype=int)[0]
    # tet_set
    tet_set_np = np.array(meshData['tet_set'], dtype=int)
    # bou_tag
    bou_tag_dirichlet_np = np.array(meshData['bou_tag_dirichlet'], dtype=int)
    bou_tag_neumann_np = np.array(meshData['bou_tag_neumann'], dtype=int)

    Body_ = Body(pos_np, tet_np, edge_np, fiber_tet_np, sheet_tet_np, num_edge_set_np, edge_set_np, num_tet_set_np,
                tet_set_np, bou_tag_dirichlet_np, bou_tag_neumann_np)
    return Body_


@ti.kernel
def get_fiber_field():
    for i in body.elements:
        id0, id1, id2, id3 = body.elements[i][0], body.elements[i][1], body.elements[i][2], body.elements[i][3]
        fiber_field_vertex[2 * i] = body.vertex[id0] + body.vertex[id1] + body.vertex[id2] + body.vertex[id3]
        fiber_field_vertex[2 * i] /= 4.0
        fiber = body.tet_fiber[i]
        fiber_field_vertex[2 * i + 1] = fiber_field_vertex[2 * i] + fiber * 0.1


if __name__ == "__main__":
    ti.init(arch=ti.cuda, default_fp=ti.f32, kernel_profiler=True)

    body = read_data()
    body.translation(0., 5.0, 0.)
    num_per_tet_set_np = np.array(meshData['sum_tet_set'], dtype=int)
    vert_fiber_np = np.array(meshData['vert_fiber'], dtype=float)
    vert_fiber_np = vert_fiber_np.reshape((-1, 3))
    dynamics_sys = xpbd.XPBD_SNH_with_active(body=body, num_pts_np=num_per_tet_set_np, vert_fiber_np=vert_fiber_np)
    # dynamics_sys.numPosIters = 20
    ep_sys = ep.diffusion_reaction(body=body)
    # ep_sys.apply_stimulation()
    # print(body.tet_fiber)
    body.set_Ta(60)

    # draw fiber field
    num_tet = body.num_tet
    fiber_field_vertex = ti.Vector.field(3, dtype=float, shape=(2 * num_tet))
    get_fiber_field()

    # ---------------------------------------------------------------------------- #
    #                                      gui                                     #
    # ---------------------------------------------------------------------------- #
    open_gui = True
    # set parameter
    windowLength = 1024
    lengthScale = min(windowLength, 512)
    light_distance = lengthScale / 25.

    # init the window, canvas, scene and camera
    window = ti.ui.Window("body show", (windowLength, windowLength), vsync=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    # initial camera position
    camera.position(-0.4, 6.02, 2.42)
    camera.lookat(-0.258, 5.643, 1.5)
    camera.up(0, 1, 0)

    # dynamics_sys.update_Jacobi()
    while window.running and open_gui:

        # ep_sys.update(1)
        dynamics_sys.update()
        # print(body.tet_Ta)

        # set the camera, you can move around by pressing 'wasdeq'
        camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.LMB)
        scene.set_camera(camera)

        # set the light
        scene.point_light(pos=(-light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
        scene.point_light(pos=(light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
        scene.ambient_light(color=(0.5, 0.5, 0.5))

        # draw
        # scene.particles(pos, radius=0.02, color=(0, 1, 1))
        scene.mesh(body.vertex, indices=body.surfaces, color=(1.0, 0, 0), two_sided=False)
        # scene.lines(fiber_field_vertex, color=(0., 1.0, 0.), width=1.0)

        # show the frame
        canvas.scene(scene)
        window.show()

    # print(camera.curr_position)
    # print(camera.curr_lookat)
    # print(camera.curr_up)


