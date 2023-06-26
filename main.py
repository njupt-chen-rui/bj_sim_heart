import taichi as ti
import taichi.math as tm
import numpy as np
from Geometry.body import Body
import Electrophysiology.d_r_LV as ep
import Dynamics.XPBD.XPBD_SNH as xpbd
from data.LV1 import meshData
# from data.cube import meshData


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


if __name__ == "__main__":
    ti.init(arch=ti.cuda, default_fp=ti.f32, kernel_profiler=True)

    body = read_data()
    body.translation(0., 20.5, 0.)
    num_per_tet_set_np = np.array(meshData['sum_tet_set'], dtype=int)
    dynamics_sys = xpbd.XPBD_SNH_with_active(body=body, num_pts_np=num_per_tet_set_np)
    ep_sys = ep.diffusion_reaction(body=body)
    ep_sys.apply_stimulation()
    # print(body.tet_fiber)


    # """
    # ---------------------------------------------------------------------------- #
    #                                      gui                                     #
    # ---------------------------------------------------------------------------- #
    # set parameter
    windowLength = 1024
    lengthScale = min(windowLength, 512)
    light_distance = lengthScale / 25.

    x_min = min(body.vertex[i][0] for i in range(body.vertex.shape[0]))
    x_max = max(body.vertex[i][0] for i in range(body.vertex.shape[0]))
    y_min = min(body.vertex[i][1] for i in range(body.vertex.shape[0]))
    y_max = max(body.vertex[i][1] for i in range(body.vertex.shape[0]))
    z_min = min(body.vertex[i][2] for i in range(body.vertex.shape[0]))
    z_max = max(body.vertex[i][2] for i in range(body.vertex.shape[0]))
    center = np.array([(x_min + x_max) / 2., (y_min + y_max) / 2., (z_min + z_max) / 2.])

    # init the window, canvas, scene and camera
    window = ti.ui.Window("body show", (windowLength, windowLength), vsync=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    # initial camera position
    camera.position(20, 60, 100)
    camera.lookat(0.5, 0.3, 0.5)
    camera.fov(55)
    # dynamics_sys.update_Jacobi()
    while window.running:

        # ep_sys.update(1)
        dynamics_sys.update()
        # print(body.tet_Ta)

        # set the camera, you can move around by pressing 'wasdeq'
        camera.track_user_inputs(window, movement_speed=0.2, hold_key=ti.ui.LMB)
        scene.set_camera(camera)

        # set the light
        scene.point_light(pos=(-light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
        scene.point_light(pos=(light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
        scene.ambient_light(color=(0.5, 0.5, 0.5))

        # draw
        # scene.particles(pos, radius=0.02, color=(0, 1, 1))
        scene.mesh(body.vertex, indices=body.surfaces, color=(1.0, 0, 0), two_sided=False)
        # scene.mesh(body.vertex, indices=body.surfaces, two_sided=False, per_vertex_color=ep_sys.vertex_color)
        # show the frame
        canvas.scene(scene)
        window.show()

    # """
