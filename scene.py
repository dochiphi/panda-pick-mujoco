import mujoco

PANDA_SCENE = "mujoco_menagerie/franka_emika_panda/scene.xml"

def build_model(cube_pos=(0.5, 0.1, 0.025)):
    spec = mujoco.MjSpec.from_file(PANDA_SCENE)

    # Red cube (5 cm), free to move
    cube = spec.worldbody.add_body(name="cube", pos=list(cube_pos))
    cube.add_freejoint()
    cube.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.025, 0.025, 0.025],
                  rgba=[0.9, 0.1, 0.1, 1], mass=0.1)

    # Camera 1 m above the table, looking straight down
    spec.worldbody.add_camera(name="top_cam", pos=[0.5, 0.0, 1.0], fovy=60)

    return spec.compile()
