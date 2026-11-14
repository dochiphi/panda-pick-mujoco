import mujoco

PANDA_SCENE = "mujoco_menagerie/franka_emika_panda/scene.xml"

# Distractor cubes: (name, rgba). Parked outside the camera view unless placed.
DISTRACTORS = [("blue", [0.1, 0.2, 0.9, 1]),
               ("green", [0.1, 0.7, 0.2, 1]),
               ("orange", [1.0, 0.35, 0.0, 1])]

def build_model(cube_pos=(0.5, 0.1, 0.025)):
    spec = mujoco.MjSpec.from_file(PANDA_SCENE)

    # Red cube (5 cm), free to move
    cube = spec.worldbody.add_body(name="cube", pos=list(cube_pos))
    cube.add_freejoint()
    cube.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.025, 0.025, 0.025],
                  rgba=[0.9, 0.1, 0.1, 1], mass=0.1)

    # Camera 1 m above the table, looking straight down
    spec.worldbody.add_camera(name="top_cam", pos=[0.5, 0.0, 1.0], fovy=60)

    # Second red cube (identical to the target), parked outside the camera view
    c2 = spec.worldbody.add_body(name="cube2", pos=[0.5, -1.0, 0.025])
    c2.add_freejoint()
    c2.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.025, 0.025, 0.025],
                rgba=[0.9, 0.1, 0.1, 1], mass=0.1)

    for i, (name, rgba) in enumerate(DISTRACTORS):
        b = spec.worldbody.add_body(name=f"distractor_{name}", pos=[0.3 + 0.2 * i, 1.0, 0.025])
        b.add_freejoint()
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.025, 0.025, 0.025], rgba=rgba, mass=0.1)

    return spec.compile()
