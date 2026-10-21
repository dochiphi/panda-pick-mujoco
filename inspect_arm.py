import mujoco

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/scene.xml")

print("Joints:", model.njnt)
for i in range(model.njnt):
    print(" ", i, model.joint(i).name, "range:", model.jnt_range[i])

print("\nActuators:", model.nu)
for i in range(model.nu):
    print(" ", i, model.actuator(i).name, "ctrl range:", model.actuator_ctrlrange[i])
