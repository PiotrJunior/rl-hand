import genesis as gs
import torch
gs.init(backend=gs.cpu)

scene = gs.Scene(
    # show_viewer=True,
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(1, 1, 2),
        camera_lookat=(0, 0, 1),
        res=(1920, 1080),
        max_FPS=120,
    ),
)

plane = scene.add_entity(gs.morphs.Plane())

orca = scene.add_entity(
    gs.morphs.URDF(
        file='orcahand_description/v2/models/urdf/orcahand_right.urdf',
        pos=(0.0, 0.0, 1.0),
        fixed=True,
    ),
)

camera = scene.add_camera(
    res=(480, 480),
    pos=(1, 1, 2),
    lookat=(0, 0, 1),
    fov=45,
    # GUI=True,
)

scene.build()

import cv2
print("Scene initialized. Starting main loop...")

print("Liczba stopni swobody (DOFs):", orca.n_dofs)

print("--- NAZWY DOF / STAWÓW ---")
joint_names = [
    "right_wrist",
    "right_p-abd",
    "right_p-mcp",
    "right_p-pip",
    "right_r-abd",
    "right_r-mcp",
    "right_r-pip",
    "right_m-abd",
    "right_m-mcp",
    "right_m-pip",
    "right_i-abd",
    "right_i-mcp",
    "right_i-pip",
    "right_t-cmc",
    "right_t-abd",
    "right_t-mcp",
    "right_t-pip",
]

for i, joint in enumerate(orca.joints):
    # Sprawdzamy czy dany staw ma stopnie swobody (nie jest zamrożony/stały)
    print(f"Staw {i}: {joint.name} | Typ: {joint.type}")
    # if joint.n_dofs > 0:
        # print(f"DOF idx: {joint.dof_start_idx} | Staw: {joint.name} | Typ: {joint.type}")

# 2. Alternatywnie: prosta lista samych nazw powiązanych bezpośrednio z indeksami DOF
# print("\n--- KORTEŻ NAZW DOF ---")
# dof_names = orca.dof_names
# for idx, name in enumerate(dof_names):
#     print(f"Indeks {idx}: {name}")

# 2. Sterowanie pozycją stawów (Position Control)
# Zakładamy, że robot ma np. 6 stawów. Podajemy pozycje w radianach.
# target_positions = [0.0, 0.5, -0.5, 0.0, 1.0, 0.0]
# orca.control_dofs_position(target_positions)

# 3. Jeśli wolisz sterować prędkością (Velocity Control):
# orca.control_dofs_velocity([0.1, 0.1, -0.1, 0.0, 0.0, 0.0])

# 4. Jeśli chcesz "zresetować" stan robota natychmiastowo (bez fizyki):
# orca.set_dofs_position([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

for i in range(1000):
    # action = torch.zeros(orca.n_dofs)
    # action[0] = -1
    # action[1:3] = -1 
    # orca.control_dofs_position(action)
    scene.step()
    # rgb, _, _, _ = camera.render(rgb=True)
    # cv2.waitKey(10)  # Krótkie opóźnienie, aby upewnić się, że kamera jest gotowa
