"""
Module to preview the environment.
"""

import os
import mujoco
from mujoco import viewer

# Resolve menagerie path
menagerie = os.environ.get("MUJOCO_MENAGERIE_PATH")
if menagerie is None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    menagerie = os.path.join(repo_root, "dev", "mujoco_menagerie")

panda_dir = os.path.join(menagerie, "franka_emika_panda")

# Read our scene XML as-is (with relative include="panda.xml")
with open("envs/assets/pickplace_scene.xml") as f:
    xml = f.read()

# Write temp copy INTO panda_dir, so all relative paths resolve correctly
temp_scene_path = os.path.join(panda_dir, "_pickplace_temp.xml")

try:
    with open(temp_scene_path, "w") as f:
        f.write(xml)
    model = mujoco.MjModel.from_xml_path(temp_scene_path)
finally:
    if os.path.exists(temp_scene_path):
        os.remove(temp_scene_path)

data = mujoco.MjData(model)
print(f"Loaded: nq={model.nq}, nv={model.nv}, nu={model.nu}, nbody={model.nbody}")

viewer.launch(model, data)