"""
Module to ensure that mujoco sim is working.
"""

import os
import sys
import time
import numpy as np
import mujoco
from mujoco import viewer


if __name__ == '__main__':
    # Read MuJoCo Menagerie path from environment variable
    menagerie_path = os.environ.get("MUJOCO_MENAGERIE_PATH")
    if menagerie_path is None:
        print("ERROR: MUJOCO_MENAGERIE_PATH environment variable is not set.")
        sys.exit(1)

    # Build full path to Franka scene XML
    scene_path = os.path.join(menagerie_path, "franka_emika_panda", "scene.xml")
    if not os.path.exists(scene_path):
        print(f"ERROR: Scene file not found at {scene_path}")
        sys.exit(1)

    # Load the model and data
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    # Launch the passive viewer for non-blocking threading
    with mujoco.viewer.launch_passive(model, data) as viewer_handle:
        # Physics + pertubation loop
        step_count = 0
        while viewer_handle.is_running():
            # Now
            step_start = time.time()
            
            # Random pertubation in every 100 step
            if step_count % 100 == 0:
                # Random changes inside of the data.ctrl[:]
                data.ctrl[:] = np.random.uniform(low=-0.5, high=0.5, size=model.nu)
            
            # Step physics
            mujoco.mj_step(model, data)
            
            # Synchronize the viewer
            viewer_handle.sync()
            step_count += 1
            
            # Run close to real-time
            time_until_next = model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)