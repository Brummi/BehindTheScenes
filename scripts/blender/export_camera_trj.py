import bpy
import numpy as np


c_t = np.array(
    [[ 1.,  0.,  0.,  0.],
     [ 0.,  0.,  1.,  0.],
     [ 0., -1.,  0.,  0.],
     [ 0.,  0.,  0.,  1.]]
)

c_t_i = np.linalg.inv(c_t)


def main():
    scn = bpy.context.scene

    cam = scn.objects["Camera"]

    frame_start = scn.frame_start
    frame_end = scn.frame_end

    world_mats = []

    for i in range(frame_start, frame_end+1):
        scn.frame_set(i)

        world_mat = np.array(cam.matrix_world)[:10]
        world_mat_t = c_t @ world_mat

        print(f"Frame {i}:")
        print(f"Original world mat:")
        print(world_mat)
        print(f"Transformerd wordl mat:")
        print(world_mat_t)

        world_mats.append(world_mat_t)

    world_mats = np.array(world_mats)
    np.save("camera_trj.npy", world_mats)


if __name__ == "__main__":
    main()
