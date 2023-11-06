from envs import Factory
from tqdm import tqdm
import numpy as np
from envs import Container

if __name__ == '__main__':
    data_num = 1000

    box_num = 20
    box_range = [10, 60]

    # box_num = 10
    # box_range = [10, 80]
    # target_container_size = [120, 120, 5000]
    # gripper_width = 10

    box_num = 10
    box_range = [1, 8]
    target_container_size = [10, 10, 5000]
    target_container_size = [14, 14, 5000]

    gripper_width = 1

    fact_type = "tap_fake"
    fact_type = "box"
    data_type = 'rand'

    data_folder = f"./tapnet/data/{fact_type}/{data_type}/{box_num}/[{target_container_size[0]}_{target_container_size[1]}]_[{box_range[0]}_{box_range[1]}]_{gripper_width}"
    # os.makedirs(save_path, exist_ok=True)


    factory = Factory(fact_type, data_type, target_container_size, gripper_width, None, data_folder)
    # factory.load_order(0)
    np.random.seed(666)
    # # TODO
    for i in tqdm(range(data_num)):
        factory.reset()
        factory.new_order(box_range, box_num)
        factory.save_order(i)

    pass