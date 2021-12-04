import os

from tqdm import tqdm
import cv2


if __name__ == "__main__":
    ori_data_path = os.path.join(os.path.dirname(__file__), "Adobe5K")
    img_lst = os.listdir(ori_data_path)

    save_path = "Adobe5K_gray"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for img_name in tqdm(img_lst):
        img = cv2.imread(os.path.join(ori_data_path, img_name))
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(os.path.join(save_path, img_name), img_gray)
