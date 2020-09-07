import os
from ..core.data import Data


class KITTIData(Data):
    dirs = []

    def __init__(self, data_dir, stat_log_dir=None,
                 development=True, fast_dir=None):
        super().__init__(data_dir, stat_log_dir,
                         development=development,
                         fast_dir=fast_dir)

    def get_raw_dirs(self):
        top_dir = os.path.join(self.current_dir, 'grid_map')
        dirs = []
        image_00_folder = os.path.join(top_dir, '00') #added
        image_01_folder = os.path.join(top_dir, '01') #added
        image_02_folder = os.path.join(top_dir, '02') #added
        image_03_folder = os.path.join(top_dir, '03') #added
        image_04_folder = os.path.join(top_dir, '04') #added
        image_05_folder = os.path.join(top_dir, '05') #added
        image_06_folder = os.path.join(top_dir, '06') #added
        image_07_folder = os.path.join(top_dir, '07') #added
        image_08_folder = os.path.join(top_dir, '08') #added
        image_09_folder = os.path.join(top_dir, '09') #added
        image_10_folder = os.path.join(top_dir, '10') #added

        image_0000_folder = os.path.join(top_dir, '0000')  # added
        image_0002_folder = os.path.join(top_dir, '0002')  # added
        image_0003_folder = os.path.join(top_dir, '0003')  # added
        image_0004_folder = os.path.join(top_dir, '0004')  # added
        image_0005_folder = os.path.join(top_dir, '0005')  # added
        image_0006_folder = os.path.join(top_dir, '0006')  # added
        image_0007_folder = os.path.join(top_dir, '0007')  # added
        image_0008_folder = os.path.join(top_dir, '0008')  # added
        dirs.extend([image_00_folder, image_01_folder, image_02_folder, image_03_folder,
                     image_04_folder, image_05_folder, image_06_folder, image_07_folder,
                     image_0000_folder, image_0002_folder, image_0003_folder, image_0004_folder,
                     image_0005_folder, image_0006_folder, image_0007_folder, image_0008_folder])
        return dirs
