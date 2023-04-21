import os

class DefaultConfig:
    @staticmethod
    def absolute_project_root_path():
        return os.path.dirname(__file__).replace("\\", "/")
