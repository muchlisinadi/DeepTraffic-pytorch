import os
import shutil

from torchvision import datasets
from torchvision.datasets.utils import (check_integrity,
                                        download_and_extract_archive,
                                        extract_archive, verify_str_arg)


class Encrypted2Class(datasets.MNIST):
    """`Encrypted-MNIST <https://github.com/echowei/DeepTraffict>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``Encrypted2ClassSessionAllLayers/raw/train-images-idx3-ubyte``
            and  ``Encrypted2ClassSessionAllLayers/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    url = "http://raw.githubusercontent.com/echowei/DeepTraffic/master/2.encrypted_traffic_classification/3.PerprocessResults/2class.zip"
    md5 = "81788848d634b10153733adebbf3997c"
    resources = [
        ("train-images-idx3-ubyte.gz", ""),
        ("train-labels-idx1-ubyte.gz", ""),
        ("t10k-images-idx3-ubyte.gz", ""),
        ("t10k-labels-idx1-ubyte.gz", ""),
    ]
    classes = ["Benign", "Malign"]

    @property
    def sub_raw_folder(self) -> str:
        raise NotImplementedError

    @property
    def raw_folder(self) -> str:
        return os.path.join(
            self.root, "EncryptedTraffic", "raw", "2class", self.sub_raw_folder
        )

    @property
    def raw_parent_folder(self) -> str:
        return os.path.join(self.root, "EncryptedTraffic", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(
            self.root, "EncryptedTraffic", "processed", "2class", self.sub_raw_folder
        )

    def download(self) -> None:
        """Download the EMNIST data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_parent_folder, exist_ok=True)

        download_and_extract_archive(
            self.url, download_root=self.raw_parent_folder, md5=self.md5
        )
        sub_folder = os.path.join(self.raw_parent_folder, "2class")
        for gzip_folder in os.listdir(sub_folder):
            gzip_folder = os.path.join(sub_folder, gzip_folder)
            for gzip_file in os.listdir(gzip_folder):
                if gzip_file.endswith(".gz"):
                    extract_archive(os.path.join(gzip_folder, gzip_file), gzip_folder)


class Encrypted2ClassSessionAllLayers(Encrypted2Class):
    """`Encrypted-MNIST <https://github.com/echowei/DeepTraffict>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``Encrypted2ClassSessionAllLayers/raw/train-images-idx3-ubyte``
            and  ``Encrypted2ClassSessionAllLayers/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    sub_raw_folder = "SessionAllLayers"


class Encrypted2ClassSessionL7(Encrypted2Class):
    """`Encrypted-MNIST <https://github.com/echowei/DeepTraffict>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``Encrypted2ClassSessionL7/raw/train-images-idx3-ubyte``
            and  ``Encrypted2ClassSessionL7/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    sub_raw_folder = "SessionL7"


class Encrypted2ClassFlowAllLayers(Encrypted2Class):
    """`Encrypted-MNIST <https://github.com/echowei/DeepTraffict>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``Encrypted2ClassFlowAllLayers/raw/train-images-idx3-ubyte``
            and  ``Encrypted2ClassFlowAllLayers/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    sub_raw_folder = "FlowAllLayers"


class Encrypted2ClassFlowL7(Encrypted2Class):
    """`Encrypted-MNIST <https://github.com/echowei/DeepTraffict>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``Encrypted2ClassFlowL7/raw/train-images-idx3-ubyte``
            and  ``Encrypted2ClassFlowL7/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    sub_raw_folder = "FlowL7"
