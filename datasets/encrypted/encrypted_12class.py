import os
import shutil

from torchvision import datasets
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
    extract_archive,
    verify_str_arg,
)


class Encrypted12Class(datasets.MNIST):
    """`Encrypted-MNIST <https://github.com/echowei/DeepTraffict>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``Encrypted12ClassSessionAllLayers/raw/train-images-idx3-ubyte``
            and  ``Encrypted12ClassSessionAllLayers/raw/t10k-images-idx3-ubyte`` exist.
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

    url = "http://raw.githubusercontent.com/echowei/DeepTraffic/master/2.encrypted_traffic_classification/3.PerprocessResults/12class.zip"
    md5 = "8490cffd069679ac8626da90dd080917"
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
            self.root, "EncryptedTraffic", "raw", "12class", self.sub_raw_folder
        )

    @property
    def raw_parent_folder(self) -> str:
        return os.path.join(self.root, "EncryptedTraffic", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(
            self.root, "EncryptedTraffic", "processed", "12class", self.sub_raw_folder
        )

    def download(self) -> None:
        """Download the EMNIST data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_parent_folder, exist_ok=True)

        download_and_extract_archive(
            self.url, download_root=self.raw_parent_folder, md5=self.md5
        )
        sub_folder = os.path.join(self.raw_parent_folder, "12class")
        for gzip_folder in os.listdir(sub_folder):
            gzip_folder = os.path.join(sub_folder, gzip_folder)
            for gzip_file in os.listdir(gzip_folder):
                if gzip_file.endswith(".gz"):
                    extract_archive(os.path.join(gzip_folder, gzip_file), gzip_folder)


class Encrypted12ClassSessionAllLayers(Encrypted12Class):
    """`Encrypted-MNIST <https://github.com/echowei/DeepTraffict>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``Encrypted12ClassSessionAllLayers/raw/train-images-idx3-ubyte``
            and  ``Encrypted12ClassSessionAllLayers/raw/t10k-images-idx3-ubyte`` exist.
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


class Encrypted12ClassSessionL7(Encrypted12Class):
    """`Encrypted-MNIST <https://github.com/echowei/DeepTraffict>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``Encrypted12ClassSessionL7/raw/train-images-idx3-ubyte``
            and  ``Encrypted12ClassSessionL7/raw/t10k-images-idx3-ubyte`` exist.
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


class Encrypted12ClassFlowAllLayers(Encrypted12Class):
    """`Encrypted-MNIST <https://github.com/echowei/DeepTraffict>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``Encrypted12ClassFlowAllLayers/raw/train-images-idx3-ubyte``
            and  ``Encrypted12ClassFlowAllLayers/raw/t10k-images-idx3-ubyte`` exist.
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


class Encrypted12ClassFlowL7(Encrypted12Class):
    """`Encrypted-MNIST <https://github.com/echowei/DeepTraffict>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``Encrypted12ClassFlowL7/raw/train-images-idx3-ubyte``
            and  ``Encrypted12ClassFlowL7/raw/t10k-images-idx3-ubyte`` exist.
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
