import os
import shutil

from torchvision import datasets
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
    extract_archive,
    verify_str_arg,
)


class Encrypted6Class(datasets.MNIST):
    """`Encrypted-MNIST <https://github.com/echowei/DeepTraffict>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``Encrypted6ClassVpnSessionAllLayers/raw/train-images-idx3-ubyte``
            and  ``Encrypted6ClassVpnSessionAllLayers/raw/t10k-images-idx3-ubyte`` exist.
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
    classes = []

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


class Encrypted6ClassVpnSessionAllLayers(Encrypted6Class):
    """`Encrypted-MNIST <https://github.com/echowei/DeepTraffict>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``Encrypted6ClassVpnSessionAllLayers/raw/train-images-idx3-ubyte``
            and  ``Encrypted6ClassVpnSessionAllLayers/raw/t10k-images-idx3-ubyte`` exist.
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

    sub_raw_folder = "VpnSessionAllLayers"
    classes = [
        "Vpn_Chat",
        "Vpn_Email",
        "Vpn_File",
        "Vpn_P2p",
        "Vpn_Streaming",
        "Vpn_Voip",
    ]


class Encrypted6ClassVpnSessionL7(Encrypted6Class):
    """`Encrypted-MNIST <https://github.com/echowei/DeepTraffict>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``Encrypted6ClassVpnSessionL7/raw/train-images-idx3-ubyte``
            and  ``Encrypted6ClassVpnSessionL7/raw/t10k-images-idx3-ubyte`` exist.
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

    sub_raw_folder = "VpnSessionL7"
    classes = [
        "Vpn_Chat",
        "Vpn_Email",
        "Vpn_File",
        "Vpn_P2p",
        "Vpn_Streaming",
        "Vpn_Voip",
    ]


class Encrypted6ClassVpnFlowAllLayers(Encrypted6Class):
    """`Encrypted-MNIST <https://github.com/echowei/DeepTraffict>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``Encrypted6ClassVpnFlowAllLayers/raw/train-images-idx3-ubyte``
            and  ``Encrypted6ClassVpnFlowAllLayers/raw/t10k-images-idx3-ubyte`` exist.
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

    sub_raw_folder = "VpnFlowAllLayers"
    classes = [
        "Vpn_Chat",
        "Vpn_Email",
        "Vpn_File",
        "Vpn_P2p",
        "Vpn_Streaming",
        "Vpn_Voip",
    ]


class Encrypted6ClassVpnFlowL7(Encrypted6Class):
    """`Encrypted-MNIST <https://github.com/echowei/DeepTraffict>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``Encrypted6ClassVpnFlowL7/raw/train-images-idx3-ubyte``
            and  ``Encrypted6ClassVpnFlowL7/raw/t10k-images-idx3-ubyte`` exist.
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

    sub_raw_folder = "VpnFlowL7"
    classes = [
        "Vpn_Chat",
        "Vpn_Email",
        "Vpn_File",
        "Vpn_P2p",
        "Vpn_Streaming",
        "Vpn_Voip",
    ]


class Encrypted6ClassNovpnSessionAllLayers(Encrypted6Class):
    """`Encrypted-MNIST <https://github.com/echowei/DeepTraffict>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``Encrypted6ClassNovpnSessionAllLayers/raw/train-images-idx3-ubyte``
            and  ``Encrypted6ClassNovpnSessionAllLayers/raw/t10k-images-idx3-ubyte`` exist.
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

    sub_raw_folder = "NovpnSessionAllLayers"
    classes = ["Chat", "Email", "File", "P2p", "Streaming", "Voip"]


class Encrypted6ClassNovpnSessionL7(Encrypted6Class):
    """`Encrypted-MNIST <https://github.com/echowei/DeepTraffict>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``Encrypted6ClassNovpnSessionL7/raw/train-images-idx3-ubyte``
            and  ``Encrypted6ClassNovpnSessionL7/raw/t10k-images-idx3-ubyte`` exist.
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

    sub_raw_folder = "NovpnSessionL7"
    classes = ["Chat", "Email", "File", "P2p", "Streaming", "Voip"]


class Encrypted6ClassNovpnFlowAllLayers(Encrypted6Class):
    """`Encrypted-MNIST <https://github.com/echowei/DeepTraffict>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``Encrypted6ClassNovpnFlowAllLayers/raw/train-images-idx3-ubyte``
            and  ``Encrypted6ClassNovpnFlowAllLayers/raw/t10k-images-idx3-ubyte`` exist.
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

    sub_raw_folder = "NovpnFlowAllLayers"
    classes = ["Chat", "Email", "File", "P2p", "Streaming", "Voip"]


class Encrypted6ClassNovpnFlowL7(Encrypted6Class):
    """`Encrypted-MNIST <https://github.com/echowei/DeepTraffict>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``Encrypted6ClassNovpnFlowL7/raw/train-images-idx3-ubyte``
            and  ``Encrypted6ClassNovpnFlowL7/raw/t10k-images-idx3-ubyte`` exist.
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

    sub_raw_folder = "NovpnFlowL7"
    classes = ["Chat", "Email", "File", "P2p", "Streaming", "Voip"]
