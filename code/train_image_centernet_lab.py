"""Train image centernet under an explicit run tag (Wave 2 label-efficiency runs), then evaluate on test.

Same protocol as train_image_centernet.py. Differences: the run folder is runs/image_centernet_<tag>/ (so runs on
different label subsets never share a folder), and --pretrained loads an SSL ResNet-18 into the backbone exactly
as train_image_centernet_ssl.py does (without it the model is the plain from-scratch CenterNet).
The labeled subset is chosen by the dataset root:  PHYSICAL_AI_ROOT=~/SSL_Wave2/roots/p10  (see make_label_splits.py).

    PHYSICAL_AI_ROOT=~/SSL_Wave2/roots/p10 python train_image_centernet_lab.py --tag lab_p10_scratch
    PHYSICAL_AI_ROOT=~/SSL_Wave2/roots/p10 python train_image_centernet_lab.py --tag lab_p10_vicreg200 \
        --pretrained ~/SSL_Wave1b/runs/ssl_vicreg/backbone_e200.pth
Copy next to train_image_centernet_ssl.py in the CenterNet repository.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.runner import main                                   # noqa: E402
from train_image_centernet_ssl import CenterNetSSL               # noqa: E402


def pop_arg(flag):
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag)
    value = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    return value


def build_model(task, num_classes):
    return CenterNetSSL(num_classes)       # = CenterNet unless --pretrained is given


if __name__ == "__main__":
    tag = pop_arg("--tag")
    if not tag:
        raise SystemExit("--tag <name> is required (run folder runs/image_centernet_<tag>)")
    main("train", task="image", version=f"centernet_{tag}", build_model=build_model)
