import torch
from torchvision import datasets

# from mmVision.model.tinyVGG import TinyVGG
from mmVision import settings as st
from mmVision.model.pyt import Vision



train_dir = st.MERGED_BALANCED_1000_TRAIN_ONLY_IFCB_DIR
test_dir = st.MERGED_BALANCED_1000_TEST_ONLY_IFCB_DIR

train_dir = st.IFCB_SOSIK_TRAIN_DIR_BALANCED_1000
test_dir = st.IFCB_SOSIK_TEST_DIR_BALANCED_1000

vision = Vision(train_dir=train_dir,
                test_dir=test_dir,
                valid_dir=None,
                epochs=2,
                batch_size=32,
                color_channel=3,
                image_size=(224, 224),
                learning_rate=0.001
                )
vision.fit()