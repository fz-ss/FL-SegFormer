# FL-SegFormer

## Requirements

For this project, we used python 3.8.5. We recommend setting up a new virtual
environment:

```shell
python -m venv ~/venv/ERF
source ~/venv/ERF/bin/activate
```

In that environment, the requirements can be installed with:

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # requires the other packages to be installed first
```

Further, please download the MiT weights from SegFormer using the
following script. If problems occur with the automatic download, please follow
the instructions for a manual download within the script.

```shell
sh tools/download_checkpoints.sh
```


## Testing & Predictions

The provided HRDA checkpoint trained on GTA→Cityscapes
(already downloaded by `tools/download_checkpoints.sh`) can be tested on the
Cityscapes validation set using:

```shell
sh test.sh work_dirs/gtaHR2csHR_ERF_mic_hrda_246ef
```

The predictions are saved for inspection to
`work_dirs/gtaHR2csHR_ERF_mic_hrda_246ef/preds`
and the mIoU of the model is printed to the console. The provided checkpoint
should achieve 73.79 mIoU. Refer to the end of
`work_dirs/gtaHR2csHR_ERF_mic_hrda_246ef/20220215_002056.log` for
more information such as the class-wise IoU.

If you want to visualize the LR predictions, HR predictions, or scale
attentions of HRDA on the validation set, please refer to [test.sh](test.sh) for
further instructions.

## Training

For convenience, we provide an [annotated config file](configs/hrda/gtaHR2csHR_hrda.py)
of the final ERF. A training job can be launched using:

```shell
python tools/train.py --config configs/ERF_mic/gtaHR2csHR_ERF_mic_hrda.py
```

The logs and checkpoints are stored in `work_dirs/`.

For the other experiments in our paper, we use a script to automatically
generate and train the configs:

```shell
python run_experiments.py --exp <ID>
```

More information about the available experiments and their assigned IDs, can be
found in [experiments.py](experiments.py). The generated configs will be stored
in `configs/generated/`.

When evaluating a model trained on Synthia→Cityscapes, please note that the
evaluation script calculates the mIoU for all 19 Cityscapes classes. However,
Synthia contains only labels for 16 of these classes. Therefore, it is a common
practice in UDA to report the mIoU for Synthia→Cityscapes only on these 16
classes. As the Iou for the 3 missing classes is 0, you can do the conversion
`mIoU16 = mIoU19 * 19 / 16`.


## Checkpoints

Below, we provide checkpoints of ERF for different benchmarks.
We provide the checkpoint with the median validation performance here.

* [One Drive](https://onedrive.live.com/?id=2D168DA887100C7A%21sfb20426bf8004cf9a626ae6cba551e03&cid=2D168DA887100C7A)


## Framework Structure

This project is based on [mmsegmentation version 0.16.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0).
For more information about the framework structure and the config system,
please refer to the [mmsegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/index.html)
and the [mmcv documentation](https://mmcv.readthedocs.ihttps://arxiv.org/abs/2007.08702o/en/v1.3.7/index.html).

## Acknowledgements

ERF is based on the following open-source projects. We thank their
authors for making the source code publicly available.

* [DAFormer](https://github.com/lhoyer/DAFormer)
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
* [DACS](https://github.com/vikolss/DACS)
* [MIC](https://github.com/lhoyer/MIC)
