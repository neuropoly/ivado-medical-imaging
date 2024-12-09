# Data preparation

These scripts prepare the data for training. It takes as input the [Spine Generic Public Database (Multi-Subject)](https://github.com/spine-generic/data-multi-subject) and outputs BIDS-compatible datasets with segmentation labels for each subject. More specifically, for each subject, the segmentation is run in one volume (T1w), then all volumes are registered to the T1w volume so that all volumes are in the same voxel space and the unique segmentation can be used across volumes.

## Dependencies

In its current state, this pipeline uses [SCT development version](https://github.com/neuropoly/spinalcordtoolbox#install-from-github-development). Once the pipeline is finalized, a stable version of SCT will be associated with this pipeline and indicated here. For now, please use the latest development version of SCT.

## How to run

#### Activate environment

See [README](../README.md)
~~~
source PATH_TO_YOUR_VENV/venv-ivadomed/bin/activate
~~~

#### Initial steps, check for folder integrity

- Copy the file `config_template.yml` and rename it as `config.yml`.
- Edit the file `config.yml` and modify the values according to your needs.
- Make sure input files are present:
~~~
sct_run_batch -script check_input_files.sh -config config.yml
~~~

#### Run first processing

Loop across subjects and run full processing:

~~~
sct_run_batch -script prepare_data.sh -config config.yml
~~~

#### Perform QC

##### Spinal cord segmentations

- Open qc/index.html
- Search only for "deepseg" QC entries (use "search" field)
- Take a screenshot of the browser when you spot a problem (wait for the segmentation to appear before taking the screenshot)
- If the data are of **very** bad quality, also take a screenshot (this time, wait for the segmentation to disappear)
- Copy all screenshots under qc_feedback/

##### Registration of MT scans

- Search for "register_multimodal"
- Take a screenshot of the browser when you spot a problem (wait for the segmentation to appear before taking the screenshot)
- If the data are of **very** bad quality, also take a screenshot (this time, wait for the segmentation to disappear)
- Copy all screenshots under qc_feedback/

##### Manually correct the segmentations

Check the following files under e.g. `result/sub-balgrist01/anat/tmp`:

| Image  | Segmentation  |
|:---|:---|
| sub-XX_acq-T1w_MTS_crop_r.nii.gz | sub-XX_acq-T1w_MTS_crop_r_seg.nii.gz|
| sub-XX_T1w_reg.nii.gz | sub-XX_T1w_reg_seg.nii.gz |
| sub-XX_T2w_reg.nii.gz | sub-XX_T2w_reg_seg.nii.gz |
| sub-XX_T2star_mean_reg.nii.gz | sub-XX_T2star_mean_reg_seg.nii.gz |

- Open the segmentation with `fsleyes`
- Manually correct it:
  - If the segmentation is leaking, remove the leak (use CMD+F to switch the overlay on/off)
  - If the segmentation exists in one slice but only consists of a few pixels, because the image quality is bad or because it is no more covering the cord (e.g. brainstem), remove all pixels in the current slice (better to have no segmentation than partial segmentation).
  - If the spinal cord is only partially visible (this can happen in T2star scans due to the registration), zero all pixels in the slice.
- Save with suffix `-manual`.
- Move to a folder named seg_manual/$FILENAME. E.g.: `~/data-multi-subject/derivatives/seg_manual/sub-amu01_acq-T1w_MTS_crop_r_seg-manual.nii.gz`

#### Exclude images

If some images are of unacceptable quality, they could be excluded from the final output dataset. List images to exclude in **config.yml** using the field `exclude-list`. 

#### Re-run processing (using manually-corrected segmentations)

Make sure to place your manually-corrected segmentations in the directory specified by `config.yml`, then re-run:

~~~
sct_run_batch -script prepare_data.sh -config config.yml
~~~

#### Copy files, final QC

Copy final files to anat/, copy json sidecars, move segmentations to derivatives/ and generate another QC:

~~~
sct_run_batch -script final_qc.sh config.yml
~~~

- Open the new QC: qc2/index.html
- Make sure that:
  - the final segmentation properly overlays on each contrast,
  - there is no missing slice (can happen for t2s data),
  - each contrast has sufficient image quality.
- If you spot any problem, take a screenshot of the browser and copy screenshots under qc2_feedback/

#### Clean temporary files

Once QC and manual correction is done, remove tmp/ folder:

~~~
sct_run_batch -script delete_tmp_files.sh -config config.yml
~~~
