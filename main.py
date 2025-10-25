from ajisai.data import AjisaiTransform, AMDIMDataset, create_fewshot_split

transform = AjisaiTransform()
fewshot_info = create_fewshot_split(
    src_dir="/kaggle/input/kengo-k423/MangoLeafBD Dataset",
    output_dir="/kaggle/working/fewshot_dataset"
)