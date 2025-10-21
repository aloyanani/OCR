import json
import time
from argparse import ArgumentParser
import os
import csv

import numpy as np
import torch.cuda
from pdf2image import convert_from_path
from PIL import Image

from armenian_ocr import OcrWrapper

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-dd", "--detection_dir", type=str, help="Path to the detection model directory.")
    parser.add_argument("-rd", "--recognition_dir", type=str, help="Path to the recognition model directory.")
    parser.add_argument("-d", "--document_path", type=str, help="Path to the document (image or PDF).")
    parser.add_argument("-o", "--output_path", type=str, help="Path to the output JSON file.")
    parser.add_argument("-l", "--layout", action="store_true", help="Detect layout.")
    parser.add_argument("-t", "--timer", action="store_true", help="Show processing times.")
    parser.add_argument("-cuda", "--cuda", action="store_true", help="Use cuda.")
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    ocr = OcrWrapper()
    ocr.load(
        det_model_dir=args.detection_dir,
        rec_model_dir=args.recognition_dir,
        device=device,
    )

    start = time.time()

    document_extension = args.document_path.split(".")[-1].lower()

    # Prepare list of images and names
    image_infos = []
    if document_extension == "pdf":
        pil_images = convert_from_path(args.document_path)
        for i, pil_image in enumerate(pil_images):
            np_image = np.array(pil_image)
            image_name = f"{os.path.splitext(os.path.basename(args.document_path))[0]}_page_{i+1}.png"
            image_infos.append((np_image, image_name))
    elif document_extension in ["png", "jpg", "jpeg"]:
        pil_image = Image.open(args.document_path)
        np_image = np.array(pil_image)

        if np_image.shape[-1] == 4:  # handle alpha channel
            pil_image.load()
            blended_image = Image.new(mode="RGB", size=pil_image.size, color=(255, 255, 255))
            blended_image.paste(im=pil_image, mask=pil_image.split()[3])
            np_image = np.array(blended_image)

        image_name = os.path.basename(args.document_path)
        image_infos.append((np_image, image_name))
    else:
        raise NotImplementedError(f"Documents with {document_extension} extension are not supported")

    # Prepare CSV
    csv_file = "image_heatmap_mapping.csv"
    os.makedirs("heatmaps", exist_ok=True)
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_file", "heatmap_file"])

        predictions_all = []
        for np_image, image_name in image_infos:
            # Pass image name to predict()
            boxes, heatmap_file = ocr.predict(
                image=np_image, 
                predict_layout=args.layout, 
                timer=args.timer
            )

            predictions_all.append(boxes)
            writer.writerow([image_name, heatmap_file])

    print(f"OCR process took {time.time() - start:.2f} seconds")
    # Save JSON predictions
    with open(args.output_path, "w", encoding="utf-8") as fp:
        json.dump(predictions_all, fp, ensure_ascii=False)
