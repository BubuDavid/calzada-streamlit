import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import requests
import torch
from PIL import Image
from transformers import YolosForObjectDetection, YolosImageProcessor


def object_detection(image=None, url=None):
    if not image and not url:
        raise Exception("No detection available")
    if not image:
        image = Image.open(requests.get(url, stream=True).raw)

    model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")
    image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    logits = outputs.logits
    bboxes = outputs.pred_boxes

    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(
        outputs, threshold=0.9, target_sizes=target_sizes
    )[0]
    return results, image, model


def show_results(results, image, model):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    def draw_bbox(ax, bbox, obj_name):
        bbox = bbox.detach().cpu().numpy()
        x1, y1, x2, y2 = bbox
        color = obj_name[0]
        try:
            ax.add_line(mlines.Line2D([x1, x2], [y1, y1], color=color))
            ax.add_line(mlines.Line2D([x1, x1], [y1, y2], color=color))
            ax.add_line(mlines.Line2D([x1, x2], [y2, y2], color=color))
            ax.add_line(mlines.Line2D([x2, x2], [y1, y2], color=color))
        except:
            ax.add_line(mlines.Line2D([x1, x2], [y1, y1], color="r"))
            ax.add_line(mlines.Line2D([x1, x1], [y1, y2], color="r"))
            ax.add_line(mlines.Line2D([x1, x2], [y2, y2], color="r"))
            ax.add_line(mlines.Line2D([x2, x2], [y1, y2], color="r"))

    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        obj_name = model.config.id2label[label.item()]
        draw_bbox(ax, box, obj_name),
        plt.text(
            box[0],
            box[1],
            f"{obj_name}: {round(score.item(), 3)}",
            bbox=dict(facecolor="yellow", alpha=0.5),
        )

    return fig


if __name__ == "__main__":
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    results, image, model = object_detection(url=url)
    show_results(results=results, image=image, model=model)

