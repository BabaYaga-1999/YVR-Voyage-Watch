import datetime

class Record:
    # Class dictionary mapping issue type codes to names
    issue_types = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
        5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
        10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird",
        15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
        20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
        25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
        30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
        35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
        40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon",
        45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
        50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut",
        55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
        60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
        65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven",
        70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
        75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush"
    }

    def __init__(self, area, issue_type, level):
        self.created_at = datetime.datetime.now().strftime("%d %B %Y at %H:%M:%S %Z")
        self.area = area
        self.issue_type = self.issue_types.get(issue_type, "Unknown Issue Type")
        self.level = level
        self.images = []

    def add_image(self, image_path):
        self.images.append(image_path)

    def __str__(self):
        return (f"Record created at {self.created_at}, Area: {self.area}, "
                f"Issue Type: {self.issue_type}, Level: {self.level}, "
                f"Images: {self.images}")

# Example usage
record = Record("Downtown", 3, 2)  # '3' corresponds to "motorcycle"
record.add_image("image1.jpg")
record.add_image("image2.jpg")

print(record)