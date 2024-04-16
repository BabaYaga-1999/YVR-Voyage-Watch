import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore, storage
import datetime

image1 = "FallDetected\FallDetected_2024-04-11_16-48-35.png"
image2 = "FallDetected\FallDetected_2024-04-11_16-48-36.png"

# Initialize Firebase
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'finalproject-3896f.appspot.com'
})

# Get a reference to the Firestore service and Storage
db = firestore.client()
bucket = storage.bucket()


class Record:
    issue_types = {
        3: "motorcycle"
    }

    def __init__(self, area, issue_type, level):
        self.createdAt = datetime.datetime.now().strftime("%d %B %Y at %H:%M:%S %Z")
        self.area = area
        self.issueType = self.issue_types.get(issue_type, "Unknown Issue Type")
        self.level = level
        self.images = []
        self.claimBy = ""

    def add_image(self, image_path):
        self.images.append(image_path)

    def upload_record(self):
        # Upload images to Firebase Storage and save URLs
        image_urls = []
        for img in self.images:
            blob = bucket.blob(f"notificationImages/{img.split('/')[-1]}")
            blob.upload_from_filename(img)
            url = blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')
            image_urls.append(url)

        # Save data and image URLs to Firestore
        doc_ref = db.collection('notifications').document()
        doc_ref.set({
            "claimBy": self.claimBy,
            "createdAt": self.createdAt,
            "area": self.area,
            "issueType": self.issueType,
            "level": self.level,
            "images": image_urls
        })


record = Record("Downtown", 3, 1)  # '3' corresponds to "motorcycle"
record.add_image(image1)  
record.add_image(image2) 
record.upload_record()
