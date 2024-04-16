import firebase_admin
from firebase_admin import credentials, firestore, storage
import datetime
from  models.record import Record

# Firebase initialization
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'finalproject-3896f.appspot.com'
})

# Get a reference to the Firestore service and Storage
db = firestore.client()
bucket = storage.bucket()

def upload_record(record):
    # Upload images to Firebase Storage and save URLs
    image_urls = []
    for img in record.images:
        blob = bucket.blob(f"notificationImages/{img.split('/')[-1]}")
        blob.upload_from_filename(img)
        url = blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')
        image_urls.append(url)

    # Save data and image URLs to Firestore
    doc_ref = db.collection('notifications').document()
    doc_ref.set({
        "claimBy": record.claimBy,
        "createdAt": record.createdAt,
        "area": record.area,
        "issueType": record.issueType,
        "level": record.level,
        "images": image_urls
    })

# Example usage
image1 = "FallDetected\\FallDetected_2024-04-11_16-48-35.png"
image2 = "FallDetected\\FallDetected_2024-04-11_16-48-36.png"
record = Record("Downtown", 3, 2)
record.add_image(image1)
record.add_image(image2)
upload_record(record)
