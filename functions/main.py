import firebase_admin
from firebase_admin import firestore, storage
from firebase_functions import https_fn, options

# Initialize Firebase Admin SDK
firebase_admin.initialize_app()

@https_fn.on_request(
    memory=options.MemoryOption.GB_2,
    timeout_sec=300,
    cors=options.CorsOptions(cors_origins="*", cors_methods=["get", "post"])
)
def processViolationImage(req: https_fn.Request) -> https_fn.Response:
    """
    Processes a violation image.

    Args:
        req (https_fn.Request): The HTTP request object.

    Returns:
        A JSON response indicating success or failure.
    """
    if req.method != 'POST':
        return https_fn.Response('Method not allowed', status=405)

    try:
        # Get the image from the request
        image = req.files.get('image')
        if not image:
            return https_fn.Response('No image provided', status=400)

        # Upload the image to Firebase Storage
        bucket = storage.bucket()
        blob = bucket.blob(f"violations/{image.filename}")
        blob.upload_from_string(
            image.read(),
            content_type=image.content_type
        )

        # Get the image URL
        image_url = blob.public_url

        # Save the violation details to Firestore
        db = firestore.client()
        db.collection('violations').add({
            'image_url': image_url,
            'timestamp': firestore.SERVER_TIMESTAMP
        })

        return https_fn.Response('{"message": "Violation processed successfully"}', status=200, headers={"Content-Type": "application/json"})

    except Exception as e:
        return https_fn.Response(f'{"error": "An unexpected error occurred: {str(e)}"}', status=500, headers={"Content-Type": "application/json"})
